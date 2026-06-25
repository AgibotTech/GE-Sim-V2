
from typing import Optional, Tuple, Callable
from einops import rearrange

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torchvision import transforms

from diffusers.utils import is_torch_version, logging
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.attention import FeedForward
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.normalization import AdaLayerNormSingle

from .transformer_cosmos import (
    CosmosAdaLayerNormZero,
    CosmosEmbedding,
    CosmosPatchEmbed,
    CosmosLearnablePositionalEmbed,
    CosmosAdaLayerNorm,
    CosmosRotaryPosEmbed,
)
from .pose_expert import PoseTransformerBlock, PoseRotaryPosEmbed

logger = logging.get_logger(__name__)


class MultiViewCosmosAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CosmosAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        n_view: int = 1,
        cross_view_attn: bool = False,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)
            if cross_view_attn:
                query = rearrange(query, '(b v) n l c -> b n (v l) c', v=n_view)
                key = rearrange(key, '(b v) n l c -> b n (v l) c', v=n_view)
                value = rearrange(value, '(b v) n l c -> b n (v l) c', v=n_view)
        else:
            query = rearrange(query, '(b v) n l c -> b n (v l) c', v=n_view)


        # 4. Prepare for GQA
        query_idx = torch.tensor(query.size(3), device=query.device)
        key_idx = torch.tensor(key.size(3), device=key.device)
        value_idx = torch.tensor(value.size(3), device=value.device)
        key = key.repeat_interleave(query_idx // key_idx, dim=3)
        value = value.repeat_interleave(query_idx // value_idx, dim=3)
        
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).type_as(query)

        if cross_view_attn or image_rotary_emb is None:
            hidden_states = rearrange(hidden_states, 'b (v l) c -> (b v) l c', v=n_view)

        # 6. Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class CosmosTransformerBlock(nn.Module):
    '''
    multi-view version of CosmosTransformerBlock
    '''
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        mlp_ratio: float = 4.0,
        adaln_lora_dim: int = 256,
        qk_norm: str = "rms_norm",
        out_bias: bool = False,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = CosmosAdaLayerNormZero(in_features=hidden_size, hidden_features=adaln_lora_dim)
        self.attn1 = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qk_norm=qk_norm,
            elementwise_affine=True,
            out_bias=out_bias,
            # processor=CosmosAttnProcessor2_0(),
            processor=MultiViewCosmosAttnProcessor2_0(),
        )

        self.norm2 = CosmosAdaLayerNormZero(in_features=hidden_size, hidden_features=adaln_lora_dim)
        self.attn2 = Attention(
            query_dim=hidden_size,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qk_norm=qk_norm,
            elementwise_affine=True,
            out_bias=out_bias,
            # processor=CosmosAttnProcessor2_0(),
            processor=MultiViewCosmosAttnProcessor2_0(),
        )

        self.norm3 = CosmosAdaLayerNormZero(in_features=hidden_size, hidden_features=adaln_lora_dim)
        self.ff = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu", bias=out_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        extra_pos_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        n_view: int = 1,
        cross_view_attn: bool = False,
    ) -> torch.Tensor:
        if extra_pos_emb is not None:
            hidden_states = hidden_states + extra_pos_emb

        # 1. Self Attention
        norm_hidden_states, gate = self.norm1(hidden_states, embedded_timestep, temb)
        attn_output = self.attn1(norm_hidden_states, 
                                 image_rotary_emb=image_rotary_emb,
                                 n_view=n_view,
                                 cross_view_attn=cross_view_attn
            )
        hidden_states = hidden_states + gate * attn_output

        # 2. Cross Attention
        norm_hidden_states, gate = self.norm2(hidden_states, embedded_timestep, temb)
        attn_output = self.attn2(norm_hidden_states, 
                                 encoder_hidden_states=encoder_hidden_states,
                                 attention_mask=attention_mask,
                                 n_view=n_view
            )
        hidden_states = hidden_states + gate * attn_output

        # 3. Feed Forward
        norm_hidden_states, gate = self.norm3(hidden_states, embedded_timestep, temb)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate * ff_output

        return hidden_states


class MultiViewCosmosTransformer3DModelState(ModelMixin, ConfigMixin):
    r"""
    A Transformer model for video-like data used in [Cosmos](https://github.com/NVIDIA/Cosmos).

    Args:
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        num_attention_heads (`int`, defaults to `32`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each attention head.
        num_layers (`int`, defaults to `28`):
            The number of layers of transformer blocks to use.
        mlp_ratio (`float`, defaults to `4.0`):
            The ratio of the hidden layer size to the input size in the feedforward network.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        adaln_lora_dim (`int`, defaults to `256`):
            The hidden dimension of the Adaptive LayerNorm LoRA layer.
        max_size (`Tuple[int, int, int]`, defaults to `(128, 240, 240)`):
            The maximum size of the input latent tensors in the temporal, height, and width dimensions.
        patch_size (`Tuple[int, int, int]`, defaults to `(1, 2, 2)`):
            The patch size to use for patchifying the input latent tensors in the temporal, height, and width
            dimensions.
        rope_scale (`Tuple[float, float, float]`, defaults to `(2.0, 1.0, 1.0)`):
            The scaling factor to use for RoPE in the temporal, height, and width dimensions.
        concat_padding_mask (`bool`, defaults to `True`):
            Whether to concatenate the padding mask to the input latent tensors.
        extra_pos_embed_type (`str`, *optional*, defaults to `learnable`):
            The type of extra positional embeddings to use. Can be one of `None` or `learnable`.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embed", "final_layer", "norm"]
    _no_split_modules = ["CosmosTransformerBlock"]
    _keep_in_fp32_modules = ["learnable_pos_embed"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        num_layers: int = 28,
        mlp_ratio: float = 4.0,
        text_embed_dim: int = 1024,
        adaln_lora_dim: int = 256,
        max_size: Tuple[int, int, int] = (128, 240, 240),
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        rope_scale: Tuple[float, float, float] = (2.0, 1.0, 1.0),
        concat_padding_mask: bool = True,
        extra_pos_embed_type: Optional[str] = "learnable",
        use_view_embed: bool = True,
        pose_expert: bool = False,
        pose_in_channels: int = 64,
        pose_num_attention_heads: int = 16,
        pose_attention_head_dim: int = 32,
    ) -> None:
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim

        # 1. Patch Embedding
        patch_embed_in_channels = in_channels + 1 if concat_padding_mask else in_channels  # stupid design
        self.patch_embed = CosmosPatchEmbed(patch_embed_in_channels, hidden_size, patch_size, bias=False)

        # 2. Positional Embedding
        self.rope = CosmosRotaryPosEmbed(
            hidden_size=attention_head_dim, max_size=max_size, patch_size=patch_size, rope_scale=rope_scale
        )

        self.learnable_pos_embed = None
        if extra_pos_embed_type == "learnable":
            self.learnable_pos_embed = CosmosLearnablePositionalEmbed(
                hidden_size=hidden_size,
                max_size=max_size,
                patch_size=patch_size,
            )

        # 3. Time Embedding
        self.time_embed = CosmosEmbedding(hidden_size, hidden_size)

        self.use_view_embed = use_view_embed
        if self.use_view_embed:
            max_view = 3
            self.view_embed = nn.Parameter(torch.randn(max_view, hidden_size))
            self.view_ada = nn.Sequential(
                                            nn.SiLU(),
                                            # nn.Linear(hidden_size, 6 * hidden_size, bias=True)
                                            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
                                        )

        # 4. Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CosmosTransformerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=text_embed_dim,
                    mlp_ratio=mlp_ratio,
                    adaln_lora_dim=adaln_lora_dim,
                    qk_norm="rms_norm",
                    out_bias=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.video_buffer_weights = nn.Parameter(torch.ones(num_layers))
        self.video_buffer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        # 5. Output norm & projection
        self.norm_out = CosmosAdaLayerNorm(hidden_size, adaln_lora_dim)
        self.proj_out = nn.Linear(
            hidden_size, patch_size[0] * patch_size[1] * patch_size[2] * out_channels, bias=False
        )

        self.gradient_checkpointing = False

        self.pose_expert = pose_expert
        if self.pose_expert:
            self.pose_inner_dim = pose_num_attention_heads * pose_attention_head_dim
            self.pose_proj_in = nn.Linear(pose_in_channels, self.pose_inner_dim)
            self.pose_scale_shift_table = nn.Parameter(
                torch.randn(2, self.pose_inner_dim) / self.pose_inner_dim**0.5
            )
            self.pose_time_embed = AdaLayerNormSingle(self.pose_inner_dim, use_additional_conditions=False)
            self.pose_rope = PoseRotaryPosEmbed(dim=pose_attention_head_dim, base_seq_length=57)
            self.pose_blocks = nn.ModuleList(
                [
                    PoseTransformerBlock(
                        dim=self.pose_inner_dim,
                        num_attention_heads=pose_num_attention_heads,
                        attention_head_dim=pose_attention_head_dim,
                        cross_attention_dim=hidden_size,
                        qk_norm="rms_norm",
                        activation_fn="gelu-approximate",
                        attention_bias=True,
                        attention_out_bias=True,
                        processor=MultiViewCosmosAttnProcessor2_0(),
                    )
                    for _ in range(num_layers)
                ]
            )
            self.pose_norm_out = nn.LayerNorm(self.pose_inner_dim, eps=1e-6, elementwise_affine=False)
            self.pose_proj_out = nn.Linear(self.pose_inner_dim, pose_in_channels)

        self._pose_only_forward = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        fps: Optional[int] = None,
        condition_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        n_view: int = 1,
        return_pose: bool = False,
        pose_states: Optional[torch.Tensor] = None,
        pose_timestep: Optional[torch.Tensor] = None,
        pose_mask: Optional[torch.Tensor] = None,
        use_ve: Optional[bool] = False,
        store_buffer: Optional[bool] = False,
        video_pose_buffer: Optional[list] = None,
        compute_video_trunk: Optional[bool] = None,
        is_inference: Optional[bool] = True,
    ) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        # 1. Concatenate padding mask if needed & prepare attention mask
        if condition_mask is not None:
            hidden_states = torch.cat([hidden_states, condition_mask], dim=1)

        if self.config.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(hidden_states.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            hidden_states = torch.cat(
                [hidden_states, padding_mask.unsqueeze(2).repeat(batch_size, 1, num_frames, 1, 1)], dim=1
            )

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, S]

        # 2. Generate positional embeddings
        image_rotary_emb = self.rope(hidden_states, fps=fps)
        extra_pos_emb = self.learnable_pos_embed(hidden_states) if self.config.extra_pos_embed_type else None

        # 3. Patchify input
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states.flatten(1, 3)  # [B, T, H, W, C] -> [B, THW, C]

        if store_buffer:
            video_pose_buffer = []
            fused_feature = None
        # 4. Timestep embeddings
        if timestep.ndim == 1:
            temb, embedded_timestep = self.time_embed(hidden_states, timestep)
        elif timestep.ndim == 5:
            assert timestep.shape == (batch_size, 1, num_frames, 1, 1), (
                f"Expected timestep to have shape [B, 1, T, 1, 1], but got {timestep.shape}"
            )
            timestep = timestep.flatten()
            temb, embedded_timestep = self.time_embed(hidden_states, timestep)
            # We can do this because num_frames == post_patch_num_frames, as p_t is 1
            temb, embedded_timestep = (
                x.view(batch_size, post_patch_num_frames, 1, 1, -1)
                .expand(-1, -1, post_patch_height, post_patch_width, -1)
                .flatten(1, 3)
                for x in (temb, embedded_timestep)
            )  # [BT, C] -> [B, T, 1, 1, C] -> [B, T, H, W, C] -> [B, THW, C]
        else:
            assert False

        if self.use_view_embed:
            embedded_view = self.view_embed[:n_view].unsqueeze(0).repeat(batch_size//n_view, 1, 1) # b v c
            embedded_view = rearrange(embedded_view, 'b v c -> (b v) c').unsqueeze(1)
            vemb = self.view_ada(embedded_view)
            temb = temb + vemb

        pose_hidden_states = None
        pose_rotary_emb = None
        pose_temb = None
        pose_embedded_timestep = None
        if return_pose:
            pose_batch_size = pose_states.shape[0]
            pose_states = pose_states.to(hidden_states.dtype)
            pose_mask = pose_mask.to(hidden_states.dtype)
            if pose_mask is not None:
                pose_states = pose_states * pose_mask

            pose_seq_length = pose_states.shape[1]
            pose_hidden_states = self.pose_proj_in(pose_states)

            pose_rotary_emb = self.pose_rope(pose_states, pose_seq_length)

            pose_timestep = pose_timestep.to(hidden_states.device)
            pose_temb, pose_embedded_timestep = self.pose_time_embed(
                pose_timestep.flatten(),
                batch_size=pose_batch_size,
                hidden_dtype=pose_hidden_states.dtype,
            )
            pose_temb = pose_temb.view(pose_batch_size, -1, pose_temb.size(-1))
            pose_embedded_timestep = pose_embedded_timestep.view(
                pose_batch_size, -1, pose_embedded_timestep.size(-1)
            )

        # Ignore use_ve; always run video trunk (controlled by compute_video_trunk/store_buffer)
        # 5. Transformer blocks
        for block_idx, block in enumerate(self.transformer_blocks):
            # decide whether to compute video trunk this pass
            if self.gradient_checkpointing and not is_inference:
                if compute_video_trunk:
                    hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        embedded_timestep,
                        temb,
                        image_rotary_emb,
                        extra_pos_emb,
                        attention_mask,
                        n_view,
                        block_idx % 3 == 0,
                    )
                    if store_buffer:
                        weight = self.video_buffer_weights[block_idx].to(hidden_states.dtype)
                        if fused_feature is None:
                            fused_feature = hidden_states * weight
                        else:
                            fused_feature = fused_feature + hidden_states * weight
                else:
                    hidden_states = video_pose_buffer[0]
            else:
                if compute_video_trunk:
                    hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        embedded_timestep=embedded_timestep,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        extra_pos_emb=extra_pos_emb,
                        attention_mask=attention_mask,
                        n_view=n_view,
                        cross_view_attn=block_idx % 3 == 0
                    )
                    if store_buffer:
                        weight = self.video_buffer_weights[block_idx].to(hidden_states.dtype)
                        if fused_feature is None:
                            fused_feature = hidden_states * weight
                        else:
                            fused_feature = fused_feature + hidden_states * weight
                else:
                    hidden_states = video_pose_buffer[0]
    # 
        if return_pose:        
            for block_idx, block in enumerate(self.transformer_blocks):
                if len(video_pose_buffer) == 1:
                    pose_input_hidden = video_pose_buffer[0]

                pose_context = rearrange(pose_input_hidden, '(b v) l c -> b (v l) c', v=n_view)

                if self.gradient_checkpointing:
                    pose_hidden_states = self._gradient_checkpointing_func(
                        self.pose_blocks[block_idx],
                        pose_hidden_states,
                        pose_context,
                        pose_temb,
                        pose_rotary_emb,
                        None,
                    )
                else:
                    pose_hidden_states = self.pose_blocks[block_idx](
                        pose_hidden_states,
                        pose_context,
                        pose_temb,
                        rotary_emb=pose_rotary_emb,
                        encoder_attention_mask=None,
                    )

        # 6. Output norm & projection & unpatchify (can be skipped for pose-only second pass)
        skip_video_tail = (
            return_pose
            and not store_buffer
            and (video_pose_buffer is not None)
            and getattr(self, "_pose_only_forward", False)
            and self.training
        )

        final_output = {}
        if not skip_video_tail:
            hidden_states = self.norm_out(hidden_states, embedded_timestep, temb)
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.unflatten(2, (p_h, p_w, p_t, -1))
            hidden_states = hidden_states.unflatten(1, (post_patch_num_frames, post_patch_height, post_patch_width))
            # NOTE: The permutation order here is not the inverse operation of what happens when patching as usually expected.
            # It might be a source of confusion to the reader, but this is correct
            hidden_states = hidden_states.permute(0, 7, 1, 6, 2, 4, 3, 5)
            hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

            video_output = hidden_states
            final_output["video"] = video_output

        if store_buffer:        
            fused_feature = self.video_buffer_norm(fused_feature)
            video_pose_buffer.append(fused_feature)
            final_output['video_pose_buffer'] = video_pose_buffer

        if return_pose:
            pose_scale_shift_values = self.pose_scale_shift_table[None, None] + pose_embedded_timestep[:, :, None]
            pose_shift, pose_scale = pose_scale_shift_values[:, :, 0], pose_scale_shift_values[:, :, 1]
            pose_hidden_states = self.pose_norm_out(pose_hidden_states)
            pose_hidden_states = pose_hidden_states * (1 + pose_scale) + pose_shift
            pose_output = self.pose_proj_out(pose_hidden_states)
            if pose_mask is not None:
                pose_output = pose_output * pose_mask
            final_output["pose"] = pose_output

        if not return_dict:
            return (final_output,)

        return Transformer2DModelOutput(sample=final_output)

    def _set_gradient_checkpointing(
        self, enable: bool = True, gradient_checkpointing_func: Callable = torch.utils.checkpoint.checkpoint
    ) -> None:
        is_gradient_checkpointing_set = False

        for name, module in self.named_modules():
            if hasattr(module, "gradient_checkpointing"):
                logger.debug(f"Setting `gradient_checkpointing={enable}` for '{name}'")
                module._gradient_checkpointing_func = gradient_checkpointing_func
                module.gradient_checkpointing = enable
                is_gradient_checkpointing_set = True

        if not is_gradient_checkpointing_set:
            raise ValueError(
                f"The module {self.__class__.__name__} does not support gradient checkpointing. Please make sure to "
                f"use a module that supports gradient checkpointing by creating a boolean attribute `gradient_checkpointing`."
            )

    def enable_gradient_checkpointing(self, gradient_checkpointing_func: Optional[Callable] = None) -> None:
        """
        Activates gradient checkpointing for the current model (may be referred to as *activation checkpointing* or
        *checkpoint activations* in other frameworks).

        Args:
            gradient_checkpointing_func (`Callable`, *optional*):
                The function to use for gradient checkpointing. If `None`, the default PyTorch checkpointing function
                is used (`torch.utils.checkpoint.checkpoint`).
        """
        if not self._supports_gradient_checkpointing:
            raise ValueError(
                f"{self.__class__.__name__} does not support gradient checkpointing. Please make sure to set the boolean attribute "
                f"`_supports_gradient_checkpointing` to `True` in the class definition."
            )

        if gradient_checkpointing_func is None:

            def _gradient_checkpointing_func(module, *args):
                ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                return torch.utils.checkpoint.checkpoint(
                    module.__call__,
                    *args,
                    **ckpt_kwargs,
                )

            gradient_checkpointing_func = _gradient_checkpointing_func

        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)
