import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from diffusers.models.attention import FeedForward
from diffusers.models.normalization import RMSNorm
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention_processor import Attention


class PoseRotaryPosEmbed(nn.Module):
    """1D rotary embedding for pose sequences."""

    def __init__(self, dim: int, base_seq_length: int = 57, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.base_seq_length = base_seq_length
        self.theta = theta

    def forward(self, hidden_states: torch.Tensor, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        grid = torch.arange(seq_length, dtype=torch.float32, device=hidden_states.device)
        grid = grid / self.base_seq_length
        grid = grid.unsqueeze(-1)

        start = 1.0
        end = self.theta
        freqs = self.theta ** torch.linspace(
            math.log(start, self.theta),
            math.log(end, self.theta),
            self.dim // 2,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        freqs = freqs * math.pi / 2.0
        freqs = freqs * (grid * 2 - 1)

        cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

        if self.dim % 2 != 0:
            cos_padding = torch.ones_like(cos_freqs[:, : self.dim % 2])
            sin_padding = torch.zeros_like(sin_freqs[:, : self.dim % 2])
            cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
            sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

        return cos_freqs, sin_freqs


@maybe_allow_in_graph
class PoseTransformerBlock(nn.Module):
    """Transformer block for pose prediction branch."""

    def __init__(
        self,
        dim: int = 512,
        num_attention_heads: int = 16,
        attention_head_dim: int = 32,
        cross_attention_dim: int = 2048,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        processor: Optional[Attention] = None,
    ) -> None:
        super().__init__()

        attn_processor = processor if processor is not None else None

        self.norm1 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            processor=attn_processor,
        )

        self.norm2 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            processor=attn_processor,
        )

        self.ff = FeedForward(dim, activation_fn=activation_fn)
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        n_view: int = 1,
    ) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        norm_hidden_states = self.norm1(hidden_states)

        num_ada_params = self.scale_shift_table.shape[0]
        ada_values = self.scale_shift_table[None, None] + temb.reshape(
            batch_size, temb.size(1), num_ada_params, -1
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)

        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            image_rotary_emb=rotary_emb,
            n_view=n_view,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa

        attn_encoder_hidden_states = encoder_hidden_states

        attn_hidden_states = self.attn2(
            hidden_states,
            encoder_hidden_states=attn_encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            n_view=n_view,
        )
        hidden_states = hidden_states + attn_hidden_states

        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp

        return hidden_states
