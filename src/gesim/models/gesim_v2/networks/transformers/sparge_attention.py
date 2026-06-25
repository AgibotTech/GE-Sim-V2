"""SpargeAttn sparse attention integration for Cosmos multi-view transformer.

Plug-and-play replacement of F.scaled_dot_product_attention with
spas_sage2_attn_meansim_topk_cuda from the SpargeAttn library.
Only attn1 (self-attention) processors are replaced; attn2 (cross-attention
with text encoder, which uses attention_mask) is left untouched.
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange

from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb

from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
from spas_sage_attn.core import spas_sage2_attn_meansim_cuda
import spas_sage_attn.core as _spas_core

logger = logging.getLogger(__name__)

# RTX 4090 (SM 8.9) falls through SpargeAttn's arch dispatch to an FP8 path
# that crashes when Sage2++ is unavailable.  Force SM 8.9 → SM 8.6 so the
# library uses the proven Ampere (f16-accumulation) kernel instead.
_orig_get_cuda_arch_versions = _spas_core.get_cuda_arch_versions

def _patched_get_cuda_arch_versions():
    return [("sm86" if a == "sm89" else a)
            for a in _orig_get_cuda_arch_versions()]

_spas_core.get_cuda_arch_versions = _patched_get_cuda_arch_versions


class SpargeAttentionState:
    """Global mutable state read by every SpargeAttn processor."""
    enabled: bool = False
    current_step: int = 0
    config: dict = {}


class MultiViewCosmosSpasAttnProcessor2_0:
    """Drop-in replacement for MultiViewCosmosAttnProcessor2_0 that routes
    the attention call through SpargeAttn when conditions are met.
    """

    def __init__(self, layer_idx: int = 0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Requires PyTorch 2.0+.")
        self.layer_idx = layer_idx

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

        query_idx = torch.tensor(query.size(3), device=query.device)
        key_idx = torch.tensor(key.size(3), device=key.device)
        value_idx = torch.tensor(value.size(3), device=value.device)
        key = key.repeat_interleave(query_idx // key_idx, dim=3)
        value = value.repeat_interleave(query_idx // value_idx, dim=3)

        # --- attention dispatch ---
        hidden_states = self._dispatch_attention(
            query, key, value,
            attention_mask=attention_mask,
            cross_view_attn=cross_view_attn,
        )

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).type_as(query)

        if cross_view_attn or image_rotary_emb is None:
            hidden_states = rearrange(hidden_states, 'b (v l) c -> (b v) l c', v=n_view)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    def _dispatch_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cross_view_attn: bool,
    ) -> torch.Tensor:
        """Choose between SpargeAttn sparse kernel and standard SDPA.

        When SpargeAttn is enabled, incompatible conditions raise errors
        instead of silently falling back to SDPA (except for the intentional
        dense_block / dense_timestep / skip_cross_view fallbacks).
        """
        cfg = SpargeAttentionState.config
        seq_len = query.size(2)  # (B, H, N, D) in HND layout
        head_dim = query.size(3)

        dense_fallback = (
            self.layer_idx < cfg.get("dense_block", 0)
            or SpargeAttentionState.current_step < cfg.get("dense_timestep", 0)
            or (cross_view_attn and cfg.get("skip_cross_view", False))
        )

        if dense_fallback:
            return F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            )

        if not query.is_cuda:
            raise RuntimeError(
                f"[SpargeAttn] layer {self.layer_idx}: query tensor is not on CUDA. "
                f"SpargeAttn requires CUDA tensors."
            )
        if attention_mask is not None:
            raise RuntimeError(
                f"[SpargeAttn] layer {self.layer_idx}: attention_mask is not None. "
                f"SpargeAttn does not support attention masks."
            )
        if seq_len < 128:
            raise RuntimeError(
                f"[SpargeAttn] layer {self.layer_idx}: seq_len={seq_len} < 128. "
                f"SpargeAttn requires seq_len >= 128."
            )
        if head_dim not in (64, 128):
            raise RuntimeError(
                f"[SpargeAttn] layer {self.layer_idx}: head_dim={head_dim} not in (64, 128). "
                f"SpargeAttn requires head_dim of 64 or 128."
            )

        mode = cfg.get("mode", "topk")
        q_c = query.contiguous()
        k_c = key.contiguous()
        v_c = value.contiguous()
        if mode == "cdfthreshd":
            return spas_sage2_attn_meansim_cuda(
                q_c, k_c, v_c,
                simthreshd1=-0.1,
                cdfthreshd=cfg.get("cdfthreshd", 0.1),
                tensor_layout="HND",
            )
        else:
            return spas_sage2_attn_meansim_topk_cuda(
                q_c, k_c, v_c,
                topk=cfg.get("topk", 0.5),
                tensor_layout="HND",
            )


def enable_sparge_attention(transformer, sparge_config: dict) -> None:
    """Patch all attn1 (self-attention) processors with SpargeAttn processors."""
    count = 0
    for idx, block in enumerate(transformer.transformer_blocks):
        if hasattr(block, "attn1") and hasattr(block.attn1, "set_processor"):
            block.attn1.set_processor(MultiViewCosmosSpasAttnProcessor2_0(layer_idx=idx))
            count += 1

    SpargeAttentionState.enabled = True
    SpargeAttentionState.config = sparge_config
    SpargeAttentionState.current_step = 0
    mode = sparge_config.get('mode', 'topk')
    if mode == 'cdfthreshd':
        sparsity_str = f"cdfthreshd={sparge_config.get('cdfthreshd', 0.1)}"
    else:
        sparsity_str = f"topk={sparge_config.get('topk', 0.5)}"
    logger.info(
        "[SpargeAttn] Patched %d attn1 processors  |  mode=%s  %s  "
        "dense_block=%s  dense_timestep=%s  skip_cross_view=%s",
        count,
        mode,
        sparsity_str,
        sparge_config.get("dense_block", 0),
        sparge_config.get("dense_timestep", 0),
        sparge_config.get("skip_cross_view", False),
    )
