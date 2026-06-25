"""Triton-fused RoPE for the Cosmos "split-half" variant.

Replaces ``diffusers.models.embeddings.apply_rotary_emb`` (with
``use_real=True, use_real_unbind_dim=-2``) with a single fused Triton kernel
that avoids intermediate fp32 allocations and multiple kernel launches.

The split-half rotation for element index *i* with half_dim = D/2::

    i <  D/2:  out[i] = x[i] * cos[i] - x[i + D/2] * sin[i]
    i >= D/2:  out[i] = x[i] * cos[i] + x[i - D/2] * sin[i]

Usage::

    from gesim.models.gesim_v2.networks.transformers.triton_rope import enable_triton_rope
    enable_triton_rope()    # patch all attention processor modules
    disable_triton_rope()   # restore originals
"""

import inspect
import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def _diffusers_apply_rotary_emb_fallback(
    x,
    freqs_cis,
    *,
    use_real: bool,
    use_real_unbind_dim: int,
    sequence_dim: int,
):
    """Call diffusers' ``apply_rotary_emb`` with only kwargs its version supports.

    Older diffusers builds omit ``sequence_dim``; passing it raises
    ``TypeError: ... unexpected keyword argument 'sequence_dim'``.
    """
    from diffusers.models.embeddings import apply_rotary_emb as fn

    kwargs = {
        "use_real": use_real,
        "use_real_unbind_dim": use_real_unbind_dim,
    }
    if "sequence_dim" in inspect.signature(fn).parameters:
        kwargs["sequence_dim"] = sequence_dim
    return fn(x, freqs_cis, **kwargs)


@triton.jit
def _rope_cosmos_kernel(
    X, OUT, COS, SIN,
    seq_len,
    stride_x,
    stride_cos,
    HD: tl.constexpr,
    HALF_HD: tl.constexpr,
):
    row = tl.program_id(0)
    seq = row % seq_len

    offs = tl.arange(0, HD)
    x_base = X + row * stride_x
    out_base = OUT + row * stride_x
    cos_base = COS + seq * stride_cos
    sin_base = SIN + seq * stride_cos

    x_val = tl.load(x_base + offs).to(tl.float32)
    cos_val = tl.load(cos_base + offs).to(tl.float32)
    sin_val = tl.load(sin_base + offs).to(tl.float32)

    is_first = offs < HALF_HD
    partner_offs = tl.where(is_first, offs + HALF_HD, offs - HALF_HD)
    partner = tl.load(x_base + partner_offs).to(tl.float32)
    x_rot = tl.where(is_first, -partner, partner)

    result = x_val * cos_val + x_rot * sin_val
    tl.store(out_base + offs, result)


def triton_apply_rotary_emb(
    x, freqs_cis, use_real=True, use_real_unbind_dim=-1, sequence_dim=2,
):
    """Drop-in replacement for ``diffusers.models.embeddings.apply_rotary_emb``.

    The Cosmos split-half variant (use_real=True, use_real_unbind_dim=-2,
    sequence_dim=2) is routed to a fused Triton kernel.  All other parameter
    combinations fall back transparently to the original PyTorch implementation.
    """
    if not (use_real and use_real_unbind_dim == -2 and sequence_dim == 2 and x.is_cuda):
        return _diffusers_apply_rotary_emb_fallback(
            x,
            freqs_cis,
            use_real=use_real,
            use_real_unbind_dim=use_real_unbind_dim,
            sequence_dim=sequence_dim,
        )

    cos, sin = freqs_cis  # each [S, D]
    cos = cos.to(x.device).contiguous()
    sin = sin.to(x.device).contiguous()

    *_, S, D = x.shape
    if D not in (64, 128):
        return _diffusers_apply_rotary_emb_fallback(
            x,
            freqs_cis,
            use_real=use_real,
            use_real_unbind_dim=use_real_unbind_dim,
            sequence_dim=sequence_dim,
        )

    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    total_rows = x_c.numel() // D

    _rope_cosmos_kernel[(total_rows,)](
        x_c, out, cos, sin,
        S,
        D,              # stride between rows (contiguous)
        cos.stride(0),
        HD=D,
        HALF_HD=D // 2,
        num_warps=4,
    )

    return out.view_as(x)


_MODULES_TO_PATCH = [
    "gesim.models.gesim_v2.networks.transformers.sparge_attention",
    "gesim.models.gesim_v2.networks.transformers.transformer_cosmos_multiview_PE",
]


def enable_triton_rope() -> int:
    """Monkey-patch ``apply_rotary_emb`` in all attention processor modules."""
    import importlib

    count = 0
    for name in _MODULES_TO_PATCH:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(mod, "apply_rotary_emb"):
            if not hasattr(mod, "_orig_apply_rotary_emb"):
                mod._orig_apply_rotary_emb = mod.apply_rotary_emb
            mod.apply_rotary_emb = triton_apply_rotary_emb
            count += 1
    logger.info("[TritonRoPE] Patched %d modules with Triton-fused RoPE.", count)
    return count


def disable_triton_rope() -> int:
    """Restore original ``apply_rotary_emb`` in all previously patched modules."""
    import importlib

    count = 0
    for name in _MODULES_TO_PATCH:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(mod, "_orig_apply_rotary_emb"):
            mod.apply_rotary_emb = mod._orig_apply_rotary_emb
            del mod._orig_apply_rotary_emb
            count += 1
    logger.info("[TritonRoPE] Restored %d modules to original RoPE.", count)
    return count
