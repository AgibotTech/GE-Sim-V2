"""
Triton-based norm replacement via Liger Kernel.

Replaces diffusers RMSNorm (used for QK-norm in every attention block and in
CosmosEmbedding) with LigerRMSNorm, which runs a fused Triton kernel instead
of the PyTorch reference loop.

Optionally (fuse_layernorm=True) also replaces nn.LayerNorm(elementwise_affine=False)
instances (used inside CosmosAdaLayerNorm / CosmosAdaLayerNormZero) with
LigerLayerNorm using a frozen all-ones weight as an identity affine, keeping
the same numerical behaviour while benefiting from the fused Triton kernel.

Usage::

    from gesim.models.gesim_v2.networks.transformers.liger_norms import enable_liger_norms
    n = enable_liger_norms(transformer)                      # RMSNorm only
    n = enable_liger_norms(transformer, fuse_layernorm=True) # + LayerNorm
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def enable_liger_norms(model: nn.Module, fuse_layernorm: bool = False) -> int:
    """
    Walk *model* and replace compatible norm modules with Triton equivalents.

    Replacements performed
    ----------------------
    - ``diffusers.models.normalization.RMSNorm`` (elementwise_affine=True)
      → ``liger_kernel.transformers.LigerRMSNorm``
    - ``nn.LayerNorm`` (elementwise_affine=False)  [only when fuse_layernorm=True]
      → ``liger_kernel.transformers.LigerLayerNorm`` with frozen all-ones weight

    Parameters
    ----------
    model : nn.Module
        The transformer model to patch in-place.
    fuse_layernorm : bool, optional
        When True, also replace affine-free LayerNorm modules (e.g. inside
        CosmosAdaLayerNorm / CosmosAdaLayerNormZero) with LigerLayerNorm.
        Defaults to False.

    Returns
    -------
    int
        Number of modules replaced.

    Raises
    ------
    ImportError
        If ``liger-kernel`` is not installed.
    """
    try:
        from liger_kernel.transformers import LigerRMSNorm
    except ImportError:
        raise ImportError(
            "liger-kernel is required for Triton norm acceleration. "
            "Install with:  pip install liger-kernel"
        )

    LigerLayerNorm = None
    if fuse_layernorm:
        try:
            from liger_kernel.transformers import LigerLayerNorm
        except ImportError:
            logger.warning(
                "[Liger] LigerLayerNorm not available in this liger-kernel version; "
                "skipping LayerNorm fusion."
            )

    try:
        from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm
    except ImportError:
        DiffusersRMSNorm = None

    if DiffusersRMSNorm is None:
        logger.warning("[Liger] diffusers RMSNorm not found, nothing to replace.")
        return 0

    rms_count, ln_count = _replace_recursive(model, LigerRMSNorm, DiffusersRMSNorm, LigerLayerNorm)
    logger.info("[Liger] Replaced %d RMSNorm modules with Triton kernels.", rms_count)
    if fuse_layernorm:
        logger.info("[Liger] Replaced %d LayerNorm modules with Triton kernels.", ln_count)
    return rms_count + ln_count


def _replace_recursive(
    module: nn.Module,
    LigerRMSNorm,
    DiffusersRMSNorm,
    LigerLayerNorm,
):
    rms_count = 0
    ln_count = 0
    for name, child in list(module.named_children()):
        # Already a Liger module – skip
        if LigerRMSNorm is not None and isinstance(child, LigerRMSNorm):
            continue
        if LigerLayerNorm is not None and isinstance(child, LigerLayerNorm):
            continue

        if isinstance(child, DiffusersRMSNorm) and child.elementwise_affine:
            hidden_size = child.dim[0]
            new_norm = LigerRMSNorm(hidden_size, eps=child.eps)
            # Move to same device/dtype BEFORE copying weight to avoid CPU Triton error
            new_norm = new_norm.to(device=child.weight.device, dtype=child.weight.dtype)
            new_norm.weight.data.copy_(child.weight.data)
            setattr(module, name, new_norm)
            rms_count += 1
        elif (
            LigerLayerNorm is not None
            and isinstance(child, nn.LayerNorm)
            and not child.elementwise_affine
        ):
            hidden_size = child.normalized_shape[0]
            new_norm = LigerLayerNorm(hidden_size, eps=child.eps, bias=False)
            # Identity scale: keep numerical equivalence to affine-free LayerNorm
            new_norm.weight.data.fill_(1.0)
            new_norm.weight.requires_grad_(False)
            new_norm = new_norm.to(device=next(module.parameters(), torch.tensor(0.0)).device,
                                   dtype=next(module.parameters(), torch.tensor(0.0)).dtype)
            setattr(module, name, new_norm)
            ln_count += 1
        else:
            r, l = _replace_recursive(child, LigerRMSNorm, DiffusersRMSNorm, LigerLayerNorm)
            rms_count += r
            ln_count += l

    return rms_count, ln_count
