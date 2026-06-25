from typing import Dict, Optional, Union
import logging
import numpy as np
import random
from collections import OrderedDict

import torch
import torchvision
from torchvision import transforms
from accelerate import Accelerator
from diffusers.utils.torch_utils import is_compiled_module
from safetensors import safe_open
from einops import rearrange

logger = logging.getLogger(__name__)


def unwrap_model(accelerator: Accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def align_device_and_dtype(
    x: Union[torch.Tensor, Dict[str, torch.Tensor]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    if isinstance(x, torch.Tensor):
        if device is not None:
            x = x.to(device)
        if dtype is not None:
            x = x.to(dtype)
    elif isinstance(x, dict):
        if device is not None:
            x = {k: align_device_and_dtype(v, device, dtype) for k, v in x.items()}
        if dtype is not None:
            x = {k: align_device_and_dtype(v, device, dtype) for k, v in x.items()}
    return x


def seed_everything(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_safetensor(path, device_map='cpu'):
    tensors = {}
    with safe_open(path, framework="pt", device=0) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k).to(device_map)
    return tensors


def load_state_dict_skip_mismatch(model, state_dict):
    """
    Loads a state dictionary into a model, skipping layers with shape mismatches.

    Args:
        model (nn.Module): The model to load the state into.
        state_dict (OrderedDict): The state dictionary to load.

    Returns:
        list: A list of keys for the layers that were skipped due to shape mismatch.
    """
    model_state_dict = model.state_dict()
    mismatched_keys = []
    
    # 1. Create a new state_dict with matching keys and shapes
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k in model_state_dict and v.shape == model_state_dict[k].shape:
            new_state_dict[k] = v
        else:
            mismatched_keys.append(k)
            logger.warning(
                "Skipping %s due to shape mismatch. Loaded shape: %s, Model shape: %s",
                k,
                v.shape,
                getattr(model_state_dict.get(k, "N/A"), "shape", "N/A"),
            )


    # 2. Load the new state_dict
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    return mismatched_keys, missing_keys, unexpected_keys


def load_state_dict(model, path, strict=False, device_map='cpu', ignore_keys=None):
    if path.endswith('safetensors'):
        state_dict = load_safetensor(path, device_map)
    else:
        state_dict = torch.load(path, device_map=device_map)

    if ignore_keys is not None:
        for k in ignore_keys:
            state_dict.pop(k)

    mismatched_keys, missing_keys, unexpected_keys = load_state_dict_skip_mismatch(model, state_dict)
    logger.info(
        "Loaded state dict from %s. missing_keys=%s, unexpected_keys=%s, mismatched_keys=%s",
        path,
        missing_keys,
        unexpected_keys,
        mismatched_keys,
    )

    return model


def apply_color_jitter_to_video(tensor, jitter=None, same_jitter_within_view=False, n_view=3):
    """
    Apply ColorJitter augmentation to a Tensor of shape (B, C, T, H, W) with values in [-1, 1].

    Args:
        tensor (torch.Tensor): Input video data of shape (B, C, T, H, W), value range [-1, 1]
        jitter (ColorJitter): A torchvision.transforms.ColorJitter instance (optional)

    Returns:
        torch.Tensor: Augmented video Tensor with the same shape, values still in [-1, 1]
    """
    B, C, T, H, W = tensor.shape
    assert C == 3, "ColorJitter only supports 3-channel RGB images"

    # Default jitter parameters
    if jitter is None:
        jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)

    # First map [-1, 1] to [0, 1]
    tensor = (tensor + 1.0) / 2.0

    if same_jitter_within_view:
        tensor = rearrange(tensor, '(b v) c t h w -> b (v t) c h w', v=n_view, t=T)
        for b in range(B // n_view):
            tensor[b] = jitter(tensor[b])
        tensor = rearrange(tensor, 'b (v t) c h w -> (b v) c t h w', v=n_view, t=T)
    else:
        tensor = rearrange(tensor, 'b c t h w -> b t c h w')
        for b in range(B):
            tensor[b, :, :] = jitter(tensor[b, :, :])
        tensor = rearrange(tensor, 'b t c h w -> b c t h w')

    # Then map back to [-1, 1]
    tensor = tensor * 2.0 - 1.0

    return tensor


def get_latents(vae,
                mem: torch.Tensor,
                video: torch.Tensor,
                patch_size: int = 1,
                patch_size_t: int = 1,
                device: Optional[torch.device] = None,
                dtype: Optional[torch.dtype] = None,
                generator: Optional[torch.Generator] = None,
                enc_only: bool = False,
                return_unpack: bool = False,
                sampling: bool = True,
                sep_mem: bool = True,
            ):
    """
    mem: (b v) c m h w [-1,1]
    video: (b v) c f h w [-1,1]
    Returns:
        mem_latents: (b v m) (1 h_latent w_latent) c
        video_latents: (b v) (f_latent h_latent w_latent) c
    """

    device = device or vae.device

    # bv c m h w -> (bv m) c 1 h w
    mem_size = mem.shape[2]
    if sep_mem:
        mem = rearrange(mem, 'b c m h w -> (b m) c h w').unsqueeze(2)

    if not enc_only:
        if sampling:
            video_latents = vae.encode(video).latent_dist.sample(generator=generator)
        else:
            video_latents = vae.encode(video).latent_dist.mode()
        video_latents = video_latents.to(dtype=dtype)
        video_latents = _normalize_latents(video_latents, vae.config.latents_mean, vae.config.latents_std)
    else:
        video_latents = vae.encode(video)
        video_latents = video_latents.to(dtype=dtype)
    video_latents_pack = _pack_latents(video_latents, patch_size, patch_size_t)

    if not enc_only:
        if sampling:
            mem_latents = vae.encode(mem).latent_dist.sample(generator=generator)
        else:
            mem_latents = vae.encode(mem).latent_dist.mode()
        mem_latents = mem_latents.to(dtype=dtype)
        mem_latents = _normalize_latents(mem_latents, vae.config.latents_mean, vae.config.latents_std)
    else:
        mem_latents = vae.encode(mem)
        mem_latents = mem_latents.to(dtype=dtype)
    mem_latents_pack = _pack_latents(mem_latents, patch_size, patch_size_t)

    if return_unpack:
        if sep_mem:
            return mem_latents_pack, video_latents_pack, rearrange(mem_latents, '(b m) c f h w -> b c (m f) h w', m=mem_size), video_latents
        else:
            return mem_latents_pack, video_latents_pack, mem_latents, video_latents
    else:
        return mem_latents_pack, video_latents_pack, None, None


def _normalize_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0,
    reverse=False,
) -> torch.Tensor:
    # Normalize latents across the channel dimension [B, C, F, H, W]
    if not isinstance(latents_mean, torch.Tensor):
        latents_mean = torch.tensor(latents_mean)
    if not isinstance(latents_std, torch.Tensor):
        latents_std = torch.tensor(latents_std)
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    if not reverse:
        latents = (latents - latents_mean) * scaling_factor / latents_std
    else:
        latents = latents * latents_std / scaling_factor + latents_mean
    return latents


def unpack_latents(
        latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
    ) -> torch.Tensor:
    # Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
    # are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of
    # what happens in the `_pack_latents` method.
    batch_size = latents.size(0)
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents


def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
    # The patch dimensions are then permuted and collapsed into the channel dimension of shape:
    # [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
    # dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features
    batch_size, num_channels, num_frames, height, width = latents.shape
    post_patch_num_frames = num_frames // patch_size_t
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size
    latents = latents.reshape(
        batch_size,
        -1,
        post_patch_num_frames,
        patch_size_t,
        post_patch_height,
        patch_size,
        post_patch_width,
        patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


def save_video(tensor, save_path, fps=30):
    """
    Input tensor: shape c,t,h,w ranging -1-1
    """
    tensor = torch.clamp(tensor, min=-1, max=1)
    tensor = ((tensor+1)/2*255).to(torch.uint8)
    tensor = rearrange(tensor, 'c t h w -> t h w c')
    torchvision.io.write_video(save_path, tensor, fps=fps)


def save_image(tensor, save_path):
    """
    Input tensor: shape c,h,w ranging -1-1
    """
    tensor = torch.clamp(tensor, min=-1, max=1)
    tensor = ((tensor+1)/2*255).to(torch.uint8)
    torchvision.io.write_png(tensor, save_path)
