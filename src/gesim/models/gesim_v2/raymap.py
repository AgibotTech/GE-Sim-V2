"""Camera ray-map construction from per-frame camera-to-world matrices."""

import numpy as np
import torch


def _prepare_intrinsics(intrinsic, batch_size, num_cams):
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    if intrinsic.shape == (3, 3):
        intrinsic = np.broadcast_to(intrinsic, (batch_size, num_cams, 3, 3)).copy()
    elif intrinsic.shape == (num_cams, 3, 3):
        intrinsic = np.broadcast_to(intrinsic[None], (batch_size, num_cams, 3, 3)).copy()
    elif intrinsic.shape == (batch_size, 3, 3):
        intrinsic = np.broadcast_to(intrinsic[:, None], (batch_size, num_cams, 3, 3)).copy()
    elif intrinsic.shape != (batch_size, num_cams, 3, 3):
        raise ValueError(
            f"Unsupported intrinsic shape {intrinsic.shape}, expected (3,3), "
            f"({num_cams},3,3), ({batch_size},3,3) or ({batch_size},{num_cams},3,3)"
        )
    return intrinsic


def prepare_ray_map(intrinsic, c2w, H, W):
    intrinsic = torch.as_tensor(intrinsic, dtype=torch.float32)
    c2w = torch.as_tensor(c2w, dtype=torch.float32)

    batch_size = intrinsic.shape[0]
    fx = intrinsic[:, 0, 0].view(batch_size, 1, 1)
    fy = intrinsic[:, 1, 1].view(batch_size, 1, 1)
    cx = intrinsic[:, 0, 2].view(batch_size, 1, 1)
    cy = intrinsic[:, 1, 2].view(batch_size, 1, 1)

    i, j = torch.meshgrid(
        torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=c2w.device),
        torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=c2w.device),
        indexing="ij",
    )
    i = i.t().unsqueeze(0).repeat(batch_size, 1, 1)
    j = j.t().unsqueeze(0).repeat(batch_size, 1, 1)

    dirs = torch.stack(((i - cx) / fx, (j - cy) / fy, torch.ones_like(i)), dim=-1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:, None, None, :3, :3], dim=-1)
    rays_o = c2w[:, :3, 3].view(batch_size, 1, 1, 3).repeat(1, H, W, 1)
    viewdir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    return rays_o, viewdir


def raymap_from_c2w(
    intrinsic,
    c2w,
    res=(512, 384),
):
    """Build the 6-channel ray map straight from per-frame c2w (no FK, no hand-eye).

    Mirrors the trainer's ``prepare_ray_map`` + the ``cat_rays`` packing path so the
    server's conditioning is byte-identical to training when the client uploads
    a real per-frame extrinsic via ``set_episode_traj``.

    Inputs
    ------
    intrinsic : (V, 3, 3) or (1, V, 3, 3) float
        Per-view pinhole K (already at model resolution 384x512).
    c2w       : (V, T, 4, 4) float
        Per-view, per-frame camera-to-world. Comes straight from ``episode_c2w``.
    res       : (H, W) tuple

    Returns
    -------
    rays : (1, 6, V, T, H, W) torch.float32
        Same layout as the trainer's ray-map packing so it drops in unchanged.
    """
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    c2w = np.asarray(c2w, dtype=np.float32)
    if c2w.ndim != 4 or c2w.shape[-2:] != (4, 4):
        raise ValueError(f"c2w must be (V, T, 4, 4), got {c2w.shape}")
    V, T = c2w.shape[:2]
    h, w = res
    intrinsic = _prepare_intrinsics(intrinsic, batch_size=1, num_cams=V)[0]  # (V, 3, 3)

    output = np.zeros((1, 6 * V, T, h, w), dtype=np.float32)
    for it in range(T):
        rays_o, rays_d = prepare_ray_map(
            intrinsic,  # (V, 3, 3)
            c2w[:, it],  # (V, 4, 4)
            H=h,
            W=w,
        )
        rays = torch.cat((rays_o, rays_d), dim=-1).permute(0, 3, 1, 2).cpu().numpy()
        output[0, :, it] = rays.reshape(6 * V, h, w)

    output = torch.from_numpy(output).reshape(1, V, 6, T, h, w).permute(0, 2, 1, 3, 4, 5)
    return output  # (1, 6, V, T, H, W)
