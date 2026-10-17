"""G02 trajectory-band renderer (gradient background), standalone module.

Extracted verbatim from ``MixAC.get_traj`` in
``lib/unified_dataset_v2/ac_state_dataset.py`` (structure adapted from the
opensource repo's ``gesim.conditioning.band_g02``) so the renderer can be
reused outside the dataset class — eval scripts, pipelines, GPU-side
rendering — while staying pixel-identical to the training recipe:

* radial-gradient background (0.20 grey at the nearest projected EEF origin
  ramping to 0.50 at the image diagonal; 4x-downsampled grid + bilinear up);
* disk radius fades linearly with camera-frame Z over the robot's
  ``traj_radius_depth_range``;
* per (view, frame) depth-sorted overlay: far disk -> near disk -> far axis
  sticks -> near axis sticks.

Inputs are the trainer-side artefacts: quat-form pose ``(T, 16)``
``[L_xyz, L_qxyzw, L_grip, R_xyz, R_qxyzw, R_grip]`` (raw gripper values),
``w2c`` ``(V, T, 4, 4)``, ``intrinsic`` ``(V, 3, 3)``. Output:
``(3, V, T, H, W)`` float32 in ``[0, 1]`` on the input pose's device.

``device`` selects where the rasterisation runs (e.g. ``"cuda"`` — same
vectorised recipe, ~10x faster for long episodes); the result is moved back
to the input pose's device so callers see no difference.
"""

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F

from gesim.conditioning.band import _quat_xyzw_to_matrix_torch

def quaternion_to_matrix(q):
    return _quat_xyzw_to_matrix_torch(q[..., [1, 2, 3, 0]])

# Rasterisation batch over T. Keep at the trainer's value: the stick sample
# count N is shared across a chunk, so a different chunking could change line
# sampling density on extreme chunks.
_T_CHUNK = 64


def quat_pose_to_transform_matrix(quat):
    """(T, 7) ``[xyz, qxyzw]`` -> (T, 4, 4). Mirrors
    ``MixAC.get_transformation_matrix_from_quat`` (xyzw reordered to wxyz
    before the pytorch3d-style conversion)."""
    rot_quat = quat[:, 3:]
    rot_quat = rot_quat[:, [3, 0, 1, 2]]
    rot = quaternion_to_matrix(rot_quat)
    trans = quat[:, :3]
    output = torch.eye(4).unsqueeze(0).repeat(quat.shape[0], 1, 1)
    output[:, :3, :3] = rot
    output[:, :3, 3] = trans
    return output


def draw_disks_vec(out_chunk, uvs_arm, rad_arm, dep_arm, col_arm, xs, ys, H, W):
    """Filled-disk rasteriser, batched over ``(V, ch)``. Mutates ``out_chunk``.

    Direct masked indexed assignment beats a full ``torch.where`` over the
    ``(3, V, ch, H, W)`` output tensor on CPU by ~3×.
    """
    V, ch = uvs_arm.shape[:2]
    u_o = uvs_arm[..., 0, 0]
    v_o = uvs_arm[..., 0, 1]
    valid = (
        (rad_arm > 0) & (dep_arm > 0)
        & (u_o >= 0) & (u_o < W)
        & (v_o >= 0) & (v_o < H)
    )
    if not valid.any():
        return
    rad_b = rad_arm.view(V, ch, 1, 1)
    dx = xs - u_o.float().view(V, ch, 1, 1)
    dy = ys - v_o.float().view(V, ch, 1, 1)
    mask = (dx * dx + dy * dy) <= rad_b * rad_b
    mask = mask & valid.view(V, ch, 1, 1)
    if not mask.any():
        return
    vi, ti, yi, xi = mask.nonzero(as_tuple=True)
    col_per_px = col_arm[vi, ti]                       # (M, 3)
    out_chunk[0, vi, ti, yi, xi] = col_per_px[:, 0]
    out_chunk[1, vi, ti, yi, xi] = col_per_px[:, 1]
    out_chunk[2, vi, ti, yi, xi] = col_per_px[:, 2]


def draw_sticks_vec(out_chunk, uvs_arm, dep_arm, cs_arm, H, W,
                    thickness: int = 8, max_samples: int = 256):
    """Rasterise the 3 axis-tip lines for one arm, batched over ``(V, ch)``.

    Lines are sampled parametrically (``t∈[0, 1]``) and a
    ``thickness×thickness`` square stamp is written at every sample. The
    sample count is ``clamp(max(|du|, |dv|) + 1, 2, max_samples)`` so short
    lines stay cheap and long lines stay un-aliased.
    """
    V, ch = uvs_arm.shape[:2]
    device = out_chunk.device
    origin = uvs_arm[..., 0, :].float()                # (V, ch, 2)
    valid_origin = (
        (dep_arm > 0)
        & (origin[..., 0] >= 0) & (origin[..., 0] < W)
        & (origin[..., 1] >= 0) & (origin[..., 1] < H)
    )
    if not valid_origin.any():
        return
    hw = thickness // 2
    off = torch.arange(-hw, thickness - hw, device=device)
    L = off.numel()
    v_idx_base = torch.arange(V, device=device).view(V, 1, 1, 1, 1)
    t_idx_base = torch.arange(ch, device=device).view(1, ch, 1, 1, 1)
    for stk_idx in range(1, 4):
        tip = uvs_arm[..., stk_idx, :].float()
        du = tip[..., 0] - origin[..., 0]
        dv = tip[..., 1] - origin[..., 1]
        seg_len = torch.maximum(du.abs(), dv.abs())
        if seg_len.numel() == 0:
            continue
        n = int(seg_len.max().item()) + 1
        N = max(2, min(n, max_samples))
        ts = torch.linspace(0.0, 1.0, N, device=device).view(1, 1, N)
        u_pts = (origin[..., 0:1] + ts * du.unsqueeze(-1)).round().long()
        v_pts = (origin[..., 1:2] + ts * dv.unsqueeze(-1)).round().long()
        u_full = u_pts.view(V, ch, N, 1, 1) + off.view(1, 1, 1, 1, L)
        v_full = v_pts.view(V, ch, N, 1, 1) + off.view(1, 1, 1, L, 1)
        u_full = u_full.expand(V, ch, N, L, L)
        v_full = v_full.expand(V, ch, N, L, L)
        in_bounds = (u_full >= 0) & (u_full < W) & (v_full >= 0) & (v_full < H)
        mask = valid_origin.view(V, ch, 1, 1, 1) & in_bounds
        if not mask.any():
            continue
        v_idx = v_idx_base.expand_as(u_full)
        t_idx = t_idx_base.expand_as(u_full)
        vi = v_idx[mask]
        ti = t_idx[mask]
        yi = v_full[mask]
        xi = u_full[mask]
        col_full = cs_arm[..., stk_idx - 1, :].view(V, ch, 1, 1, 1, 3) \
                                              .expand(V, ch, N, L, L, 3)
        ci = col_full[mask]
        out_chunk[0, vi, ti, yi, xi] = ci[:, 0]
        out_chunk[1, vi, ti, yi, xi] = ci[:, 1]
        out_chunk[2, vi, ti, yi, xi] = ci[:, 2]


@torch.no_grad()
def render_traj_g02(pose, w2c, intrinsic, *, robot, gripper_offset,
                    sample_size, z_near=None, z_far=None, device=None):
    """Render the (3, V, T, H, W) gradient-background trajectory band.

    pose:      (T, 16) quat-form ``[L_xyz, L_qxyzw, L_grip, R_xyz, R_qxyzw, R_grip]``
    w2c:       (V, T, 4, 4); intrinsic: (V, 3, 3)
    robot:     robot object owning ``pose_rotation_matrix()``,
               ``grip_colormap_norm()`` and ``traj_radius_depth_range``
    gripper_offset: z-offset (metres) applied to the EEF pose
    z_near / z_far: optional overrides of ``robot.traj_radius_depth_range``
    """
    out_device = pose.device
    if device is not None:
        dev = torch.device(device)
        pose = pose.to(dev)
        w2c = w2c.to(dev)
        intrinsic = intrinsic.to(dev)
    h, w = sample_size
    device = pose.device

    # ── 1. Pose & projection (vectorised over (V, T, arm)) ───────────
    pose_l_mat = quat_pose_to_transform_matrix(pose[:, 0:7].cpu()).to(device).unsqueeze(0)   # (1, T, 4, 4)
    pose_r_mat = quat_pose_to_transform_matrix(pose[:, 8:15].cpu()).to(device).unsqueeze(0)
    rz = robot.pose_rotation_matrix()
    if rz is not None:
        rz = rz.to(dtype=pose_l_mat.dtype, device=device)
        pose_l_mat = pose_l_mat @ rz
        pose_r_mat = pose_r_mat @ rz

    correct_matrix = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, gripper_offset],
        [0, 0, 0, 1],
    ], dtype=torch.float32, device=device).view(1, 1, 4, 4)
    ee2cam_l = (w2c @ pose_l_mat) @ correct_matrix  # (V, T, 4, 4)
    ee2cam_r = (w2c @ pose_r_mat) @ correct_matrix

    ee_key_pts = torch.tensor([
        [0, 0, 0, 1],
        [0.1, 0, 0, 1],
        [0, 0.1, 0, 1],
        [0, 0, 0.1, 1],
    ], dtype=torch.float32, device=device).T  # (4, 4) column vectors

    pts_l = ee2cam_l @ ee_key_pts                 # (V, T, 4, 4)
    pts_r = ee2cam_r @ ee_key_pts
    K = intrinsic.unsqueeze(1)                    # (V, 1, 3, 3)
    eps = 1e-6
    ppl = K @ pts_l[:, :, :3, :]
    ppr = K @ pts_r[:, :, :3, :]
    uvs_l = (ppl[:, :, :2, :] / ppl[:, :, 2:3, :].clamp(min=eps)).permute(0, 1, 3, 2)
    uvs_r = (ppr[:, :, :2, :] / ppr[:, :, 2:3, :].clamp(min=eps)).permute(0, 1, 3, 2)
    uvs_l = uvs_l.round().to(torch.int64)         # (V, T, 4, 2)
    uvs_r = uvs_r.round().to(torch.int64)

    depth_l = pts_l[:, :, 2, 0]                   # (V, T) camera-Z of origin
    depth_r = pts_r[:, :, 2, 0]

    # ── 2. Disk radius (matches existing _r_pix exactly) ──────────────
    z_near_def, z_far_def = robot.traj_radius_depth_range
    z_near = z_near_def if z_near is None else float(z_near)
    z_far = z_far_def if z_far is None else float(z_far)
    z_span = max(z_far - z_near, 1e-6)
    rad_l = torch.where(
        depth_l > z_near,
        (1.0 - (depth_l - z_near) / z_span).clamp(0.0, 1.0) * 100.0,
        torch.zeros_like(depth_l),
    )
    rad_r = torch.where(
        depth_r > z_near,
        (1.0 - (depth_r - z_near) / z_span).clamp(0.0, 1.0) * 100.0,
        torch.zeros_like(depth_r),
    )

    # ── 3. Per-frame colours — gripper colormap is owned by the robot ─
    norm_l = robot.grip_colormap_norm(pose[:, 7])
    norm_r = robot.grip_colormap_norm(pose[:, 15])
    # ``cm.Greens`` / ``cm.Reds`` are CPU-only — pull the per-frame scalars
    # off-device once, then ship the (T, 3) palette back.
    col_l_np = np.asarray(cm.Greens(norm_l.detach().cpu().numpy())[:, :3], dtype=np.float32)
    col_r_np = np.asarray(cm.Reds(norm_r.detach().cpu().numpy())[:, :3], dtype=np.float32)
    col_l = torch.from_numpy(col_l_np).to(device)   # (T, 3) RGB in [0, 1]
    col_r = torch.from_numpy(col_r_np).to(device)
    # Stick palette: same RGB int tuples the cv2 version fed in, normalised
    # to [0, 1]. Order matches the original [color_list_l, color_list_r].
    sc_l = torch.tensor([[0, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=torch.float32, device=device)
    sc_r = torch.tensor([[1, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=torch.float32, device=device)

    # ── 4. Output buffer + (H, W) pixel grids ────────────────────────
    V, T = w2c.shape[0], w2c.shape[1]
    out = torch.empty((3, V, T, h, w), dtype=torch.float32, device=device)
    ys = torch.arange(h, dtype=torch.float32, device=device).view(1, 1, h, 1)
    xs = torch.arange(w, dtype=torch.float32, device=device).view(1, 1, 1, w)

    # Gradient endpoints: dark near, mid-grey far.
    _BG_NEAR, _BG_FAR = 0.20, 0.50
    diag = float(np.hypot(h, w))
    inv_diag = 1.0 / diag
    # Background is a smooth ramp — compute it on a 4× downsampled grid
    # then bilinearly upsample. ~16× less arithmetic for the same look.
    _BG_DS = 4
    h_d, w_d = h // _BG_DS, w // _BG_DS
    ys_d = torch.arange(h_d, dtype=torch.float32, device=device).view(1, 1, h_d, 1) * _BG_DS
    xs_d = torch.arange(w_d, dtype=torch.float32, device=device).view(1, 1, 1, w_d) * _BG_DS

    # ── 5. Per-T-chunk rasterisation ─────────────────────────────────
    # Chunking keeps peak memory bounded (the (V, ch, H, W) tensors used
    # inside draw_disks_vec / draw_sticks_vec scale linearly with ``ch``).
    for t0 in range(0, T, _T_CHUNK):
        t1 = min(t0 + _T_CHUNK, T)
        ch = t1 - t0

        ul_o = uvs_l[:, t0:t1, 0, 0].float()
        vl_o = uvs_l[:, t0:t1, 0, 1].float()
        ur_o = uvs_r[:, t0:t1, 0, 0].float()
        vr_o = uvs_r[:, t0:t1, 0, 1].float()

        # 5a. Gradient bg on downsampled grid → bilinear up to (H, W).
        dl_sq = ((xs_d - ul_o.view(V, ch, 1, 1)) ** 2
                 + (ys_d - vl_o.view(V, ch, 1, 1)) ** 2)
        dr_sq = ((xs_d - ur_o.view(V, ch, 1, 1)) ** 2
                 + (ys_d - vr_o.view(V, ch, 1, 1)) ** 2)
        d_min = torch.minimum(dl_sq, dr_sq).sqrt_() * inv_diag
        d_min.clamp_(0.0, 1.0)
        bg_d = (_BG_NEAR + (_BG_FAR - _BG_NEAR) * d_min).view(V * ch, 1, h_d, w_d)
        bg = F.interpolate(bg_d, size=(h, w), mode="bilinear", align_corners=False).view(V, ch, h, w)
        out[:, :, t0:t1] = bg.unsqueeze(0).expand(3, V, ch, h, w)

        # 5b. Depth sort: per (V, ch) pick the FAR vs NEAR arm.
        depth_l_c = depth_l[:, t0:t1]
        depth_r_c = depth_r[:, t0:t1]
        left_is_far = depth_l_c > depth_r_c                # (V, ch)

        def _sel(a, b, left_is_far=left_is_far):
            m = left_is_far
            while m.ndim < a.ndim:
                m = m.unsqueeze(-1)
            return torch.where(m, a, b), torch.where(m, b, a)

        col_l_VT = col_l[t0:t1].unsqueeze(0).expand(V, ch, 3)
        col_r_VT = col_r[t0:t1].unsqueeze(0).expand(V, ch, 3)
        sc_l_VT  = sc_l.view(1, 1, 3, 3).expand(V, ch, 3, 3)
        sc_r_VT  = sc_r.view(1, 1, 3, 3).expand(V, ch, 3, 3)

        far_uvs, near_uvs = _sel(uvs_l[:, t0:t1], uvs_r[:, t0:t1])  # (V, ch, 4, 2)
        far_rad, near_rad = _sel(rad_l[:, t0:t1], rad_r[:, t0:t1])  # (V, ch)
        far_dep, near_dep = _sel(depth_l_c, depth_r_c)              # (V, ch)
        far_cc,  near_cc  = _sel(col_l_VT, col_r_VT)                # (V, ch, 3)
        far_cs,  near_cs  = _sel(sc_l_VT, sc_r_VT)                  # (V, ch, 3, 3)

        out_chunk = out[:, :, t0:t1]
        # 5c. Disks: far first, then near.
        for uvs_arm, rad_arm, dep_arm, col_arm in (
            (far_uvs,  far_rad,  far_dep,  far_cc),
            (near_uvs, near_rad, near_dep, near_cc),
        ):
            draw_disks_vec(out_chunk, uvs_arm, rad_arm, dep_arm, col_arm, xs, ys, h, w)
        # 5d. Sticks: same depth order, 3 axes each.
        for uvs_arm, dep_arm, cs_arm in (
            (far_uvs,  far_dep,  far_cs),
            (near_uvs, near_dep, near_cs),
        ):
            draw_sticks_vec(out_chunk, uvs_arm, dep_arm, cs_arm, h, w)

    return out.to(out_device)
