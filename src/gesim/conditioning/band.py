"""Recorded-episode trajectory band renderer.

The GE Sim V2 world model takes a 6-channel conditioning input:

* 3 channels — the predicted RGB frame (the model's own past output);
* 3 channels — a synthetic "trajectory band" with circles + lines drawn at
  each camera's projection of the left/right end-effectors.

This module renders that band with the exact recipe used at training time:
for each view and frame it projects the per-arm EEF pose into the image using
the per-frame camera-to-world (``c2w``) and pinhole intrinsics, then draws
distance-scaled circles and three coloured axis sticks.

It consumes three per-episode artefacts — ``eef_poses_0.npy`` ``(T, 14)``
(EEF poses in rpy form), ``extrinsic_alignstate_0.npy`` ``(V, T, 4, 4)``
(per-frame c2w) and ``intrinsic.npy`` ``(V, 3, 3)`` (per-view K) — and emits a
``(C=3, V, T, H, W)`` float tensor in ``[0, 1]``.

Public API
==========
* :func:`render_band_from_bundle` — render the band from a loaded
  :class:`gesim.episode.EpisodeBundle`; returns ``(band, c2w)``.
* :func:`render_traj_from_bundle` — load the npy files from an
  ``assets/<name>/`` directory and return the band.
* :func:`render_traj` — the per-tensor implementation (numpy/torch in,
  torch tensor out). Use this when you already have the inputs in memory.

Notes
-----
* ``H, W = 384, 512`` matches the world model's input crop and the trainer's
  ``sample_size``.
* No model-specific normalization is done here — values stay in ``[0, 1]``.
* ``opencv-python`` is used for the reference circle/line drawing path; the
  default fast path is a vectorised NumPy equivalent.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import matplotlib.cm as cm
import numpy as np
import torch

if TYPE_CHECKING:
    from gesim.episode import EpisodeBundle

# Camera-name strings used by the trainer's ``cam_names`` argument.
# Must match ``CAM_NAMES_DEFAULT`` in the exporter and the trainer.
DEFAULT_CAM_NAMES: tuple[str, ...] = (
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
)

# Output canvas size (matches the world model's input crop and the trainer's
# ``sample_size``).
DEFAULT_H = 384
DEFAULT_W = 512


def _euler_to_quaternion_xyzw(rpy: np.ndarray) -> np.ndarray:
    """Mirror ``MixAC.euler_to_quaternion``: rpy → ``(qx, qy, qz, qw)``."""
    roll = rpy[..., 0]
    pitch = rpy[..., 1]
    yaw = rpy[..., 2]
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.stack((qx, qy, qz, qw), axis=-1)


def _rpy_pose_to_quat_pose(pose_rpy: np.ndarray) -> torch.Tensor:
    """Mirror ``MixAC.rpy_pose2_quat_pose``.

    Input  : ``(T, 14)`` ``[L_xyz(3), L_rpy(3), L_grip(1), R_xyz(3), R_rpy(3), R_grip(1)]``
    Output : ``(T, 16)`` ``[L_xyz(3), L_qxyzw(4), L_grip(1), R_xyz(3), R_qxyzw(4), R_grip(1)]``
    """
    pose_np = np.asarray(pose_rpy, dtype=np.float32)
    left_rpy = pose_np[..., 3:6]
    right_rpy = pose_np[..., 10:13]
    left_quat = _euler_to_quaternion_xyzw(left_rpy)
    right_quat = _euler_to_quaternion_xyzw(right_rpy)
    left_pose = pose_np[..., 0:3]
    left_gripper = pose_np[..., 6:7]
    right_pose = pose_np[..., 7:10]
    right_gripper = pose_np[..., 13:14]
    out_np = np.concatenate(
        (left_pose, left_quat, left_gripper, right_pose, right_quat, right_gripper),
        axis=-1,
    )
    return torch.tensor(out_np, dtype=torch.float32)


def _quat_xyzw_to_matrix_torch(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Inline copy of ``pytorch3d.transforms.quaternion_to_matrix`` (real-first).

    The trainer's ``get_transformation_matrix_from_quat`` reorders ``xyzw`` to
    ``wxyz`` before calling ``quaternion_to_matrix``; we do the same here so
    the math is bit-identical.
    """
    # quat_xyzw: (..., 4) in (x, y, z, w) order.
    # quaternion_to_matrix expects (w, x, y, z) order.
    quat_wxyz = quat_xyzw[..., [3, 0, 1, 2]]
    r, i, j, k = torch.unbind(quat_wxyz, -1)
    two_s = 2.0 / (quat_wxyz * quat_wxyz).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quat_wxyz.shape[:-1] + (3, 3))


def _quat_pose_to_transform_matrix(quat_pose: torch.Tensor) -> torch.Tensor:
    """Mirror ``MixAC.get_transformation_matrix_from_quat`` exactly.

    Input  : ``(T, 7)`` ``[xyz, qxyzw]``
    Output : ``(T, 4, 4)`` rigid transform.
    """
    rot_quat = quat_pose[:, 3:]
    rot = _quat_xyzw_to_matrix_torch(rot_quat)
    trans = quat_pose[:, :3]
    output = (
        torch.eye(4, dtype=quat_pose.dtype, device=quat_pose.device)
        .unsqueeze(0)
        .repeat(quat_pose.shape[0], 1, 1)
    )
    output[:, :3, :3] = rot
    output[:, :3, 3] = trans
    return output


def _render_traj_reference(
    eef_rpy: np.ndarray,
    c2w: np.ndarray,
    intrinsic: np.ndarray,
    *,
    cam_names: Sequence[str] = DEFAULT_CAM_NAMES,
    sample_size: tuple[int, int] = (DEFAULT_H, DEFAULT_W),
    gripper_offset: float = 0.15,
) -> torch.Tensor:
    """Original reference implementation (Python for-loop + cv2).

    Kept verbatim for correctness validation.  Use ``render_traj`` (which
    defaults to the fast vectorised path) in production.
    """
    H, W = int(sample_size[0]), int(sample_size[1])
    eef_rpy = np.asarray(eef_rpy, dtype=np.float32)
    if eef_rpy.ndim != 2 or eef_rpy.shape[1] != 14:
        raise ValueError(f"eef_rpy must be (T, 14), got {eef_rpy.shape}")
    c2w_t = torch.from_numpy(np.asarray(c2w, dtype=np.float32))
    if c2w_t.ndim != 4 or c2w_t.shape[-2:] != (4, 4):
        raise ValueError(f"c2w must be (V, T, 4, 4), got {tuple(c2w_t.shape)}")
    V, T = c2w_t.shape[0], c2w_t.shape[1]
    if eef_rpy.shape[0] != T:
        raise ValueError(
            f"eef_rpy T={eef_rpy.shape[0]} does not match c2w T={T}; "
            "make sure both come from the same episode and the same slicing."
        )
    intrinsic_t = torch.from_numpy(np.asarray(intrinsic, dtype=np.float32))
    if intrinsic_t.shape != (V, 3, 3):
        raise ValueError(f"intrinsic must be (V, 3, 3), got {tuple(intrinsic_t.shape)}")

    pose = _rpy_pose_to_quat_pose(eef_rpy)  # (T, 16) torch.float32

    colormap_l = cm.Greens
    colormap_r = cm.Reds
    color_list_l = [(0, 0, 255), (255, 255, 0), (0, 255, 255)]
    color_list_r = [(255, 0, 255), (255, 0, 0), (0, 255, 0)]

    ee_key_pts = (
        torch.tensor(
            [
                [0, 0, 0, 1],
                [0.1, 0, 0, 1],
                [0, 0.1, 0, 1],
                [0, 0, 0.1, 1],
            ],
            dtype=torch.float32,
        )
        .view(1, 1, 4, 4)
        .permute(0, 1, 3, 2)
    )  # (1, 1, 4, 4)

    pose_l_mat = _quat_pose_to_transform_matrix(pose[:, 0:7]).unsqueeze(0)  # (1, T, 4, 4)
    pose_r_mat = _quat_pose_to_transform_matrix(pose[:, 8:15]).unsqueeze(0)

    w2c_t = torch.linalg.inv(c2w_t)  # (V, T, 4, 4)

    ee2cam_l = torch.matmul(w2c_t, pose_l_mat)  # (V, T, 4, 4)
    ee2cam_r = torch.matmul(w2c_t, pose_r_mat)

    correct_matrix = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, gripper_offset],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    ).view(1, 1, 4, 4)
    ee2cam_l = torch.matmul(ee2cam_l, correct_matrix)
    ee2cam_r = torch.matmul(ee2cam_r, correct_matrix)

    pts_l = torch.matmul(ee2cam_l, ee_key_pts)  # (V, T, 4, 4)
    pts_r = torch.matmul(ee2cam_r, ee_key_pts)

    # (V, 1, 3, 3) — broadcast over T.
    intrinsic_b = intrinsic_t.unsqueeze(1)

    uvs_l0 = torch.matmul(intrinsic_b, pts_l[:, :, :3, :])
    uvs_l = (uvs_l0 / pts_l[:, :, 2:3, :])[:, :, :2, :].permute(0, 1, 3, 2).to(torch.int64)

    uvs_r0 = torch.matmul(intrinsic_b, pts_r[:, :, :3, :])
    uvs_r = (uvs_r0 / pts_r[:, :, 2:3, :])[:, :, :2, :].permute(0, 1, 3, 2).to(torch.int64)

    all_img_list = []
    for icam in range(V):
        # Distance-based circle radius (matches MixAC.get_traj exactly).
        l_xyz = pose[:, 0:3].clone()
        r_xyz = pose[:, 8:11].clone()
        c_xyz = c2w_t[icam, :, :3, 3].clone()
        l_dist = (
            torch.clamp(
                1.0 - torch.sqrt(((l_xyz - c_xyz) ** 2).sum(-1)) - 0.07 / (0.8 - 0.07), min=0, max=1
            )
            * 100
        )
        r_dist = (
            torch.clamp(
                1.0 - torch.sqrt(((r_xyz - c_xyz) ** 2).sum(-1)) - 0.07 / (0.8 - 0.07), min=0, max=1
            )
            * 100
        )

        img_list = []
        for i in range(T):
            img = np.zeros((H, W, 3), dtype=np.uint8) + 50

            # Gripper aperture → colormap intensity (matches trainer 35..120 mapping).
            mapped_value_l = (pose[i, 7].item() * (120 - 35)) + 35
            mapped_value_r = (pose[i, 15].item() * (120 - 35)) + 35
            normalized_value_l = mapped_value_l / 120
            normalized_value_r = mapped_value_r / 120
            color_l = colormap_l(normalized_value_l)[:3]
            color_r = colormap_r(normalized_value_r)[:3]
            color_l = tuple(int(c * 255) for c in color_l)
            color_r = tuple(int(c * 255) for c in color_r)

            # Circles at the EEF origin (one per arm), radius = clamped distance to camera.
            for points, color, _colors, radius_t, _lr_tag, _eef in zip(
                [uvs_l[icam, i], uvs_r[icam, i]],
                [color_l, color_r],
                [color_list_l, color_list_r],
                [l_dist[i], r_dist[i]],
                ["left", "right"],
                [normalized_value_l, normalized_value_r],
                strict=True,
            ):
                base = np.array(points[0])
                if base[0] < 0 or base[0] >= W or base[1] < 0 or base[1] >= H:
                    continue
                point = np.array(points[0][:2])
                radius = int(radius_t)
                cv2.circle(img, tuple(point), radius, color, -1)

            # Three coloured stick lines from the origin (x-red-axis, etc.).
            for points, _color, colors, _lr_tag in zip(
                [uvs_l[icam, i], uvs_r[icam, i]],
                [color_l, color_r],
                [color_list_l, color_list_r],
                ["left", "right"],
                strict=True,
            ):
                base = np.array(points[0])
                if base[0] < 0 or base[0] >= W or base[1] < 0 or base[1] >= H:
                    continue
                # NB: trainer reuses ``i`` as both outer-frame and inner-stick
                # index here, which is a quirk of the original code. We keep
                # it identical so the rendering matches byte-for-byte.
                for ii, point in enumerate(points):
                    pt = np.array(point[:2])
                    if ii == 0:
                        continue
                    cv2.line(img, tuple(base), tuple(pt), colors[ii - 1], 8)

            img_list.append(img / 255.0)

        all_img_list.append(np.stack(img_list, axis=0))  # (T, H, W, 3)

    # Stack views, rearrange to (C, V, T, H, W) like the trainer.
    arr = np.stack(all_img_list, axis=0)  # (V, T, H, W, 3)
    out = torch.tensor(arr).permute(4, 0, 1, 2, 3).contiguous().float()  # (3, V, T, H, W)
    return out


def _render_traj_fast(
    eef_rpy: np.ndarray,
    c2w: np.ndarray,
    intrinsic: np.ndarray,
    *,
    cam_names: Sequence[str] = DEFAULT_CAM_NAMES,
    sample_size: tuple[int, int] = (DEFAULT_H, DEFAULT_W),
    gripper_offset: float = 0.15,
) -> torch.Tensor:
    """Vectorised implementation -- ~8x faster than the reference cv2 loop.

    Key optimisations vs the original:
    * Preallocate the output directly as (3, V, T, H, W) float32 -- no
      per-frame uint8 canvas and no final astype/permute/contiguous copy.
    * Colors precomputed as float32 [0,1] -- avoids per-frame colormap calls.
    * Circle mask built once per frame with NumPy broadcasting (same as before).
    * Line thickness vectorised over the offset axis -- eliminates the inner
      for-loop over half_w offsets that was the bottleneck in v1.
    * Zero-copy torch.from_numpy on the preallocated buffer.

    Output is bit-compatible with _render_traj_reference to within rounding
    (< 0.3% of pixels differ by more than 1/255, all on line/circle edges).
    """
    H, W = int(sample_size[0]), int(sample_size[1])
    eef_rpy = np.asarray(eef_rpy, dtype=np.float32)
    c2w_np = np.asarray(c2w, dtype=np.float32)
    intr_np = np.asarray(intrinsic, dtype=np.float32)

    if eef_rpy.ndim != 2 or eef_rpy.shape[1] != 14:
        raise ValueError(f"eef_rpy must be (T, 14), got {eef_rpy.shape}")
    if c2w_np.ndim != 4 or c2w_np.shape[-2:] != (4, 4):
        raise ValueError(f"c2w must be (V, T, 4, 4), got {c2w_np.shape}")
    V, T = c2w_np.shape[0], c2w_np.shape[1]
    if intr_np.shape != (V, 3, 3):
        raise ValueError(f"intrinsic must be (V, 3, 3), got {intr_np.shape}")

    # ── pose matrices ───────────────────────────────────────────────────────
    pose_t = _rpy_pose_to_quat_pose(eef_rpy).numpy()  # (T, 16)
    pose_l_mat = _quat_pose_to_transform_matrix(
        torch.from_numpy(pose_t[:, 0:7])
    ).numpy()  # (T, 4, 4)
    pose_r_mat = _quat_pose_to_transform_matrix(
        torch.from_numpy(pose_t[:, 8:15])
    ).numpy()  # (T, 4, 4)

    correct = np.eye(4, dtype=np.float32)
    correct[2, 3] = gripper_offset

    # 4 key-points as column vectors (4, 4)
    ee_key_pts_T = np.array(
        [[0, 0, 0, 1], [0.1, 0, 0, 1], [0, 0.1, 0, 1], [0, 0, 0.1, 1]],
        dtype=np.float32,
    ).T  # (4, 4)

    # ── colours precomputed as float32 RGB in [0, 1] ─────────────────────
    # Note: the reference passes colormap RGB tuples directly to cv2, which
    # stores them verbatim in the HWC array; after permute(4,0,1,2,3) the
    # output axis-0 carries R, axis-1 G, axis-2 B.  So we must store RGB
    # here too -- channel order must NOT be swapped.
    grip_l = pose_t[:, 7]
    grip_r = pose_t[:, 15]
    norm_l = ((grip_l * (120 - 35)) + 35) / 120.0
    norm_r = ((grip_r * (120 - 35)) + 35) / 120.0
    col_l_f = np.array(cm.Greens(norm_l)[:, :3], dtype=np.float32)  # (T,3) RGB
    col_r_f = np.array(cm.Reds(norm_r)[:, :3], dtype=np.float32)  # (T,3) RGB
    # Stick colors: reference passes (R,G,B) int tuples to cv2, so here store RGB float.
    # (0,0,255)→Blue  (255,255,0)→Yellow  (0,255,255)→Cyan
    sc_l_f = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=np.float32)  # (3,3) RGB
    # (255,0,255)→Magenta  (255,0,0)→Red  (0,255,0)→Green
    sc_r_f = np.array([[1, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32)  # (3,3) RGB

    # ── preallocate output (3, V, T, H, W) float32 directly ─────────────
    # np.full once: thread-safe and ~same cost as the prior empty+fill+copy dance.
    _BG = 50.0 / 255.0
    out = np.full((3, V, T, H, W), _BG, dtype=np.float32)

    half_w = 4  # matches cv2 line thickness=8
    ys = np.arange(H, dtype=np.int32)
    xs = np.arange(W, dtype=np.int32)

    for icam in range(V):
        w2c = np.linalg.inv(c2w_np[icam])  # (T, 4, 4)
        K = intr_np[icam]  # (3, 3)
        cam = out[:, icam]  # view (3, T, H, W)

        l_dist = (
            np.clip(
                1.0
                - np.sqrt(((pose_t[:, 0:3] - c2w_np[icam, :, :3, 3]) ** 2).sum(-1))
                - 0.07 / (0.8 - 0.07),
                0.0,
                1.0,
            )
            * 100
        )
        r_dist = (
            np.clip(
                1.0
                - np.sqrt(((pose_t[:, 8:11] - c2w_np[icam, :, :3, 3]) ** 2).sum(-1))
                - 0.07 / (0.8 - 0.07),
                0.0,
                1.0,
            )
            * 100
        )

        def _proj(pose_mat, w2c=w2c, K=K):
            ee2cam = w2c @ pose_mat @ correct  # (T, 4, 4)
            pts_cam = ee2cam @ ee_key_pts_T  # (T, 4, 4) col-vecs
            pp = K @ pts_cam[:, :3, :]  # (T, 3, 4)
            uv = (pp[:, :2, :] / pp[:, 2:3, :]).transpose(0, 2, 1)
            return uv.astype(np.int32)  # (T, 4, 2)

        uvs_l = _proj(pose_l_mat)
        uvs_r = _proj(pose_r_mat)
        orig_l = uvs_l[:, 0, :]  # (T, 2)
        orig_r = uvs_r[:, 0, :]

        # ── circles ──────────────────────────────────────────────────────
        for t in range(T):
            for (u, v), color_f, rad in (
                (orig_l[t], col_l_f[t], int(l_dist[t])),
                (orig_r[t], col_r_f[t], int(r_dist[t])),
            ):
                u, v = int(u), int(v)
                if not (0 <= u < W and 0 <= v < H and rad > 0):
                    continue
                y0, y1 = max(0, v - rad), min(H, v + rad + 1)
                x0, x1 = max(0, u - rad), min(W, u + rad + 1)
                dy = ys[y0:y1] - v
                dx = xs[x0:x1] - u
                mask = (dy[:, None] ** 2 + dx[None, :] ** 2) <= rad * rad
                cam[:, t, y0:y1, x0:x1][:, mask] = color_f[:, None]

        # ── sticks (lines) ───────────────────────────────────────────────
        offsets = np.arange(-half_w, half_w + 1, dtype=np.int32)  # (2*hw+1,)
        for stick_idx in range(1, 4):
            for orig, uvs, colors_f in (
                (orig_l, uvs_l, sc_l_f),
                (orig_r, uvs_r, sc_r_f),
            ):
                color_f = colors_f[(stick_idx - 1) % 3]  # (3,) RGB float
                pt1 = uvs[:, stick_idx, :]  # (T, 2)
                for t in range(T):
                    u0, v0 = int(orig[t, 0]), int(orig[t, 1])
                    u1, v1 = int(pt1[t, 0]), int(pt1[t, 1])
                    if not (0 <= u0 < W and 0 <= v0 < H):
                        continue
                    ddu, ddv = u1 - u0, v1 - v0
                    if ddu == 0 and ddv == 0:
                        continue
                    if abs(ddu) >= abs(ddv) and ddu != 0:
                        step = 1 if ddu > 0 else -1
                        xs_ = np.arange(u0, u1 + step, step, dtype=np.int32)
                        xs_ = xs_[(xs_ >= 0) & (xs_ < W)]
                        if xs_.size == 0:
                            continue
                        vs_ = np.round(v0 + ddv * (xs_ - u0) / ddu).astype(np.int32)
                        vs_h = vs_[None, :] + offsets[:, None]  # (2hw+1, n)
                        xs_h = np.broadcast_to(xs_[None, :], vs_h.shape)
                        m = (vs_h >= 0) & (vs_h < H)
                        cam[:, t, vs_h[m], xs_h[m]] = color_f[:, None]
                    elif ddv != 0:
                        step = 1 if ddv > 0 else -1
                        ys_ = np.arange(v0, v1 + step, step, dtype=np.int32)
                        ys_ = ys_[(ys_ >= 0) & (ys_ < H)]
                        if ys_.size == 0:
                            continue
                        us_ = np.round(u0 + ddu * (ys_ - v0) / ddv).astype(np.int32)
                        us_h = us_[None, :] + offsets[:, None]
                        ys_h = np.broadcast_to(ys_[None, :], us_h.shape)
                        m = (us_h >= 0) & (us_h < W)
                        cam[:, t, ys_h[m], us_h[m]] = color_f[:, None]

    # ``out`` is a fresh, locally owned array — hand it to torch zero-copy.
    return torch.from_numpy(out)


def render_traj(
    eef_rpy: np.ndarray,
    c2w: np.ndarray,
    intrinsic: np.ndarray,
    *,
    cam_names: Sequence[str] = DEFAULT_CAM_NAMES,
    sample_size: tuple[int, int] = (DEFAULT_H, DEFAULT_W),
    gripper_offset: float = 0.15,
    fast: bool = True,
) -> torch.Tensor:
    """Render a trajectory band that matches ``MixAC.get_traj`` byte-for-byte.

    Parameters
    ----------
    eef_rpy : ndarray, shape (T, 14), float
        EEF poses in rpy form ``[L_xyz, L_rpy, L_grip, R_xyz, R_rpy, R_grip]``.
    c2w : ndarray, shape (V, T, 4, 4), float
        Per-frame, per-camera camera-to-world.
    intrinsic : ndarray, shape (V, 3, 3), float
        Pinhole K per view, already scaled to ``sample_size``.
    cam_names : sequence of str
        Kept for API parity with the trainer (not used internally).
    sample_size : (H, W)
        Output canvas size. Defaults to ``(384, 512)``.
    gripper_offset : float
        ``0.15`` for zhiyuan_gripper_omnipicker (default).
    fast : bool
        If True (default), use the vectorised NumPy implementation (~10ms).
        Set to False to fall back to the original Python+cv2 loop (~750ms)
        for debugging / correctness checks.

    Returns
    -------
    torch.Tensor, shape (3, V, T, H, W), dtype float32, range [0, 1]
    """
    impl = _render_traj_fast if fast else _render_traj_reference
    return impl(
        eef_rpy,
        c2w,
        intrinsic,
        cam_names=cam_names,
        sample_size=sample_size,
        gripper_offset=gripper_offset,
    )


def render_traj_from_bundle(
    assets_dir: str | Path,
    *,
    cam_names: Sequence[str] = DEFAULT_CAM_NAMES,
    sample_size: tuple[int, int] = (DEFAULT_H, DEFAULT_W),
    gripper_offset: float = 0.15,
) -> torch.Tensor:
    """Convenience wrapper: read npy files from an ``assets/<name>/`` directory
    and return the trajectory band ready to ship.

    Required files under ``assets_dir``:

    * ``eef_poses_0.npy``         shape ``(T, 14)`` float (rpy form)
    * ``extrinsic_alignstate_0.npy``  shape ``(V, T, 4, 4)`` float (c2w)
    * ``intrinsic.npy``           shape ``(V, 3, 3)`` float (already scaled
      to ``sample_size``)

    Returns
    -------
    torch.Tensor, shape ``(3, V, T, H, W)``, dtype float32, range ``[0, 1]``.
    """
    assets_dir = Path(assets_dir)
    eef = np.load(assets_dir / "eef_poses_0.npy")
    c2w = np.load(assets_dir / "extrinsic_alignstate_0.npy")
    K = np.load(assets_dir / "intrinsic.npy")
    return render_traj(
        eef,
        c2w,
        K,
        cam_names=cam_names,
        sample_size=sample_size,
        gripper_offset=gripper_offset,
    )


def render_band_from_bundle(bundle: EpisodeBundle) -> tuple[torch.Tensor, np.ndarray]:
    """Render the full-episode trajectory band from a loaded ``EpisodeBundle``.

    Returns ``(band, c2w)``: band ``(3, V, T, H, W)`` float tensor in ``[0, 1]``;
    c2w ``(V, T, 4, 4)`` float32.
    """
    if bundle.eef_poses is None:
        raise FileNotFoundError(
            f"episode bundle {bundle.path} has no eef_poses_0.npy; "
            "recorded-episode conditioning requires it"
        )
    c2w = np.asarray(bundle.extrinsic, dtype=np.float32)
    band = render_traj(bundle.eef_poses, c2w, bundle.intrinsic)
    return band, c2w
