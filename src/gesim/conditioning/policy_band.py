"""Closed-loop trajectory band renderer driven by a 16-D joint policy.

This module is the closed-loop counterpart to :mod:`gesim.conditioning.band`:
that one consumes the **recorded** ``eef_poses_0.npy`` + per-frame ``c2w`` from
an episode bundle, while this one consumes a **fresh** policy action chunk and
synthesises both:

  * the 14-D EEF poses (rpy form) the band renderer expects, via FK on the
    policy's 7 arm joints (left/right);
  * a per-frame ``(V, T, 4, 4)`` ``c2w`` where the head camera stays on the
    episode's recorded mount and the wrist cameras follow the FK-derived
    motion of ``Link7_l`` / ``Link7_r`` (the wrists are rigidly attached
    to those links).

Geometry recap
==============
Body assumption: the robot's torso, head and waist are stationary at
inference time; only the two arms move. Therefore:

  * ``c2w_head[t] = c2w_head[0]``  for every t (read straight from
    ``extrinsic_alignstate_0.npy[0, 0]``).
  * ``c2w_wrist[t] = T_base_link7[t] @ T_link7_camera``
    where ``T_link7_camera`` is the rigid wrist→camera mount, recovered
    once at episode start as
    ``T_link7_camera = inv(T_base_link7[0]) @ extrinsic_alignstate_0[wrist, 0]``.
    ``T_base_link7[t]`` comes from running FK on the policy action's 7 arm
    joints (left or right). Head/waist angles stay pinned to the episode's
    recorded values, read from ``state_joints_0.npy``.

The closed-loop band uses the same drawing recipe as the recorded path while
the wrist ``c2w`` evolves with the policy the way the real robot's wrist
cameras would.

Bundle requirements
===================
``render_policy_traj_from_bundle`` expects an ``assets/<name>/`` directory
containing:

  * ``intrinsic.npy``                — ``(V, 3, 3)`` float, baked at 384×512
  * ``extrinsic_alignstate_0.npy``   — ``(V, T, 4, 4)`` float, c2w per view
  * ``state_eef_poses_0.npy``        — ``(T, 14)`` float, **state**-FK EEF rpy
  * ``state_joints_0.npy``           — ``(T, 20)`` float, trainer-layout joint rows

FK runs in the bundled compiled Genie-01 (G01) kinematics (see
:mod:`gesim.conditioning.kinematics`).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from gesim.conditioning.band import (
    DEFAULT_CAM_NAMES,
    DEFAULT_H,
    DEFAULT_W,
    render_traj,
)

if TYPE_CHECKING:
    from gesim.episode import EpisodeBundle


# The last 4 columns of ``state_joints_0`` are the head/waist joints, held
# constant through closed loop (only the arms move).
_HEAD_WAIST_SLICE = slice(16, 20)


def _default_kinematics():
    """The bundled compiled Genie-01 (G01) FK backend."""
    from gesim.conditioning.kinematics import CompiledKinematics

    return CompiledKinematics()


def _xyzquat_to_mat(xyzquat_xyzw: np.ndarray) -> np.ndarray:
    """``(7,)`` ``[xyz, qx, qy, qz, qw]`` → ``(4, 4)`` rigid transform."""
    from scipy.spatial.transform import Rotation as _R

    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = _R.from_quat(xyzquat_xyzw[3:]).as_matrix()
    out[:3, 3] = xyzquat_xyzw[:3]
    return out


def _eef_rpy_to_mat14(eef_rpy_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One row of ``(14,)`` EEF rpy → two ``(4, 4)`` transforms ``(T_left, T_right)``."""
    from scipy.spatial.transform import Rotation as _R

    out_l = np.eye(4, dtype=np.float64)
    out_l[:3, :3] = _R.from_euler("xyz", eef_rpy_row[3:6]).as_matrix()
    out_l[:3, 3] = eef_rpy_row[0:3]
    out_r = np.eye(4, dtype=np.float64)
    out_r[:3, :3] = _R.from_euler("xyz", eef_rpy_row[10:13]).as_matrix()
    out_r[:3, 3] = eef_rpy_row[7:10]
    return out_l, out_r


def _policy_actions16_to_eef14(
    actions16: np.ndarray,
    head_waist_const: np.ndarray,
    *,
    kinematics,
) -> np.ndarray:
    """``(L, 16)`` policy actions + held head/waist → ``(L, 14)`` EEF rpy.

    The action→joint mapping and FK both run inside the compiled kinematics; this
    only assembles the 14-D EEF-rpy rows the band renderer consumes.
    """
    from scipy.spatial.transform import Rotation as _R

    if actions16.ndim != 2 or actions16.shape[1] < 16:
        raise ValueError(f"actions16 must be (L, >=16); got {actions16.shape}")
    if head_waist_const.shape != (4,):
        raise ValueError(f"head_waist_const must be (4,); got {head_waist_const.shape}")

    L = int(actions16.shape[0])
    out = np.zeros((L, 14), dtype=np.float32)
    for t in range(L):
        a = actions16[t]
        l_pose, r_pose = kinematics.fk_action(a, head_waist_const)
        out[t, 0:3] = l_pose[:3]
        out[t, 3:6] = _R.from_quat(l_pose[3:]).as_euler("xyz", degrees=False)
        out[t, 6:7] = a[7]
        out[t, 7:10] = r_pose[:3]
        out[t, 10:13] = _R.from_quat(r_pose[3:]).as_euler("xyz", degrees=False)
        out[t, 13:14] = a[15]
    return out


def _wrist_c2w_from_eef(
    eef_rpy: np.ndarray,
    eef0_rpy: np.ndarray,
    c2w0: np.ndarray,
    *,
    is_left: bool,
) -> np.ndarray:
    """``(L, 14)`` EEF rpy + episode-start anchor → ``(L, 4, 4)`` wrist c2w.

    ``T_link7_camera = inv(T_base_link7[0]) @ c2w_wrist[0]``
    ``c2w_wrist[t]   = T_base_link7[t] @ T_link7_camera``

    All link7 transforms come from the EEF rpy poses (which are themselves
    FK outputs at link7_l / link7_r in the trainer's URDF).
    """
    L = int(eef_rpy.shape[0])
    base_link7_0_l, base_link7_0_r = _eef_rpy_to_mat14(eef0_rpy)
    base_link7_0 = base_link7_0_l if is_left else base_link7_0_r
    T_link7_camera = np.linalg.inv(base_link7_0) @ c2w0
    out = np.zeros((L, 4, 4), dtype=np.float32)
    for t in range(L):
        l_mat, r_mat = _eef_rpy_to_mat14(eef_rpy[t])
        base_link7_t = l_mat if is_left else r_mat
        out[t] = (base_link7_t @ T_link7_camera).astype(np.float32)
    return out


def render_policy_traj_from_bundle(
    actions16: np.ndarray,
    bundle_dir: str | Path,
    *,
    cam_names: Sequence[str] = DEFAULT_CAM_NAMES,
    sample_size: tuple[int, int] = (DEFAULT_H, DEFAULT_W),
    gripper_offset: float = 0.15,
    kinematics=None,
    bundle_cache: dict | None = None,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Render a closed-loop trajectory band + per-frame c2w from a PI05 chunk.

    Parameters
    ----------
    actions16 : ``(L, 16)`` float
        Policy action chunk in ``[L7, L_grip, R7, R_grip]`` layout (PI05).
    bundle_dir : path
        Episode assets dir (``assets/<name>/``). Must contain
        ``intrinsic.npy``, ``extrinsic_alignstate_0.npy``,
        ``state_joints_0.npy`` and ``state_eef_poses_0.npy``.
    cam_names, sample_size, gripper_offset : passed to
        ``gesim.conditioning.band.render_traj``.
    kinematics : internal — a cached ``CompiledKinematics`` (built if None).
    bundle_cache : optional dict with pre-loaded npy arrays, keyed by filename stem.
        Provide to avoid re-reading files on every sub-chunk call within the same
        episode.  Build once per episode with ``load_bundle_cache(bundle_dir)``.
        Expected keys: ``state_joints``, ``state_eef``, ``c2w_episode``, ``intrinsic``.

    Returns
    -------
    traj : ``torch.float32`` ``(3, V, L, H, W)`` in ``[0, 1]``
        Same layout the world-model server expects from ``set_episode_traj``.
    c2w  : ``np.float32`` ``(V, L, 4, 4)``
        Per-frame camera-to-world. Head row is constant, wrist rows follow
        the FK-derived link7 motion.
    eef_rpy : ``np.float32`` ``(L, 14)``
        FK output (debug-friendly; same array fed to ``render_traj``).
    """
    actions16 = np.asarray(actions16, dtype=np.float32)
    bundle = Path(bundle_dir)

    if bundle_cache is not None:
        state_joints = bundle_cache["state_joints"]
        state_eef = bundle_cache["state_eef"]
        c2w_episode = bundle_cache["c2w_episode"]
        intrinsic = bundle_cache["intrinsic"]
    else:
        state_joints = np.load(bundle / "state_joints_0.npy").astype(np.float32)
        state_eef = np.load(bundle / "state_eef_poses_0.npy").astype(np.float32)
        c2w_episode = np.load(bundle / "extrinsic_alignstate_0.npy").astype(np.float32)
        intrinsic = np.load(bundle / "intrinsic.npy").astype(np.float32)

    if state_joints.shape[1] < 20:
        raise ValueError(
            f"state_joints_0.npy expected (T, 20); got {state_joints.shape}. Re-export the bundle."
        )
    head_waist_const = state_joints[0, _HEAD_WAIST_SLICE].astype(np.float32)

    if kinematics is None:
        kinematics = _default_kinematics()

    eef_rpy = _policy_actions16_to_eef14(
        actions16,
        head_waist_const=head_waist_const,
        kinematics=kinematics,
    )

    # Per-view c2w: head stays put, wrists follow link7 FK.
    L = int(actions16.shape[0])
    V = c2w_episode.shape[0]
    if V != 3:
        raise ValueError(f"expected V=3 in extrinsic_alignstate_0; got V={V}")
    c2w_out = np.zeros((V, L, 4, 4), dtype=np.float32)
    c2w_out[0] = np.broadcast_to(c2w_episode[0, 0], (L, 4, 4))
    c2w_out[1] = _wrist_c2w_from_eef(
        eef_rpy, state_eef[0], c2w_episode[1, 0].astype(np.float64), is_left=True
    )
    c2w_out[2] = _wrist_c2w_from_eef(
        eef_rpy, state_eef[0], c2w_episode[2, 0].astype(np.float64), is_left=False
    )

    traj = render_traj(
        eef_rpy,
        c2w_out,
        intrinsic,
        cam_names=cam_names,
        sample_size=sample_size,
        gripper_offset=gripper_offset,
    )
    return traj, c2w_out, eef_rpy


def load_bundle_cache(bundle_dir: str | Path) -> dict:
    """Pre-load all npy assets for one episode bundle.

    Returns a dict suitable for passing as ``bundle_cache`` to
    ``render_policy_traj_from_bundle``.  Call once per episode, reuse across
    all sub-chunk calls within that episode to avoid repeated disk I/O.
    """
    bundle = Path(bundle_dir)
    return {
        "state_joints": np.load(bundle / "state_joints_0.npy").astype(np.float32),
        "state_eef": np.load(bundle / "state_eef_poses_0.npy").astype(np.float32),
        "c2w_episode": np.load(bundle / "extrinsic_alignstate_0.npy").astype(np.float32),
        "intrinsic": np.load(bundle / "intrinsic.npy").astype(np.float32),
    }


class PolicyBandRenderer:
    """Renders per-chunk trajectory bands from policy actions for one episode.

    Construction loads the FK backend, joint map, and bundle cache; build once
    per episode and call :meth:`render` per action chunk.
    """

    def __init__(self, bundle: EpisodeBundle):
        self._kinematics = _default_kinematics()
        self._bundle_cache = load_bundle_cache(bundle.path)
        self._bundle_path = bundle.path

    def render(self, actions: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """``(L, 16)`` actions -> ``(band (3, V, L, H, W), c2w (V, L, 4, 4))``.

        Returns
        -------
        band : torch.Tensor, dtype float32, shape (3, V, L, H, W), values in [0, 1]
        c2w  : numpy.ndarray, dtype float32, shape (V, L, 4, 4)
        """
        band, c2w, _ = render_policy_traj_from_bundle(
            actions,
            self._bundle_path,
            kinematics=self._kinematics,
            bundle_cache=self._bundle_cache,
        )
        return band, c2w
