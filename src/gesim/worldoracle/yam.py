"""YAM (I2RT bimanual station) forward kinematics and camera calibration conversion.

The training export uses LeRobot v3.0 with robot_type yam_bimanual:
``observation.state`` / ``action`` are 14-d ``[L_j1..6, L_grip, R_j1..6, R_grip]`` (radians,
gripper opening in [0, 1], 1 = open); three letterboxed 224x224 AV1 streams
``observation.images.{top,left_wrist,right_wrist}``.

World frame = left arm base (URDF ``base`` link); the right base sits at (-0.004, -0.6065,
-0.006) m (station spec 0.61 m, fitted against the 46-d XDOF export). FK uses the public
I2RT ``yam.urdf`` chain (joint signs +1, offsets 0).

EEF frame for GE-Sim (pose16 / action map): the flange (joint6 frame) rotated 180 deg about
X, i.e. the station's tcp-site orientation, so +Z is the approach direction (G02 convention:
``gripper_offset`` is applied along +Z) and Y is the finger-opening axis. ``gripper_offset``
= +0.14465 m puts the disk at the tool-centre point.

Calibration file (``yam_calib.json``, ~600 B per episode): per episode ``band`` (letterbox
content rows), ``top`` (``pose`` = rotvec+t of world_T_top, ``f``, ``c`` in the full 224x224
frame), ``left_wrist``/``right_wrist`` (``X`` = rotvec+t of flange_T_cam, ``f``).
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation as Rot

YAM_CAMS = (
    "observation.images.top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
YAM_CAM_SHORT = {
    "observation.images.top": "top",
    "observation.images.left_wrist": "left_wrist",
    "observation.images.right_wrist": "right_wrist",
}

# ── I2RT yam.urdf (i2rt/robot_models/arm/yam/v1) joint chain: (origin xyz, origin rpy, axis) ──
_YAM_JOINTS = [
    (
        [-2.49777947129965e-07, -9.17402363407408e-13, 0.0679999999995413],
        [0, 1.5708, 3.1415963267949],
        [-1, 0, 0],
    ),
    ([-0.0455, -0.0339, -0.02], [0, 0, -3.14159], [0, 1, 0]),
    ([4.08431e-07, -0.0688, 0.264], [0, 0, -3.14159], [0, 1, 0]),
    ([-0.0600003, -0.0688, -0.244999], [0, 0, 0], [0, 1, 0]),
    ([-0.0405003, 0.0338995, -0.0739989], [0, 0, 0], [1, 0, 0]),
    ([0.0404996, 2.39858e-07, -0.0356], [0, 0, 0], [0, 0, -1]),
]


def _origin(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = Rot.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


_ORIGINS = np.stack([_origin(x, r) for x, r, _ in _YAM_JOINTS])
_AXES = np.stack([np.asarray(a, float) / np.linalg.norm(a) for _, _, a in _YAM_JOINTS])

BASE_T_RIGHT = np.eye(4)
BASE_T_RIGHT[:3, 3] = [-0.004, -0.6065, -0.006]
EEF_FROM_FLANGE = np.diag([1.0, -1.0, -1.0, 1.0])
TCP_OFFSET = 0.14465  # along the EEF +Z (approach)
TRAJ_RADIUS_DEPTH_RANGE = (0.050, 2.2)


def fk_flange(q):
    """Batched URDF FK: q (N, 6) rad -> base_T_flange (N, 4, 4)."""
    q = np.asarray(q, dtype=np.float64)
    N = q.shape[0]
    T = np.tile(np.eye(4), (N, 1, 1))
    for j in range(6):
        Tj = np.tile(np.eye(4), (N, 1, 1))
        Tj[:, :3, :3] = Rot.from_rotvec(np.outer(q[:, j], _AXES[j])).as_matrix()
        T = T @ _ORIGINS[j] @ Tj
    return T


def joints_to_flanges(j14):
    """(N, 14) joints -> world_T_flange (N, 2, 4, 4) for [left, right]."""
    j14 = np.asarray(j14, dtype=np.float64)
    out = np.zeros((len(j14), 2, 4, 4))
    out[:, 0] = fk_flange(j14[:, 0:6])
    out[:, 1] = BASE_T_RIGHT @ fk_flange(j14[:, 7:13])
    return out


def joints_to_eef14(j14):
    """(N, 14) joints -> GE-Sim 14-d pose ``[L_xyz, L_rpy, L_grip, R_xyz, R_rpy, R_grip]``
    in the world frame with the GE-Sim EEF convention (rpy = scipy 'xyz' extrinsic, matching
    ``build_lerobot_end_eef`` / ``rpy_pose2_quat_pose``)."""
    fl = joints_to_flanges(j14)
    out = np.zeros((len(j14), 14), np.float32)
    for k, off in ((0, 0), (1, 7)):
        eef = fl[:, k] @ EEF_FROM_FLANGE
        out[:, off : off + 3] = eef[:, :3, 3]
        out[:, off + 3 : off + 6] = Rot.from_matrix(eef[:, :3, :3]).as_euler("xyz", degrees=False)
    out[:, 6] = j14[:, 6]
    out[:, 13] = j14[:, 13]
    return out


def _se3(p6):
    T = np.eye(4)
    T[:3, :3] = Rot.from_rotvec(np.asarray(p6[:3], float)).as_matrix()
    T[:3, 3] = p6[3:6]
    return T


@lru_cache(maxsize=4)
def load_yam_calib(path):
    """yam_calib.json -> {episode_index: entry} with matrices unpacked."""
    with open(path) as f:
        raw = json.load(f)
    recs = raw.get("episodes", raw)
    out = {}
    for k, r in recs.items():
        if not r.get("ok", True) or "top" not in r:
            continue
        top = r["top"]
        K_top = np.array(
            [[top["f"][0], 0, top["c"][0]], [0, top["f"][1], top["c"][1]], [0, 0, 1.0]]
        )
        entry = dict(
            band=(int(r["band"][0]), int(r["band"][1])),
            world_T_top=_se3(top["pose"]),
            K224={"top": K_top},
            X={},
        )
        for cam in ("left_wrist", "right_wrist"):
            w = r[cam]
            entry["X"][cam] = _se3(w["X"])
            entry["K224"][cam] = np.array([[w["f"], 0, 112.0], [0, w["f"], 112.0], [0, 0, 1.0]])
        out[int(k)] = entry
    return out


def yam_cameras(entry, cam_names, flanges):
    """Per-frame cameras for the requested cams.

    entry:    calibration entry from ``load_yam_calib``
    cam_names: iterable of lerobot camera keys (``observation.images.*``)
    flanges:  world_T_flange (T, 2, 4, 4) at the sampled frames
    Returns intrinsics (V, 3, 3) in BAND-CROP pixel coordinates (rows r0..r1 cropped away
    before ``transform_videos``), c2w (V, T, 4, 4), w2c (V, T, 4, 4) as float32 numpy.
    """
    r0, _ = entry["band"]
    T = flanges.shape[0]
    K, c2w = [], []
    for cam in cam_names:
        short = YAM_CAM_SHORT.get(cam, cam)
        Kc = entry["K224"][short].copy()
        if not (np.isfinite(Kc).all() and Kc[0, 0] > 1.0 and Kc[1, 1] > 1.0):
            raise ValueError(
                f"YAM calibration for episode {entry.get('ep')} camera {short} has a degenerate "
                f"focal ({Kc[0, 0]:.3f}, {Kc[1, 1]:.3f}); "
                "regenerate yam_calib.json with merge_calib.py"
            )
        Kc[1, 2] -= r0
        K.append(Kc)
        if short == "top":
            c2w.append(np.tile(entry["world_T_top"], (T, 1, 1)))
        else:
            k = 0 if short == "left_wrist" else 1
            c2w.append(flanges[:, k] @ entry["X"][short])
    K = np.stack(K).astype(np.float32)
    c2w = np.stack(c2w).astype(np.float64)
    w2c = np.linalg.inv(c2w)
    return K, c2w.astype(np.float32), w2c.astype(np.float32)
