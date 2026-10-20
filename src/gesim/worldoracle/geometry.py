"""WorldOracle geometry, without a training-repository dependency."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from gesim.conditioning.band import _quat_pose_to_transform_matrix, _rpy_pose_to_quat_pose

from . import yam
from .band import render_traj_g02

CAMERAS = ("top_head", "hand_left", "hand_right")


def field(rows, descriptions, name):
    spec = descriptions[name]
    indices = spec.get("indices", list(range(spec.get("start", 0), spec.get("end", 0))))
    if not indices:
        raise ValueError(f"Empty required field: {name}")
    return rows[:, indices]


class BandRobot:
    def __init__(self, embodiment):
        self.yam = embodiment == "yam"
        self.traj_radius_depth_range = (0.05, 2.2 if self.yam else 1.286)
        self.offset = 0.14465 if self.yam else 0.21

    def pose_rotation_matrix(self):
        if self.yam:
            return None
        return torch.tensor(
            [
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).view(1, 1, 4, 4)

    def grip_colormap_norm(self, g):
        t = g.clamp(0, 1) if self.yam else 1 - ((g + 0.91) / 0.91).clamp(0, 1)
        return (t * 85 + 35) / 120


def yam_geometry(action, state, schema, calibration):
    desc = schema["features"]["action"]["field_descriptions"]
    joints = np.concatenate(
        [
            field(action, desc, k)
            for k in (
                "joint_pos_action_left",
                "gripper_pos_action_left",
                "joint_pos_action_right",
                "gripper_pos_action_right",
            )
        ],
        axis=-1,
    )
    entry = yam.load_yam_calib(str(calibration))[-1]
    pose = yam.joints_to_eef14(joints)
    state_desc = schema["features"]["observation.state"]["field_descriptions"]
    observed = np.concatenate(
        [
            field(state, state_desc, k)
            for k in (
                "joint_pos_obs_left",
                "gripper_pos_obs_left",
                "joint_pos_obs_right",
                "gripper_pos_obs_right",
            )
        ],
        axis=-1,
    )
    # YAM training obtains wrist cameras from observed state, while the band
    # encodes commanded EEF poses. Do not replace observed cameras by action FK.
    K, c2w, _ = yam.yam_cameras(entry, yam.YAM_CAMS, yam.joints_to_flanges(observed))
    band_h = entry["band"][1] - entry["band"][0] + 1
    K[:, 0, :] *= 512 / 224
    K[:, 1, :] *= 384 / band_h
    return pose, K, c2w, c2w.copy(), band_h


def _g02_body_frames(rows, desc, prefix):
    """Fixed G02 waist/head geometry in metres; no robot-description file needed."""
    waist = field(rows, desc, prefix + "waist/position")
    head = field(rows, desc, prefix + "head/position")

    def joint(xyz, axis, angle):
        t = np.tile(np.eye(4), (len(rows), 1, 1))
        t[:, :3, 3] = xyz
        t[:, :3, :3] = Rotation.from_euler(axis, angle).as_matrix()
        return t

    body = np.tile(np.eye(4), (len(rows), 1, 1))
    for i, (xyz, axis) in enumerate(
        zip(
            [(0.102, 0, 0.144), (0, 0, 0.36), (0, 0, 0.38), (0, 0, 0), (0, 0, 0.149)],
            "yyyxz",
            strict=False,
        )
    ):
        body = body @ joint(xyz, axis, waist[:, i])
    arm_base = body @ joint((0, 0, 0.3085), "x", -np.pi / 2)
    for i, (xyz, axis) in enumerate(
        zip([(-0.013, 0, 0.496), (0.03335, 0, 0.07), (-0.03035, 0, 0)], "zxy", strict=False)
    ):
        body = body @ joint(xyz, axis, head[:, i])
    return arm_base, body


def _g02_end_frames(rows, desc, prefix):
    xyz = field(rows, desc, prefix + "end/position").reshape(-1, 2, 3)
    quat = field(rows, desc, prefix + "end/orientation").reshape(-1, 2, 4)
    if not np.allclose(np.linalg.norm(quat, axis=-1), 1.0, atol=1e-3):
        raise ValueError(f"G02 {prefix}end/orientation must contain unit XYZW quaternions")
    t = np.tile(np.eye(4), (len(rows), 2, 1, 1))
    t[:, :, :3, 3] = xyz
    t[:, :, :3, :3] = Rotation.from_quat(quat.reshape(-1, 4)).as_matrix().reshape(-1, 2, 3, 3)
    return t


def g02_geometry(action, state, schema, calibration, fps=30.0):
    """Dataset end poses with action-aligned cameras and integrated velocity rays."""
    da = schema["features"]["action"]["field_descriptions"]
    ds = schema["features"]["observation.state"]["field_descriptions"]
    arm_base, action_head = _g02_body_frames(action, da, "action/")
    _, state_head = _g02_body_frames(state, ds, "state/")
    # action/end is arm-base-relative; state/end is robot-base-relative.
    # Both orientations are XYZW, despite the stale Euler schema description.
    fa = np.concatenate(
        [arm_base[:, None] @ _g02_end_frames(action, da, "action/"), action_head[:, None]], axis=1
    )
    fs = np.concatenate([_g02_end_frames(state, ds, "state/"), state_head[:, None]], axis=1)
    pose = np.zeros((len(action), 14), dtype=np.float32)
    for arm, off in ((0, 0), (1, 7)):
        pose[:, off : off + 3] = fa[:, arm, :3, 3]
        pose[:, off + 3 : off + 6] = Rotation.from_matrix(fa[:, arm, :3, :3]).as_euler("xyz")
        side = "left" if arm == 0 else "right"
        pose[:, off + 6] = field(action, da, f"action/{side}_effector/position")[:, 0]
    c2w = np.tile(np.eye(4), (3, len(state), 1, 1))
    K = []
    for v, (cam, tag) in enumerate(
        zip(
            CAMERAS,
            ("head_front_rgbd_aligned", "hand_left_rgbd_aligned", "hand_right_rgbd_aligned"),
            strict=False,
        )
    ):
        c2w[v, :, :3, :3] = field(state, ds, f"extrinsic_end_T_{tag}/rotation_matrix").reshape(
            -1, 3, 3
        )
        c2w[v, :, :3, 3] = field(state, ds, f"extrinsic_end_T_{tag}/translation_vector")
        intr = calibration["cameras"][cam]["intrinsics"]
        # WorldOracle calibration K is expressed at the decoded 640px width.
        # Wrist 1280x1056 metadata refers to the physical sensor, not these pixels.
        k = np.array([[intr["Fx"], 0, intr["Cx"]], [0, intr["Fy"], intr["Cy"]], [0, 0, 1.0]])
        source_h = 400 if cam == "top_head" else 528
        k[0] *= 512 / 640
        k[1] *= 384 / source_h
        K.append(k)
    # Action-map poses use the same slice-first anchor as training. Frame zero
    # remains in every inference memory window, so the reference never changes.
    base = np.tile(np.eye(4), (len(state), 1, 1))
    base[:, :3, :3] = Rotation.from_quat(field(state, ds, "state/robot/orientation")).as_matrix()
    base[:, :3, 3] = field(state, ds, "state/robot/position")
    anchor = np.linalg.inv(base[0]) @ base
    state_pose = pose.copy()
    for transforms, target in ((fa, pose), (fs, state_pose)):
        for arm, off in ((0, 0), (1, 7)):
            eef = anchor @ transforms[:, arm]
            target[:, off : off + 3] = eef[:, :3, 3]
            target[:, off + 3 : off + 6] = Rotation.from_matrix(eef[:, :3, :3]).as_euler("xyz")

    # Match integrate_base_velocity: integrate every raw frame, not just sampled
    # history/future indices. The body-frame planar increment composes on the right.
    velocity = field(action, da, "action/robot/velocity")
    if velocity.shape[1] not in (2, 6) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Expected 2/6-channel base velocity and positive native FPS")
    integrated = np.tile(np.eye(4), (len(action), 1, 1))
    for i, row in enumerate(velocity[:-1]):
        vx, vy, wz = (row[0], row[1], row[5]) if len(row) == 6 else (row[0], 0.0, row[1])
        angle = float(wz) / fps
        c, sn = np.cos(angle), np.sin(angle)
        delta = np.array(
            [
                [c, -sn, 0, float(vx) / fps],
                [sn, c, 0, float(vy) / fps],
                [0, 0, 1, 0],
                [0, 0, 0, 1.0],
            ]
        )
        integrated[i + 1] = integrated[i] @ delta

    action_q, state_q = _rpy_pose_to_quat_pose(pose), _rpy_pose_to_quat_pose(state_pose)
    wrist_corrections = []
    for offset in (0, 8):
        sa = _quat_pose_to_transform_matrix(state_q[:, offset : offset + 7])
        aa = _quat_pose_to_transform_matrix(action_q[:, offset : offset + 7])
        wrist_corrections.append(sa @ torch.linalg.inv(aa))

    def align_cameras(world):
        # This ordering deliberately follows MixAC.align_extrinsics: wrist
        # corrections use the action-map poses even for velocity-window rays.
        w2c = torch.linalg.inv(torch.from_numpy((world @ c2w).astype(np.float32)))
        hs = torch.from_numpy((world @ fs[:, 2]).astype(np.float32))
        ha = torch.from_numpy((world @ fa[:, 2]).astype(np.float32))
        w2c[0] = w2c[0] @ (hs @ torch.linalg.inv(ha))
        for v, correction in enumerate(wrist_corrections, 1):
            w2c[v] = w2c[v] @ correction
        return torch.linalg.inv(w2c).numpy()

    return pose, np.asarray(K, np.float32), align_cameras(integrated), align_cameras(anchor), None


def render_conditioning(pose, K, ray_c2w, band_c2w, indices, embodiment):
    idx = np.asarray(indices)
    robot = BandRobot(embodiment)
    band = render_traj_g02(
        _rpy_pose_to_quat_pose(pose[idx]),
        torch.from_numpy(np.linalg.inv(band_c2w[:, idx]).astype(np.float32)),
        torch.from_numpy(K),
        robot=robot,
        gripper_offset=robot.offset,
        sample_size=(384, 512),
    )
    return band, ray_c2w[:, idx]
