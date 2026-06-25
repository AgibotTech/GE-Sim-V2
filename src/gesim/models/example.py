"""Synthetic world model for API smoke tests and CI. No GPU, no torch."""

from __future__ import annotations

import numpy as np

from gesim.models.base import StepResult, WorldModel


class ExampleWorldModel(WorldModel):
    """Returns deterministic synthetic frames and echoes actions as state."""

    chunk_size = 25

    def __init__(self):
        self._intrinsic: np.ndarray | None = None
        self._first_frame: np.ndarray | None = None
        self._traj: np.ndarray | None = None
        self._step_index = 0

    @classmethod
    def from_config(cls, config: dict) -> ExampleWorldModel:
        return cls()

    def reset(self) -> None:
        self._first_frame = None
        self._traj = None
        self._step_index = 0

    def set_camera_params(self, intrinsic, extrinsic=None) -> None:
        self._intrinsic = np.asarray(intrinsic, dtype=np.float32)

    def set_episode_data(self, first_frame) -> None:
        frame = np.asarray(first_frame, dtype=np.float32)
        if frame.ndim != 4 or frame.shape[0] != 3:
            raise ValueError(f"first_frame must be (3, V, H, W), got {frame.shape}")
        self._first_frame = frame

    def set_episode_traj(self, traj, c2w) -> None:
        traj = np.asarray(traj, dtype=np.float32)
        if traj.ndim != 5 or traj.shape[0] != 3:
            raise ValueError(f"traj must be (3, V, T, H, W), got {traj.shape}")
        self._traj = traj

    def step(self, actions) -> StepResult:
        if self._first_frame is None:
            raise RuntimeError("set_episode_data was not called before step")
        if self._traj is None:
            raise RuntimeError("set_episode_traj was not called before step")
        actions = np.asarray(actions, dtype=np.float32)
        num_frames = actions.shape[0]
        # First frame with a brightness ramp so videos visibly advance.
        ramp = np.linspace(0.0, 0.25, num_frames, dtype=np.float32)
        frames = np.clip(
            self._first_frame[None] + ramp[:, None, None, None, None], 0.0, 1.0
        ).astype(np.float32)
        self._step_index += 1
        return StepResult(frames=frames, state=actions.copy())
