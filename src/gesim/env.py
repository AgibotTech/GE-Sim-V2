"""Gym-style environment over a gesim world-model server."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gesim.action_chunk import compress_action_chunk
from gesim.client.transport import WorldModelClient
from gesim.conditioning.band import render_band_from_bundle
from gesim.conditioning.policy_band import PolicyBandRenderer
from gesim.episode import EpisodeBundle
from gesim.rewards.base import RewardClient
from gesim.types import (
    ACTION_DIM,
    Observation,
    StepInfo,
    frame_to_view_images,
    head_view_frames,
    wm_state_to_policy_state,
)
from gesim.video import save_video

CONDITIONING_MODES = ("action", "episode")


class WorldModelEnv:
    """Action-conditioned world model behind a gym-style reset/step API.

    Example::

        env = WorldModelEnv("http://localhost:9000")
        obs = env.reset("assets/demo_000")
        obs, reward, state, info = env.step(actions)   # actions: (L, 16) joint-space
        env.save_video("rollout.mp4")

    By default every generated frame is kept in memory so ``save_video`` can
    write the full rollout afterwards (roughly 7 MB per frame at 3x384x512);
    pass ``keep_frames=False`` for long rollouts and consume frames from
    ``StepInfo`` instead.

    By default (``compress_actions=True``), a ``step`` chunk longer than
    ``chunk_size`` is compressed (server-parity avg-pool) down to one model chunk,
    so the world model runs a single inference per turn (e.g. a 50-action pi05
    round -> 25) — faster, at the cost of temporal resolution. Pass
    ``compress_actions=False`` to instead split the chunk into model-sized
    sub-chunks (one inference each), keeping the full horizon.
    """

    def __init__(
        self,
        server_url: str,
        *,
        reward: RewardClient | None = None,
        chunk_size: int = 25,
        keep_frames: bool = True,
        compress_actions: bool = True,
        user_name: str = "gesim",
        timeout: float = 300.0,
    ):
        self._client = WorldModelClient(server_url, user_name=user_name, timeout=timeout)
        self._reward = reward
        self._chunk_size = int(chunk_size)
        self._keep_frames = bool(keep_frames)
        self._compress_actions = bool(compress_actions)
        self._bundle: EpisodeBundle | None = None
        self._task: str = ""
        self._band_renderer: PolicyBandRenderer | None = None
        self._frames: list[np.ndarray] = []
        self._obs: Observation | None = None

    def reset(
        self,
        episode: str | Path | EpisodeBundle,
        task: str | None = None,
        *,
        conditioning: str = "action",
    ) -> Observation:
        """Start a new episode from a bundle.

        Args:
            episode: bundle directory or a loaded ``EpisodeBundle``.
            task: task instruction; defaults to the bundle's ``task.txt``.
            conditioning: ``"action"`` renders the trajectory band from the
                actions passed to ``step()`` via FK (closed loop);
                ``"episode"`` uploads the band rendered once from the bundle's
                recorded end-effector poses (training-parity replay).
        """
        if conditioning not in CONDITIONING_MODES:
            raise ValueError(
                f"conditioning must be one of {CONDITIONING_MODES}, got {conditioning!r}"
            )
        bundle = episode if isinstance(episode, EpisodeBundle) else EpisodeBundle.load(episode)
        self._bundle = bundle
        self._task = task or bundle.task
        self._frames = []

        self._client.reset()
        self._client.set_task(self._task)
        self._client.set_camera_params(bundle.intrinsic, bundle.initial_extrinsic)
        self._client.set_episode_data(bundle.first_frame)

        if conditioning == "episode":
            band, c2w = render_band_from_bundle(bundle)
            self._client.set_episode_traj(np.asarray(band), c2w)
            self._band_renderer = None
        else:
            self._band_renderer = PolicyBandRenderer(bundle)

        self._obs = bundle.first_observation(self._task)
        return self._obs

    def step(
        self, actions: np.ndarray
    ) -> tuple[Observation, np.ndarray | None, np.ndarray | None, StepInfo]:
        """Advance the world model by an action chunk.

        Args:
            actions: ``(L, 16)`` float32 absolute-joint actions in WM layout
                ``[L7_arm, L_grip, R7_arm, R_grip]``. ``L`` may exceed the model
                chunk size; the env splits it into model-sized sub-chunks (or, with
                ``compress_actions=True``, compresses it to a single chunk).

        Returns:
            ``(obs, reward, state, info)``.

            - ``reward``: ``(T,)`` per-frame success probability when a reward
              client is configured (the reward model scores this step's frames
              as one independent window), else None.
            - ``state``: ``(T, 16)`` float32 Pose-Expert robot state predicted by
              the world model, in WM layout ``[L7_arm, L_grip, R7_arm, R_grip]``,
              or None if the model returns no state. ``obs.state`` is the last row
              reordered to policy layout for the next policy call.
            - ``info``: a ``StepInfo`` carrying the generated frames and per-frame
              task progress (when a reward client is configured).
        """
        if self._bundle is None:
            raise RuntimeError("call reset() before step()")
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError(f"actions must be (L, {ACTION_DIM}), got {actions.shape}")

        # Compress a long chunk to one model chunk -> a single WM inference per turn.
        if self._compress_actions and actions.shape[0] > self._chunk_size:
            actions = compress_action_chunk(actions, self._chunk_size)

        frame_parts: list[np.ndarray] = []
        state_parts: list[np.ndarray] = []
        for start in range(0, actions.shape[0], self._chunk_size):
            chunk = actions[start : start + self._chunk_size]
            if self._band_renderer is not None:
                band, c2w = self._band_renderer.render(chunk)
                self._client.set_episode_traj(np.asarray(band), c2w)
            frames, state = self._client.step(chunk)
            n = min(chunk.shape[0], frames.shape[0])
            frame_parts.append(frames[:n])
            if state is not None:
                state_parts.append(state[:n])

        frames = np.concatenate(frame_parts, axis=0)
        state = np.concatenate(state_parts, axis=0) if state_parts else None
        if self._keep_frames:
            self._frames.append(frames)

        reward = progress = None
        if self._reward is not None:
            result = self._reward.evaluate(head_view_frames(frames), self._task)
            reward, progress = result.success, result.progress

        next_state = (
            wm_state_to_policy_state(state[-1])
            if state is not None and state.shape[-1] >= ACTION_DIM
            else wm_state_to_policy_state(actions[-1])
        )
        self._obs = Observation(
            images=frame_to_view_images(frames[-1]), state=next_state, task=self._task
        )
        return self._obs, reward, state, StepInfo(frames=frames, progress=progress)

    @property
    def frames(self) -> np.ndarray:
        """All frames generated since the last ``reset()``, ``(T, 3, V, H, W)``.

        Empty unless the env was constructed with ``keep_frames=True``.
        """
        if not self._frames:
            return np.zeros((0, 3, 0, 0, 0), dtype=np.float32)
        return np.concatenate(self._frames, axis=0)

    def save_video(self, path: str | Path, fps: int = 16) -> None:
        """Write the accumulated rollout (views tiled horizontally) to MP4."""
        frames = self.frames
        if frames.shape[0] == 0:
            raise RuntimeError(
                "no frames buffered; call step() first "
                "(and construct the env with keep_frames=True)"
            )
        save_video(frames, str(path), fps=fps)

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
