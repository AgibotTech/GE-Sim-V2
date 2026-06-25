"""World-model interface and registry.

To add a world model: subclass ``WorldModel``, implement the abstract methods,
and register its import path in ``_REGISTRY``. See docs/adding_world_models.md.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StepResult:
    """Output of one world-model step.

    Attributes:
        frames: ``(T, 3, V, H, W)`` float32 ``[0, 1]`` generated frames.
        state: ``(T, D)`` float32 predicted robot state, or None.
    """

    frames: np.ndarray
    state: np.ndarray | None


class WorldModel(ABC):
    """One episode-at-a-time world model. All array arguments are numpy."""

    chunk_size: int = 25

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> WorldModel:
        """Build and load the model from a configuration dict."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all per-episode state."""

    @abstractmethod
    def set_camera_params(self, intrinsic: np.ndarray, extrinsic: np.ndarray | None = None) -> None:
        """Store per-view intrinsics ``(V, 3, 3)`` and optional first-frame c2w ``(V, 4, 4)``."""

    @abstractmethod
    def set_episode_data(self, first_frame: np.ndarray) -> None:
        """Ingest the first observation ``(3, V, H, W)`` float32 ``[0, 1]``."""

    @abstractmethod
    def set_episode_traj(self, traj: np.ndarray, c2w: np.ndarray) -> None:
        """Ingest trajectory-band conditioning ``(3, V, T, H, W)`` + c2w ``(V, T, 4, 4)``."""

    def set_task(self, task: str) -> None:  # noqa: B027  (intentional optional no-op hook)
        """Store the task instruction. Optional; default is a no-op."""

    @abstractmethod
    def step(self, actions: np.ndarray) -> StepResult:
        """Generate the next chunk from ``(L <= chunk_size, 16)`` actions."""


# Import paths keep heavy model dependencies out of light-weight callers.
_REGISTRY: dict[str, str] = {
    "example": "gesim.models.example:ExampleWorldModel",
    "gesim_v2": "gesim.models.gesim_v2.model:GeSimV2WorldModel",
}


def available_world_models() -> list[str]:
    return sorted(_REGISTRY)


def get_world_model(name: str) -> type[WorldModel]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown world model {name!r}; available: {available_world_models()}")
    module_name, _, class_name = _REGISTRY[name].partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
