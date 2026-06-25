"""Reward-client protocol. A reward model scores generated head-camera video against a task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class RewardResult:
    """Per-frame scores for one video chunk.

    Attributes:
        success: ``(T,)`` float32 success probability per frame.
        progress: ``(T,)`` float32 task progress per frame.
    """

    success: np.ndarray
    progress: np.ndarray


@runtime_checkable
class RewardClient(Protocol):
    """Protocol for reward models that score head-camera video against a task string."""

    def evaluate(self, head_frames: np.ndarray, task: str) -> RewardResult:
        """Score ``(T, H, W, 3)`` uint8 head-camera frames against ``task``."""
        ...
