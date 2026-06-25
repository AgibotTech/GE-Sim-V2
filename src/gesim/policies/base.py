"""Policy protocol: anything that maps an Observation to a (horizon, 16) action chunk."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from gesim.types import Observation


@runtime_checkable
class Policy(Protocol):
    """Protocol for action policies driving the world model."""

    def reset(self) -> None:
        """Clear per-episode state. Called once before each rollout."""
        ...

    def infer(self, obs: Observation) -> np.ndarray:
        """Return an action chunk ``(horizon, 16)`` float32 in WM layout
        ``[L7_arm, L_grip, R7_arm, R_grip]``."""
        ...
