"""OpenPI policy client.

Talks to an openpi websocket policy server (``scripts/serve_policy.py`` in the
``third_party/openpi`` submodule). Requires the ``openpi-client`` package:

    pip install -e third_party/openpi/packages/openpi-client
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gesim.types import ACTION_DIM, Observation

# Default observation keys for pi05-style configs; override per checkpoint.
DEFAULT_IMAGE_KEYS = {
    "head": "observation.images.head",
    "left_wrist": "observation.images.hand_left",
    "right_wrist": "observation.images.hand_right",
}
DEFAULT_STATE_KEY = "observation.state"
DEFAULT_PROMPT_KEY = "prompt"


def build_openpi_payload(
    obs: Observation,
    *,
    image_keys: dict[str, str] | None = None,
    state_key: str = DEFAULT_STATE_KEY,
    prompt_key: str = DEFAULT_PROMPT_KEY,
) -> dict[str, Any]:
    """Map a gesim Observation to the flat dict an openpi policy server expects.

    ``obs.state`` is forwarded as-is (policy layout ``[L7_arm, R7_arm, L_grip, R_grip]``).

    Raises:
        KeyError: if a required camera view is missing from ``obs.images``.
    """
    image_keys = image_keys or DEFAULT_IMAGE_KEYS
    payload: dict[str, Any] = {
        state_key: np.asarray(obs.state, dtype=np.float32).reshape(-1),
        prompt_key: obs.task,
    }
    for view, key in image_keys.items():
        if view not in obs.images:
            raise KeyError(f"observation is missing view {view!r}; has {sorted(obs.images)}")
        payload[key] = np.ascontiguousarray(obs.images[view])
    return payload


def validate_actions(actions: np.ndarray) -> np.ndarray:
    """Normalize a policy response to ``(horizon, 16)`` float32."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"expected actions (horizon, {ACTION_DIM}), got {actions.shape}")
    return actions


class OpenPIPolicy:
    """Policy backed by an openpi websocket policy server.

    Sends ``Observation`` images/state/prompt to the server and returns action
    chunks ``(horizon, 16)`` float32 in WM layout ``[L7_arm, L_grip, R7_arm, R_grip]``.
    """

    def __init__(
        self,
        url: str,
        *,
        action_horizon: int | None = None,
        image_keys: dict[str, str] | None = None,
        state_key: str = DEFAULT_STATE_KEY,
        prompt_key: str = DEFAULT_PROMPT_KEY,
    ):
        try:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy
        except ImportError as exc:
            raise ImportError(
                "openpi-client is not installed. Run: "
                "pip install -e third_party/openpi/packages/openpi-client"
            ) from exc
        self._client = WebsocketClientPolicy(host=url)
        self._action_horizon = action_horizon
        self._image_keys = image_keys
        self._state_key = state_key
        self._prompt_key = prompt_key

    def reset(self) -> None:
        self._client.reset()

    def infer(self, obs: Observation) -> np.ndarray:
        """Return an action chunk ``(horizon, 16)`` float32 in WM layout.

        Layout is ``[L7_arm, L_grip, R7_arm, R_grip]``.
        """
        payload = build_openpi_payload(
            obs,
            image_keys=self._image_keys,
            state_key=self._state_key,
            prompt_key=self._prompt_key,
        )
        result = self._client.infer(payload)
        if "actions" not in result:
            raise KeyError(
                f"policy server response is missing 'actions'; got keys {sorted(result)}"
            )
        actions = validate_actions(result["actions"])
        if self._action_horizon is not None:
            actions = actions[: self._action_horizon]
        return actions
