"""HTTP transport to a gesim world-model server. Internal — use gesim.WorldModelEnv."""

from __future__ import annotations

import json

import numpy as np
import requests

from gesim.client.codecs import (
    BlockReader,
    decode_frame_jpeg,
    encode_frame_jpeg,
    pack_block,
)
from gesim.exceptions import GesimConnectionError, GesimError, ServerError, raise_for_status

_BINARY_HEADERS = {"Content-Type": "application/octet-stream"}
_CONNECT_TIMEOUT = 10.0


class WorldModelClient:
    """Speaks the gesim world-model wire protocol. One client = one episode session.

    Timeouts are (10 s connect, ``timeout`` read).
    """

    def __init__(self, server_url: str, *, user_name: str = "gesim", timeout: float = 300.0):
        self.base_url = server_url.rstrip("/")
        self.timeout = float(timeout)
        self.session = requests.Session()
        self.client_id: str | None = None
        data = self._post_json("/init", {"user_name": user_name}, timeout=30.0)
        try:
            self.client_id = data["client_id"]
        except KeyError:
            raise GesimConnectionError(
                f"malformed /init response from {self.base_url} (missing 'client_id'): {data!r}"
            ) from None

    # -- JSON endpoints ------------------------------------------------------

    def reset(self) -> None:
        self._post_json("/reset", {"client_id": self.client_id})

    def set_task(self, task: str) -> None:
        self._post_json("/set_task", {"client_id": self.client_id, "task": task})

    def set_camera_params(self, intrinsic: np.ndarray, extrinsic: np.ndarray | None = None) -> None:
        intrinsic = np.asarray(intrinsic, dtype=np.float32)
        if intrinsic.ndim != 3 or intrinsic.shape[-2:] != (3, 3):
            raise ValueError(f"intrinsic must be (V, 3, 3), got {intrinsic.shape}")
        payload = {"client_id": self.client_id, "intrinsic": intrinsic.tolist()}
        if extrinsic is not None:
            extrinsic = np.asarray(extrinsic, dtype=np.float32)
            if extrinsic.ndim != 3 or extrinsic.shape[-2:] != (4, 4):
                raise ValueError(f"extrinsic must be (V, 4, 4), got {extrinsic.shape}")
            payload["extrinsic"] = extrinsic.tolist()
        self._post_json("/set_camera_params", payload)

    def close(self) -> None:
        if self.client_id is None:
            return
        try:
            self._post_json("/close", {"client_id": self.client_id}, timeout=5.0)
        except GesimError:
            pass
        finally:
            self.client_id = None

    # -- Binary endpoints ----------------------------------------------------

    def set_episode_data(self, first_frame: np.ndarray) -> None:
        """Upload the episode's first observation, ``(3, V, H, W)`` float32 ``[0, 1]``."""
        first_frame = np.asarray(first_frame, dtype=np.float32)
        meta = {"client_id": self.client_id, "frame_shape": list(first_frame.shape)}
        body = pack_block(encode_frame_jpeg(first_frame)) + pack_block(_dumps(meta))
        self._post_binary("/set_episode_data", body)

    def set_episode_traj(self, traj: np.ndarray, c2w: np.ndarray) -> None:
        """Upload the trajectory-band conditioning.

        Args:
            traj: ``(3, V, T, H, W)`` float32 ``[0, 1]`` band images.
            c2w: ``(V, T, 4, 4)`` float32 per-frame camera-to-world.
        """
        traj = np.asarray(traj, dtype=np.float32)
        c2w = np.asarray(c2w, dtype=np.float32)
        if traj.ndim != 5 or traj.shape[0] != 3:
            raise ValueError(f"traj must be (3, V, T, H, W), got {traj.shape}")
        _, num_views, num_frames, _, _ = traj.shape
        if c2w.shape != (num_views, num_frames, 4, 4):
            raise ValueError(f"c2w must be ({num_views}, {num_frames}, 4, 4), got {c2w.shape}")
        traj_section = b"".join(
            pack_block(encode_frame_jpeg(traj[:, :, t])) for t in range(num_frames)
        )
        meta = {
            "client_id": self.client_id,
            "traj_shape": [int(x) for x in traj.shape],
            "c2w_shape": [int(x) for x in c2w.shape],
        }
        body = (
            pack_block(traj_section) + pack_block(c2w.tobytes(order="C")) + pack_block(_dumps(meta))
        )
        self._post_binary("/set_episode_traj", body, timeout=600.0)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Run one model step.

        Args:
            actions: ``(L, action_dim)`` float32 action chunk, ``L`` <= server chunk size.

        Returns:
            ``(frames, state)``: frames ``(T, 3, V, H, W)`` float32 ``[0, 1]``;
            state ``(T, D)`` float32 or None.
        """
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"actions must be (L, C), got {actions.shape}")
        meta = {"client_id": self.client_id, "action_shape": list(actions.shape)}
        body = pack_block(actions.tobytes()) + pack_block(_dumps(meta))
        response = self._post_binary("/step", body)

        reader = BlockReader(response)
        resp_meta = json.loads(reader.read_block().decode("utf-8"))
        if "frame_shape" not in resp_meta:
            raise ServerError(
                200, f"malformed /step response (missing 'frame_shape'): {resp_meta!r}"
            )
        frame_shape = resp_meta["frame_shape"]  # [T, 3, V, H, W]
        frames = np.stack(
            [
                decode_frame_jpeg(reader.read_block(), frame_shape[1:])
                for _ in range(frame_shape[0])
            ],
            axis=0,
        )
        state_bytes = reader.read_block()
        state = None
        if resp_meta.get("state_shape"):
            state = (
                np.frombuffer(state_bytes, dtype=np.float32)
                .reshape(resp_meta["state_shape"])
                .copy()
            )
        return frames, state

    # -- Internals -----------------------------------------------------------

    def _post_json(self, path: str, payload: dict, *, timeout: float | None = None) -> dict:
        try:
            resp = self.session.post(
                f"{self.base_url}{path}",
                json=payload,
                timeout=(_CONNECT_TIMEOUT, self.timeout if timeout is None else timeout),
            )
        except requests.exceptions.Timeout as exc:
            raise GesimConnectionError(
                f"timed out waiting for world-model server at {self.base_url}"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise GesimConnectionError(
                f"cannot reach world-model server at {self.base_url}"
            ) from exc
        raise_for_status(resp)
        return resp.json()

    def _post_binary(self, path: str, body: bytes, *, timeout: float | None = None) -> bytes:
        try:
            resp = self.session.post(
                f"{self.base_url}{path}",
                data=body,
                headers=_BINARY_HEADERS,
                timeout=(_CONNECT_TIMEOUT, self.timeout if timeout is None else timeout),
            )
        except requests.exceptions.Timeout as exc:
            raise GesimConnectionError(
                f"timed out waiting for world-model server at {self.base_url}"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise GesimConnectionError(
                f"cannot reach world-model server at {self.base_url}"
            ) from exc
        raise_for_status(resp)
        return resp.content

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def _dumps(obj: dict) -> bytes:
    return json.dumps(obj).encode("utf-8")
