"""HTTP server exposing a WorldModel over the gesim wire protocol.

Endpoints: POST /init /reset /set_task /set_camera_params /set_episode_data
/set_episode_traj /step /close; GET /healthz. A live dashboard is served at
``GET /`` (status at ``/api/status``, frame preview at ``/api/preview.jpg``).
The server hosts ONE episode session at a time — a new /init replaces the
previous session. Only /step is serialized with a lock; the model owns all
episode state.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from gesim.client.codecs import BlockReader, decode_frame_jpeg, encode_frame_jpeg, pack_block
from gesim.models.base import WorldModel
from gesim.server.dashboard import DASHBOARD_HTML
from gesim.server.session import Session
from gesim.server.status import ServerStatus

logger = logging.getLogger("gesim.server")


class WorldModelServer:
    """Owns the model, the active session, and the HTTP listener."""

    # Concurrency contract: the protocol assumes a single sequential client.
    # /init and /step share _step_lock so a reconnect cannot reset the model
    # mid-step; the remaining handlers are unguarded by design.

    def __init__(
        self,
        model: WorldModel,
        host: str = "0.0.0.0",
        port: int = 9000,
        *,
        model_name: str = "world_model",
    ):
        self.model = model
        self.host = host
        self.port = port
        self.session: Session | None = None
        self.status = ServerStatus(model_name, model.chunk_size)
        self._step_lock = threading.Lock()
        handler = _make_handler(self)
        self._httpd = ThreadingHTTPServer((host, port), handler)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Serve in a background thread (used by tests and embedding callers)."""
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        logger.info("listening on http://%s:%d — dashboard at /", self.host, self.port)

    def serve_forever(self) -> None:
        logger.info("listening on http://%s:%d — dashboard at /", self.host, self.port)
        self._httpd.serve_forever()

    def stop(self) -> None:
        """Stop accepting requests. Does not drain in-flight handler threads."""
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    # -- Request handlers ------------------------------------------------

    def handle_init(self, payload: dict) -> dict:
        with self._step_lock:
            if self.session is not None:
                logger.info("replacing active session %s", self.session.client_id)
            self.session = Session(user_name=str(payload.get("user_name", "anonymous")))
            self.model.reset()
        self.status.on_init()
        return {"success": True, "client_id": self.session.client_id}

    def handle_reset(self, payload: dict) -> dict:
        self._require_session(payload)
        self.model.reset()
        return {"success": True}

    def handle_set_task(self, payload: dict) -> dict:
        session = self._require_session(payload)
        session.task = str(payload.get("task", ""))
        self.model.set_task(session.task)
        self.status.on_task(session.task)
        return {"success": True}

    def handle_set_camera_params(self, payload: dict) -> dict:
        self._require_session(payload)
        intrinsic = np.asarray(payload["intrinsic"], dtype=np.float32)
        extrinsic = payload.get("extrinsic")
        if extrinsic is not None:
            extrinsic = np.asarray(extrinsic, dtype=np.float32)
        self.model.set_camera_params(intrinsic, extrinsic)
        return {"success": True}

    def handle_close(self, payload: dict) -> dict:
        self._require_session(payload)
        self.session = None
        self.model.reset()
        self.status.on_close()
        return {"success": True}

    def handle_set_episode_data(self, body: bytes) -> dict:
        reader = BlockReader(body)
        jpeg = reader.read_block()
        meta = json.loads(reader.read_block().decode("utf-8"))
        self._require_session(meta)
        frame = decode_frame_jpeg(jpeg, meta["frame_shape"])
        self.model.set_episode_data(frame)
        return {"success": True}

    def handle_set_episode_traj(self, body: bytes) -> dict:
        reader = BlockReader(body)
        traj_section = reader.read_block()
        c2w_bytes = reader.read_block()
        meta = json.loads(reader.read_block().decode("utf-8"))
        self._require_session(meta)
        channels, num_views, num_frames, height, width = meta["traj_shape"]
        frame_shape = (channels, num_views, height, width)
        traj_reader = BlockReader(traj_section)
        frames = [
            decode_frame_jpeg(traj_reader.read_block(), frame_shape) for _ in range(num_frames)
        ]
        traj = np.stack(frames, axis=2)  # (3, V, T, H, W)
        c2w = np.frombuffer(c2w_bytes, dtype=np.float32).reshape(meta["c2w_shape"]).copy()
        self.model.set_episode_traj(traj, c2w)
        return {"success": True}

    def handle_step(self, body: bytes) -> bytes:
        reader = BlockReader(body)
        action_bytes = reader.read_block()
        meta = json.loads(reader.read_block().decode("utf-8"))
        actions = np.frombuffer(action_bytes, dtype=np.float32).reshape(meta["action_shape"]).copy()
        if actions.shape[0] > self.model.chunk_size:
            raise ValueError(
                f"action chunk {actions.shape[0]} exceeds model chunk_size {self.model.chunk_size}"
            )
        self.status.set_phase("running")
        with self._step_lock:
            self._require_session(meta)
            result = self.model.step(actions)

        frames = np.asarray(result.frames, dtype=np.float32)
        resp_meta = {
            "frame_shape": list(frames.shape),
            "state_shape": list(result.state.shape) if result.state is not None else None,
        }
        out = pack_block(json.dumps(resp_meta).encode("utf-8"))
        last_jpeg = b""
        for t in range(frames.shape[0]):
            last_jpeg = encode_frame_jpeg(frames[t])
            out += pack_block(last_jpeg)
        state_bytes = (
            np.asarray(result.state, dtype=np.float32).tobytes()
            if result.state is not None
            else b""
        )
        out += pack_block(state_bytes)

        state_row = result.state[-1] if result.state is not None and len(result.state) else None
        self.status.on_step(
            frames=int(frames.shape[0]),
            state_row=state_row,
            action_row=actions[-1] if len(actions) else None,
            preview=last_jpeg or None,
        )
        return out

    def _require_session(self, payload: dict) -> Session:
        client_id = payload.get("client_id")
        if self.session is None or self.session.client_id != client_id:
            raise PermissionError("unknown client_id; POST /init first")
        return self.session


_JSON_ROUTES = {
    "/init": "handle_init",
    "/reset": "handle_reset",
    "/set_task": "handle_set_task",
    "/set_camera_params": "handle_set_camera_params",
    "/close": "handle_close",
}
_BINARY_JSON_ROUTES = {
    "/set_episode_data": "handle_set_episode_data",
    "/set_episode_traj": "handle_set_episode_traj",
}


def _make_handler(server: WorldModelServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            logger.debug("%s " + fmt, self.address_string(), *args)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/healthz":
                self._send_json(200, {"status": "ok"})
            elif path == "/":
                self._send_bytes(200, DASHBOARD_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/status":
                self._send_json(200, server.status.snapshot())
            elif path == "/api/preview.jpg":
                preview = server.status.preview()
                if preview is None:
                    self._send_json(204, {})
                else:
                    self._send_bytes(200, preview, "image/jpeg")
            else:
                self._send_json(404, {"error": f"unknown path {self.path}"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                if self.path in _JSON_ROUTES:
                    payload = json.loads(body.decode("utf-8")) if body else {}
                    result = getattr(server, _JSON_ROUTES[self.path])(payload)
                    self._send_json(200, result)
                elif self.path in _BINARY_JSON_ROUTES:
                    result = getattr(server, _BINARY_JSON_ROUTES[self.path])(body)
                    self._send_json(200, result)
                elif self.path == "/step":
                    response = server.handle_step(body)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(response)))
                    self.end_headers()
                    self.wfile.write(response)
                else:
                    self._send_json(404, {"error": f"unknown path {self.path}"})
            except PermissionError as exc:
                self._send_json(403, {"error": str(exc)})
            except (KeyError, ValueError) as exc:  # includes json.JSONDecodeError
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:  # surface model failures to the client
                logger.exception("request failed: %s", self.path)
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def _send_json(self, code: int, obj: dict):
            self._send_bytes(code, json.dumps(obj).encode("utf-8"), "application/json")

        def _send_bytes(self, code: int, data: bytes, content_type: str):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            if code == 200 and content_type.startswith(("image/", "application/json")):
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if data:
                self.wfile.write(data)

    return Handler
