import socket
import urllib.request

import numpy as np
import pytest

from gesim import WorldModelEnv
from gesim.models.base import get_world_model
from gesim.server.app import WorldModelServer


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def server():
    model = get_world_model("example").from_config({})
    srv = WorldModelServer(model, host="127.0.0.1", port=_free_port(), model_name="example")
    srv.start()
    yield srv
    srv.stop()


def _get(url: str):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, resp.headers.get("Content-Type", ""), resp.read()


def test_replay_roundtrip(server, demo_bundle_dir, tmp_path):
    # compress_actions=False to exercise the multi-chunk split (30 -> 25 + 5).
    env = WorldModelEnv(f"http://127.0.0.1:{server.port}", compress_actions=False)
    obs = env.reset(demo_bundle_dir, conditioning="episode")
    assert obs.state.shape == (16,)
    assert obs.task

    actions = np.load(demo_bundle_dir / "actions_0.npy").astype(np.float32)[:30]
    obs, reward, state, info = env.step(actions)

    assert reward is None
    assert info.frames.shape[0] == 30  # 25 + 5 across two server chunks
    assert info.frames.shape[1] == 3
    assert state is not None and state.shape == (30, 16)
    # Example model echoes actions as state; env reorders the last row to policy layout.
    last = actions[-1]
    assert obs.state[0] == pytest.approx(last[0])
    assert obs.state[14] == pytest.approx(last[7])

    out = tmp_path / "rollout.mp4"
    env.save_video(out)
    assert out.is_file() and out.stat().st_size > 0
    env.close()


def test_compress_actions_default(server, demo_bundle_dir):
    # Default compress_actions=True: a 30-action chunk -> one 25-row model chunk.
    env = WorldModelEnv(f"http://127.0.0.1:{server.port}")
    env.reset(demo_bundle_dir, conditioning="episode")
    actions = np.load(demo_bundle_dir / "actions_0.npy").astype(np.float32)[:30]
    _obs, _reward, state, info = env.step(actions)
    assert info.frames.shape[0] == 25  # compressed to a single model chunk
    assert state is not None and state.shape == (25, 16)
    env.close()


def test_step_before_reset_raises(server):
    env = WorldModelEnv(f"http://127.0.0.1:{server.port}")
    with pytest.raises(RuntimeError):
        env.step(np.zeros((25, 16), dtype=np.float32))
    env.close()


def test_dashboard_endpoints(server, demo_bundle_dir):
    import json

    base = f"http://127.0.0.1:{server.port}"

    status, ctype, body = _get(f"{base}/")
    assert status == 200 and "text/html" in ctype and b"world model" in body

    status, ctype, body = _get(f"{base}/api/status")
    snap = json.loads(body)
    assert snap["model"] == "example" and snap["phase"] == "idle"
    assert snap["step_count"] == 0 and snap["has_preview"] is False

    env = WorldModelEnv(base, user_name="dash-test", compress_actions=False)
    env.reset(demo_bundle_dir, conditioning="episode")
    env.step(np.load(demo_bundle_dir / "actions_0.npy").astype(np.float32)[:30])

    snap = json.loads(_get(f"{base}/api/status")[2])
    assert snap["step_count"] == 2  # 30 actions -> 25 + 5 across two server steps
    assert snap["frames_generated"] == 30
    assert snap["has_preview"] is True
    assert snap["state"] is not None and len(snap["state"]) == 16
    assert snap["action"] is not None and len(snap["action"]) == 16
    assert snap["task"]

    status, ctype, body = _get(f"{base}/api/preview.jpg")
    assert status == 200 and ctype == "image/jpeg" and body[:2] == b"\xff\xd8"  # JPEG SOI
    env.close()
