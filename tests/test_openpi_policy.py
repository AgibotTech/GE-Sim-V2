import numpy as np
import pytest

from gesim.policies.openpi import build_openpi_payload, validate_actions
from gesim.types import Observation


def _obs() -> Observation:
    rng = np.random.default_rng(0)
    images = {
        "head": rng.integers(0, 255, (384, 512, 3), dtype=np.uint8),
        "left_wrist": rng.integers(0, 255, (384, 512, 3), dtype=np.uint8),
        "right_wrist": rng.integers(0, 255, (384, 512, 3), dtype=np.uint8),
    }
    return Observation(
        images=images, state=np.arange(16, dtype=np.float32), task="pick up the cube"
    )


def test_build_payload_default_keys():
    payload = build_openpi_payload(_obs())
    assert payload["prompt"] == "pick up the cube"
    assert payload["observation.state"].shape == (16,)
    assert payload["observation.images.head"].dtype == np.uint8
    assert payload["observation.images.hand_left"].shape == (384, 512, 3)
    assert "observation.images.hand_right" in payload


def test_build_payload_missing_view():
    obs = _obs()
    images = dict(obs.images)
    del images["head"]
    broken = Observation(images=images, state=obs.state, task=obs.task)
    with pytest.raises(KeyError):
        build_openpi_payload(broken)


def test_validate_actions_shape():
    ok = np.zeros((50, 16), dtype=np.float32)
    assert validate_actions(ok).shape == (50, 16)
    with pytest.raises(ValueError):
        validate_actions(np.zeros((50, 7), dtype=np.float32))
    with pytest.raises(ValueError):
        validate_actions(np.zeros(16, dtype=np.float32))


def test_validate_actions_unbatch():
    batched = np.zeros((1, 50, 16), dtype=np.float32)
    assert validate_actions(batched).shape == (50, 16)
