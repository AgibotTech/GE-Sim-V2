import numpy as np
import pytest

from gesim.action_chunk import compress_action_chunk


def test_compress_shape_and_endpoints():
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, size=(50, 16)).astype(np.float32)
    out = compress_action_chunk(actions, 25)
    assert out.shape == (25, 16)
    # continuous (arm) dims keep their endpoints exactly
    arm = [i for i in range(16) if i not in (7, 15)]
    assert np.allclose(out[0, arm], actions[0, arm])
    assert np.allclose(out[-1, arm], actions[-1, arm])


def test_compress_gripper_is_nearest_neighbour():
    actions = np.zeros((50, 16), dtype=np.float32)
    actions[25:, 7] = 1.0  # left gripper closes halfway through
    actions[:, 15] = np.linspace(0, 1, 50)  # right gripper ramps
    out = compress_action_chunk(actions, 25)
    # gripper values are sampled from the originals (never averaged to new values)
    assert set(np.unique(out[:, 7])).issubset({0.0, 1.0})


def test_passthrough_and_guard():
    actions = np.ones((25, 16), dtype=np.float32)
    assert np.array_equal(compress_action_chunk(actions, 25), actions)
    with pytest.raises(ValueError):
        compress_action_chunk(actions, 50)  # L < target_len


def test_target_one():
    actions = np.arange(40 * 16, dtype=np.float32).reshape(40, 16)
    out = compress_action_chunk(actions, 1)
    assert out.shape == (1, 16) and np.array_equal(out[0], actions[0])
