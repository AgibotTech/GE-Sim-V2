import numpy as np
import pytest

from gesim.models.base import StepResult, WorldModel, available_world_models, get_world_model


def test_registry_lists_models():
    names = available_world_models()
    assert "example" in names and "gesim_v2" in names


def test_get_example_model():
    cls = get_world_model("example")
    model = cls.from_config({})
    assert isinstance(model, WorldModel)
    assert model.chunk_size == 25


def test_unknown_model():
    with pytest.raises(KeyError):
        get_world_model("nope")


def test_example_model_step_shapes():
    model = get_world_model("example").from_config({})
    model.reset()
    model.set_camera_params(np.zeros((3, 3, 3), dtype=np.float32))
    model.set_episode_data(np.zeros((3, 3, 384, 512), dtype=np.float32))
    model.set_episode_traj(
        np.zeros((3, 3, 25, 384, 512), dtype=np.float32),
        np.zeros((3, 25, 4, 4), dtype=np.float32),
    )
    actions = np.zeros((25, 16), dtype=np.float32)
    result = model.step(actions)
    assert isinstance(result, StepResult)
    assert result.frames.shape == (25, 3, 3, 384, 512)
    assert result.frames.dtype == np.float32
    assert result.state.shape == (25, 16)


def test_example_model_requires_setup():
    model = get_world_model("example").from_config({})
    model.reset()
    with pytest.raises(RuntimeError):
        model.step(np.zeros((25, 16), dtype=np.float32))
