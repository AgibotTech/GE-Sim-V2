import numpy as np

from gesim.episode import EpisodeBundle
from gesim.types import (
    VIEW_NAMES,
    Observation,
    frame_to_view_images,
    head_view_frames,
    wm_state_to_policy_state,
)


def test_bundle_load(demo_bundle_dir):
    bundle = EpisodeBundle.load(demo_bundle_dir)
    assert bundle.task
    v = bundle.intrinsic.shape[0]
    assert bundle.intrinsic.shape == (v, 3, 3)
    assert bundle.extrinsic.ndim == 4 and bundle.extrinsic.shape[-2:] == (4, 4)
    assert bundle.first_frame.shape[0] == 3 and bundle.first_frame.ndim == 4
    assert bundle.first_frame.dtype == np.float32
    assert 0.0 <= bundle.first_frame.min() and bundle.first_frame.max() <= 1.0
    assert bundle.initial_state.shape == (16,)
    assert bundle.actions is not None and bundle.actions.shape[1] == 16
    assert bundle.initial_extrinsic.shape == (v, 4, 4)


def test_first_observation(demo_bundle_dir):
    bundle = EpisodeBundle.load(demo_bundle_dir)
    obs = bundle.first_observation()
    assert isinstance(obs, Observation)
    assert set(obs.images) == set(VIEW_NAMES)
    for img in obs.images.values():
        assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
    assert obs.state.shape == (16,)
    assert obs.task == bundle.task


def test_frame_to_view_images():
    frame = np.zeros((3, 3, 8, 10), dtype=np.float32)
    frame[:, 1] = 1.0
    imgs = frame_to_view_images(frame)
    assert list(imgs) == list(VIEW_NAMES)
    assert imgs["head"].shape == (8, 10, 3)
    assert imgs["left_wrist"].max() == 255 and imgs["head"].max() == 0


def test_head_view_frames():
    frames = np.zeros((4, 3, 3, 8, 10), dtype=np.float32)
    frames[:, :, 0] = 1.0  # head view all-white; other views black
    head = head_view_frames(frames)
    assert head.shape == (4, 8, 10, 3) and head.dtype == np.uint8
    assert head.min() == 255
    frames[:, :, 0] = 0.0
    assert head_view_frames(frames).max() == 0


def test_wm_state_to_policy_state():
    v = np.arange(16, dtype=np.float32)
    out = wm_state_to_policy_state(v)
    # WM [L7(0:7), Lg(7), R7(8:15), Rg(15)] -> policy [L7, R7, Lg, Rg]
    assert out[:7].tolist() == list(range(7))
    assert out[7:14].tolist() == list(range(8, 15))
    assert out[14] == 7.0 and out[15] == 15.0
