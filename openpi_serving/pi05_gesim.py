"""openpi pi05 config + transforms for the gesim policy checkpoint.

Runs inside the openpi environment (imports openpi as a library); it does NOT
import the gesim package. Mirrors the pi05 training/inference pipeline so the
checkpoint produces equivalent actions:

source inference chain (reproduced here via openpi's standard pi05 transforms):
  inject default prompt
  -> camera remap top_head/hand_left/hand_right -> base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb
  -> pad state 16->32, zero dims 16:31
  -> quantile (q01/q99) Normalize          (openpi: use_quantile_norm, auto-on for pi05)
  -> resize images to 224x224              (openpi ModelTransformFactory)
  -> PaligemmaTokenizer(200) + discrete state tokens
  -> model
  -> Unnormalize
  -> slice to 16 dims, reorder [L7,R7,Lg,Rg] -> [L7,Lg,R7,Rg]  (gesim WM layout)

The 16-D action/state layout convention:
  model/internal: [L7_arm(7), R7_arm(7), L_grip(1), R_grip(1)]
  served output:  [L7_arm(7), L_grip(1), R7_arm(7), R_grip(1)]  (what gesim env.step expects)
"""

from __future__ import annotations

import dataclasses

import numpy as np
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.training import config as _config
from openpi import transforms as _transforms

# Default task prompt for task_5636 (used only if an observation omits "prompt").
DEFAULT_PROMPT = "Pick up the kettle on the table with right arm and pour the water into the cup."

# Raw observation keys sent by gesim.policies.OpenPIPolicy (dot-separated, flat).
HEAD_KEY = "observation.images.head"
LEFT_KEY = "observation.images.hand_left"
RIGHT_KEY = "observation.images.hand_right"
STATE_KEY = "observation.state"

REAL_DIM = 16   # meaningful action/state dims; the model is padded to 32
MODEL_DIM = 32


def _to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    """Coerce one camera image to uint8 HWC RGB."""
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _square_resize_224(img_hwc_uint8: np.ndarray) -> np.ndarray:
    """Aspect-squashing resize to 224x224 (matches the training-time PIL ResizeImages(224,224)).

    Done here so openpi's downstream resize_with_pad(224,224) is a no-op, keeping
    the image preprocessing identical to training instead of aspect-padding.
    """
    from PIL import Image

    im = Image.fromarray(img_hwc_uint8, mode="RGB").resize((224, 224), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


@dataclasses.dataclass(frozen=True)
class GesimInputs(_transforms.DataTransformFn):
    """Map a gesim observation to the openpi pi05 model input dict."""

    def __call__(self, data: dict) -> dict:
        images = {
            "base_0_rgb": _square_resize_224(_to_hwc_uint8(data[HEAD_KEY])),
            "left_wrist_0_rgb": _square_resize_224(_to_hwc_uint8(data[LEFT_KEY])),
            "right_wrist_0_rgb": _square_resize_224(_to_hwc_uint8(data[RIGHT_KEY])),
        }
        image_mask = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        }

        # Pad state 16 -> 32 and zero the padding (training pads state to the model width and zeroes the unused tail).
        state = np.asarray(data[STATE_KEY], dtype=np.float32).reshape(-1)
        state = _transforms.pad_to_dim(state, MODEL_DIM)
        state[REAL_DIM:] = 0.0

        inputs: dict = {"image": images, "image_mask": image_mask, "state": state}

        prompt = data.get("prompt")
        if prompt is not None:
            inputs["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else str(prompt)
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        return inputs


@dataclasses.dataclass(frozen=True)
class GesimOutputs(_transforms.DataTransformFn):
    """Slice to 16 dims and reorder to gesim WM layout [L7, L_grip, R7, R_grip]."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])[:, :REAL_DIM]  # (horizon, 16) in [L7, R7, L_grip, R_grip]
        out = np.empty_like(actions)
        out[:, 0:7] = actions[:, 0:7]     # L arm
        out[:, 7] = actions[:, 14]        # L gripper
        out[:, 8:15] = actions[:, 7:14]   # R arm
        out[:, 15] = actions[:, 15]       # R gripper
        return {"actions": out}


def make_config(
    asset_id: str = "gesim", action_horizon: int = 50, compile_mode: str | None = None
) -> _config.TrainConfig:
    """Build the in-process TrainConfig for serving the ported pi05 checkpoint.

    ``compile_mode`` defaults to None (eager): the policy server makes one
    inference per call, and torch.compile's first-call "max-autotune" pass takes
    minutes and blocks the asyncio loop past the websocket keepalive. Eager mode
    is responsive; set e.g. "max-autotune" only for throughput benchmarking.
    """
    return _config.TrainConfig(
        name="pi05_gesim",
        exp_name="serve",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=MODEL_DIM,
            action_horizon=action_horizon,
            pytorch_compile_mode=compile_mode,
        ),
        data=_config.SimpleDataConfig(
            assets=_config.AssetsConfig(asset_id=asset_id),
            data_transforms=lambda model: _transforms.Group(
                inputs=[GesimInputs()],
                outputs=[GesimOutputs()],
            ),
        ),
    )
