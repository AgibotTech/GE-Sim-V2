"""Episode bundle loading.

An episode bundle is a directory exported from a recorded robot episode:

    intrinsic.npy                (V, 3, 3) per-view pinhole intrinsics, baked at 512x384
    extrinsic_alignstate_0.npy   (V, T, 4, 4) per-frame camera-to-world
    cur_head.png / cur_left.png / cur_right.png   first-frame RGB per camera
    task.txt                     natural-language task instruction
    actions_0.npy                (T, 16) recorded absolute-joint actions   [replay]
    eef_poses_0.npy              (T, 14) action-FK end-effector poses      [replay conditioning]
    state_joints_0.npy           (T, 20) joints [L7, R7, L_grip, R_grip, head2, waist2]
    state_eef_poses_0.npy        (T, 14) state-FK end-effector poses       [closed-loop cond.]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gesim.types import Observation, frame_to_view_images

# Index-aligned with gesim.types.VIEW_NAMES (head, left_wrist, right_wrist).
_CAMERA_STEMS = ("cur_head", "cur_left", "cur_right")
_REQUIRED_FILES = ("intrinsic.npy", "extrinsic_alignstate_0.npy", "task.txt")


def _chw_to_float01(chw: np.ndarray) -> np.ndarray:
    """Single camera ``(3, H, W)`` -> float32 [0, 1].

    - Integer (e.g. uint8) 0–255: divide by 255 once.
    - Float with max > 1.5: treat as 0–255 float, divide by 255 once.
    - Float already in ~[0, 1]: clip only.
    - Clear [-1, 1] latents: strict gate then ``(x+1)/2``.
    """
    x = np.asarray(chw)
    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError(f"Expected CHW (3, H, W), got {x.shape}")
    if np.issubdtype(x.dtype, np.integer):
        return (x.astype(np.float32) / 255.0).clip(0.0, 1.0)
    xf = x.astype(np.float32)
    maxv = float(np.max(xf))
    minv = float(np.min(xf))
    if maxv > 1.5:
        return (xf / 255.0).clip(0.0, 1.0)
    # True [-1, 1] tensors: require a strong negative tail (avoid false positives).
    if minv < -0.35 and maxv <= 1.02:
        return ((xf + 1.0) * 0.5).clip(0.0, 1.0)
    return xf.clip(0.0, 1.0)


def _canonical_one_camera_to_chw(img: np.ndarray) -> np.ndarray:
    """Return one camera as ``(3, H, W)`` float32 in **[0, 1]**.

    Accepts ``(3, H, W)`` CHW or ``(H, W, 3)`` HWC (uint8 / float).
    """
    x = np.asarray(img)
    if x.ndim != 3:
        raise ValueError(f"One camera image must be 3D, got shape {x.shape}")
    if x.shape[0] == 3:
        return _chw_to_float01(x)
    if x.shape[2] == 3:
        chw = np.transpose(x, (2, 0, 1))
        return _chw_to_float01(chw)
    raise ValueError(f"Cannot interpret camera array shape {x.shape}; expected (3,H,W) or (H,W,3).")


def _load_camera_image(assets_dir: Path, stem: str) -> np.ndarray:
    """Load one ``cur_*`` camera image from ``assets_dir`` as uint8 HWC RGB.

    Prefers ``<stem>.png`` (new format). Falls back to ``<stem>.npy`` for
    backward compatibility with older bundles. Numpy fallback may store either
    CHW or HWC; we normalize to HWC here so callers always see ``(H, W, 3)``
    uint8.

    Args:
        assets_dir: Bundle folder, e.g. ``Path("assets/demo_000")``.
        stem: File stem without extension, e.g. ``"cur_head"``.

    Returns:
        ``np.ndarray`` shape ``(H, W, 3)``, dtype ``uint8``, values 0–255.

    Raises:
        FileNotFoundError: if neither ``<stem>.png`` nor ``<stem>.npy`` exists.
    """
    base = Path(assets_dir)
    png_path = base / f"{stem}.png"
    npy_path = base / f"{stem}.npy"

    if png_path.is_file():
        # PIL is already a project dependency (used by server.py + world_simulator.py).
        from PIL import Image

        with Image.open(png_path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        return arr

    if npy_path.is_file():
        arr = np.load(npy_path)
        if arr.dtype != np.uint8:
            # Older/foreign bundles may have stored float; clip + cast to keep the contract.
            if np.issubdtype(arr.dtype, np.floating):
                if float(arr.max()) <= 1.0 + 1e-3:
                    arr = (np.clip(arr, 0.0, 1.0) * 255.0).round()
                else:
                    arr = np.clip(arr, 0.0, 255.0)
            arr = arr.astype(np.uint8)
        # CHW → HWC so downstream code can rely on a single layout.
        if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"{npy_path} has unexpected shape {arr.shape}; expected (H,W,3) or (3,H,W)."
            )
        return arr

    raise FileNotFoundError(
        f"No {png_path.name} or {npy_path.name} under {base.resolve()}. "
        "expected <stem>.png (preferred) or <stem>.npy in the bundle directory"
    )


def _load_optional(path: Path) -> np.ndarray | None:
    return np.load(path).astype(np.float32) if path.is_file() else None


@dataclass(frozen=True)
class EpisodeBundle:
    """In-memory view of an episode bundle directory. See module docstring for the layout."""

    path: Path
    task: str
    intrinsic: np.ndarray  # (V, 3, 3)
    extrinsic: np.ndarray  # (V, T, 4, 4)
    first_frame: np.ndarray  # (3, V, H, W) float32 [0, 1]
    initial_state: np.ndarray  # (16,) float32, policy layout [L7, R7, L_grip, R_grip]
    head_waist: np.ndarray | None  # (4,) [head_yaw, head_pitch, body_pitch, lift_body]
    actions: np.ndarray | None  # (T, 16)
    eef_poses: np.ndarray | None  # (T, 14)
    state_joints: np.ndarray | None  # (T, 20)
    state_eef_poses: np.ndarray | None  # (T, 14)

    @classmethod
    def load(cls, path: str | Path) -> EpisodeBundle:
        base = Path(path)
        missing = [n for n in _REQUIRED_FILES if not (base / n).is_file()]
        missing += [
            f"{stem}.png/.npy"
            for stem in _CAMERA_STEMS
            if not ((base / f"{stem}.png").is_file() or (base / f"{stem}.npy").is_file())
        ]
        if missing:
            raise FileNotFoundError(f"episode bundle {base.resolve()} is missing: {missing}")

        task = (base / "task.txt").read_text(encoding="utf-8").strip()
        if not task:
            raise ValueError(f"empty task.txt in {base.resolve()}")

        cameras = [_load_camera_image(base, stem) for stem in _CAMERA_STEMS]
        chw = [_canonical_one_camera_to_chw(img) for img in cameras]
        if len({c.shape for c in chw}) != 1:
            raise ValueError(f"camera shapes differ: {[c.shape for c in chw]}")
        first_frame = np.stack(chw, axis=1)  # (3, V, H, W)

        state_joints = _load_optional(base / "state_joints_0.npy")
        if state_joints is not None and state_joints.shape[1] >= 16:
            initial_state = state_joints[0, :16].copy()
            head_waist = state_joints[0, 16:20].copy() if state_joints.shape[1] >= 20 else None
        else:
            warnings.warn(
                f"{base / 'state_joints_0.npy'} missing or has fewer than 16 columns; "
                "initial_state falls back to zeros",
                stacklevel=2,
            )
            initial_state = np.zeros(16, dtype=np.float32)
            head_waist = None

        return cls(
            path=base,
            task=task,
            intrinsic=np.load(base / "intrinsic.npy").astype(np.float32),
            extrinsic=np.load(base / "extrinsic_alignstate_0.npy").astype(np.float32),
            first_frame=first_frame,
            initial_state=initial_state,
            head_waist=head_waist,
            actions=_load_optional(base / "actions_0.npy"),
            eef_poses=_load_optional(base / "eef_poses_0.npy"),
            state_joints=state_joints,
            state_eef_poses=_load_optional(base / "state_eef_poses_0.npy"),
        )

    @property
    def initial_extrinsic(self) -> np.ndarray:
        """First-frame per-view camera-to-world, shape ``(V, 4, 4)``."""
        return self.extrinsic[:, 0]

    def first_observation(self, task: str | None = None) -> Observation:
        return Observation(
            images=frame_to_view_images(self.first_frame),
            state=self.initial_state.copy(),
            task=task or self.task,
        )
