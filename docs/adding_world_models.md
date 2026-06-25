# Adding a world model

A world model hosts one episode session at a time and turns action chunks into
predicted multi-view video plus robot state. New models implement the
`WorldModel` ABC and add one line to the registry.

## The `WorldModel` ABC

From `src/gesim/models/base.py`:

```python
@dataclass(frozen=True)
class StepResult:
    frames: np.ndarray            # (T, 3, V, H, W) float32 [0, 1]
    state: np.ndarray | None      # (T, D) float32 predicted robot state, or None


class WorldModel(ABC):
    """One episode-at-a-time world model. All array arguments are numpy."""

    chunk_size: int = 25

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> "WorldModel":
        """Build and load the model from a configuration dict."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all per-episode state."""

    @abstractmethod
    def set_camera_params(self, intrinsic: np.ndarray, extrinsic: np.ndarray | None = None) -> None:
        """Store per-view intrinsics ``(V, 3, 3)`` and optional first-frame c2w ``(V, 4, 4)``."""

    @abstractmethod
    def set_episode_data(self, first_frame: np.ndarray) -> None:
        """Ingest the first observation ``(3, V, H, W)`` float32 ``[0, 1]``."""

    @abstractmethod
    def set_episode_traj(self, traj: np.ndarray, c2w: np.ndarray) -> None:
        """Ingest trajectory-band conditioning ``(3, V, T, H, W)`` + c2w ``(V, T, 4, 4)``."""

    def set_task(self, task: str) -> None:
        """Store the task instruction. Optional; default is a no-op."""

    @abstractmethod
    def step(self, actions: np.ndarray) -> StepResult:
        """Generate the next chunk from ``(L <= chunk_size, 16)`` actions."""
```

## Per-method contract

| Method | Input shapes | Output |
|---|---|---|
| `from_config(config)` | the YAML config as a dict | a constructed `WorldModel` |
| `reset()` | — | clears all per-episode buffers |
| `set_camera_params(intrinsic, extrinsic)` | `intrinsic (V, 3, 3)`, optional `extrinsic (V, 4, 4)` first-frame c2w | — |
| `set_episode_data(first_frame)` | `(3, V, H, W)` float32 `[0, 1]` | — |
| `set_episode_traj(traj, c2w)` | `traj (3, V, T, H, W)` float32 `[0, 1]`, `c2w (V, T, 4, 4)` | — |
| `set_task(task)` | `str` | — (optional) |
| `step(actions)` | `(L <= chunk_size, 16)` float32, WM layout `[L7_arm, L_grip, R7_arm, R_grip]` | `StepResult(frames (T, 3, V, H, W), state (T, D) | None)` |

`V = 3` views (head, left wrist, right wrist) at `384x512`. The session protocol
calls `reset`, `set_camera_params`, `set_episode_data`, `set_episode_traj`,
`set_task`, then `step` repeatedly. `chunk_size` (25 for `gesim_v2`) caps `L`;
the client splits larger chunks for you.

## Registry

`get_world_model` resolves a name to a class through `_REGISTRY` in
`src/gesim/models/base.py`, which maps a name to a `module:ClassName` import
path so heavy model dependencies stay out of light-weight callers:

```python
_REGISTRY: dict[str, str] = {
    "example": "gesim.models.example:ExampleWorldModel",
    "gesim_v2": "gesim.models.gesim_v2.model:GeSimV2WorldModel",
}
```

Add your model by giving it an entry, e.g.
`"my_model": "gesim.models.my_model:MyWorldModel"`. The server then accepts it
via `python -m gesim.server --model my_model`
(`--model` choices come from `available_world_models()`).

## Config

The server loads the `--config` YAML into a dict and passes it to
`from_config`. Follow the [`configs/gesim_v2.yaml`](../configs/gesim_v2.yaml)
pattern; checkpoint entries are local paths. Models that need no checkpoints
(like `example`) can ignore the config and run without `--config`.

## Reference implementation

`ExampleWorldModel` (`src/gesim/models/example.py`) is a minimal, numpy-only,
GPU-free implementation: it validates the input shapes and returns correctly
shaped synthetic frames, echoing the actions as state. Use it as the template
for a new model and for CPU-only tests.

## Smoke test

Serve the example model and replay the demo bundle against it — no GPU or
checkpoints required:

```bash
python -m gesim.server --model example &
python examples/replay.py --server http://localhost:9000 \
    --episode assets/demo_000 --output-dir outputs/example_replay
```

A successful run writes `video.mp4` and `metrics.json` to the output directory.
