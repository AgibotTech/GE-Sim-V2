# Adding a reward client

`WorldModelEnv` can score each step's generated frames against the task. No
reward model ships with this repo — you attach your own by passing any object
that implements the `RewardClient` protocol.

## The protocol

```python
# src/gesim/rewards/base.py
from typing import Protocol, runtime_checkable
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardResult:
    success: np.ndarray   # (T,) float32 success probability per frame
    progress: np.ndarray  # (T,) float32 task progress per frame


@runtime_checkable
class RewardClient(Protocol):
    def evaluate(self, head_frames: np.ndarray, task: str) -> RewardResult:
        """Score (T, H, W, 3) uint8 head-camera frames against `task`."""
        ...
```

The env extracts the head-camera view of each step's frames, calls
`evaluate(head_frames, task)`, and surfaces the result as the `reward` return of
`step` (the `success` array) and `StepInfo.progress`.

## Example

```python
import numpy as np
from gesim import WorldModelEnv
from gesim.rewards import RewardResult


class MyReward:
    """Toy reward: mean head-frame brightness as 'progress'."""

    def evaluate(self, head_frames: np.ndarray, task: str) -> RewardResult:
        t = head_frames.shape[0]
        score = head_frames.reshape(t, -1).mean(axis=1).astype(np.float32) / 255.0
        return RewardResult(success=score, progress=score)


env = WorldModelEnv("http://localhost:9000", reward=MyReward())
obs = env.reset("assets/demo_000", conditioning="episode")
obs, reward, state, info = env.step(actions)  # reward: (T,) from MyReward
```

`MyReward` needs no registration — the env accepts any object structurally
matching `RewardClient`. A real reward client typically forwards the frames to a
trained reward/value model (local or over HTTP) and returns its per-frame
scores.
