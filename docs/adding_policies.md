# Adding a policy

A policy is anything that maps an `Observation` to an action chunk. Policies are
defined by structural typing (`typing.Protocol`): any object with the two
methods below works as a policy — there is no base class to subclass and no
registry to update.

## The `Policy` protocol

From `src/gesim/policies/base.py`:

```python
@runtime_checkable
class Policy(Protocol):
    """Protocol for action policies driving the world model."""

    def reset(self) -> None:
        """Clear per-episode state. Called once before each rollout."""
        ...

    def infer(self, obs: Observation) -> np.ndarray:
        """Return an action chunk ``(horizon, 16)`` float32 of joint-space
        targets in WM layout ``[L7_arm, L_grip, R7_arm, R_grip]`` (7 arm joints
        + 1 gripper per arm), not end-effector poses."""
        ...
```

## Contract

**Input** — an `Observation` (`src/gesim/types.py`):

- `images: dict[str, np.ndarray]` — per-view `uint8` `(H, W, 3)` RGB arrays,
  keyed by `head`, `left_wrist`, `right_wrist`.
- `state: np.ndarray` — `(16,)` float32 proprioception in **policy** layout
  `[L7_arm, R7_arm, L_grip, R_grip]`.
- `task: str` — the natural-language instruction.

**Output** — a `(horizon, 16)` float32 array in **WM** layout
`[L7_arm, L_grip, R7_arm, R_grip]` (left arm 7 joints, left gripper, right arm 7
joints, right gripper). `horizon` is the policy's choice; `WorldModelEnv.step`
splits it into model-sized chunks internally.

Note the input state and the output action use different 16-D orderings: the
observation state is in policy layout, while the returned actions are in WM
layout.

## Example: a random policy

This complete policy implements the protocol and runs against the `example`
world model:

```python
import numpy as np

from gesim.types import ACTION_DIM, Observation


class RandomPolicy:
    """Emits small random joint deltas around the current state."""

    def __init__(self, horizon: int = 50, scale: float = 0.02, seed: int = 0):
        self._horizon = horizon
        self._scale = scale
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        # No per-episode state to clear for a stateless random policy.
        pass

    def infer(self, obs: Observation) -> np.ndarray:
        # obs.state is policy layout [L7, R7, L_grip, R_grip]; reorder to WM
        # layout [L7, L_grip, R7, R_grip] before adding noise.
        s = np.asarray(obs.state, dtype=np.float32).reshape(-1)
        base = np.empty(ACTION_DIM, dtype=np.float32)
        base[0:7] = s[0:7]      # left arm
        base[7] = s[14]         # left gripper
        base[8:15] = s[7:14]    # right arm
        base[15] = s[15]        # right gripper
        noise = self._rng.normal(0.0, self._scale, (self._horizon, ACTION_DIM))
        return (base[None] + noise).astype(np.float32)
```

Drive the example world model with it:

```bash
python -m gesim.server --model example &
```

```python
from gesim import WorldModelEnv

env = WorldModelEnv("http://localhost:9000")
policy = RandomPolicy()

obs = env.reset("assets/demo_000", conditioning="action")
policy.reset()
for _ in range(4):
    obs, reward, state, info = env.step(policy.infer(obs))
env.save_video("random_rollout.mp4")
```

`OpenPIPolicy` in `src/gesim/policies/openpi.py` is the production implementation
of the same protocol.
