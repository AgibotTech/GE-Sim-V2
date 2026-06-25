# Replay a recorded episode

Replay drives the world model with the actions recorded in an episode bundle
and compares the model's predicted video against what the real robot did. It is
the open-loop evaluation path: no policy is involved — the recorded
`actions_0.npy` are streamed straight into `WorldModelEnv.step`.

## Episode bundle format

An episode bundle is a directory exported from a recorded robot episode:

| File | Shape / type | Purpose |
|---|---|---|
| `intrinsic.npy` | `(V, 3, 3)` | Per-view pinhole intrinsics, baked at 512x384. |
| `extrinsic_alignstate_0.npy` | `(V, T, 4, 4)` | Per-frame camera-to-world. |
| `cur_head.png` / `cur_left.png` / `cur_right.png` | RGB | First-frame image per camera. |
| `task.txt` | text | Natural-language task instruction. |
| `actions_0.npy` | `(T, 16)` | Recorded absolute-joint actions (replay). |
| `eef_poses_0.npy` | `(T, 14)` | Action-FK end-effector poses (replay conditioning). |
| `state_joints_0.npy` | `(T, 20)` | Joints `[L7, R7, L_grip, R_grip, head2, waist2]`. |
| `state_eef_poses_0.npy` | `(T, 14)` | State-FK end-effector poses (closed-loop conditioning). |

`V = 3` views (head, left wrist, right wrist); frames are `384x512`; the action
dimension is 16. The bundled `assets/demo_000` is a complete example.

## Conditioning modes

The world model requires an uploaded trajectory band as conditioning. There are
two ways to produce it, selected by `WorldModelEnv.reset(..., conditioning=...)`:

- **`episode`** — render the band once from the bundle's recorded end-effector
  poses (`eef_poses_0.npy`) and per-frame extrinsics. This matches the band the
  model saw during training, so it is the correct choice for **replay**.
- **`action`** — render the band per `step()` chunk from the actions via forward
  kinematics. Use this for **closed loop**, where actions come from a live policy
  rather than the recording. It uses the robot model, a compiled FK library
  bundled in the package (see [closed_loop.md](closed_loop.md)).

Replay uses `conditioning="episode"`.

## Running

Start the world-model server, then run the replay example:

```bash
# terminal 1: world-model server
MODEL=gesim_v2 CONFIG=configs/gesim_v2.yaml bash scripts/serve_world_model.sh

# terminal 2: replay
python examples/replay.py --server http://localhost:9000 \
    --episode assets/demo_000 --output-dir outputs/replay
```

Useful flags:

- `--max-frames N` — replay only the first `N` recorded actions (`0` = full
  episode).
- `--chunk-size 25`, `--fps 16`.

To score frames, attach a `RewardClient` in the example
(`WorldModelEnv(reward=...)`); none is bundled (see `adding_rewards.md`).

For a CPU-only smoke test, start the server with `--model example` instead
(`python -m gesim.server --model example`); it returns correctly shaped
synthetic frames with no GPU or checkpoints.

## Outputs

The example writes the following into `--output-dir`:

| File | Contents |
|---|---|
| `video.mp4` | The full rollout, three views tiled horizontally. |
| `state.npy` | `(T, 16)` Pose-Expert predicted state in WM layout (when produced). |
| `reward.npy` | `(T,)` per-frame success probability (only when a reward client is attached). |
| `metrics.json` | Task, frame count, average chunk time, final reward. |
