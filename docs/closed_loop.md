# Closed-loop policy rollout

In a closed-loop rollout a live policy drives the world model: the policy looks
at the current observation, emits an action chunk, the world model renders the
predicted next frames, and those frames become the next observation. No physical
robot is in the loop.

## Processes

A closed-loop run uses up to three processes:

1. **World-model server** — generates video and state from action chunks.

   ```bash
   MODEL=gesim_v2 CONFIG=configs/gesim_v2.yaml bash scripts/serve_world_model.sh
   ```

2. **openpi policy server** — serves the pi05 checkpoint over a websocket. See
   [`configs/openpi/README.md`](../configs/openpi/README.md) for picking or
   defining the train config and for the GPU verification checklist.

   ```bash
   OPENPI_CKPT=checkpoints/pi05_gesim_g01op_test bash scripts/serve_policy_pi05.sh
   ```

Then run the example:

```bash
python examples/closed_loop.py --server http://localhost:9000 \
    --policy ws://localhost:8000 --episode assets/demo_000 --steps 8
```

## One loop iteration

Each round of the loop, for one policy action chunk:

1. `policy.infer(obs)` returns a joint-space action chunk `(horizon, 16)` —
   `[L7_arm, L_grip, R7_arm, R_grip]`, 50 rows by default — not end-effector poses.
2. `env.step(actions)` compresses the chunk to one model chunk of 25 (default;
   see below) — or, with `compress_actions=False`, splits it into model-sized
   sub-chunks of 25.
3. For each (sub-)chunk, the env renders the trajectory band from those actions via
   forward kinematics: the head camera stays on the episode's recorded mount
   while the wrist cameras follow the FK-derived motion of the arm links. The
   band and its per-frame camera-to-world are uploaded with `set_episode_traj`.
4. The world model `step`s on the chunk and returns `(T, 3, V, H, W)` frames
   plus the Pose-Expert predicted state.
5. The next `Observation` is built from the last generated frame, with
   proprioception taken from the Pose-Expert state (reordered to policy layout),
   and fed back into `policy.infer` on the next round.

## One inference per turn (`compress_actions`)

**By default**, a `step` chunk longer than `chunk_size` is compressed to a single
model chunk, so the world model runs **once per turn** (a 50-action pi05 round →
25). The compression matches the server's preprocessing — gripper dims are
nearest-neighbour sampled (preserving open/close timing) while arm dims keep
their endpoints and average-pool the interior.

To keep the full horizon instead — splitting a long chunk into model-sized
sub-chunks, one inference each — pass `WorldModelEnv(compress_actions=False)` (or
`--no-compress` on the example). That preserves temporal resolution (50
frames/turn) at the cost of more WM inferences.

When a `RewardClient` is attached (`WorldModelEnv(reward=...)`), the head view of
each step's frames is scored, and `step` returns per-frame success (`reward`) and
progress (`StepInfo.progress`). No reward model is bundled — see
`adding_rewards.md`.

## Robot model

Closed-loop conditioning (`conditioning="action"`) renders the band from policy
actions via forward kinematics. The Genie-01 (G01) kinematics ship as a compiled
FK library inside the package (`gesim/conditioning/_g01_fk.so`): it takes a 16-D
policy action plus the held head/waist joints and returns the two end-effector poses.
No URDF or robot geometry is published, and closed loop needs no extra setup.

## Serving pi05

The pi05 checkpoint may need a custom openpi train config (16-D actions, three
camera keys, joint norm stats). The full recipe — defining `pi05_gesim`,
serving, and a checkpoint-load verification snippet — is in
[`configs/openpi/README.md`](../configs/openpi/README.md), which also contains a
GPU verification checklist to run before publishing results.

## Troubleshooting

- **`ImportError: openpi-client is not installed`** when constructing
  `OpenPIPolicy`: install the lightweight client from the submodule:

  ```bash
  pip install -e third_party/openpi/packages/openpi-client
  ```
