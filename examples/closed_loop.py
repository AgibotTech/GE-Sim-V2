"""Closed-loop rollout: an openpi policy drives the world model.

Usage (three terminals):
    python -m gesim.server --model gesim_v2 --config configs/gesim_v2.yaml
    bash scripts/serve_policy_pi05.sh        # openpi policy server
    python examples/closed_loop.py --server http://localhost:9000 \
        --policy ws://localhost:8000 --episode assets/demo_000 --steps 8
"""

import argparse
from pathlib import Path

import numpy as np

from gesim import EpisodeBundle, WorldModelEnv
from gesim.policies import OpenPIPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:9000")
    parser.add_argument("--policy", default="ws://127.0.0.1:8000")
    parser.add_argument("--episode", default="assets/demo_000")
    parser.add_argument("--task", default="", help="defaults to the bundle's task.txt")
    parser.add_argument("--steps", type=int, default=8, help="policy inference rounds")
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=50,
        help="actions consumed per policy round (multiple of 25 recommended)",
    )
    parser.add_argument("--output-dir", default="outputs/closed_loop")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="disable per-turn action compression; split each round into model-sized sub-chunks",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = EpisodeBundle.load(args.episode)
    policy = OpenPIPolicy(args.policy, action_horizon=args.action_horizon)

    # Attach a RewardClient here (reward=...) to score frames; none is bundled.
    with WorldModelEnv(args.server, compress_actions=not args.no_compress) as env:
        obs = env.reset(bundle, task=args.task or None, conditioning="action")
        policy.reset()
        print(f"[closed_loop] task: {obs.task!r}")
        print(f"[closed_loop] watch progress on the world-model dashboard: {args.server}")

        rewards = []
        for step in range(args.steps):
            actions = policy.infer(obs)
            obs, reward, _state, info = env.step(actions)
            if reward is not None:
                rewards.append(reward)
                print(
                    f"[closed_loop] step {step}: {info.frames.shape[0]} frames, "
                    f"success={float(reward[-1]):.3f}"
                )
            else:
                print(f"[closed_loop] step {step}: {info.frames.shape[0]} frames")

        env.save_video(out_dir / "closed_loop.mp4", fps=args.fps)
        if rewards:
            np.save(out_dir / "reward.npy", np.concatenate(rewards, axis=0))
        print(f"[closed_loop] done -> {out_dir}")


if __name__ == "__main__":
    main()
