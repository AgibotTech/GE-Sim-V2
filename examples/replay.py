"""Replay a recorded episode through the world model.

Usage:
    python -m gesim.server --model gesim_v2 --config configs/gesim_v2.yaml &
    python examples/replay.py --server http://localhost:9000 \
        --episode assets/demo_000 --output-dir outputs/replay
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from gesim import EpisodeBundle, WorldModelEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:9000")
    parser.add_argument("--episode", default="assets/demo_000")
    parser.add_argument("--output-dir", default="outputs/replay")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = full episode")
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--fps", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = EpisodeBundle.load(args.episode)
    if bundle.actions is None:
        raise FileNotFoundError(f"{bundle.path} has no actions_0.npy; cannot replay")
    actions = bundle.actions
    if args.max_frames > 0:
        actions = actions[: args.max_frames]

    # Attach a RewardClient here (reward=...) to score frames; none is bundled.
    with WorldModelEnv(args.server) as env:
        env.reset(bundle, conditioning="episode")
        print(f"[replay] task: {bundle.task!r}, frames: {len(actions)}")
        print(f"[replay] watch progress on the world-model dashboard: {args.server}")

        states, rewards, chunk_times = [], [], []
        for start in range(0, len(actions), args.chunk_size):
            chunk = actions[start : start + args.chunk_size]
            t0 = time.time()
            _obs, reward, state, _info = env.step(chunk)
            chunk_times.append(time.time() - t0)
            if state is not None:
                states.append(state)
            if reward is not None:
                rewards.append(reward)
            print(f"[replay] frames {start}-{start + len(chunk)}  ({chunk_times[-1]:.2f}s)")

        env.save_video(out_dir / "video.mp4", fps=args.fps)
        if states:
            np.save(out_dir / "state.npy", np.concatenate(states, axis=0))
        if rewards:
            np.save(out_dir / "reward.npy", np.concatenate(rewards, axis=0))
        metrics = {
            "task": bundle.task,
            "frames": int(env.frames.shape[0]),
            "avg_chunk_time_s": float(np.mean(chunk_times)),
            "reward_final": float(rewards[-1][-1]) if rewards else None,
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"[replay] done -> {out_dir}")


if __name__ == "__main__":
    main()
