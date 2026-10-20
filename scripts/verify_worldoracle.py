#!/usr/bin/env python3
"""Independently check complete WorldOracle prediction folders; no GPU required."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset-root", type=Path, required=True)
    p.add_argument("--prediction-root", type=Path, required=True)
    p.add_argument("--tasks", nargs="+")
    p.add_argument("--episodes", nargs="+")
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()

    def load(path):
        return json.loads(path.read_text())

    manifest = args.asset_root / "manifest.json"
    if manifest.exists():
        paths = [args.asset_root / e["path"] for e in load(manifest)["episodes"]]
    else:
        paths = [
            p.parent / name
            for p in sorted(args.asset_root.glob("*/task.json"))
            for name in load(p)["episodes"]
        ]
    paths = [
        p
        for p in paths
        if (not args.tasks or p.parent.name in args.tasks)
        and (not args.episodes or p.name in args.episodes)
    ]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Empty selection or duplicate manifest entries")
    results = []
    for ep in paths:
        n = int(load(ep / "metadata.json")["num_frames"])
        out = args.prediction_root / ep.parent.name / ep.name
        record = load(out / "rollout.json")
        if record["status"] != "complete" or record["checkpoint_sha256"] != args.checkpoint_sha256:
            raise ValueError(f"Incomplete run or wrong checkpoint: {out}")
        if record["expected_frames"] != n or record["frames_per_camera"] != n:
            raise ValueError(f"Incorrect recorded frame count: {out}")
        h = hashlib.sha256()
        for cam in ("top_head", "hand_left", "hand_right"):
            files = sorted((out / "video" / cam).glob("frame_*"))
            if [f.name for f in files] != [f"frame_{i:06d}.jpg" for i in range(n)]:
                raise ValueError(f"Incorrect frame names/count: {out}/{cam}")
            for f in files:
                with Image.open(f) as im:
                    im.load()
                    if im.mode != "RGB" or im.size != (512, 384):
                        raise ValueError(f"Incorrect RGB image dimensions: {f}")
                h.update(f.relative_to(out).as_posix().encode())
                h.update(hashlib.sha256(f.read_bytes()).digest())
        results.append(
            {
                "task": ep.parent.name,
                "episode": ep.name,
                "frames_per_view": n,
                "images_verified": n * 3,
                "ordered_image_tree_sha256": h.hexdigest(),
            }
        )
    report = {"status": "passed", "checkpoint_sha256": args.checkpoint_sha256, "episodes": results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
