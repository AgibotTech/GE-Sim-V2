"""Portable WorldOracle rollout and strict output verification.

Only initial images and the provided action/state trajectories are read.
Training code, ground-truth future videos, and a Pose Expert are not required.
"""

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from einops import rearrange
from PIL import Image

from .geometry import CAMERAS, g02_geometry, render_conditioning, yam_geometry


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def episodes(root, tasks=None, names=None):
    manifest = root / "manifest.json"
    if manifest.exists():
        paths = [root / e["path"] for e in read_json(manifest)["episodes"]]
    else:
        # Some older YAM distributions provide task manifests only.
        paths = [
            p.parent / name
            for p in sorted(root.glob("*/task.json"))
            for name in read_json(p)["episodes"]
        ]
    result = [
        p for p in paths if (not tasks or p.parent.name in tasks) and (not names or p.name in names)
    ]
    if not result:
        raise ValueError("No matching episodes")
    if len(set(result)) != len(result):
        raise ValueError("Duplicate episode paths in manifest")
    return result


def load_episode(ep, root, cfg, release):
    import pyarrow.parquet as pq

    meta = read_json(ep / "metadata.json")
    schema = read_json(ep.parent / "task.json")
    table = pq.read_table(ep / "trajectory.parquet", columns=["action", "observation.state"])
    action = np.stack(table["action"].to_pylist()).astype(np.float32)
    state = np.stack(table["observation.state"].to_pylist()).astype(np.float32)
    expected = int(meta["num_frames"])
    if float(meta["fps"]) != float(cfg["prompt_fps"]):
        raise ValueError("Episode FPS does not match the packaged FPS-only prompt embedding")
    if len(action) != expected or len(state) != expected:
        raise ValueError(f"Trajectory length mismatch for {ep}")
    if not (np.isfinite(action).all() and np.isfinite(state).all()):
        raise ValueError(f"Non-finite trajectory for {ep}")
    if cfg["embodiment"] == "yam":
        geometry = yam_geometry(action, state, schema, release / cfg["calibration"])
    else:
        geometry = g02_geometry(
            action, state, schema, read_json(ep / "calibration.json"), fps=float(meta["fps"])
        )
    pose, K, rays, band, band_h = geometry
    if not all(np.isfinite(x).all() for x in (pose, K, rays, band)):
        raise ValueError(f"Non-finite geometry for {ep}")
    frames = []
    for camera in CAMERAS:
        with Image.open(ep / "observations" / f"{camera}.png") as im:
            rgb = np.asarray(im.convert("RGB"))
        if band_h is not None:
            rgb = cv2.resize(rgb, (224, band_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.resize(rgb, (512, 384), interpolation=cv2.INTER_LINEAR)
        frames.append(torch.from_numpy(rgb).permute(2, 0, 1))
    seed = torch.stack(frames).float().div(127.5).sub(1)
    return meta, geometry, seed


def load_pipeline(cfg, release, base_model, device, vae_slicing=False):
    from safetensors.torch import load_file

    from gesim.models.gesim_v2.networks.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
    from gesim.models.gesim_v2.pipelines.pipeline_cosmos_acwm_PE_eff import ACWMCosmos2PosePipeline
    from gesim.models.gesim_v2.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )

    from .model import VideoOnlyTransformer

    model = VideoOnlyTransformer(
        in_channels=26,
        out_channels=16,
        num_attention_heads=16,
        attention_head_dim=128,
        num_layers=28,
        mlp_ratio=4.0,
        text_embed_dim=1024,
        adaln_lora_dim=256,
        max_size=[128, 240, 240],
        patch_size=[1, 2, 2],
        rope_scale=[1.0, 3.0, 3.0],
        concat_padding_mask=True,
        extra_pos_embed_type=None,
        use_view_embed=True,
        pose_expert=False,
    ).to(dtype=torch.bfloat16)
    weights = release / cfg["checkpoint"]
    digest = sha256(weights)
    print(json.dumps({"checkpoint": str(weights), "sha256": digest}), flush=True)
    # A wrong model or missing key is fatal: no warm-start fallback or random head.
    model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    print("Checkpoint strict load: all keys and shapes matched", flush=True)
    vae = AutoencoderKLWan.from_pretrained(str(base_model / "vae"), torch_dtype=torch.bfloat16)
    if vae_slicing:
        vae.enable_slicing()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(base_model / "scheduler"))
    pipe = ACWMCosmos2PosePipeline(
        text_encoder=None,
        tokenizer=None,
        transformer=model.eval(),
        vae=vae.eval(),
        scheduler=scheduler,
    )
    pipe.to(device)
    embeds = torch.load(release / cfg["prompt_embeds"], map_location="cpu", weights_only=True)
    if not isinstance(embeds, torch.Tensor):
        raise ValueError("prompt_embeds.pt must contain a tensor")
    if embeds.ndim == 2:
        embeds = embeds[None]
    if embeds.ndim != 3 or embeds.shape[-1] != 1024:
        raise ValueError(f"Unexpected prompt embedding shape: {tuple(embeds.shape)}")
    # Legacy unified caches may contain a repeated training batch. Each rollout
    # is one episode; match the original trainer's prompt_embeds[:1] selection.
    embeds = embeds[:1]
    return pipe, embeds.to(device, dtype=torch.bfloat16), digest


def ray_features(K, cameras, fmt):
    from gesim.models.gesim_v2.raymap import raymap_from_c2w

    rays = raymap_from_c2w(K, cameras, res=(384, 512))
    if fmt == "plucker":
        origin, direction = rays[:, :3], rays[:, 3:]
        rays = torch.cat([direction, torch.cross(origin, direction, dim=1)], dim=1)
    elif fmt != "od":
        raise ValueError(f"Unsupported ray format {fmt}")
    return rays


def verify_episode(path, count):
    for camera in CAMERAS:
        files = sorted((path / "video" / camera).glob("frame_*.jpg"))
        wanted = [f"frame_{i:06d}.jpg" for i in range(count)]
        if [f.name for f in files] != wanted:
            raise ValueError(f"{path}/{camera}: missing, extra, or misnumbered frames")
        for f in files:
            with Image.open(f) as im:
                im.load()
                if im.size != (512, 384) or im.mode != "RGB":
                    raise ValueError(f"Invalid image: {f}")
    return {"frames_per_camera": count, "cameras": list(CAMERAS), "images_checked": count * 3}


def model_conditioning(geometry, indices, cfg, dtype=torch.bfloat16):
    """Build the exact normalized, resized tensor consumed by the transformer."""
    from gesim.models.gesim_v2.utils.geometry_utils import resize_traj_and_ray

    pose, K, ray_c2w, band_c2w, _ = geometry
    band, cameras = render_conditioning(pose, K, ray_c2w, band_c2w, indices, cfg["embodiment"])
    # Training applies Normalize(mean=.5, std=.5) to trajectory RGB as well
    # as observations. Raw renderer parity alone does not test this contract.
    band = band.sub(0.5).div(0.5)
    rays = ray_features(K, cameras, cfg["ray_map_format"])
    cond = torch.cat([band[None], rays], dim=1).to(dtype=dtype)
    cond = resize_traj_and_ray(
        cond,
        mem_size=int(cfg["n_previous"]),
        future_size=int(cfg["chunk_size"]) // 4 + 1,
        height=48,
        width=64,
    )
    return rearrange(cond, "b c v t h w -> (b v) c t h w")


@torch.inference_mode()
def rollout(ep, cfg, release, root, out, pipe, embeds, digest, max_chunks=0):
    meta, geometry, seed = load_episode(ep, root, cfg, release)
    pose, K, ray_c2w, band_c2w, _ = geometry
    n = int(meta["num_frames"])
    chunk = int(cfg["chunk_size"])
    prev = int(cfg["n_previous"])
    dest = out / ep.parent.name / ep.name
    if dest.exists():
        raise FileExistsError(f"Output exists; use a fresh output root: {dest}")
    dest.mkdir(parents=True)
    for camera in CAMERAS:
        (dest / "video" / camera).mkdir(parents=True)
    # Disk-backed RGB history preserves the original uint8 feedback without
    # accumulating many gigabytes of float32 RAM on long WorldOracle episodes.
    history_path = dest / ".history.uint8"
    history = np.memmap(history_path, mode="w+", dtype="uint8", shape=(n, 3, 3, 384, 512))
    started = time.time()
    produced = 0
    chunks = math.ceil(n / chunk)
    if max_chunks:
        chunks = min(chunks, max_chunks)
    for i in range(chunks):
        future = np.minimum(np.arange(i * chunk, (i + 1) * chunk), n - 1)
        if i == 0:
            mem_idx = np.zeros(prev, np.int64)
            video = seed[:, None].repeat(1, prev, 1, 1, 1)
        else:
            mem_idx = torch.linspace(0, produced - 1, prev).long().numpy()
            video = (
                torch.from_numpy(np.array(history[mem_idx]))
                .permute(1, 0, 2, 3, 4)
                .float()
                .div(127.5)
                .sub(1)
            )
        # Keep video and conditioning memory on the SAME episode indices.
        indices = np.concatenate([mem_idx, future])
        cond = model_conditioning(geometry, indices, cfg).to(pipe.device)
        result = pipe(
            video=video.to(pipe.device),
            cond_to_concat=cond,
            prompt=None,
            prompt_embeds=embeds,
            height=384,
            width=512,
            num_frames=chunk,
            n_view=3,
            n_prev=prev,
            num_inference_steps=int(cfg["num_inference_steps"]),
            guidance_scale=1.0,
            fps=float(meta["fps"]),
            merge_view_into_width=False,
            postprocess_video=False,
            show_progress=False,
            return_pose=False,
            distilled_sampling=False,
            liger_norm=False,
            liger_layernorm=False,
            triton_rope=False,
            sparge_config=None,
        ).frames
        if not torch.isfinite(result).all():
            raise ValueError(f"Non-finite model output in chunk {i}")
        frames = (
            ((result.clamp(-1, 1) + 1) * 127.5).round().byte().cpu().permute(2, 0, 1, 3, 4).numpy()
        )
        take = min(chunk, n - produced)
        history[produced : produced + take] = frames[:take]
        for j in range(take):
            for v, camera in enumerate(CAMERAS):
                rgb = frames[j, v].transpose(1, 2, 0)
                Image.fromarray(rgb).save(
                    dest / "video" / camera / f"frame_{produced + j:06d}.jpg", quality=95
                )
        produced += take
        print(
            f"{ep.parent.name}/{ep.name} chunk {i + 1}/{chunks}, frames {produced}/{n}, "
            f"elapsed {time.time() - started:.1f}s",
            flush=True,
        )
    history.flush()
    del history
    history_path.unlink()
    validation = verify_episode(dest, produced)
    report = {
        "status": "complete" if produced == n else "partial_debug",
        "task": ep.parent.name,
        "episode": ep.name,
        "expected_frames": n,
        "checkpoint_sha256": digest,
        "config": cfg,
        "fps": meta["fps"],
        "prompt_sha256": sha256(release / cfg["prompt_embeds"]),
        "conditioning_version": cfg.get("conditioning_version", "training-aligned-v2"),
        "elapsed_seconds": time.time() - started,
        **validation,
    }
    (dest / "rollout.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--release-root", type=Path, required=True)
    p.add_argument(
        "--base-model",
        type=Path,
        required=True,
        help="GE-Sim checkpoint directory containing vae/ and scheduler/",
    )
    p.add_argument("--asset-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--tasks", nargs="+")
    p.add_argument("--episodes", nargs="+")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true", help="CPU geometry and input validation only")
    p.add_argument(
        "--max-chunks", type=int, default=0, help="Debug only; zero means complete episodes"
    )
    p.add_argument(
        "--vae-slicing", action="store_true", help="Decode views sequentially to reduce peak VRAM"
    )
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    selected = episodes(args.asset_root, args.tasks, args.episodes)
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        reports = []
        for ep in selected:
            meta, geometry, _ = load_episode(ep, args.asset_root, cfg, args.release_root)
            pose, K, rays, cameras, _ = geometry
            band, _ = render_conditioning(pose, K, rays, cameras, [0], cfg["embodiment"])
            preview = rearrange(band[:, :, 0], "c v h w -> h (v w) c").numpy()
            Image.fromarray((preview.clip(0, 1) * 255).astype(np.uint8)).save(
                args.out_root / f"{ep.parent.name}_{ep.name}_conditioning.png"
            )
            reports.append(
                {
                    "task": ep.parent.name,
                    "episode": ep.name,
                    "frames": meta["num_frames"],
                    "intrinsic": K.tolist(),
                }
            )
        (args.out_root / "preflight.json").write_text(
            json.dumps({"status": "geometry_only", "episodes": reports}, indent=2) + "\n"
        )
        print("Geometry preflight complete; no model inference was performed.", flush=True)
        return
    pipe, embeds, digest = load_pipeline(
        cfg, args.release_root, args.base_model, args.device, args.vae_slicing
    )
    reports = []
    for ep in selected:
        reports.append(
            rollout(
                ep,
                cfg,
                args.release_root,
                args.asset_root,
                args.out_root,
                pipe,
                embeds,
                digest,
                args.max_chunks,
            )
        )
        (args.out_root / "validation.json").write_text(
            json.dumps({"episodes": reports}, indent=2) + "\n"
        )
    print(
        "Complete" if all(r["status"] == "complete" for r in reports) else "Partial debug output",
        flush=True,
    )
