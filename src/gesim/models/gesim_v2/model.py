"""
GE Sim V2 world model: Cosmos backbone with a Pose Expert (PE) head.

Wraps the ge_sim_v2 ``ACWMCosmos2PosePipeline`` behind the ``WorldModel`` ABC.

Memory protocol (mirrors the training-time validation rollout):

  Chunk 0 — we only have **one** real observation:
    * memory video:  repeat the current observation ``N_PREV=4`` times.
    * memory traj :  repeat ``episode_traj[..., 0:1]`` 4× (band drawn at frame 0).
    * memory c2w  :  repeat ``episode_c2w[:, 0:1]`` 4× (frame 0 extrinsics).
    * future window: ``episode_traj[..., 0:CHUNK]`` / ``action_chunk[0:CHUNK]``.
    Training uses 4 *distinct* sequential GT frames; we tell the model "static
    first frame" by replicating, which keeps memory self-consistent.

  Chunk i ≥ 1 — autoregressive:
    * ``select_mem = torch.linspace(0, T_gen-1, N_PREV).long()`` over the
      *whole* prediction history (no truncation, matching training).
    * memory video = ``generated_frames[select_mem]``.
    * memory traj  = ``episode_traj[..., select_mem]``  (same indices ⇒ video
                                                      & band stay in sync).
    * memory c2w   = ``episode_c2w[:, select_mem]``.
    * future       = ``episode_traj[..., i*CHUNK : (i+1)*CHUNK]`` and matching
                     ``episode_c2w`` slice + ``action_chunk[0:CHUNK]``.

Camera parameters (``set_camera_params``):
  Used only as a fallback when the client does **not** call
  ``set_episode_traj``. With ``set_episode_traj`` populated, all on-the-fly
  hand-eye FK is bypassed — the band and the ray map both come from the
  uploaded ``traj`` / ``c2w``, so server conditioning is bit-equivalent to
  what the model saw during training.

PE output:
  16-dim abs-joint prediction per future frame.

Action input:
  16-dim abs-joint, layout ``[L7_arm, L_grip, R7_arm, R_grip]``.
"""

import json
import logging
import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from gesim.checkpoints import resolve
from gesim.models.base import StepResult, WorldModel
from gesim.models.gesim_v2.networks.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from gesim.models.gesim_v2.networks.transformers.transformer_cosmos_multiview_PE import (
    MultiViewCosmosTransformer3DModelState,
)
from gesim.models.gesim_v2.pipelines.pipeline_cosmos_acwm_PE_eff import ACWMCosmos2PosePipeline
from gesim.models.gesim_v2.raymap import raymap_from_c2w
from gesim.models.gesim_v2.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from gesim.models.gesim_v2.utils.geometry_utils import resize_traj_and_ray
from gesim.models.gesim_v2.utils.torch_utils import load_state_dict

logger = logging.getLogger("gesim.gesim_v2")

# A consolidated GE Sim V2 checkpoint folder (the ``checkpoint`` key) holds every
# weight under conventional names; these expand to the explicit keys below. Any
# explicit key, if set, overrides the folder convention. ``base_model`` is the
# folder itself — it provides the ``vae/`` and ``scheduler/`` subdirs. A single
# ``model.safetensors`` carries both the distilled video backbone and the
# pose-expert keys (they share no parameter names, so they merge losslessly).
_CKPT_FILES = {
    "backbone_checkpoint": "model.safetensors",
    "prompt_embeds": "prompt_embeds.pt",
    "pe_norm_stats": "norm_stats.json",
}

# Required once a ``checkpoint`` folder has been expanded. ``pe_checkpoint`` is
# optional: it overlays pose-expert weights from a SEPARATE file (the legacy
# two-file form); the consolidated ``model.safetensors`` already contains them.
# Each accepts a local path or an hf:// URI.
_REQUIRED_KEYS = (
    "base_model",
    "backbone_checkpoint",
    "prompt_embeds",
    "pe_norm_stats",
)

# Back-compat: older configs named the video backbone ``cosmos_checkpoint``.
_KEY_ALIASES = {"cosmos_checkpoint": "backbone_checkpoint"}


def _expand_checkpoint(config: dict) -> dict:
    """Expand a single ``checkpoint`` folder into the explicit artifact keys.

    A folder laid out as ``vae/  scheduler/  backbone.safetensors
    pose_expert.safetensors  prompt_embeds.pt  norm_stats.json`` is addressed by a
    single ``checkpoint`` path. Explicit keys (and the legacy ``cosmos_checkpoint``
    alias) take precedence, so a caller can still override any single artifact.
    """
    cfg = dict(config)
    for old, new in _KEY_ALIASES.items():
        if cfg.get(old) and not cfg.get(new):
            cfg[new] = cfg[old]
    root = cfg.get("checkpoint")
    if root:
        root = str(root).rstrip("/")
        cfg.setdefault("base_model", root)  # holds vae/ + scheduler/
        for key, name in _CKPT_FILES.items():
            cfg.setdefault(key, f"{root}/{name}")
    return cfg

# ── Video/model resolution ───────────────────────────────────────────────────
SAMPLE_H = 384
SAMPLE_W = 512
N_VIEW = 3
N_PREV = 4
CHUNK_SIZE = 25
TEMPORAL_DOWN_RATIO = 4
SPATIAL_DOWN_RATIO = 8

# ── Pose Expert dims ─────────────────────────────────────────────────────────
POSE_DIM = 16  # 16-D abs-joint, layout [L7_arm, L_grip, R7_arm, R_grip]
# (matches the trainer's joint-pose conditioning + the policy's reordered action output).
POSE_N_PREV = 4  # history tokens for PE

# ── Pose Expert norm stats ───────────────────────────────────────────────────
# Norm stats for the 16-D abs-joint PE head. Layout matches the training-time
# joint sequence: ``[L7_arm, L_grip, R7_arm, R_grip]``.


def _load_pe_joint_norm_stats(path: str) -> dict | None:
    """Return ``{'action': {'mean','std'}, 'state': {'mean','std'}}`` numpy arrays.

    Both the PE input (history_action / history_pose) and the PE output need
    the **same** stats the trainer used. ``stat['action']`` and ``stat['state']``
    can differ (action is the teacher-forced policy command, state is the
    FK-derived observed joints). We load both and let the caller choose.
    """
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for key in ("action", "state"):
        if key not in raw or "mean" not in raw[key] or "std" not in raw[key]:
            logger.warning("PE joint norm stats %s missing '%s.mean/std'.", path, key)
            return None
        out[key] = {
            "mean": np.asarray(raw[key]["mean"], dtype=np.float32),
            "std": np.asarray(raw[key]["std"], dtype=np.float32),
        }
        if out[key]["mean"].shape != (POSE_DIM,) or out[key]["std"].shape != (POSE_DIM,):
            logger.warning(
                "PE joint norm stats %s shape mismatch: got mean %s, std %s, want (%d,).",
                key,
                out[key]["mean"].shape,
                out[key]["std"].shape,
                POSE_DIM,
            )
            return None
    return out


# Distillation sigma schedule for the 4-step distilled sampler.
# Corresponds to TrigFlow [pi/2, atan(15), atan(5), atan(5/3)] converted to
# normalized sigma: sigma = sigma_EDM / (sigma_EDM + 1).
DISTILL_SIGMA_SCHEDULE = [1.0, 0.9375, 0.8333, 0.625]
_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
FPS = 30  # must match the validation config; the pipeline default (16) is wrong


def _parse_sigma_schedule(schedule, num_steps=None):
    if schedule is None:
        return None
    if isinstance(schedule, str):
        values = [float(v.strip()) for v in schedule.split(",") if v.strip()]
    else:
        values = [float(v) for v in schedule]
    if not values:
        return None
    if num_steps is None:
        return values

    n_steps = int(num_steps)
    if n_steps <= 0:
        raise ValueError(
            f"num_inference_steps must be positive for distilled sampling, got {n_steps}"
        )
    non_terminal = values[:-1] if abs(values[-1]) <= 1e-8 else values
    if len(non_terminal) < n_steps:
        raise ValueError(
            f"distill_sigma_schedule only has {len(non_terminal)} non-terminal sigma values, "
            f"but num_inference_steps={n_steps}"
        )
    return non_terminal[:n_steps]


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


class GeSimV2WorldModel(WorldModel):
    """Cosmos backbone + Pose Expert world model."""

    chunk_size = CHUNK_SIZE

    def __init__(self, config=None):
        self.config = _expand_checkpoint(config or {})

        self.device: torch.device | None = None
        self.dtype: torch.dtype = torch.bfloat16

        self.vae = None
        self.transformer = None
        self.scheduler = None
        self.pipe: ACWMCosmos2PosePipeline | None = None
        self.prompt_embeds: torch.Tensor | None = None

        # Generated frame history — ALL frames produced during the current episode.
        # For chunk i>0, n_prev memory frames are sampled via linspace from this
        # list, matching the autoregressive rollout strategy.
        self.generated_frames: list[torch.Tensor] = []  # each: (V, 3, H_m, W_m), [-1,1], CPU
        # Initial observation buffer (only ever holds 1 real frame in this build:
        # the open-loop and closed-loop drivers send a single first frame).
        # We keep the deque shape so the rest of the code is unchanged, but the
        # chunk-0 memory video is just that one frame replicated N_PREV times.
        self.frame_buffer: deque = deque(maxlen=N_PREV)

        # Camera params
        self.intrinsic: torch.Tensor | None = None  # (V, 3, 3) on device
        self.c2w: torch.Tensor | None = None  # (V, 4, 4) numpy, initial-frame c2w
        self.w2c: torch.Tensor | None = None  # (V, 4, 4) numpy, initial-frame w2c

        # Pose / action history buffers – each entry: (POSE_DIM,) tensor.
        # Both store **unnormalised** abs-joint values so the per-step
        # normalisation in ``inference`` always uses fresh stats.
        self.pose_history_buffer: deque = deque(maxlen=POSE_N_PREV)
        self.action_history_buffer: deque = deque(maxlen=POSE_N_PREV)

        # Norm stats for PE input/output (16-D abs-joint, layout
        # [L7, L_grip, R7, R_grip]). Loaded from disk during ``_load``.
        self._pe_joint_norm_stats: dict | None = None

        # Episode-level pre-rendered conditioning (set via set_episode_traj).
        # When populated, ``inference()`` slices these per chunk; otherwise we
        # raise (the legacy hand-eye-FK path no longer runs in ``inference``).
        self._episode_traj: torch.Tensor | None = None  # (1, 3, V, T_all, H, W) [0,1]
        self._episode_c2w: torch.Tensor | None = None  # (V, T_all, 4, 4) c2w per frame
        self._chunk_index: int = 0

        # Acceleration flags (defaults match the distilled-inference recipe)
        self.sparge_config: dict | None = None
        self.liger_norm: bool = True
        self.liger_layernorm: bool = True
        self.triton_rope: bool = False

    @classmethod
    def from_config(cls, config: dict) -> "GeSimV2WorldModel":
        # __init__ expands a single ``checkpoint`` folder into the explicit keys.
        model = cls(config)
        missing = [k for k in _REQUIRED_KEYS if not model.config.get(k)]
        if missing:
            raise ValueError(
                f"gesim_v2 config needs a 'checkpoint' folder or these keys: {missing}. "
                "See the example config gesim_v2.yaml shipped in the repository's "
                "configs/ directory; each accepts a local path or an hf:// URI."
            )
        model._load()
        return model

    def _get_pe_joint_norm_stats(self) -> dict | None:
        """Return the cached PE 16-D abs-joint norm stats (loaded during ``_load``)."""
        return self._pe_joint_norm_stats

    def _normalize_joint(self, x: torch.Tensor, key: str) -> torch.Tensor:
        """``(x - mean) / std`` along the last 16 dims; passthrough if stats missing."""
        stats = self._get_pe_joint_norm_stats()
        if stats is None:
            return x
        mean = torch.as_tensor(stats[key]["mean"], device=x.device, dtype=x.dtype)
        std = torch.as_tensor(stats[key]["std"], device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def _denormalize_joint(self, x: torch.Tensor, key: str) -> torch.Tensor:
        """``x * std + mean`` along the last 16 dims; passthrough if stats missing."""
        stats = self._get_pe_joint_norm_stats()
        if stats is None:
            return x
        mean = torch.as_tensor(stats[key]["mean"], device=x.device, dtype=x.dtype)
        std = torch.as_tensor(stats[key]["std"], device=x.device, dtype=x.dtype)
        return x * std + mean

    # ── Model initialisation ─────────────────────────────────────────────────

    def _load(self):
        """Load all model components onto the device."""
        device_str = self.config.get("device", "cuda")
        self.device = torch.device(device_str)
        dtype_str = self.config.get("dtype", "bfloat16")
        if dtype_str not in _DTYPES:
            raise ValueError(f"unsupported dtype {dtype_str!r}; choose one of {sorted(_DTYPES)}")
        self.dtype = _DTYPES[dtype_str]

        base_model = resolve(self.config["base_model"])
        backbone_ckpt = resolve(self.config["backbone_checkpoint"])
        pe_ckpt_ref = self.config.get("pe_checkpoint")
        pe_ckpt = resolve(pe_ckpt_ref) if pe_ckpt_ref else None

        logger.info("Loading PE joint norm stats …")
        self._pe_joint_norm_stats = _load_pe_joint_norm_stats(resolve(self.config["pe_norm_stats"]))
        if self._pe_joint_norm_stats is not None:
            am = self._pe_joint_norm_stats["action"]["mean"]
            logger.info("PE joint norm stats loaded; action.mean head4=%s", am[:4].tolist())

        logger.info("Loading VAE …")
        self.vae = AutoencoderKLWan.from_pretrained(
            os.path.join(base_model, "vae"), torch_dtype=self.dtype
        ).to(self.device)
        self.vae.eval()

        logger.info("Loading scheduler …")
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            os.path.join(base_model, "scheduler")
        )

        logger.info("Building transformer (PE variant) …")
        transformer_config = dict(
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
            pose_expert=True,
            pose_in_channels=POSE_DIM,
            pose_num_attention_heads=16,
            pose_attention_head_dim=32,
        )
        self.transformer = MultiViewCosmosTransformer3DModelState(**transformer_config).to(
            self.device, dtype=self.dtype
        )

        # Load the world model: the distilled video backbone plus the pose-expert
        # keys (the consolidated ``model.safetensors`` carries both — they share no
        # parameter names, so a single load gets everything).
        logger.info("Loading world model: %s", backbone_ckpt)
        self.transformer = load_state_dict(self.transformer, backbone_ckpt)

        # Legacy two-file form: overlay the pose-expert head from a SEPARATE file.
        # Such a file may also carry a full backbone copy (saved during joint
        # training); we take ONLY the pose-expert / video-buffer keys so the distilled
        # backbone loaded above is preserved. Skipped when the backbone file already
        # holds the pose keys (``pe_checkpoint`` unset or the same path).
        if pe_ckpt and pe_ckpt != backbone_ckpt:
            logger.info("Overlaying pose-expert checkpoint (pose keys only): %s", pe_ckpt)
            # Imported lazily: only needed for the separate-file pose overlay.
            from safetensors import safe_open

            _pose_prefixes = ("pose_", "video_buffer")
            _sd = {}
            with safe_open(pe_ckpt, framework="pt") as _f:
                for _k in _f.keys():
                    if any(_k.startswith(_p) or _k == _p or _p in _k for _p in _pose_prefixes):
                        _sd[_k] = _f.get_tensor(_k).to(device=self.device, dtype=self.dtype)
            if _sd:
                _missing, _unexpected = self.transformer.load_state_dict(_sd, strict=False)
                _pose_missing = [k for k in _missing if any(p in k for p in _pose_prefixes)]
                logger.info(
                    "PE pose keys loaded: %d, pose keys still missing: %d",
                    len(_sd), len(_pose_missing),
                )
                if _pose_missing:
                    logger.info("Still missing: %s ...", _pose_missing[:5])
            else:
                logger.warning(
                    "pose-expert checkpoint has no pose keys – PE head stays as loaded "
                    "from the backbone file."
                )

        # For inference we run both video and PE branches
        self.transformer._pose_only_forward = False
        self.transformer.eval()

        logger.info("Loading unified prompt embeddings …")
        embed_path = resolve(self.config["prompt_embeds"])
        raw = torch.load(embed_path, map_location="cpu", weights_only=True)
        self.prompt_embeds = raw.to(self.device, dtype=self.dtype)
        if self.prompt_embeds.ndim == 2:
            self.prompt_embeds = self.prompt_embeds.unsqueeze(0)  # (1, seq, dim)
        # The embed file may contain multiple embeddings (e.g. one per environment);
        # the pipeline expects batch_size=1 so that batch_size * n_view == n_view.
        self.prompt_embeds = self.prompt_embeds[:1]  # (1, seq, dim)

        # Acceleration flags (defaults match the distilled-inference recipe)
        self.liger_norm = self.config.get("liger_norm", True)
        self.liger_layernorm = self.config.get("liger_layernorm", True)
        self.triton_rope = self.config.get("triton_rope", True)
        sparge_enabled = self.config.get("sparge_attention", True)
        if sparge_enabled:
            self.sparge_config = {
                "enabled": True,
                "mode": "cdfthreshd",
                "cdfthreshd": 0.90,
                "dense_block": 1,
                "dense_timestep": 1,
                "skip_cross_view": True,
            }
        else:
            self.sparge_config = None
        logger.info(
            "Acceleration: liger_norm=%s  liger_layernorm=%s  triton_rope=%s  sparge=%s",
            self.liger_norm,
            self.liger_layernorm,
            self.triton_rope,
            sparge_enabled,
        )

        logger.info("Building pipeline …")
        self.pipe = ACWMCosmos2PosePipeline(
            text_encoder=None,
            tokenizer=None,
            transformer=self.transformer,
            vae=self.vae,
            scheduler=self.scheduler,
        )

        logger.info("Ready.")

    # ── Camera params ────────────────────────────────────────────────────────

    def set_camera_params(
        self,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray | None = None,
    ) -> None:
        """Store the per-view pinhole ``K`` (already at the model crop, 512×384).

        The intrinsics are expected to already be baked at the 512×384 model
        resolution from the dataset's calibration JSON (no server-side
        native→model scaling is applied).

        ``extrinsic`` is stashed at episode-start only for diagnostics. Per-frame
        ``c2w`` for ``cond_to_concat`` always comes from ``set_episode_traj``.
        """
        intrinsic = np.asarray(intrinsic, dtype=np.float32).copy()
        if intrinsic.ndim != 3 or intrinsic.shape[-2:] != (3, 3):
            raise ValueError(f"intrinsic must be (V, 3, 3); got {intrinsic.shape}")
        self.intrinsic = intrinsic
        if extrinsic is not None:
            extrinsic = np.asarray(extrinsic, dtype=np.float32)
            self.c2w = extrinsic
            self.w2c = np.linalg.inv(extrinsic)

    # ── ABC surface (numpy adapters) ─────────────────────────────────────────

    def reset(self) -> None:
        self.reset_buffers()

    def set_episode_data(self, first_frame: np.ndarray) -> None:
        self._set_episode_data_torch(torch.from_numpy(np.ascontiguousarray(first_frame)))

    def set_episode_traj(self, traj: np.ndarray, c2w: np.ndarray) -> None:
        self._set_episode_traj_torch(
            torch.from_numpy(np.ascontiguousarray(traj)),
            torch.from_numpy(np.ascontiguousarray(c2w)),
        )

    def step(self, actions: np.ndarray) -> StepResult:
        """Run one action chunk through the world model.

        Chunks shorter than ``chunk_size`` still run a full-length generation internally;
        outputs are truncated to the request length.
        """
        frames, state = self.inference(
            torch.from_numpy(np.ascontiguousarray(actions, dtype=np.float32))
        )
        n = int(actions.shape[0])
        return StepResult(
            frames=frames[:n].cpu().numpy().astype(np.float32),
            state=state[:n].cpu().numpy().astype(np.float32) if state is not None else None,
        )

    # ── Episode data (pre-rendered conditioning) ────────────────────────────

    def _set_episode_data_torch(
        self,
        cur_obs: torch.Tensor | None = None,
    ) -> None:
        cur_obs_f = cur_obs.to(device=self.device, dtype=torch.float32)
        obs_model = self._preprocess_obs(cur_obs_f)  # (V, 3, H_m, W_m) in [-1,1]
        self._update_frame_buffer(obs_model)
        self._chunk_index = 0

    def _set_episode_traj_torch(
        self,
        traj: torch.Tensor,
        c2w: torch.Tensor,
    ) -> None:
        """Pre-rendered trajectory band + per-frame c2w for the current episode.

        With this slot populated, ``inference()`` consumes ``_episode_traj``
        for the band and ``_episode_c2w`` for the ray map — bypassing the
        legacy hand-eye-FK path. That makes the model's visual conditioning
        bit-equivalent to what the trainer saw on the same episode.

        Args:
            traj: ``(3, V, T, H, W)`` float32 in ``[0, 1]``. Client-rendered
                via the same trajectory-band recipe as the trainer. Stored as
                ``(1, 3, V, T, H, W)`` so the rest of the pipeline can keep its
                leading batch dim. We **do not** apply ``* 2 - 1`` here; the
                per-chunk consumer does that, matching the FK-fallback path.
            c2w: ``(V, T, 4, 4)`` float32, per-frame camera-to-world. Used
                directly by ``raymap_from_c2w`` to build the model's 6-channel
                ray map — same formula as the trainer's ``prepare_ray_map``.
        """
        if traj.ndim != 5 or traj.shape[0] != 3:
            raise ValueError(f"traj must be (3, V, T, H, W); got {tuple(traj.shape)}")
        # Keep on CPU to avoid holding a multi-hundred-MB tensor on GPU for
        # the whole episode; we slice and move per-chunk inside ``inference``.
        traj_b = traj.unsqueeze(0).contiguous()  # (1, 3, V, T, H, W)
        self._episode_traj = traj_b.to(dtype=self.dtype).cpu()
        if c2w.ndim != 4 or c2w.shape[-2:] != (4, 4):
            raise ValueError(f"c2w must be (V, T, 4, 4); got {tuple(c2w.shape)}")
        self._episode_c2w = c2w.float().cpu().contiguous()

    # ── Buffer management ────────────────────────────────────────────────────

    def reset_buffers(self) -> None:
        """Clear all rolling buffers. Call at every episode reset."""
        self.frame_buffer.clear()
        self.generated_frames.clear()
        self.pose_history_buffer.clear()
        self.action_history_buffer.clear()
        self._episode_traj = None
        self._episode_c2w = None
        self._chunk_index = 0

    # ── Observation preprocessing ────────────────────────────────────────────

    def _preprocess_obs(self, cur_obs: torch.Tensor) -> torch.Tensor:
        """Resize and normalise a single observation.

        Args:
            cur_obs: (3, V, H, W) or (V, 3, H, W) float32 in [0, 1].

        Returns:
            (V, 3, SAMPLE_H, SAMPLE_W) float32 in [-1, 1].
        """
        # Ensure (V, 3, H, W)
        if cur_obs.shape[0] == 3 and cur_obs.ndim == 4:
            cur_obs = cur_obs.permute(1, 0, 2, 3)  # (V, 3, H, W)

        V = cur_obs.shape[0]
        flat = cur_obs.reshape(V, 3, cur_obs.shape[2], cur_obs.shape[3])
        flat = F.interpolate(flat, (SAMPLE_H, SAMPLE_W), mode="bilinear", align_corners=False)
        flat = flat * 2.0 - 1.0  # [0,1] → [-1,1]
        return flat  # (V, 3, SAMPLE_H, SAMPLE_W)

    def _update_frame_buffer(self, frame: torch.Tensor) -> None:
        """Append (V, 3, H_m, W_m) frame (CPU) to the rolling buffer."""
        self.frame_buffer.append(frame.cpu())

    # NOTE: ``_get_memory_frames`` was removed. Memory tensors are now built
    # inline inside ``inference()`` so that the memory video, memory traj,
    # and memory c2w can share a single ``select_mem`` index list (chunk i ≥ 1)
    # or a single "frame 0" anchor (chunk 0). Splitting that across helpers
    # was the original source of the temporal-misalignment train/eval gap.

    # ── Main inference ───────────────────────────────────────────────────────

    @torch.no_grad()
    def inference(
        self,
        action_chunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate next video chunk + PE state prediction.

        Args:
            action_chunk: (L <= CHUNK_SIZE=25, 16) tensor of abs-joint actions,
                          layout ``[L7_arm, L_grip, R7_arm, R_grip]``.

        Returns:
            frames: (CHUNK_SIZE, 3, V, H_m, W_m) RGB float32 in [0, 1].
            state:  (CHUNK_SIZE, POSE_DIM) predicted abs-joint poses.
        """
        assert self.pipe is not None, "Call from_config() first."

        # ── 0. Sanity: client must have set up frame_buffer + episode_traj ────
        if self._chunk_index == 0 and len(self.frame_buffer) == 0:
            raise RuntimeError(
                "GeSimV2WorldModel: frame_buffer is empty on the first chunk. "
                "After reset(), the client must set the initial first frame "
                "(3, V, H, W) via set_episode_data before the first step."
            )
        if self._episode_traj is None or self._episode_c2w is None:
            raise RuntimeError(
                "GeSimV2WorldModel: ``set_episode_traj`` was never called. Upload a "
                "(3, V, T, H, W) pre-rendered trajectory band + (V, T, 4, 4) per-frame "
                "c2w via set_episode_traj before the first step."
            )

        action_np = action_chunk.float().cpu().numpy()
        joints = np.concatenate([action_np[:, :7], action_np[:, 8:15]], axis=-1)
        frames = joints.shape[0]  # ≤ CHUNK_SIZE
        is_chunk0 = self._chunk_index == 0

        # Memory **video** is the same in both modes.
        if is_chunk0:
            cur_one = self.frame_buffer[-1]  # (V, 3, H, W)
            mem_video = cur_one.unsqueeze(1).repeat(1, N_PREV, 1, 1, 1).to(self.device)
        else:
            T_gen = len(self.generated_frames)
            select_mem = torch.linspace(0, T_gen - 1, N_PREV).long()
            mem_video = torch.stack([self.generated_frames[int(i)] for i in select_mem], dim=1).to(
                self.device
            )

        T_total = int(self._episode_traj.shape[3])  # episode-traj length

        if T_total >= CHUNK_SIZE * 2:
            # ── Open-loop / replay (``inference.py``): episode_traj is the full
            # recorded episode. Slice the i-th chunk and pull memory from the
            # same indices as ``mem_video`` for chunk≥1 (chunk 0 anchors at frame 0).
            t0 = self._chunk_index * CHUNK_SIZE
            t1 = min(t0 + frames, T_total)
            future_traj = self._episode_traj[:, :, :, t0:t1].clone()
            future_c2w = self._episode_c2w[:, t0:t1].clone()
            if t1 - t0 < frames:
                pad_n = frames - (t1 - t0)
                future_traj = torch.cat(
                    [future_traj, future_traj[:, :, :, -1:].repeat(1, 1, 1, pad_n, 1, 1)],
                    dim=3,
                )
                future_c2w = torch.cat(
                    [future_c2w, future_c2w[:, -1:].repeat(1, pad_n, 1, 1)], dim=1
                )
                logger.warning(
                    "episode_traj exhausted at chunk=%d; padded %d band/c2w frame(s) "
                    "by repeating last entry.",
                    self._chunk_index,
                    pad_n,
                )
            if is_chunk0:
                mem_traj = self._episode_traj[:, :, :, 0:1].repeat(1, 1, 1, N_PREV, 1, 1)
                mem_c2w = self._episode_c2w[:, 0:1].repeat(1, N_PREV, 1, 1)
            else:
                mem_traj = self._episode_traj[:, :, :, select_mem].clone()
                mem_c2w = self._episode_c2w[:, select_mem].clone()
        else:
            # ── Closed-loop: client re-uploads ``L`` frames per step and they
            # represent **only the future window** — there is no episode history
            # to slice mem from. Anchor mem on the current chunk's first frame
            # (the policy's "now"); good enough for the conditioning the model needs.
            future_traj = self._episode_traj[:, :, :, :frames].clone()
            future_c2w = self._episode_c2w[:, :frames].clone()
            if future_traj.shape[3] < frames:
                pad_n = frames - future_traj.shape[3]
                future_traj = torch.cat(
                    [future_traj, future_traj[:, :, :, -1:].repeat(1, 1, 1, pad_n, 1, 1)],
                    dim=3,
                )
                future_c2w = torch.cat(
                    [future_c2w, future_c2w[:, -1:].repeat(1, pad_n, 1, 1)], dim=1
                )
            mem_traj = future_traj[:, :, :, 0:1].repeat(1, 1, 1, N_PREV, 1, 1)
            mem_c2w = future_c2w[:, 0:1].repeat(1, N_PREV, 1, 1)

        # ── 3. Concatenate memory + future for traj and c2w ──────────────────
        traj = torch.cat([mem_traj, future_traj], dim=3).to(dtype=self.dtype)
        traj = traj * 2.0 - 1.0
        c2w_stack = torch.cat([mem_c2w, future_c2w], dim=1)
        rays = raymap_from_c2w(self.intrinsic, c2w_stack, res=(384, 512))
        rays = rays.to(dtype=self.dtype)

        memory = mem_video  # (V, N_PREV, 3, H, W)

        cond_raw = torch.cat([traj, rays], dim=1)  # (1, 9, V, T, H, W)
        cond_raw = cond_raw.to(self.device)

        # Resize to latent resolution (resize_traj_and_ray asserts orig_t > mem+future)
        cond_to_concat = resize_traj_and_ray(
            cond_raw,
            mem_size=N_PREV,
            future_size=CHUNK_SIZE // TEMPORAL_DOWN_RATIO + 1,
            height=SAMPLE_H // SPATIAL_DOWN_RATIO,
            width=SAMPLE_W // SPATIAL_DOWN_RATIO,
        )  # (1, 9, V, T_lat, H_lat, W_lat)
        cond_to_concat = rearrange(cond_to_concat, "b c v t h w -> (b v) c t h w")

        # Build PE input the same way the trainer does:
        #   pose_states = cat([history_action(4), future_action(25),
        #                      history_pose(4), future_pose_noisy(25)], dim=1)
        # All four blocks are 16-D normalised abs-joint values, layout
        # ``[L7_arm, L_grip, R7_arm, R_grip]`` (the policy's action output layout).
        # The pipeline overwrites ``future_pose_noisy`` with sampled noise on the first denoising
        # step, so we just zero-init it here for safety.
        chunk_now = int(action_chunk.shape[0])  # may be < CHUNK_SIZE for partial chunks
        action_norm_now = self._normalize_joint(
            action_chunk.to(device=self.device, dtype=torch.float32), key="action"
        )  # (chunk_now, 16)
        # Pad to CHUNK_SIZE if needed (PE expects exactly 25 future tokens).
        if chunk_now < CHUNK_SIZE:
            pad = action_norm_now[-1:].repeat(CHUNK_SIZE - chunk_now, 1)
            future_action = torch.cat([action_norm_now, pad], dim=0)
        else:
            future_action = action_norm_now[:CHUNK_SIZE]

        # History buffers are stored unnormalised. PE training never sees fewer than
        # ``POSE_N_PREV`` real history rows (the dataloader guarantees this); during
        # closed-loop the first few /step calls have an empty buffer, so pad with the
        # **current chunk's first row** rather than zeros — that matches training-time
        # "robot held still before the chunk started" much better than a 0-vector OOD prefix.
        def _pad_history(buf: deque, key: str, fallback_row: torch.Tensor) -> torch.Tensor:
            rows = [r.to(device=self.device, dtype=torch.float32) for r in buf]
            while len(rows) < POSE_N_PREV:
                rows.insert(0, fallback_row)
            stacked = torch.stack(rows[-POSE_N_PREV:], dim=0)
            return self._normalize_joint(stacked, key=key)

        # Both action history and pose history fall back to the current action's first row.
        # That row is in joint space (PE input layout [L7, L_grip, R7, R_grip]) and matches
        # the chunk's starting pose, which is the closest stand-in we have for "true" history.
        first_action_row = action_chunk[0].to(device=self.device, dtype=torch.float32)
        history_action = _pad_history(
            self.action_history_buffer, key="action", fallback_row=first_action_row
        )
        history_pose = _pad_history(
            self.pose_history_buffer, key="state", fallback_row=first_action_row
        )
        future_pose_noisy = torch.zeros(
            CHUNK_SIZE, POSE_DIM, device=self.device, dtype=torch.float32
        )

        pose_states = (
            torch.cat([history_action, future_action, history_pose, future_pose_noisy], dim=0)
            .unsqueeze(0)
            .to(dtype=self.dtype)
        )  # (1, 58, 16) — 58 = 2 * POSE_N_PREV + 2 * CHUNK_SIZE pose tokens
        pose_mask = torch.ones_like(pose_states)
        pose_timesteps = torch.zeros(1, 58, device=self.device, dtype=self.dtype)
        history_token_num = 2 * POSE_N_PREV + CHUNK_SIZE  # 33

        num_steps = int(self.config.get("num_inference_steps", 4))

        # Build inference_sigmas: use distill schedule when num_steps matches
        # the distillation training schedule length (4 steps), otherwise let the
        # pipeline fall back to linspace.
        distill_sigmas = self.config.get("distill_sigma_schedule", None)
        if distill_sigmas is None and num_steps == len(DISTILL_SIGMA_SCHEDULE):
            distill_sigmas = DISTILL_SIGMA_SCHEDULE
        distill_sigmas = _parse_sigma_schedule(distill_sigmas, num_steps)
        distilled_sampling = _as_bool(
            self.config.get("distilled_sampling", distill_sigmas is not None)
        )

        output = self.pipe(
            video=memory,  # (V, N_PREV, 3, H_m, W_m) = (b*v, t, c, h, w)
            cond_to_concat=cond_to_concat,  # (V, 9, T_lat, H_lat, W_lat)
            prompt=None,
            prompt_embeds=self.prompt_embeds,
            height=SAMPLE_H,
            width=SAMPLE_W,
            num_frames=CHUNK_SIZE,
            n_view=N_VIEW,
            n_prev=N_PREV,
            num_inference_steps=num_steps,
            guidance_scale=1.0,
            fps=FPS,
            merge_view_into_width=False,
            postprocess_video=False,
            show_progress=True,
            return_pose=True,
            pose_states=pose_states,
            pose_mask=pose_mask,
            pose_timesteps=pose_timesteps,
            pose_history_token_num=history_token_num,
            pose_buffer_store_step=None,  # store at last denoising step
            # We de-normalise ourselves below using the abs-joint stats the trainer
            # used; the pipeline's built-in path is hardcoded for 14-D EEF.
            denormalize_pose=False,
            inference_sigmas=distill_sigmas,
            distilled_sampling=distilled_sampling,
            sparge_config=self.sparge_config,
            liger_norm=self.liger_norm,
            liger_layernorm=self.liger_layernorm,
            triton_rope=self.triton_rope,
        )

        # ── 6. Parse outputs ──────────────────────────────────────────────────
        # frames: (b*v=V, 3, T, H_m, W_m) in [-1, 1]
        raw_frames = output.frames  # (V, 3, T, H_m, W_m) in [-1, 1]

        # Store generated frames into history for future memory sampling.
        # raw_frames are in (V, C, T, H, W) format; split along T.
        T_gen = raw_frames.shape[2]
        for t in range(T_gen):
            self.generated_frames.append(raw_frames[:, :, t].cpu())  # (V, 3, H, W) in [-1, 1]

        frames = rearrange(raw_frames, "v c t h w -> t c v h w")
        frames = (frames.clamp(-1.0, 1.0) + 1.0) / 2.0  # → [0, 1]

        # PE poses: (1, 58, 16) — pipeline returned the **normalised** abs-joint prediction
        # (we passed denormalize_pose=False). Slice future tokens, de-normalise with the
        # `state` stats, and remember last row for the next chunk's history.
        pose_pred = getattr(output, "poses", None)
        if pose_pred is not None:
            state_norm = pose_pred[:, history_token_num:, :].squeeze(0)  # (25, 16)
            state = self._denormalize_joint(state_norm.to(dtype=torch.float32), key="state")
            self.pose_history_buffer.append(state[-1].detach().cpu())
        else:
            # Fallback: use the unnormalised action chunk so the next chunk still has
            # *something* in the history slot.
            state = action_chunk.to(device=self.device, dtype=torch.float32)
            self.pose_history_buffer.append(state[-1].detach().cpu())

        # Always log the action we just consumed so the next chunk's history_action
        # matches what we conditioned the world model on.
        self.action_history_buffer.append(
            action_chunk[-1].detach().to(device=self.device, dtype=torch.float32).cpu()
        )

        self._chunk_index += 1
        return frames.float(), state.float()
