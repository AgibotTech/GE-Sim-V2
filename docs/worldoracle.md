# WorldOracle inference

Run GE-Sim 2.0 G02 and YAM checkpoints on the [WorldOracle](https://worldoracle-benchmark.github.io/) benchmark.

## Setup

Download the checkpoints and supporting files from the shared model repository:

```bash
hf download agibot-world/Genie-Envisioner-Sim-v2.0 \
  --include "checkpoints/gesim_community_v2.0.1_g02_2B/**" \
            "checkpoints/gesim_community_v2.0.1_yam_2B/**" \
            "checkpoints/gesim_community_v2.0.1_g01op_distill_2B/vae/**" \
            "checkpoints/gesim_community_v2.0.1_g01op_distill_2B/scheduler/**" \
            "requirements-tested.txt" \
  --local-dir /path/to/release
```

From the GE-Sim-V2 repository, install the inference dependencies on Linux with Python 3.10:

```bash
pip install -e '.[worldoracle]' -c /path/to/release/requirements-tested.txt
```

Download the WorldOracle inference data. Both models reuse the GE-Sim `vae/` and
`scheduler/` from `checkpoints/gesim_community_v2.0.1_g01op_distill_2B`.

The YAM camera calibration used for evaluation was estimated by us from the validation set.

## Inference

G02:

```bash
python scripts/infer_worldoracle.py \
  --config configs/worldoracle_g02.yaml \
  --release-root /path/to/release \
  --base-model /path/to/release/checkpoints/gesim_community_v2.0.1_g01op_distill_2B \
  --asset-root /path/to/inference_dataset_g02 \
  --out-root /path/to/predictions/g02 --vae-slicing
```

YAM (set `calibration` in the config to your local calibration JSON):

```bash
python scripts/infer_worldoracle.py \
  --config configs/worldoracle_yam.yaml \
  --release-root /path/to/release \
  --base-model /path/to/release/checkpoints/gesim_community_v2.0.1_g01op_distill_2B \
  --asset-root /path/to/inference_dataset_yam \
  --out-root /path/to/predictions/yam --vae-slicing
```

The profiles use 20 inference steps at 30 FPS. By default, all episodes are processed;
use `--tasks` and `--episodes` to select a subset. Use a fresh output directory.
Predictions are saved under `<out>/<task>/<episode>/video/<camera>/`, with a
`rollout.json` for each episode and a top-level `validation.json`.
