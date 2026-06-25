# Installation

## Requirements

- Python >= 3.10.
- A CUDA GPU is required for the `gesim_v2` world-model server. The client
  (`WorldModelEnv`, policies, rewards) runs anywhere, including CPU-only
  machines and CI.

Clone with submodules so the `openpi` policy server is available:

```bash
git clone --recursive <repo-url> gesim
cd gesim
```

## Install variants

```bash
pip install -e ".[server]"   # world-model server (GPU): adds diffusers,
                             # transformers, accelerate, safetensors, ...
pip install -e .             # client only (runs anywhere)
pip install -e ".[dev]"      # development: pytest + ruff
```

The policy client is a separate lightweight package from the submodule:

```bash
pip install -e third_party/openpi/packages/openpi-client
```

`pinocchio` (the `pin` pip package) is a base dependency: it provides the
forward kinematics used to render the closed-loop trajectory band, so it is
installed by all variants above.

## Accelerated kernels

The world-model server runs without any of the kernels below, but they
substantially speed up inference. Each maps to a flag in
[`configs/gesim_v2.yaml`](../configs/gesim_v2.yaml); set the flag to `false` to
run without the corresponding kernel.

Install a CUDA build of PyTorch first (the CUDA 12.6 build is shown; pick the
one matching your driver):

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126
```

Building the kernels from source against a non-default CUDA additionally needs
the CUDA toolkit and compilers (e.g. `cuda-nvcc`, `gcc`/`g++`, `cmake`,
`ninja`). The steps below are a recipe; run only the ones you need.

**Flash-Attention** (recommended for the world model at full speed):

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
(cd flash-attention && MAX_JOBS=4 python setup.py install)
```

**SpargeAttn** (optional — maps to `sparge_attention`):

```bash
git clone https://github.com/thu-ml/SpargeAttn.git
(cd SpargeAttn && MAX_JOBS=16 python setup.py install)
```

**PyTorch3D** (optional — used by some FK utilities):

```bash
git clone https://github.com/facebookresearch/pytorch3d.git
(cd pytorch3d && MAX_JOBS=16 python setup.py install)
```

The remaining acceleration flags — `liger_norm`, `liger_layernorm`,
`triton_rope` — are toggled in the config and require their respective Python
packages (`liger-kernel`, `triton`); all are optional and default to fused
kernels when available.

## Checkpoints

The released checkpoints live on Hugging Face at
[`agibot-world/Genie-Envisioner-Sim-v2.0`](https://huggingface.co/agibot-world/Genie-Envisioner-Sim-v2.0):

```bash
huggingface-cli download agibot-world/Genie-Envisioner-Sim-v2.0 \
    --include "checkpoints/**" --local-dir .
```

That fetches `checkpoints/gesim_community_v2.0.1_g01op_distill_2B` (the world
model) and `checkpoints/pi05_gesim_g01op_test` (the pi05 policy, see
[`../configs/openpi/README.md`](../configs/openpi/README.md)).

Set `checkpoint` in [`configs/gesim_v2.yaml`](../configs/gesim_v2.yaml) to the
downloaded world-model folder:

```yaml
checkpoint: checkpoints/gesim_community_v2.0.1_g01op_distill_2B
```

The folder is self-contained — a single path is all the world model needs. The
checkpoint is named
`gesim_<channel>_v<version>_<robot+gripper>_<variant>_<size>` (e.g.
`gesim_community_v2.0.1_g01op_distill_2B` = community release v2.0.1, Genie-01 +
OmniPicker, distilled, 2B backbone) so different releases stay unambiguous.

## Closed-loop robot model

Closed-loop rollouts (`conditioning="action"`) render the trajectory band from
policy actions via forward kinematics. The Genie-01 (G01) kinematics ship as a
compiled FK library inside the package (`gesim/conditioning/_g01_fk.so`) — no URDF
or robot geometry is published — so closed loop needs no extra setup. Replay
(`conditioning="episode"`) needs no robot model at all.
