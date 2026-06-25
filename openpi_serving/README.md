# Serving the pi05 policy with openpi

This directory serves the released pi05 policy checkpoint with **open-source
openpi** — its inference transforms are reproduced with openpi's own pi05
transform stack, built **in-process**; the `third_party/openpi` submodule is
**not modified**.

```
pi05_gesim.py   in-process TrainConfig + GesimInputs/GesimOutputs
serve_pi05.py   create_trained_policy + websocket server (port 8000)
```

The released checkpoint is on Hugging Face as `checkpoints/pi05_gesim_g01op_test`
(repo [`agibot-world/Genie-Envisioner-Sim-v2.0`](https://huggingface.co/agibot-world/Genie-Envisioner-Sim-v2.0)):

```bash
huggingface-cli download agibot-world/Genie-Envisioner-Sim-v2.0 \
    --include "checkpoints/pi05_gesim_g01op_test/**" --local-dir .
```

## 0. Why openpi needs its own environment

openpi pins `transformers==4.53.2`, `numpy<2`, and ships a `transformers_replace`
monkeypatch — incompatible with the world-model env (transformers 5.x, numpy 2.x).
The policy server is a separate process anyway, so give it a dedicated
environment. The world-model server keeps its own env; they talk over
websocket/HTTP.

## 1. Set up the openpi environment

Full install (supported, heavy — pulls jax/lerobot you don't need for PyTorch
inference but it is the maintained path):

```bash
cd third_party/openpi
# If the lerobot git dep trips on a missing git-LFS test artifact, prefix with
# GIT_LFS_SKIP_SMUDGE=1. Use --frozen to install from the committed uv.lock
# (avoids a slow/failing universal re-resolution of torch).
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
```

**Required overlay (pi05 PyTorch):** openpi's pi05 model patches transformers'
gemma/paligemma; copy the overlay into the installed transformers or
`PI0Pytorch.__init__` raises "transformers_replace is not installed correctly":

```bash
cp -r src/openpi/models_pytorch/transformers_replace/* \
      .venv/lib/python3.11/site-packages/transformers/
```

PyTorch-inference-only (lighter, fresh env):

```bash
conda create -n openpi python=3.11 -y && conda activate openpi
pip install "torch==2.7.1" "transformers==4.53.2" "numpy<2" sentencepiece pillow
pip install -e third_party/openpi/packages/openpi-client
pip install -e third_party/openpi            # the openpi package itself
# then apply the transformers_replace overlay as above
```

GPU: the pi05 model (gemma_2b + gemma_300m, bf16) needs roughly 9–10 GB free.

## 2. Serve

```bash
# inside the openpi env
OPENPI_CKPT=checkpoints/pi05_gesim_g01op_test PORT=8000 bash scripts/serve_policy_pi05.sh
# or directly:
PYTHONPATH=openpi_serving python openpi_serving/serve_pi05.py \
    --checkpoint checkpoints/pi05_gesim_g01op_test --port 8000
```

## 3. Drive the closed loop

In the world-model env, with the world-model server up and the policy server
serving on :8000:

```bash
POLICY_URL=ws://127.0.0.1:8000 bash local/run_closed_loop_test.sh
# or directly: python examples/closed_loop.py --policy ws://127.0.0.1:8000 ...
```

`gesim.policies.OpenPIPolicy` sends the observation; `GesimOutputs` already
returns the WM action layout `[L7, L_grip, R7, R_grip]`, so no reorder is needed
client-side.
