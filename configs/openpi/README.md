# Serving the pi05 policy with openpi

`gesim.policies.OpenPIPolicy` talks to an openpi websocket policy server. The
released pi05 checkpoint runs on **open-source openpi**.

The checkpoint is released on Hugging Face as `checkpoints/pi05_gesim_g01op_test`
(repo [`agibot-world/Genie-Envisioner-Sim-v2.0`](https://huggingface.co/agibot-world/Genie-Envisioner-Sim-v2.0)).
openpi needs its own environment — see
[`openpi_serving/README.md`](../../openpi_serving/README.md) for setup. In short:

```bash
# 1. serve the policy (in the openpi env)
OPENPI_CKPT=checkpoints/pi05_gesim_g01op_test bash scripts/serve_policy_pi05.sh

# 2. drive the closed loop (world-model env)
bash local/run_closed_loop_test.sh
```

`GesimOutputs` returns the WM action layout `[L7_arm, L_grip, R7_arm, R_grip]`,
so `OpenPIPolicy` needs no client-side reorder.

## Verification checklist (GPU required)

1. `serve_pi05.py` serves; `examples/closed_loop.py --steps 1 --policy
   ws://127.0.0.1:8000` returns a `(50, 16)` chunk with plausible joint values.
2. The world-model server consumes the chunk and returns 50 frames.
