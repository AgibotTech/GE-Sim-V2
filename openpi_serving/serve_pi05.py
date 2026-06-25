"""Serve the ported pi05 checkpoint over openpi's websocket policy server.

Runs in the openpi environment. Builds the gesim pi05 TrainConfig in-process
(the openpi submodule is never modified) and serves the checkpoint.

Usage:
    python openpi_serving/serve_pi05.py --checkpoint checkpoints/pi05_gesim_g01op_test --port 8000

The checkpoint dir must contain:
    model.safetensors
    assets/<asset-id>/norm_stats.json     (asset-id default: gesim)
"""

from __future__ import annotations

import argparse
import logging

from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server

from pi05_gesim import DEFAULT_PROMPT, make_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="openpi checkpoint dir")
    parser.add_argument("--asset-id", default="gesim")
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None, help="cuda | cuda:0 | cpu (default: auto)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = make_config(asset_id=args.asset_id, action_horizon=args.action_horizon)
    logging.info("loading pi05 policy from %s", args.checkpoint)
    policy = _policy_config.create_trained_policy(
        config,
        args.checkpoint,
        default_prompt=DEFAULT_PROMPT,
        pytorch_device=args.device,
    )
    logging.info("serving on ws://%s:%d (asset_id=%s, horizon=%d)",
                 args.host, args.port, args.asset_id, args.action_horizon)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, metadata=policy.metadata
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
