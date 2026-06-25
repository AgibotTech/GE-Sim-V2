"""World-model server launcher.

Example::

    python -m gesim.server --model gesim_v2 --config configs/gesim_v2.yaml

Preview the dashboard with synthetic data (no real model or rollout)::

    python -m gesim.server --demo
    # then open http://localhost:9000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from gesim.models.base import available_world_models, get_world_model
from gesim.server.app import WorldModelServer


def _inject_demo(server: WorldModelServer) -> None:
    """Populate the dashboard with representative synthetic data for previewing."""
    from gesim.client.codecs import encode_frame_jpeg

    server.status.on_init()
    server.status.on_task(
        "Pick up the kettle on the table with right arm and pour the water into the cup."
    )

    # Preview image: the demo bundle's first frame if present, else a gradient.
    rng = np.random.default_rng(0)
    try:
        from gesim.episode import EpisodeBundle

        frame = EpisodeBundle.load("assets/demo_000").first_frame
    except Exception:
        ramp = np.linspace(0.1, 0.9, 512, dtype=np.float32)
        frame = np.broadcast_to(ramp, (3, 3, 384, 512)).copy()
    preview = encode_frame_jpeg(frame)

    # A plausible commanded action and a slightly different predicted state.
    action = rng.uniform(-1.4, 1.4, size=16).astype(np.float32)
    for _ in range(3):
        state = action + rng.normal(0.0, 0.05, size=16).astype(np.float32)
        server.status.on_step(frames=25, state_row=state, action_row=action, preview=preview)


def main() -> None:
    parser = argparse.ArgumentParser(description="gesim world-model server")
    parser.add_argument("--model", default="gesim_v2", choices=available_world_models())
    parser.add_argument(
        "--config", default=None, help="YAML model config (required for gesim_v2; see configs/)"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="serve the dashboard with synthetic data (no real model/rollout)",
    )
    args = parser.parse_args()
    if args.demo:
        args.model = "example"
    elif args.model != "example" and not args.config:
        parser.error(f"--config is required for model {args.model!r} (see configs/)")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config: dict = {}
    if args.config:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}

    model = get_world_model(args.model).from_config(config)
    model_name = "demo" if args.demo else args.model
    server = WorldModelServer(model, host=args.host, port=args.port, model_name=model_name)
    if args.demo:
        _inject_demo(server)
        logging.info("demo dashboard ready — open http://localhost:%d", args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
