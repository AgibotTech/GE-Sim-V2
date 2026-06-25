#!/usr/bin/env bash
# Launch the gesim world-model server.
# Usage: MODEL=gesim_v2 CONFIG=configs/gesim_v2.yaml bash scripts/serve_world_model.sh [extra args]
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m gesim.server --model "${MODEL:-gesim_v2}" --config "${CONFIG:-configs/gesim_v2.yaml}" "$@"
