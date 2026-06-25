#!/usr/bin/env bash
# Serve the gesim pi05 checkpoint with openpi (in-process gesim config; the
# third_party/openpi submodule is not modified).
#
# Run inside the openpi environment (see openpi_serving/README.md for setup).
# Env vars:
#   OPENPI_CKPT   openpi checkpoint dir (default: checkpoints/pi05_gesim_g01op_test)
#   PORT          websocket port (default: 8000)
#   ASSET_ID      norm-stats asset id (default: gesim)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPENPI_CKPT="${OPENPI_CKPT:-${REPO_ROOT}/checkpoints/pi05_gesim_g01op_test}"
PORT="${PORT:-8000}"
ASSET_ID="${ASSET_ID:-gesim}"

# pi05_gesim.py and serve_pi05.py import each other by module name.
export PYTHONPATH="${REPO_ROOT}/openpi_serving:${PYTHONPATH:-}"

exec python "${REPO_ROOT}/openpi_serving/serve_pi05.py" \
    --checkpoint "${OPENPI_CKPT}" \
    --asset-id "${ASSET_ID}" \
    --port "${PORT}" \
    "$@"
