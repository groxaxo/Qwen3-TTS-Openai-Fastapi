#!/usr/bin/env bash
# Default NVIDIA launch path: one public API and one private CUDA-graph worker.
set -euo pipefail
source "$(dirname "$0")/scripts/runtime-env.sh"
if [[ ! -x "$TTS_FRONTEND_ENV/bin/python" ]]; then
    echo 'API Python not found. Run ./scripts/install.sh or set TTS_FRONTEND_ENV.' >&2
    exit 1
fi
exec "$TTS_FRONTEND_ENV/bin/python" "$TTS_REPO_DIR/scripts/serve.py" "$@"
