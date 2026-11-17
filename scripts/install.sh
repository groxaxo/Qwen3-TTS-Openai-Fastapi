#!/usr/bin/env bash
# Linux NVIDIA installation. Reuses existing Conda prefixes; never removes envs.
set -euo pipefail
if [[ "${1:-}" == '--help' ]]; then
    cat <<'EOF'
Usage: ./scripts/install.sh
Install the default 1.7B CustomVoice CUDA-graph runtime into two Conda envs.
Existing envs are reused and their dependencies are updated to the tested pins.

CONDA_EXE          Existing conda executable (auto-detected)
TTS_FRONTEND_ENV   API prefix (default: <conda-base>/envs/qwen3-tts)
TTS_GRAPH_ENV      Worker prefix (default: <conda-base>/envs/qwen3-tts-graphs)
TTS_MODEL_ID      HF ID or local path (default: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
TTS_MODEL_REVISION Optional Hugging Face commit/tag for reproducible model weights
SKIP_MODEL_DOWNLOAD=1   Download on first start instead (used by Docker builds)
EOF
    exit 0
fi
source "$(dirname "$0")/runtime-env.sh"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo 'This installer targets Linux x86_64 NVIDIA GPUs. See README for CPU, ROCm and MLX.' >&2
    exit 1
fi
if [[ -z "${CONDA_EXE:-}" ]]; then echo 'Set CONDA_EXE to an existing Conda executable.' >&2; exit 1; fi
command -v git >/dev/null || { echo 'Install git first.' >&2; exit 1; }
for prefix in "$TTS_FRONTEND_ENV" "$TTS_GRAPH_ENV"; do
    if [[ ! -x "$prefix/bin/python" ]]; then
        "$CONDA_EXE" create --yes --prefix "$prefix" --override-channels -c conda-forge python=3.12 pip
    fi
    "$prefix/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "The default runtime requires Python 3.12"'
    "$prefix/bin/python" -m pip install --upgrade 'pip==25.3'
    "$prefix/bin/python" -m pip install 'torch==2.9.1' 'torchaudio==2.9.1' --index-url https://download.pytorch.org/whl/cu128
done
"$TTS_FRONTEND_ENV/bin/python" -m pip install -c "$TTS_REPO_DIR/requirements/frontend-constraints.txt" -e "$TTS_REPO_DIR[api]"
# Do not install this repository in the worker env: both distributions expose qwen_tts.
"$TTS_GRAPH_ENV/bin/python" -m pip install -r "$TTS_REPO_DIR/requirements/cuda-graphs.txt"
"$TTS_FRONTEND_ENV/bin/python" -m pip check
"$TTS_GRAPH_ENV/bin/python" -m pip check
if [[ "${SKIP_MODEL_DOWNLOAD:-0}" != 1 ]]; then
    (
        cd /tmp
        env -u PYTHONPATH "$TTS_GRAPH_ENV/bin/python" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
model = os.getenv('TTS_MODEL_NAME') or os.getenv('TTS_MODEL_ID', 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice')
if not Path(model).is_dir():
    path = snapshot_download(model, revision=os.getenv('TTS_MODEL_REVISION'))
    print(f'Model cached at {path}')
PY
    )
fi
echo "Ready. Run: $TTS_REPO_DIR/start_server.sh"
echo "API env: $TTS_FRONTEND_ENV"
echo "Worker env: $TTS_GRAPH_ENV"
