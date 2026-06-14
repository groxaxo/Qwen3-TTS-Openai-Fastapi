#!/usr/bin/env bash
# run_tts.sh — launch MLX Qwen3-TTS CustomVoice server instance(s).
#
# Two failure-isolated instances are supported out of the box:
#   fast  → 0.6B CustomVoice 8bit on :18881  (low latency, default for talk.sh)
#   hq    → 1.7B CustomVoice 8bit on :18882  (higher quality, more latency/RAM)
#
# Separate processes (not one multi-model server) are intentional: each model
# has a large Metal working set, and isolation prevents one instance from
# taking down the other. Each instance boots with TTS_WARMUP_ON_START=true to
# absorb cold graph compilation, and TTS_IDLE_TIMEOUT_SECONDS=0 so it stays up
# indefinitely.
#
# Only CustomVoice checkpoints work (the backend calls generate_custom_voice).
# Base models load but cannot synthesize the preset voices — do not use them.
#
# Default is hq-only (1.7B): a bare `./run_tts.sh` starts just the 1.7B. Run
# `both` if you also want the 0.6B fast instance (watch memory — running both
# alongside other local models can trigger an OOM SIGTERM on the 1.7B).
#
# Usage:
#   ./run_tts.sh            # default: start 1.7B (hq) only
#   ./run_tts.sh fast       # start 0.6B on :18881
#   ./run_tts.sh hq         # start 1.7B on :18882
#   ./run_tts.sh both       # start both
#   ./run_tts.sh status      # show what is listening
#   ./run_tts.sh stop        # stop both managed instances
#   MODEL=<hf-id> PORT=<n> ./run_tts.sh custom   # arbitrary CustomVoice model
set -euo pipefail
cd "$(dirname "$0")"

VENV="${VENV:-.venv}"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

FAST_MODEL="mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"
HQ_MODEL="mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
FAST_PORT=18881
HQ_PORT=18882

_listening() { lsof -nP -iTCP:"$1" -sTCP:LISTEN -t >/dev/null 2>&1; }

start_one() {
    local model="$1" port="$2" name="$3"
    if _listening "$port"; then
        echo "[run_tts] $name already listening on :$port (leaving it)"
        return 0
    fi
    echo "[run_tts] starting $name → $model on :$port"
    HOST=0.0.0.0 PORT="$port" TTS_BACKEND=mlx MLX_MODEL_ID="$model" \
        TTS_IDLE_TIMEOUT_SECONDS=0 TTS_LAZY_LOAD=false TTS_WARMUP_ON_START=true \
        nohup python -m uvicorn api.main:app --host 0.0.0.0 --port "$port" \
        > "server_${port}.log" 2>&1 &
    disown
    echo "[run_tts] $name pid $! → log: server_${port}.log (warming up…)"
}

stop_one() {
    local port="$1" name="$2" pid
    pid=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)
    if [ -n "$pid" ]; then
        kill "$pid" 2>/dev/null || true
        echo "[run_tts] stopped $name (pid $pid) on :$port"
    else
        echo "[run_tts] $name not running on :$port"
    fi
}

status_one() {
    local port="$1" name="$2"
    if _listening "$port"; then
        local h
        h=$(curl -sf -m 2 "http://127.0.0.1:$port/health" 2>/dev/null || true)
        if [ -n "$h" ]; then
            python - "$name" "$port" "$h" <<'PY'
import json, sys
name, port, raw = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.loads(raw)
    b = d.get("backend", {})
    print(f"  {name:9s} :{port}  {d.get('status','?'):12s} ready={b.get('ready')}  {b.get('model_id','?')}")
except Exception:
    print(f"  {name:9s} :{port}  (health unparseable)")
PY
        else
            echo "  $name :$port  LISTENING (no /health yet — still warming up?)"
        fi
    else
        echo "  $name :$port  not running"
    fi
}

case "${1:-hq}" in
    fast|0.6b)  start_one "$FAST_MODEL" "${PORT:-$FAST_PORT}" "fast-0.6B" ;;
    hq|1.7b)    start_one "$HQ_MODEL"   "${PORT:-$HQ_PORT}"   "hq-1.7B" ;;
    both)       start_one "$FAST_MODEL" "$FAST_PORT" "fast-0.6B"
                start_one "$HQ_MODEL"   "$HQ_PORT"   "hq-1.7B" ;;
    custom)     start_one "${MODEL:?set MODEL=<hf-id>}" "${PORT:?set PORT=<n>}" "custom" ;;
    stop)       stop_one "$FAST_PORT" "fast-0.6B"; stop_one "$HQ_PORT" "hq-1.7B" ;;
    status)     echo "[run_tts] TTS instances:"; status_one "$FAST_PORT" "fast-0.6B"; status_one "$HQ_PORT" "hq-1.7B" ;;
    *) echo "usage: $0 {fast|hq|both|status|stop|custom}" >&2; exit 2 ;;
esac
