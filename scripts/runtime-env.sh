#!/usr/bin/env bash
# Shared by the native installer and launcher. Prefixes may point to existing envs.
set -euo pipefail
TTS_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${CONDA_EXE:-}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_EXE="$(command -v conda)"
    else
        for candidate in "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do
            if [[ -x "$candidate" ]]; then CONDA_EXE="$candidate"; break; fi
        done
    fi
fi
if [[ -z "${TTS_FRONTEND_ENV:-}" || -z "${TTS_GRAPH_ENV:-}" ]]; then
    if [[ -z "${CONDA_EXE:-}" ]]; then
        echo 'Conda not found. Install Miniforge or set CONDA_EXE to your existing conda executable.' >&2
        exit 1
    fi
    tts_conda_base="$("$CONDA_EXE" info --base)"
    TTS_FRONTEND_ENV="${TTS_FRONTEND_ENV:-$tts_conda_base/envs/qwen3-tts}"
    TTS_GRAPH_ENV="${TTS_GRAPH_ENV:-$tts_conda_base/envs/qwen3-tts-graphs}"
fi
TTS_FRONTEND_ENV="$(realpath -m "$TTS_FRONTEND_ENV")"
TTS_GRAPH_ENV="$(realpath -m "$TTS_GRAPH_ENV")"
if [[ "$TTS_FRONTEND_ENV" == "$TTS_GRAPH_ENV" ]]; then
    echo 'The API and graph worker require different environments (conflicting qwen_tts packages).' >&2
    exit 1
fi
export TTS_REPO_DIR TTS_FRONTEND_ENV TTS_GRAPH_ENV
