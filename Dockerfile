# syntax=docker/dockerfile:1
# Explicit CPU fallback. The final/default target below is CUDA graphs.
FROM python:3.12-slim-bookworm AS cpu-base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/numba_cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 sox curl libgomp1 tini \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md gradio_voice_studio.py ./
COPY api ./api
COPY qwen_tts ./qwen_tts
COPY requirements ./requirements
COPY scripts/install.sh scripts/runtime-env.sh ./scripts/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -c requirements/frontend-constraints.txt -e ".[api]" && pip check
COPY . .
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser
ENV HOST=0.0.0.0 PORT=8880 WORKERS=1 TTS_BACKEND=pytorch TTS_DEVICE=cpu TTS_DTYPE=float32 TTS_ATTN=sdpa
EXPOSE 8880
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8880/health || exit 1
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "api.main"]

# Pip's cu128 wheels supply CUDA/cuDNN; no flash-attn compilation or CUDA toolkit needed.
FROM condaforge/miniforge3:25.3.0-3 AS production
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR=/tmp/numba_cache \
    TTS_FRONTEND_ENV=/opt/conda/envs/qwen3-tts \
    TTS_GRAPH_ENV=/opt/conda/envs/qwen3-tts-graphs \
    SKIP_MODEL_DOWNLOAD=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsndfile1 sox curl libgomp1 tini \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md gradio_voice_studio.py ./
COPY api ./api
COPY qwen_tts ./qwen_tts
COPY requirements ./requirements
COPY scripts/install.sh scripts/runtime-env.sh ./scripts/
RUN --mount=type=cache,target=/root/.cache/pip \
    bash scripts/install.sh && conda clean --all --yes
COPY . .
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /home/appuser/.cache/huggingface /app/voice_library /tmp/numba_cache \
    && chown -R appuser:appuser /app /home/appuser /tmp/numba_cache
USER appuser
ENV HOST=0.0.0.0 PORT=8880 WORKERS=1 TTS_BACKEND=cuda_graphs \
    HF_HOME=/home/appuser/.cache/huggingface \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
EXPOSE 8880
# Graph capture and a first model download can take several minutes.
HEALTHCHECK --interval=30s --timeout=10s --start-period=900s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8880/health || exit 1
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "/app/start_server.sh"]
