# Default NVIDIA runtime: 1.7B CustomVoice with CUDA graphs

The default installation uses [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts/tree/b35fdc7d71eb3e4d19e63f4b4a330a1aef31d682) with PyTorch CUDA graphs. TensorRT is not required. The OpenAI-compatible API, nine native speakers, instruction input, encoded audio, and incremental PCM streaming remain available.

## Install and run

Requirements: Linux x86_64, NVIDIA Ampere or newer, a driver compatible with CUDA 12.8 wheels, existing Conda/Miniforge, and Git/FFmpeg/libsndfile/sox. Tested with an RTX 3060 12 GB. Available VRAM depends on other processes. Allow at least 45 GB free for two environments, weights, and pip caches; allow 60 GB or more for Docker build caches and image layers.

```bash
# Debian/Ubuntu system dependencies:
sudo apt-get install git ffmpeg libsndfile1 sox
git clone https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi.git
cd Qwen3-TTS-Openai-Fastapi
./scripts/install.sh
./start_server.sh
```

No Conda activation is needed. The scripts discover `CONDA_EXE`, `conda` on PATH, or standard Miniforge/Miniconda/Anaconda locations. They reuse `<conda-base>/envs/qwen3-tts` and `<conda-base>/envs/qwen3-tts-graphs`, creating missing environments with Python 3.12. Reused environments must also use 3.12. Rerunning the installer updates their dependencies to the tested pins; it never deletes environments. If those names belong to other projects, set two different absolute prefixes:

```bash
export CONDA_EXE="$HOME/miniforge3/bin/conda"
export TTS_FRONTEND_ENV="$HOME/miniforge3/envs/my-qwen-api"
export TTS_GRAPH_ENV="$HOME/miniforge3/envs/my-qwen-worker"
./scripts/install.sh
CUDA_VISIBLE_DEVICES=0 HOST=0.0.0.0 PORT=8880 ./start_server.sh
```

Retain those exported settings for subsequent launches and service units. `.env.example` lists settings; for native launches load a local copy explicitly with `set -a; source .env; set +a`. Compose reads `.env` itself.

## Isolation and lifecycle

| Process | Package stack | GPU access |
|---|---|---|
| Public API | This repo's `qwen-tts`, Transformers 4.57.3 | Hidden; HTTP and encoding only |
| Private worker | `faster-qwen3-tts` at `b35fdc7…`, `qwen-tts-hf` 0.1.1.post1, Transformers 5.15.1 | One assigned GPU |

Both environments pin PyTorch/torchaudio 2.9.1. See `requirements/frontend-constraints.txt` and `requirements/cuda-graphs.txt`. Frontend transitive dependencies are resolved by pip; these are tested compatibility pins, not a complete environment lock. The API package and `qwen-tts-hf` both provide `qwen_tts`, so combining them breaks imports. A worker launched from the repository root can also shadow its installed package. The launcher strips worker `PYTHONPATH` and uses a neutral working directory.

The worker uses BF16, SDPA, a 4096-token graph allocation, at most 2048 new tokens per chunk, and 8-codec-frame streaming chunks. It preserves `non_streaming_mode=False` for synthesis. Under the launcher, the API uses punctuation-aware 200–600-character chunks to reduce repeated setup overhead compared with the old 70-character limit. Generation stays single-concurrency. `CUDA_VISIBLE_DEVICES` is passed through to the worker; logical `cuda:0` refers to that assigned card. The API has CUDA hidden.

`start_server.sh` starts the worker on `127.0.0.1:18881`, waits for graph capture and warmup of both generation paths, then starts the API on `0.0.0.0:8880`. A busy worker port causes an error; it never attaches to or kills an unrelated process. Ctrl+C/SIGTERM stops both children. If either fails, the other is stopped. Set `TTS_GRAPH_PORT` for another instance and retain `WORKERS=1`, `TTS_MAX_CONCURRENT=1`.

Direct `python -m api.main` starts only the API. Advanced deployments can manage the worker independently using its environment's Python, an absolute `scripts/qwen_graph_worker.py` path, and `/tmp` as working directory. Configure the API with `TTS_BACKEND=cuda_graphs` and `TTS_GRAPH_WORKER_URL`. Keep the worker on loopback.

## Models, cache, and offline operation

The default checkpoint is `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`. Set `TTS_MODEL_ID` to a compatible CustomVoice HF ID or an absolute local directory. `TTS_MODEL_NAME`, if set, takes precedence in both processes. Base cloning requires an explicit alternative such as `official`; this graph adapter does not implement VoiceDesign.

The installer downloads weights without using a GPU. `SKIP_MODEL_DOWNLOAD=1` defers downloading until launch. Use the same `HF_HOME` at installation and startup. Optionally set `TTS_MODEL_REVISION` to a Hugging Face commit to reproduce weights:

```bash
export HF_HOME="$HOME/.cache/huggingface"
# export TTS_MODEL_REVISION=<full-model-commit>
./scripts/install.sh
HF_HUB_OFFLINE=1 ./start_server.sh
```

Local checkpoint directories are used directly. Startup still captures graphs on the selected GPU. Increase `TTS_STARTUP_TIMEOUT` above its 900-second default for slow downloads. There is no FlashAttention compilation or TensorRT engine build.

## Docker

Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host. The image supplies the CUDA runtime through PyTorch's wheels; a host CUDA toolkit is unnecessary.

```bash
docker compose up --build -d
docker compose logs -f qwen3-tts-gpu
curl --fail http://127.0.0.1:8880/health
docker compose down
```

The default/final target is `production`: the same installer and launcher in two Conda environments, running as an unprivileged user. Compose reserves one GPU, publishes only the API port, and persists the HF cache and voice-library data in named volumes. Weights download on first start and are not baked into the image. Allow several minutes; the healthcheck startup grace is 900 seconds. `TTS_PORT` changes the published host port; `TTS_MODEL_NAME` and `TTS_MODEL_REVISION` select weights. To run offline, first populate the cache volume online, then add `HF_HUB_OFFLINE=1` to the service environment.

The explicit `cpu-base` target/`cpu` profile and the separate vLLM/ROCm Dockerfiles remain alternatives. For a specific GPU, use `docker run --gpus '"device=0"' ...` as shown in the README, or replace Compose's GPU `count: 1` with `device_ids: ["0"]`.

## systemd

Use one service around the same supervisor. Replace all absolute paths with your own:

```ini
[Unit]
Description=Qwen3-TTS CustomVoice API and CUDA-graph worker
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/absolute/path/Qwen3-TTS-Openai-Fastapi
Environment=TTS_FRONTEND_ENV=/absolute/conda/envs/qwen3-tts
Environment=TTS_GRAPH_ENV=/absolute/conda/envs/qwen3-tts-graphs
Environment=CUDA_VISIBLE_DEVICES=0
Environment=HOST=0.0.0.0
Environment=PORT=8880
ExecStart=/absolute/path/Qwen3-TTS-Openai-Fastapi/start_server.sh
Restart=on-failure
RestartSec=5
TimeoutStopSec=60
KillMode=control-group

[Install]
WantedBy=default.target
```

Save under `~/.config/systemd/user/qwen3-tts.service`, run `systemctl --user daemon-reload`, then `systemctl --user enable --now qwen3-tts.service`. Inspect `journalctl --user -u qwen3-tts.service -f`. Configure user lingering separately if it must survive logout.

## Verify and benchmark

```bash
curl --fail http://127.0.0.1:8880/health
curl --fail http://127.0.0.1:8880/v1/voices
curl --fail http://127.0.0.1:8880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","voice":"Ryan","input":"The speech service is ready.","response_format":"wav"}' \
  --output speech.wav
python scripts/benchmark_api.py --requests 4 --mode wav --output wav-benchmark.json
python scripts/benchmark_api.py --requests 4 --mode pcm --output pcm-benchmark.json
```

Use `--base-url http://host:port/v1` to include a proxy, and `--model your-proxy-model-id` when the gateway requires an explicit model. The script excludes one warmup request, records wall time and generated audio duration per request, and reports median RTF = wall seconds / audio seconds. Below 1 means faster than real time. PCM mode records first-audio latency. Disable buffering in any reverse proxy to preserve streaming.

Measured on 2026-11-17 with the same RTX 3060 12 GB, 1.7B CustomVoice checkpoint, BF16 and SDPA:

| Measurement | Previous official runtime | CUDA graphs |
|---|---:|---:|
| Warm WAV median RTF, 4 samples each | 1.646 | 0.563 |
| Ratio of median RTFs | 1× | 2.92× faster |
| PCM first audio through a proxy, 3 samples | Buffered until completion | 0.460 s median |

First-audio samples: 0.460, 1.491, 0.429 seconds. These results include the deployment's revised chunk sizes: they compare complete configurations, not isolated kernels. Generation is stochastic; results vary with hardware and text. Validation covered nonempty finite waveforms, sample rate, native voices, disconnect/recovery, and stop/restart. Listening quality and transcription accuracy were not evaluated.

## Troubleshooting and fallback

- Missing streaming methods or incompatible Transformers: confirm distinct environments, rerun the installer, and use `start_server.sh`. Never install this repository into the worker environment.
- CUDA unavailable: check `nvidia-smi`, `CUDA_VISIBLE_DEVICES`, driver compatibility, and the NVIDIA container runtime. Use the README's CPU/MLX/ROCm paths on other hardware.
- Out of VRAM: inspect other GPU users, retain one worker/generation, or select a compatible smaller CustomVoice model. The launcher does not preempt other jobs.
- Busy worker port: `TTS_GRAPH_PORT=18883 ./start_server.sh`.
- Streaming arrives at completion: use `stream=true`, `response_format=pcm`, and disable proxy buffering. WAV/compressed output remains buffered for encoding.
- Official fallback using the existing API environment: `TTS_BACKEND=official ./start_server.sh`. No graph worker launches. Select a compatible model for Base cloning or other backend-specific features.
