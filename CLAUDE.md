# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenAI-compatible FastAPI server for Qwen3-TTS text-to-speech. Drop-in replacement for OpenAI's TTS API (`/v1/audio/speech`). Supports two backends: the official Qwen3-TTS model and vLLM-Omni for faster inference.

## Common Commands

```bash
# Install dependencies
pip install -e ".[api,dev]"          # API + dev/test deps
pip install -e ".[api,vllm,dev]"     # Include vLLM backend

# Run the server
python -m api.main                   # Starts on port 8880
TTS_BACKEND=vllm_omni python -m api.main  # Use vLLM backend

# Run tests
pytest                               # All tests (verbose, short traceback by default)
pytest tests/test_api.py             # API endpoint tests only
pytest tests/test_backends.py        # Backend tests only
pytest tests/test_api.py::TestHealthEndpoint::test_health_endpoint  # Single test

# Docker
docker-compose up qwen3-tts-gpu                     # Official backend
docker-compose --profile vllm up qwen3-tts-vllm     # vLLM backend
docker-compose --profile cpu up qwen3-tts-cpu        # CPU-only
```

## Architecture

### Backend System (Strategy + Factory Pattern)

The core abstraction is `TTSBackend` (ABC in `api/backends/base.py`). Two implementations exist:

- **OfficialQwen3TTSBackend** (`api/backends/official_qwen3_tts.py`) - Uses the `qwen_tts` package directly with GPU optimizations (Flash Attention 2, torch.compile, TF32, BFloat16)
- **VllmOmniQwen3TTSBackend** (`api/backends/vllm_omni_qwen3_tts.py`) - Uses vLLM-Omni for inference

`api/backends/factory.py` manages a singleton backend instance. Selection is via the `TTS_BACKEND` env var (`"official"` or `"vllm_omni"`). The backend is initialized during FastAPI's lifespan startup in `api/main.py`.

### API Layer

`api/routers/openai_compatible.py` contains all endpoints:
- `POST /v1/audio/speech` - Main TTS endpoint
- `POST /v1/audio/voice-clone` - Voice cloning (requires Base model)
- `GET /v1/models`, `GET /v1/audio/voices` - Discovery endpoints

OpenAI voice names (alloy, echo, fable, nova, onyx, shimmer) are mapped to Qwen voices (Vivian, Ryan, Sophia, Isabella, Evan, Lily). Model aliases `tts-1` and `tts-1-hd` map to `qwen3-tts`.

### Services

- `api/services/text_processing.py` - Text normalization (URLs, emails, units, money, phone numbers, symbols) with configurable `NormalizationOptions`
- `api/services/audio_encoding.py` - Converts numpy audio arrays to MP3, Opus, AAC, FLAC, WAV, or PCM format

### Request/Response Models

Pydantic schemas are in `api/structures/schemas.py`. The main request type is `OpenAISpeechRequest` which extends the OpenAI spec with `language`, `instruct` (voice style), and `normalization_options` fields.

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `TTS_BACKEND` | `official` | Backend: `official` or `vllm_omni` |
| `TTS_WARMUP_ON_START` | `false` | Run warmup inference on startup |
| `TTS_MODEL_NAME` | (auto) | Override model name/path |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8880` | Server port |
| `WORKERS` | `1` | Uvicorn workers |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |

## Testing

Tests use `pytest` with `pytest-asyncio` (async mode: auto). Test client uses `httpx` via FastAPI's `TestClient`. The `reset_backend_after_test` fixture in `test_backends.py` calls `reset_backend()` to clear the singleton between tests.

## Adding a New Backend

1. Subclass `TTSBackend` from `api/backends/base.py`
2. Implement all abstract methods (`initialize`, `generate_speech`, `get_backend_name`, etc.)
3. Optionally override `supports_voice_cloning()` and `generate_voice_clone()`
4. Register in `api/backends/factory.py`
