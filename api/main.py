# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS OpenAI-compatible FastAPI server."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    import gradio as gr

    GRADIO_AVAILABLE = True
except ImportError:
    gr = None
    GRADIO_AVAILABLE = False

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

API_VERSION = "0.1.1"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid %s=%r; using %s", name, raw, default)
    return default


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, raw, default)
        return default
    if value < minimum:
        logger.warning("%s must be >= %d; using %d", name, minimum, default)
        return default
    return value


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "*")
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or ["*"]


HOST = os.getenv("HOST", "0.0.0.0").strip() or "0.0.0.0"
PORT = _env_int("PORT", 8880, minimum=1)
WORKERS = _env_int("WORKERS", 1, minimum=1)
TTS_BACKEND = os.getenv("TTS_BACKEND", "cuda_graphs").strip().lower() or "cuda_graphs"
TTS_WARMUP_ON_START = _env_bool("TTS_WARMUP_ON_START", False)
GPU_KEEPALIVE_INTERVAL = _env_int("GPU_KEEPALIVE_INTERVAL", 0, minimum=0)
TTS_LAZY_LOAD = _env_bool("TTS_LAZY_LOAD", True)
# Production servers must not unexpectedly terminate after being idle. Opt in
# explicitly on workstation deployments that benefit from releasing RAM/VRAM.
TTS_IDLE_TIMEOUT_SECONDS = _env_int("TTS_IDLE_TIMEOUT_SECONDS", 0, minimum=0)
CORS_ORIGINS = _cors_origins()
ENABLE_VOICE_STUDIO = _env_bool("ENABLE_VOICE_STUDIO", False)
VOICE_LIBRARY_DIR = Path(os.getenv("VOICE_LIBRARY_DIR", "./voice_library")).resolve()
STATIC_DIR = Path(__file__).parent / "static"


async def _cancel_task(task: asyncio.Task | None) -> None:
    if task is None:
        return
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    display_host = "localhost" if HOST in {"0.0.0.0", "::"} else HOST
    logger.info("Qwen3-TTS API %s starting on http://%s:%d", API_VERSION, display_host, PORT)
    logger.info("Backend=%s lazy_load=%s workers=%d", TTS_BACKEND, TTS_LAZY_LOAD, WORKERS)
    if WORKERS > 1:
        logger.warning(
            "WORKERS=%d loads one model per process and can exhaust VRAM; "
            "single-GPU deployments should normally use WORKERS=1",
            WORKERS,
        )

    if not TTS_LAZY_LOAD:
        from .backends import initialize_backend

        backend = await initialize_backend(warmup=TTS_WARMUP_ON_START)
        logger.info(
            "Backend ready: %s (%s)",
            backend.get_backend_name(),
            backend.get_model_id(),
        )
    else:
        logger.info("Backend will initialize on the first synthesis request")

    keepalive_task: asyncio.Task | None = None
    if GPU_KEEPALIVE_INTERVAL > 0:
        try:
            import torch

            if torch.cuda.is_available():
                keepalive_tensor = torch.randn(512, 512, device="cuda:0")

                async def _gpu_keepalive() -> None:
                    while True:
                        await asyncio.sleep(GPU_KEEPALIVE_INTERVAL)
                        with torch.inference_mode():
                            torch.matmul(keepalive_tensor, keepalive_tensor)

                keepalive_task = asyncio.create_task(
                    _gpu_keepalive(), name="gpu-keepalive"
                )
                logger.info("GPU keepalive enabled every %ds", GPU_KEEPALIVE_INTERVAL)
        except Exception as exc:
            logger.warning("GPU keepalive setup failed: %s", exc)

    idle_shutdown_task: asyncio.Task | None = None
    app.state.last_speech_at = time.monotonic()
    app.state.idle_timeout_seconds = TTS_IDLE_TIMEOUT_SECONDS
    app.state.speech_request_count = 0
    app.state.speech_total_samples = 0

    if TTS_IDLE_TIMEOUT_SECONDS > 0:

        async def _idle_watchdog() -> None:
            while True:
                since_last = time.monotonic() - app.state.last_speech_at
                if (
                    app.state.speech_request_count > 0
                    and since_last >= app.state.idle_timeout_seconds
                ):
                    logger.info(
                        "No successful speech request for %.0fs; shutting down",
                        since_last,
                    )
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
                await asyncio.sleep(1 if since_last < 60 else 30)

        idle_shutdown_task = asyncio.create_task(
            _idle_watchdog(), name="idle-shutdown"
        )
        logger.info("Idle shutdown enabled after %ds", TTS_IDLE_TIMEOUT_SECONDS)

    try:
        yield
    finally:
        await _cancel_task(keepalive_task)
        await _cancel_task(idle_shutdown_task)
        logger.info("Server shutdown complete")


app = FastAPI(
    title="Qwen3-TTS API",
    description=(
        "OpenAI-compatible text-to-speech API powered by Qwen3-TTS. "
        "Use POST /v1/audio/speech and discover models/voices under /v1."
    ),
    version=API_VERSION,
    lifespan=lifespan,
    openapi_url="/openapi.json",
)

# Browsers reject credentialed wildcard CORS. Keep wildcard convenient for
# local use, but only enable credentials when explicit origins are configured.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials="*" not in CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

from .routers.openai_compatible import router as openai_router

app.include_router(openai_router, prefix="/v1")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if ENABLE_VOICE_STUDIO:
    if not GRADIO_AVAILABLE:
        logger.warning(
            "ENABLE_VOICE_STUDIO=true but Gradio is unavailable; install project dependencies"
        )
    else:
        try:
            parent_dir = Path(__file__).parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))
            from gradio_voice_studio import build_app

            voice_studio_host = "localhost" if HOST in {"0.0.0.0", "::"} else HOST
            studio = build_app(
                f"http://{voice_studio_host}:{PORT}", VOICE_LIBRARY_DIR
            )
            app = gr.mount_gradio_app(app, studio, path="/voice-studio")
            logger.info("Voice Studio mounted at /voice-studio")
        except Exception as exc:
            logger.warning("Failed to mount Voice Studio: %s", exc)


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)

    studio_link = (
        '<li><a href="/voice-studio">Voice Studio</a></li>'
        if ENABLE_VOICE_STUDIO and GRADIO_AVAILABLE
        else ""
    )
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen3-TTS API</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 760px; margin: 4rem auto; padding: 0 1rem; }}
    code {{ background: #eee; padding: .15rem .35rem; border-radius: .25rem; }}
  </style>
</head>
<body>
  <h1>Qwen3-TTS OpenAI-compatible API</h1>
  <p>Server version {API_VERSION}. Use <code>POST /v1/audio/speech</code>.</p>
  <ul>
    <li><a href="/docs">Swagger API documentation</a></li>
    <li><a href="/redoc">ReDoc API documentation</a></li>
    <li><a href="/v1/models">Models</a></li>
    <li><a href="/v1/voices">Voices</a></li>
    {studio_link}
  </ul>
</body>
</html>
"""


@app.get("/health")
async def health_check():
    try:
        from .backends import get_backend

        backend = get_backend()
        device_info = backend.get_device_info()
        ready = backend.is_ready()
        return {
            "status": "healthy" if ready else "initializing",
            "backend": {
                "name": backend.get_backend_name(),
                "model_id": backend.get_model_id(),
                "ready": ready,
            },
            "device": {
                "type": device_info.get("device"),
                "gpu_available": device_info.get("gpu_available"),
                "gpu_name": device_info.get("gpu_name"),
                "vram_total": device_info.get("vram_total"),
                "vram_used": device_info.get("vram_used"),
            },
            "version": API_VERSION,
        }
    except Exception as exc:
        logger.exception("Health check failed")
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error": str(exc),
                "backend": {"name": TTS_BACKEND, "ready": False},
                "version": API_VERSION,
            },
        )


def main() -> None:
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=HOST,
        port=PORT,
        workers=WORKERS,
        reload=False,
    )


if __name__ == "__main__":
    main()
