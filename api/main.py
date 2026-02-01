# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-TTS OpenAI-Compatible FastAPI Server.

A high-performance TTS API server providing OpenAI-compatible endpoints
for the Qwen3-TTS model.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    gr = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8880"))
WORKERS = int(os.getenv("WORKERS", "1"))

# Backend configuration
TTS_BACKEND = os.getenv("TTS_BACKEND", "official")
TTS_WARMUP_ON_START = os.getenv("TTS_WARMUP_ON_START", "false").lower() == "true"

# CORS configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Voice Studio configuration
ENABLE_VOICE_STUDIO = os.getenv("ENABLE_VOICE_STUDIO", "false").lower() == "true"
VOICE_LIBRARY_DIR = Path(os.getenv("VOICE_LIBRARY_DIR", "./voice_library")).resolve()

# Get the directory containing static files
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for model initialization."""
    
    # Print startup banner
    boundary = "░" * 24
    startup_msg = f"""
{boundary}

    ╔═╗┬ ┬┌─┐┌┐┌╔═╗  ╔╦╗╔╦╗╔═╗
    ║═╬╡│││├┤ │││╚═╗───║  ║ ╚═╗
    ╚═╝└┴┘└─┘┘└┘╚═╝   ╩  ╩ ╚═╝
    
    OpenAI-Compatible TTS API
    Backend: {TTS_BACKEND}

{boundary}
"""
    logger.info(startup_msg)
    # Show localhost in logs for user-friendly access URL (server binds to 0.0.0.0)
    display_host = "localhost" if HOST == "0.0.0.0" else HOST
    logger.info(f"Server starting on http://{display_host}:{PORT}")
    logger.info(f"API Documentation: http://{display_host}:{PORT}/docs")
    logger.info(f"Web Interface: http://{display_host}:{PORT}/")
    if ENABLE_VOICE_STUDIO:
        logger.info(f"Voice Studio: http://{display_host}:{PORT}/voice-studio")
    logger.info(boundary)
    
    # Pre-load the TTS backend
    try:
        from .backends import initialize_backend
        logger.info(f"Initializing TTS backend: {TTS_BACKEND}")
        backend = await initialize_backend(warmup=TTS_WARMUP_ON_START)
        logger.info(f"TTS backend '{backend.get_backend_name()}' loaded successfully!")
        logger.info(f"Model: {backend.get_model_id()}")
        
        device_info = backend.get_device_info()
        if device_info.get("gpu_available"):
            logger.info(f"GPU: {device_info.get('gpu_name')}")
            logger.info(f"VRAM: {device_info.get('vram_total')}")
    except Exception as e:
        logger.warning(f"Backend initialization delayed: {e}")
        logger.info("Backend will be loaded on first request.")
    
    yield
    
    # Cleanup
    logger.info("Server shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Qwen3-TTS API",
    description="""
## Qwen3-TTS OpenAI-Compatible API

A high-performance text-to-speech API server powered by Qwen3-TTS, 
providing full compatibility with OpenAI's TTS API specification.

### Features
- 🎯 OpenAI API compatible endpoints
- 🌍 Multi-language support (10+ languages)
- 🎨 Multiple voice options
- 📊 Multiple audio formats (MP3, Opus, AAC, FLAC, WAV, PCM)
- ⚡ GPU-accelerated inference
- 🔧 Text normalization and sanitization

### Quick Start
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed")

response = client.audio.speech.create(
    model="qwen3-tts",
    voice="Vivian",
    input="Hello! This is Qwen3-TTS speaking."
)
response.stream_to_file("output.mp3")
```
""",
    version="0.1.0",
    lifespan=lifespan,
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from .routers.openai_compatible import router as openai_router
app.include_router(openai_router, prefix="/v1")

# Mount static files if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount Voice Studio if enabled
if ENABLE_VOICE_STUDIO:
    if not GRADIO_AVAILABLE:
        logger.warning("Voice Studio enabled but gradio is not installed. Install with: pip install gradio")
    else:
        try:
            # Import gradio_voice_studio from parent directory
            parent_dir = Path(__file__).parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))
            
            from gradio_voice_studio import build_app
            
            # Build the Voice Studio app with the current server URL
            # Use localhost when server is bound to 0.0.0.0, otherwise use the actual host
            voice_studio_host = "localhost" if HOST == "0.0.0.0" else HOST
            base_url = f"http://{voice_studio_host}:{PORT}"
            voice_studio_app = build_app(base_url, VOICE_LIBRARY_DIR)
            
            # Mount the Gradio app
            app = gr.mount_gradio_app(app, voice_studio_app, path="/voice-studio")
            logger.info(f"Voice Studio mounted at /voice-studio")
        except Exception as e:
            logger.warning(f"Failed to mount Voice Studio: {e}")
            logger.info("Voice Studio can still be run separately with 'qwen-tts-voice-studio'")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    
    # Build links dynamically
    voice_studio_link = ""
    if ENABLE_VOICE_STUDIO and GRADIO_AVAILABLE:
        voice_studio_link = '<li><a href="/voice-studio">🎙️ Voice Studio</a></li>'
    
    # Return a simple HTML page if index.html doesn't exist
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Qwen3-TTS API</title>
    <style>
        body {{ 
            font-family: 'Courier New', monospace; 
            background: #1a1a2e; 
            color: #eee; 
            padding: 40px;
            max-width: 800px;
            margin: 0 auto;
        }}
        pre {{ color: #00ff88; }}
        a {{ color: #00aaff; }}
        h1 {{ color: #fff; }}
    </style>
</head>
<body>
    <pre>
    ╔═╗┬ ┬┌─┐┌┐┌╔═╗  ╔╦╗╔╦╗╔═╗
    ║═╬╡│││├┤ │││╚═╗───║  ║ ╚═╗
    ╚═╝└┴┘└─┘┘└┘╚═╝   ╩  ╩ ╚═╝
    </pre>
    <h1>Qwen3-TTS OpenAI-Compatible API</h1>
    <p>Welcome to the Qwen3-TTS API server!</p>
    <ul>
        <li><a href="/docs">API Documentation (Swagger UI)</a></li>
        <li><a href="/redoc">API Documentation (ReDoc)</a></li>
        <li><a href="/v1/models">List Models</a></li>
        <li><a href="/v1/voices">List Voices</a></li>
        {voice_studio_link}
    </ul>
</body>
</html>
"""


@app.get("/health")
async def health_check():
    """Health check endpoint with backend information."""
    try:
        from .backends import get_backend
        
        backend = get_backend()
        device_info = backend.get_device_info()
        
        return {
            "status": "healthy" if backend.is_ready() else "initializing",
            "backend": {
                "name": backend.get_backend_name(),
                "model_id": backend.get_model_id(),
                "ready": backend.is_ready(),
            },
            "device": {
                "type": device_info.get("device"),
                "gpu_available": device_info.get("gpu_available"),
                "gpu_name": device_info.get("gpu_name"),
                "vram_total": device_info.get("vram_total"),
                "vram_used": device_info.get("vram_used"),
            },
            "version": "0.1.0",
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "backend": {
                "name": TTS_BACKEND,
                "ready": False,
            },
            "version": "0.1.0",
        }


def main():
    """Run the server using uvicorn."""
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
