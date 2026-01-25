#!/usr/bin/env python3
"""
Persistent voice cloning TTS server using Qwen3-TTS Base model.
Keeps model loaded in memory for fast inference.
"""

import io
import os
import re
import struct
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import torch
import soundfile as sf
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment
MODEL_PATH = os.environ.get("MODEL_PATH", "./models/Qwen3-TTS-12Hz-1.7B-Base")
REF_AUDIO = os.environ.get("REF_AUDIO", "./voice_refs/hai_reference.wav")
REF_TEXT = os.environ.get("REF_TEXT", "Yeah so basically I checked the system logs and found a couple of errors. Nothing critical, but you should probably take a look when you get a chance. The server has been running fine otherwise.")
PORT = int(os.environ.get("PORT", "8881"))

# Global state
model = None
voice_prompt = None


def load_model():
    """Load the voice cloning model."""
    global model, voice_prompt

    from qwen_tts import Qwen3TTSModel

    logger.info(f"Loading model from {MODEL_PATH}...")

    # Try Flash Attention 2, fall back to eager
    try:
        model = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        logger.info("Model loaded with Flash Attention 2.")
    except Exception as e:
        logger.warning(f"Flash Attention 2 not available: {e}, falling back to eager")
        model = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        logger.info("Model loaded with eager attention.")

    logger.info("Creating voice clone prompt...")
    voice_prompt = model.create_voice_clone_prompt(
        ref_audio=REF_AUDIO,
        ref_text=REF_TEXT,
        x_vector_only_mode=False,
    )
    logger.info("Voice clone prompt ready.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    load_model()
    yield
    # Cleanup on shutdown
    global model, voice_prompt
    model = None
    voice_prompt = None
    torch.cuda.empty_cache()


app = FastAPI(
    title="Voice Clone TTS Server",
    description="Qwen3-TTS voice cloning with persistent model",
    lifespan=lifespan,
)


class TTSRequest(BaseModel):
    text: str
    language: str = "English"
    stream: bool = False
    response_format: str = "wav"  # wav or pcm


def create_wav_streaming_header(
    sample_rate: int = 24000,
    num_channels: int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    """Create WAV header for streaming with unknown final size."""
    buffer = io.BytesIO()
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    buffer.write(b'RIFF')
    buffer.write(struct.pack('<I', 0xFFFFFFFF))  # Placeholder
    buffer.write(b'WAVE')
    buffer.write(b'fmt ')
    buffer.write(struct.pack('<I', 16))
    buffer.write(struct.pack('<H', 1))  # PCM
    buffer.write(struct.pack('<H', num_channels))
    buffer.write(struct.pack('<I', sample_rate))
    buffer.write(struct.pack('<I', byte_rate))
    buffer.write(struct.pack('<H', block_align))
    buffer.write(struct.pack('<H', bits_per_sample))
    buffer.write(b'data')
    buffer.write(struct.pack('<I', 0xFFFFFFFF))  # Placeholder
    return buffer.getvalue()


def convert_to_pcm(audio: np.ndarray) -> bytes:
    """Convert float audio to 16-bit PCM bytes."""
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    return pcm.tobytes()


async def generate_speech_streaming(
    text: str,
    language: str,
) -> AsyncGenerator[tuple[np.ndarray, int], None]:
    """Generate speech with sentence-level streaming."""
    global model, voice_prompt

    # Split text into sentences
    sentence_pattern = r'(?<=[.!?。！？])\s+'
    sentences = re.split(sentence_pattern, text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        sentences = [text]

    for sentence in sentences:
        try:
            wavs, sr = model.generate_voice_clone(
                text=sentence,
                language=language,
                voice_clone_prompt=voice_prompt,
            )
            if wavs and len(wavs[0]) > 0:
                yield wavs[0], sr
        except Exception as e:
            logger.error(f"Error generating sentence: {e}")
            continue


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/v1/audio/speech")
async def generate_speech_endpoint(request: TTSRequest):
    """Generate speech from text using voice cloning."""
    if model is None or voice_prompt is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Handle streaming mode
    if request.stream:
        async def audio_stream_generator():
            if request.response_format == "wav":
                yield create_wav_streaming_header(24000)

            async for audio_chunk, sr in generate_speech_streaming(
                text=request.text,
                language=request.language,
            ):
                if audio_chunk is not None and len(audio_chunk) > 0:
                    yield convert_to_pcm(audio_chunk)

        content_type = "audio/wav" if request.response_format == "wav" else "audio/pcm"
        return StreamingResponse(
            audio_stream_generator(),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                "Cache-Control": "no-cache",
                "Transfer-Encoding": "chunked",
            },
        )

    # Non-streaming mode
    try:
        wavs, sr = model.generate_voice_clone(
            text=request.text,
            language=request.language,
            voice_clone_prompt=voice_prompt,
        )

        audio = wavs[0]

        # Convert to WAV bytes
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format='WAV')
        buf.seek(0)

        return Response(
            content=buf.read(),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"},
        )

    except Exception as e:
        logger.error(f"Speech generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
