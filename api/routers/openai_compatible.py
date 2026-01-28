# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
OpenAI-compatible router for text-to-speech API.
Implements endpoints compatible with OpenAI's TTS API specification.
"""

import base64
import io
import logging
import time
from typing import List, Optional

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from ..structures.schemas import (
    OpenAISpeechRequest,
    ModelInfo,
    VoiceInfo,
    VoiceCloneRequest,
    VoiceCloneCapabilities,
)
from ..services.text_processing import normalize_text
from ..services.audio_encoding import encode_audio, get_content_type, DEFAULT_SAMPLE_RATE

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["OpenAI Compatible TTS"],
    responses={404: {"description": "Not found"}},
)


# Language code to language name mapping
LANGUAGE_CODE_MAPPING = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
}

# Available models (including language-specific variants)
AVAILABLE_MODELS = [
    ModelInfo(
        id="qwen3-tts",
        object="model",
        created=1737734400,  # 2025-01-24
        owned_by="qwen",
    ),
    ModelInfo(
        id="tts-1",
        object="model",
        created=1737734400,
        owned_by="qwen",
    ),
    ModelInfo(
        id="tts-1-hd",
        object="model",
        created=1737734400,
        owned_by="qwen",
    ),
]

# Add language-specific model variants
for lang_code in LANGUAGE_CODE_MAPPING.keys():
    AVAILABLE_MODELS.extend([
        ModelInfo(
            id=f"tts-1-{lang_code}",
            object="model",
            created=1737734400,
            owned_by="qwen",
        ),
        ModelInfo(
            id=f"tts-1-hd-{lang_code}",
            object="model",
            created=1737734400,
            owned_by="qwen",
        ),
    ])

# Model name mapping (OpenAI -> internal)
MODEL_MAPPING = {
    "tts-1": "qwen3-tts",
    "tts-1-hd": "qwen3-tts",
    "qwen3-tts": "qwen3-tts",
}

# Add language-specific model mappings
for lang_code in LANGUAGE_CODE_MAPPING.keys():
    MODEL_MAPPING[f"tts-1-{lang_code}"] = "qwen3-tts"
    MODEL_MAPPING[f"tts-1-hd-{lang_code}"] = "qwen3-tts"

# OpenAI voice mapping to Qwen voices
VOICE_MAPPING = {
    "alloy": "Vivian",
    "echo": "Ryan",
    "fable": "Sophia",
    "nova": "Isabella",
    "onyx": "Evan",
    "shimmer": "Lily",
}


def extract_language_from_model(model_name: str) -> Optional[str]:
    """
    Extract language from model name if it has a language suffix.
    
    Args:
        model_name: Model name (e.g., "tts-1-es", "tts-1-hd-fr")
    
    Returns:
        Language name if suffix found, None otherwise
    """
    # Check if model ends with a language code
    # Only extract language if the model follows the expected pattern
    for lang_code, lang_name in LANGUAGE_CODE_MAPPING.items():
        suffix = f"-{lang_code}"
        if model_name.endswith(suffix):
            # Verify it's a valid language-specific model variant
            # Should be either tts-1-{lang} or tts-1-hd-{lang}
            if model_name == f"tts-1{suffix}" or model_name == f"tts-1-hd{suffix}":
                return lang_name
    return None


async def get_tts_backend():
    """Get the TTS backend instance, initializing if needed."""
    from ..backends import get_backend, initialize_backend
    
    backend = get_backend()
    
    if not backend.is_ready():
        await initialize_backend()
    
    return backend


def get_voice_name(voice: str) -> str:
    """Map voice name to internal voice identifier."""
    # Check OpenAI voice mapping first
    if voice.lower() in VOICE_MAPPING:
        return VOICE_MAPPING[voice.lower()]
    # Otherwise use the voice name directly
    return voice


async def generate_speech(
    text: str,
    voice: str,
    language: str = "Auto",
    instruct: Optional[str] = None,
    speed: float = 1.0,
) -> tuple[np.ndarray, int]:
    """
    Generate speech from text using the configured TTS backend.
    
    Args:
        text: The text to synthesize
        voice: Voice name to use
        language: Language code
        instruct: Optional instruction for voice style
        speed: Speech speed multiplier
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    backend = await get_tts_backend()
    
    # Map voice name
    voice_name = get_voice_name(voice)
    
    # Generate speech using the backend
    try:
        audio, sr = await backend.generate_speech(
            text=text,
            voice=voice_name,
            language=language,
            instruct=instruct,
            speed=speed,
        )
        
        return audio, sr
        
    except Exception as e:
        raise RuntimeError(f"Speech generation failed: {e}")


@router.post("/audio/speech")
async def create_speech(
    request: OpenAISpeechRequest,
    client_request: Request,
):
    """
    OpenAI-compatible endpoint for text-to-speech.
    
    Generates audio from the input text using the specified voice and model.
    """
    # Validate model
    if request.model not in MODEL_MAPPING:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_model",
                "message": f"Unsupported model: {request.model}. Supported: {list(MODEL_MAPPING.keys())}",
                "type": "invalid_request_error",
            },
        )
    
    try:
        # Normalize input text
        normalized_text = normalize_text(request.input, request.normalization_options)
        
        if not normalized_text.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_input",
                    "message": "Input text is empty after normalization",
                    "type": "invalid_request_error",
                },
            )
        
        # Extract language from model name if present, otherwise use request language
        model_language = extract_language_from_model(request.model)
        language = model_language if model_language else (request.language or "Auto")
        
        # Generate speech
        audio, sample_rate = await generate_speech(
            text=normalized_text,
            voice=request.voice,
            language=language,
            instruct=request.instruct,
            speed=request.speed,
        )
        
        # Encode audio to requested format
        audio_bytes = encode_audio(audio, request.response_format, sample_rate)
        
        # Get content type
        content_type = get_content_type(request.response_format)
        
        # Return audio response
        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                "Cache-Control": "no-cache",
            },
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )


@router.get("/models")
async def list_models():
    """List all available TTS models."""
    return {
        "object": "list",
        "data": [model.model_dump() for model in AVAILABLE_MODELS],
    }


@router.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get information about a specific model."""
    for model in AVAILABLE_MODELS:
        if model.id == model_id:
            return model.model_dump()
    
    raise HTTPException(
        status_code=404,
        detail={
            "error": "model_not_found",
            "message": f"Model '{model_id}' not found",
            "type": "invalid_request_error",
        },
    )


@router.get("/audio/voices")
@router.get("/voices")
async def list_voices():
    """List all available voices for text-to-speech."""
    # Default voices (always available)
    default_voices = [
        VoiceInfo(id="Vivian", name="Vivian", language="English", description="Female voice"),
        VoiceInfo(id="Ryan", name="Ryan", language="English", description="Male voice"),
        VoiceInfo(id="Sophia", name="Sophia", language="English", description="Female voice"),
        VoiceInfo(id="Isabella", name="Isabella", language="English", description="Female voice"),
        VoiceInfo(id="Evan", name="Evan", language="English", description="Male voice"),
        VoiceInfo(id="Lily", name="Lily", language="English", description="Female voice"),
    ]
    
    # OpenAI-compatible voice aliases
    openai_voices = [
        VoiceInfo(id="alloy", name="Alloy", description="OpenAI-compatible voice (maps to Vivian)"),
        VoiceInfo(id="echo", name="Echo", description="OpenAI-compatible voice (maps to Ryan)"),
        VoiceInfo(id="fable", name="Fable", description="OpenAI-compatible voice (maps to Sophia)"),
        VoiceInfo(id="nova", name="Nova", description="OpenAI-compatible voice (maps to Isabella)"),
        VoiceInfo(id="onyx", name="Onyx", description="OpenAI-compatible voice (maps to Evan)"),
        VoiceInfo(id="shimmer", name="Shimmer", description="OpenAI-compatible voice (maps to Lily)"),
    ]
    
    default_languages = ["English", "Chinese", "Japanese", "Korean", "German", "French", "Spanish", "Russian", "Portuguese", "Italian"]
    
    try:
        backend = await get_tts_backend()
        
        # Get supported speakers from the backend
        speakers = backend.get_supported_voices()
        
        # Get supported languages
        languages = backend.get_supported_languages()
        
        # Build voice list from backend
        if speakers:
            voices = []
            for speaker in speakers:
                voice_info = VoiceInfo(
                    id=speaker,
                    name=speaker,
                    language=languages[0] if languages else "Auto",
                    description=f"Qwen3-TTS voice: {speaker}",
                )
                voices.append(voice_info.model_dump())
        else:
            voices = [v.model_dump() for v in default_voices]
        
        return {
            "voices": voices + [v.model_dump() for v in openai_voices],
            "languages": languages if languages else default_languages,
        }
        
    except Exception as e:
        logger.warning(f"Could not get voices from backend: {e}")
        # Return default voices if backend is not loaded
        return {
            "voices": [v.model_dump() for v in default_voices] + [v.model_dump() for v in openai_voices],
            "languages": default_languages,
        }


@router.get("/audio/voice-clone/capabilities")
async def get_voice_clone_capabilities():
    """
    Get voice cloning capabilities of the current backend.

    Returns whether voice cloning is supported and what modes are available.
    Voice cloning requires the Base model (Qwen3-TTS-12Hz-1.7B-Base).
    """
    try:
        backend = await get_tts_backend()

        supports_cloning = backend.supports_voice_cloning()
        model_type = backend.get_model_type() if hasattr(backend, 'get_model_type') else "unknown"

        return VoiceCloneCapabilities(
            supported=supports_cloning,
            model_type=model_type,
            icl_mode_available=supports_cloning,
            x_vector_mode_available=supports_cloning,
        )

    except Exception as e:
        logger.warning(f"Could not get voice clone capabilities: {e}")
        return VoiceCloneCapabilities(
            supported=False,
            model_type="unknown",
            icl_mode_available=False,
            x_vector_mode_available=False,
        )


@router.post("/audio/voice-clone")
async def create_voice_clone(
    request: VoiceCloneRequest,
    client_request: Request,
):
    """
    Clone a voice from reference audio and generate speech.

    This endpoint requires the Base model (Qwen3-TTS-12Hz-1.7B-Base).
    Set TTS_MODEL_NAME=Qwen/Qwen3-TTS-12Hz-1.7B-Base environment variable when starting the server.

    Two modes are available:
    - ICL mode (x_vector_only_mode=False): Requires ref_text transcript for best quality
    - X-Vector mode (x_vector_only_mode=True): No transcript needed, good quality
    """
    try:
        backend = await get_tts_backend()

        # Check if voice cloning is supported
        if not backend.supports_voice_cloning():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "voice_cloning_not_supported",
                    "message": "Voice cloning requires the Base model (Qwen3-TTS-12Hz-1.7B-Base). "
                               "Set TTS_MODEL_NAME=Qwen/Qwen3-TTS-12Hz-1.7B-Base environment variable and restart the server.",
                    "type": "invalid_request_error",
                },
            )

        # Validate ICL mode requires ref_text
        if not request.x_vector_only_mode and not request.ref_text:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "missing_ref_text",
                    "message": "ICL mode requires ref_text (transcript of reference audio). "
                               "Either provide ref_text or set x_vector_only_mode=True.",
                    "type": "invalid_request_error",
                },
            )

        # Decode base64 audio
        try:
            audio_bytes = base64.b64decode(request.ref_audio)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_audio",
                    "message": f"Failed to decode base64 audio: {e}",
                    "type": "invalid_request_error",
                },
            )

        # Load audio using soundfile
        try:
            audio_buffer = io.BytesIO(audio_bytes)
            ref_audio, ref_sr = sf.read(audio_buffer)

            # Convert to mono if stereo
            if len(ref_audio.shape) > 1:
                ref_audio = ref_audio.mean(axis=1)

            ref_audio = ref_audio.astype(np.float32)

        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "audio_processing_error",
                    "message": f"Failed to process reference audio: {e}. "
                               "Ensure the audio is a valid WAV, MP3, or other supported format.",
                    "type": "invalid_request_error",
                },
            )

        # Normalize input text
        from ..services.text_processing import normalize_text
        normalized_text = normalize_text(request.input, request.normalization_options)

        if not normalized_text.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_input",
                    "message": "Input text is empty after normalization",
                    "type": "invalid_request_error",
                },
            )

        # Generate voice clone
        audio, sample_rate = await backend.generate_voice_clone(
            text=normalized_text,
            ref_audio=ref_audio,
            ref_audio_sr=ref_sr,
            ref_text=request.ref_text,
            language=request.language or "Auto",
            x_vector_only_mode=request.x_vector_only_mode,
            speed=request.speed,
        )

        # Encode audio to requested format
        audio_bytes = encode_audio(audio, request.response_format, sample_rate)

        # Get content type
        content_type = get_content_type(request.response_format)

        # Return audio response
        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=voice_clone.{request.response_format}",
                "Cache-Control": "no-cache",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice cloning failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "processing_error",
                "message": str(e),
                "type": "server_error",
            },
        )
