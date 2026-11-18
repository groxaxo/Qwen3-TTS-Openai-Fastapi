# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
OpenAI-compatible router for text-to-speech API.
Implements endpoints compatible with OpenAI's TTS API specification.
"""

import asyncio
import base64
import inspect
import io
import json
import logging
import os
import re
import time
from pathlib import Path
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

# Concurrency cap: prevents simultaneous requests from starving GPU memory.
# Override with TTS_MAX_CONCURRENT env var (default 1 for single-GPU deployments).
try:
    _MAX_CONCURRENT = max(1, int(os.getenv("TTS_MAX_CONCURRENT", "1")))
except ValueError:
    logger.warning("Invalid TTS_MAX_CONCURRENT value; falling back to 1")
    _MAX_CONCURRENT = 1
_generation_semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

# --- Auto-chunking -----------------------------------------------------------
# Input is split at punctuation into chunks sized to a [min, max] character
# window, each synthesized separately and the audio concatenated back together.
# This keeps every generation well under the backend's wall-clock cap (and below
# the mlx-audio 0.3.x graph-compile hang threshold for long sequences), and
# lowers first-audio latency. Inputs that fit in a single chunk take the
# original code path with zero overhead.
#
# Splitting prefers sentence punctuation (. ! ?), then clause punctuation
# (, ; :), then word boundaries. Chunks are packed greedily up to max_chars and
# kept at/above min_chars where possible (a too-small piece is merged with a
# neighbour as long as the result still fits within max_chars).
#   TTS_AUTOCHUNK=false        disable entirely
#   TTS_MIN_CHUNK_CHARS=20     soft lower bound per chunk
#   TTS_MAX_CHUNK_CHARS=70     hard upper bound per chunk
#   TTS_CHUNK_GAP_MS=120       silence inserted between merged chunks
try:
    _AUTOCHUNK = os.getenv("TTS_AUTOCHUNK", "true").lower() == "true"
    _MIN_CHUNK_CHARS = max(1, int(os.getenv("TTS_MIN_CHUNK_CHARS", "20")))
    _MAX_CHUNK_CHARS = max(_MIN_CHUNK_CHARS, int(os.getenv("TTS_MAX_CHUNK_CHARS", "70")))
    _CHUNK_GAP_MS = max(0, int(os.getenv("TTS_CHUNK_GAP_MS", "120")))
except ValueError:
    logger.warning("Invalid auto-chunk env value; using defaults")
    _AUTOCHUNK, _MIN_CHUNK_CHARS, _MAX_CHUNK_CHARS, _CHUNK_GAP_MS = True, 20, 70, 120


def _pieces(text: str, max_chars: int) -> List[str]:
    """Break text into pieces each <= max_chars, splitting on sentence
    punctuation (. ! ?), then clause punctuation (, ; :), then words."""
    out: List[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) <= max_chars:
            out.append(sent)
            continue
        for clause in re.split(r"(?<=[,;:])\s+", sent):
            clause = clause.strip()
            if not clause:
                continue
            if len(clause) <= max_chars:
                out.append(clause)
                continue
            buf = ""
            for w in clause.split():
                if not buf:
                    buf = w
                elif len(buf) + 1 + len(w) <= max_chars:
                    buf += " " + w
                else:
                    out.append(buf)
                    buf = w
            if buf:
                out.append(buf)
    return out


def _split_into_chunks(text: str, min_chars: int, max_chars: int) -> List[str]:
    """Split text into chunks within a [min_chars, max_chars] window, breaking
    at punctuation. Pieces are packed greedily up to max_chars; a chunk shorter
    than min_chars is merged into a neighbour when the result still fits."""
    pieces = _pieces(text, max_chars)
    if not pieces:
        return []
    # Greedy pack up to max_chars.
    chunks: List[str] = []
    buf = ""
    for p in pieces:
        if not buf:
            buf = p
        elif len(buf) + 1 + len(p) <= max_chars:
            buf += " " + p
        else:
            chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    # Soft minimum: fold an undersized chunk into a neighbour if it still fits.
    merged: List[str] = []
    for c in chunks:
        if (
            merged
            and (len(c) < min_chars or len(merged[-1]) < min_chars)
            and len(merged[-1]) + 1 + len(c) <= max_chars
        ):
            merged[-1] = merged[-1] + " " + c
        else:
            merged.append(c)
    return [c for c in merged if c.strip()]
# -----------------------------------------------------------------------------

# Voice library: saved voice profiles used via the "clone:ProfileName" voice prefix.
# Configurable via VOICE_LIBRARY_DIR env var; defaults to ./voice_library.
VOICE_LIBRARY_DIR = Path(
    os.environ.get("VOICE_LIBRARY_DIR", "./voice_library")
).resolve()

# In-process cache for reference audio reads (profile_name -> (audio_np, sample_rate)).
# Avoids re-reading and re-decoding the same WAV file on every request.
_ref_audio_cache: dict = {}

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


def _load_voice_profile(name_or_id: str) -> dict:
    """Load a voice profile by name or profile_id from the voice library.

    Searches ``VOICE_LIBRARY_DIR/profiles/`` for a sub-directory whose
    ``meta.json`` matches the given *name_or_id* (case-insensitive name match
    or exact profile_id match).

    Returns a dict with keys:
        ref_audio_path, ref_text, x_vector_only_mode, language, name

    Raises:
        ValueError: if the profile is not found or its reference audio is missing.
    """
    profiles_dir = VOICE_LIBRARY_DIR / "profiles"
    if not profiles_dir.exists():
        raise ValueError(f"Voice library not found: {profiles_dir}")

    for child in sorted(profiles_dir.iterdir()):
        if not child.is_dir():
            continue
        meta_file = child / "meta.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        if (
            meta.get("profile_id") == name_or_id
            or meta.get("name", "").lower() == name_or_id.lower()
        ):
            ref_filename = meta.get("ref_audio_filename", "")
            if not ref_filename:
                raise ValueError(f"Profile '{name_or_id}' has no reference audio filename")
            ref_path = child / ref_filename
            if not ref_path.exists():
                raise ValueError(f"Reference audio missing: {ref_path}")
            return {
                "ref_audio_path": str(ref_path),
                "ref_text": meta.get("ref_text", ""),
                "x_vector_only_mode": meta.get("x_vector_only_mode", False),
                "language": meta.get("language", "Auto"),
                "name": meta.get("name", name_or_id),
            }

    raise ValueError(f"Voice profile not found: '{name_or_id}'")


async def get_tts_backend():
    """Get the TTS backend instance, initializing if needed.

    Honors lazy-load: if the backend hasn't been initialized yet, the
    first caller pays the model-load + warmup cost in one shot. The
    warmup is gated by the ``TTS_WARMUP_ON_START`` env var (which the
    factory re-reads on each call). This is critical for the MLX
    backend — the cold graph compile can wedge mlx-audio 0.3.x, and
    warmup absorbs that cost at first-load time instead of on the
    user's first request.
    """
    from ..backends import get_backend, initialize_backend

    backend = get_backend()

    if not backend.is_ready():
        warmup_enabled = os.getenv("TTS_WARMUP_ON_START", "false").lower() == "true"
        await initialize_backend(warmup=warmup_enabled)

    return backend


def note_speech_activity(app, samples: int = 0) -> None:
    """Reset the idle-shutdown timer on a successful speech request.

    Called from the /v1/audio/speech handler. Read-only endpoints
    like /health do NOT call this, so they don't keep a quiet
    server alive forever.

    Args:
        app: The FastAPI app instance (``request.app`` from the route).
        samples: Number of audio samples generated, for stats.
    """
    import time as _time
    state = getattr(app, "state", None)
    if state is None:
        return
    state.last_speech_at = _time.monotonic()
    state.speech_request_count = getattr(state, "speech_request_count", 0) + 1
    state.speech_total_samples = getattr(state, "speech_total_samples", 0) + int(samples)


def get_voice_name(voice: str) -> str:
    """Map voice name to internal voice identifier."""
    # Check OpenAI voice mapping first
    if voice.lower() in VOICE_MAPPING:
        return VOICE_MAPPING[voice.lower()]
    # Otherwise use the voice name directly
    return voice


def _method_accepts_kwarg(method, kwarg: str) -> bool:
    """Return True if a callable accepts a given keyword argument."""
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return False

    if kwarg in signature.parameters:
        return True

    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )


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

    # Check custom voice BEFORE applying OpenAI alias mapping,
    # so custom voices with OpenAI alias names remain accessible.
    is_custom = backend.is_custom_voice(voice)
    # Map voice name (OpenAI aliases to internal names) for built-in voices.
    voice_name = voice if is_custom else get_voice_name(voice)

    async def _synth(segment: str) -> tuple[np.ndarray, int]:
        if is_custom:
            return await backend.generate_speech_with_custom_voice(
                text=segment,
                voice=voice,
                language=language,
                speed=speed,
            )
        return await backend.generate_speech(
            text=segment,
            voice=voice_name,
            language=language,
            instruct=instruct,
            speed=speed,
        )

    # Decide chunking. A single chunk (or disabled) takes the original path.
    chunks = (
        _split_into_chunks(text, _MIN_CHUNK_CHARS, _MAX_CHUNK_CHARS)
        if _AUTOCHUNK else [text]
    )
    if len(chunks) <= 1:
        try:
            return await _synth(text)
        except Exception as e:
            raise RuntimeError(f"Speech generation failed: {e}")

    # Multi-chunk: synthesize each sentence-group, then merge with a short gap.
    logger.info(
        "Auto-chunking %d chars into %d chunks (window=%d-%d)",
        len(text), len(chunks), _MIN_CHUNK_CHARS, _MAX_CHUNK_CHARS,
    )
    try:
        audios: List[np.ndarray] = []
        sr = DEFAULT_SAMPLE_RATE
        for i, seg in enumerate(chunks):
            a, sr = await _synth(seg)
            if a is not None and len(a):
                audios.append(np.asarray(a))
        if not audios:
            raise RuntimeError("no audio produced from any chunk")
        gap_len = int(sr * _CHUNK_GAP_MS / 1000.0)
        gap = (
            np.zeros(gap_len, dtype=audios[0].dtype)
            if gap_len > 0 else None
        )
        merged: List[np.ndarray] = []
        for i, a in enumerate(audios):
            if i and gap is not None:
                merged.append(gap)
            merged.append(a)
        return np.concatenate(merged), sr
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

    **Voice library:** pass ``voice: "clone:ProfileName"`` to use a saved voice
    profile from the voice library (``VOICE_LIBRARY_DIR/profiles/``).  The
    server automatically switches to the Base model for profile-based cloning.

    **Real-time streaming:** set ``stream: true`` together with
    ``response_format: "pcm"`` to receive raw PCM chunks as the model generates
    audio (requires the *optimized* backend; other backends fall back to chunked
    delivery of fully-generated audio).
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

        # ----------------------------------------------------------------
        # Voice library: "clone:ProfileName" -> load profile + voice clone
        # ----------------------------------------------------------------
        if request.voice.lower().startswith("clone:"):
            profile_name = request.voice[len("clone:"):].strip()
            if not profile_name:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "invalid_voice",
                        "message": (
                            "The 'clone:' prefix requires a profile name, "
                            "e.g. voice='clone:MyVoice'"
                        ),
                        "type": "invalid_request_error",
                    },
                )
            try:
                profile = _load_voice_profile(profile_name)
            except ValueError as exc:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "profile_not_found",
                        "message": str(exc),
                        "type": "invalid_request_error",
                    },
                )

            backend = await get_tts_backend()

            # Check that voice cloning is supported by the current backend/model
            if not backend.supports_voice_cloning():
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "voice_cloning_not_supported",
                        "message": (
                            "Voice library cloning requires a Base model and the "
                            "optimized backend (TTS_BACKEND=optimized), or a backend "
                            "that supports voice cloning."
                        ),
                        "type": "invalid_request_error",
                    },
                )

            # ICL mode (x_vector_only_mode=False) requires a ref_text transcript
            if not profile["x_vector_only_mode"] and not profile["ref_text"]:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "missing_ref_text",
                        "message": (
                            f"Profile '{profile['name']}' is configured for ICL mode "
                            "(x_vector_only_mode=false) but has no ref_text. "
                            "Add a transcript to meta.json or set x_vector_only_mode=true."
                        ),
                        "type": "invalid_request_error",
                    },
                )

            # Normalize cache key to canonical profile name (case-insensitive safe)
            canonical_key = profile["name"].lower()

            # Cache reference audio reads to avoid repeated disk I/O
            ref_audio_path = profile["ref_audio_path"]
            file_stat = Path(ref_audio_path).stat()
            canonical_key = repr((
                canonical_key, ref_audio_path, file_stat.st_mtime_ns, file_stat.st_size,
                profile["ref_text"], profile["x_vector_only_mode"], profile["language"],
            ))
            if canonical_key not in _ref_audio_cache:
                try:
                    ref_audio_np, ref_sr = sf.read(ref_audio_path)
                    if len(ref_audio_np.shape) > 1:
                        ref_audio_np = ref_audio_np.mean(axis=1)
                    ref_audio_np = ref_audio_np.astype(np.float32)
                    if len(_ref_audio_cache) >= 32:
                        _ref_audio_cache.pop(next(iter(_ref_audio_cache)))
                    _ref_audio_cache[canonical_key] = (ref_audio_np, ref_sr)
                    logger.info(f"Reference audio cached for profile '{profile['name']}'")
                except Exception as exc:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "audio_processing_error",
                            "message": (
                                f"Failed to load reference audio for profile "
                                f"'{profile['name']}': {exc}"
                            ),
                            "type": "invalid_request_error",
                        },
                    )
            ref_audio_np, ref_sr = _ref_audio_cache[canonical_key]

            clone_lang = (
                language if language != "Auto" else profile["language"]
            )
            logger.info(
                f"Voice library clone '{profile['name']}': "
                f"lang={clone_lang}, "
                f"x_vector_only={profile['x_vector_only_mode']}, "
                f"stream={request.stream}"
            )

            if request.stream and hasattr(backend, "generate_voice_clone_streaming"):
                # Streaming: only PCM and WAV→PCM are valid
                if request.response_format not in ("pcm", "wav"):
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "invalid_format_for_streaming",
                            "message": (
                                f"Real-time streaming only supports response_format "
                                f"'pcm' (raw float32). Got '{request.response_format}'. "
                                "Use stream=false for compressed formats."
                            ),
                            "type": "invalid_request_error",
                        },
                    )
                fmt = "pcm"
                content_type = get_content_type(fmt)

                async def _clone_stream():
                    gen_start = time.time()
                    first_chunk_logged = False
                    total_samples = 0
                    chunk_count = 0
                    sample_rate = 24000
                    clone_stream_kwargs = {
                        "text": normalized_text,
                        "ref_audio": ref_audio_np,
                        "ref_audio_sr": ref_sr,
                        "ref_text": profile["ref_text"] or None,
                        "language": clone_lang,
                        "x_vector_only_mode": profile["x_vector_only_mode"],
                    }
                    if _method_accepts_kwarg(
                        backend.generate_voice_clone_streaming, "cache_key"
                    ):
                        clone_stream_kwargs["cache_key"] = canonical_key
                    async with _generation_semaphore:
                        async for pcm_chunk, sr in backend.generate_voice_clone_streaming(
                            **clone_stream_kwargs,
                        ):
                            if pcm_chunk is not None and len(pcm_chunk) > 0:
                                if not first_chunk_logged:
                                    logger.info(
                                        f"Voice clone stream TTFB: "
                                        f"{time.time()-gen_start:.3f}s"
                                    )
                                    first_chunk_logged = True
                                total_samples += len(pcm_chunk)
                                sample_rate = sr
                                chunk_count += 1
                                yield encode_audio(pcm_chunk, fmt, sr)
                                await asyncio.sleep(0)
                    gen_time = time.time() - gen_start
                    audio_dur = total_samples / sample_rate if sample_rate > 0 else 0
                    rtf = gen_time / audio_dur if audio_dur > 0 else 0
                    logger.info(
                        f"Voice clone stream done: "
                        f"total={gen_time:.2f}s audio={audio_dur:.2f}s "
                        f"RTF={rtf:.2f}x chunks={chunk_count}"
                    )

                return StreamingResponse(
                    _clone_stream(),
                    media_type=content_type,
                    headers={
                        "Content-Disposition": f"inline; filename=speech.{fmt}",
                        "Cache-Control": "no-cache",
                    },
                )
            else:
                # Non-streaming path — honor the requested format (including wav)
                gen_start = time.time()
                clone_kwargs = {
                    "text": normalized_text,
                    "ref_audio": ref_audio_np,
                    "ref_audio_sr": ref_sr,
                    "ref_text": profile["ref_text"] or None,
                    "language": clone_lang,
                    "x_vector_only_mode": profile["x_vector_only_mode"],
                    "speed": request.speed,
                }
                if _method_accepts_kwarg(backend.generate_voice_clone, "cache_key"):
                    clone_kwargs["cache_key"] = canonical_key
                async with _generation_semaphore:
                    audio, sample_rate = await backend.generate_voice_clone(**clone_kwargs)
                gen_time = time.time() - gen_start
                audio_dur = len(audio) / sample_rate if sample_rate > 0 else 0
                rtf = gen_time / audio_dur if audio_dur > 0 else 0
                logger.info(
                    f"Voice clone done: gen={gen_time:.2f}s "
                    f"audio={audio_dur:.2f}s RTF={rtf:.2f}x"
                )

                fmt = request.response_format
                audio_bytes = await asyncio.to_thread(encode_audio, audio, fmt, sample_rate)
                content_type = get_content_type(fmt)

                return Response(
                    content=audio_bytes,
                    media_type=content_type,
                    headers={
                        "Content-Disposition": f"inline; filename=speech.{fmt}",
                        "Cache-Control": "no-cache",
                    },
                )

        # ----------------------------------------------------------------
        # Real-time streaming for built-in voices (optimized backend only)
        # ----------------------------------------------------------------
        if request.stream:
            backend = await get_tts_backend()
            if hasattr(backend, "generate_speech_streaming"):
                # Streaming: only PCM and WAV→PCM are valid (compressed formats
                # produce invalid streams when per-chunk encode_audio is concatenated)
                if request.response_format not in ("pcm", "wav"):
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "invalid_format_for_streaming",
                            "message": (
                                f"Real-time streaming only supports response_format "
                                f"'pcm' (raw float32). Got '{request.response_format}'. "
                                "Use stream=false for compressed formats."
                            ),
                            "type": "invalid_request_error",
                        },
                    )
                voice_name = get_voice_name(request.voice)
                fmt = "pcm"
                content_type = get_content_type(fmt)

                async def _speech_stream():
                    gen_start = time.time()
                    first_chunk_logged = False
                    total_samples = 0
                    chunk_count = 0
                    sample_rate = 24000
                    async with _generation_semaphore:
                        async for pcm_chunk, sr in backend.generate_speech_streaming(
                            text=normalized_text,
                            voice=voice_name,
                            language=language,
                            instruct=request.instruct,
                            model=request.model,
                        ):
                            if pcm_chunk is not None and len(pcm_chunk) > 0:
                                if not first_chunk_logged:
                                    logger.info(
                                        f"TTS stream TTFB: "
                                        f"{time.time()-gen_start:.3f}s"
                                    )
                                    first_chunk_logged = True
                                total_samples += len(pcm_chunk)
                                sample_rate = sr
                                chunk_count += 1
                                yield encode_audio(pcm_chunk, fmt, sr)
                                await asyncio.sleep(0)
                    gen_time = time.time() - gen_start
                    audio_dur = total_samples / sample_rate if sample_rate > 0 else 0
                    rtf = gen_time / audio_dur if audio_dur > 0 else 0
                    logger.info(
                        f"TTS stream done: total={gen_time:.2f}s "
                        f"audio={audio_dur:.2f}s RTF={rtf:.2f}x chunks={chunk_count}"
                    )
                    # Reset the idle-shutdown timer for this in-process
                    # server. We do this on the streaming path's
                    # natural completion so the timer only resets on
                    # a real successful generation.
                    try:
                        note_speech_activity(client_request.app, samples=total_samples)
                    except Exception:
                        pass

                return StreamingResponse(
                    _speech_stream(),
                    media_type=content_type,
                    headers={
                        "Content-Disposition": f"attachment; filename=speech.{fmt}",
                        "Cache-Control": "no-cache",
                    },
                )

        # ----------------------------------------------------------------
        # Non-streaming (or streaming fallback for non-optimized backends)
        # ----------------------------------------------------------------
        # Guard against concurrent overload
        async with _generation_semaphore:
            # Generate speech
            audio, sample_rate = await generate_speech(
                text=normalized_text,
                voice=request.voice,
                language=language,
                instruct=request.instruct,
                speed=request.speed,
            )

            # Reset the idle-shutdown timer for this in-process
            # server. Read-only endpoints like /health do not touch
            # this, so a quiet server self-terminates.
            try:
                note_speech_activity(client_request.app, samples=len(audio))
            except Exception:
                pass

        # Get content type
        content_type = get_content_type(request.response_format)

        if request.stream:
            # Fallback streaming: generate fully then chunk (non-optimized backends)
            async def _pcm_chunks():
                chunk_size = 4096
                audio_bytes = await asyncio.to_thread(
                    encode_audio, audio, request.response_format, sample_rate
                )
                for i in range(0, len(audio_bytes), chunk_size):
                    yield audio_bytes[i:i + chunk_size]

            return StreamingResponse(
                _pcm_chunks(),
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                    "Cache-Control": "no-cache",
                },
            )

        # Encode audio to requested format (offloaded – pydub MP3 encoding is CPU-heavy)
        audio_bytes = await asyncio.to_thread(encode_audio, audio, request.response_format, sample_rate)

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
    """List all available voices for text-to-speech.

    Includes built-in Qwen3-TTS speakers, OpenAI-compatible aliases, and any
    saved voice profiles from the voice library (listed with a ``clone:`` prefix).
    """
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

    # Discover voice library profiles (clone: prefix voices)
    clone_voices: List[dict] = []
    profiles_dir = VOICE_LIBRARY_DIR / "profiles"
    if profiles_dir.exists():
        for child in sorted(profiles_dir.iterdir()):
            meta_file = child / "meta.json"
            if not meta_file.exists():
                continue
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                ref_audio_filename = meta.get("ref_audio_filename")
                name = meta.get("name")
                if ref_audio_filename and isinstance(name, str) and name.strip():
                    clone_name = name.strip()
                    clone_id = f"clone:{clone_name}"
                    clone_voices.append(
                        VoiceInfo(
                            id=clone_id,
                            name=clone_id,
                            description=f"Voice library profile: {clone_name}",
                        ).model_dump()
                    )
                elif ref_audio_filename:
                    logger.warning(
                        "Skipping voice profile at %s due to invalid or missing 'name' in meta.json",
                        meta_file,
                    )
            except Exception:
                pass

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
                if hasattr(backend, "is_custom_voice") and backend.is_custom_voice(speaker):
                    description = f"Custom cloned voice: {speaker}"
                else:
                    description = f"Qwen3-TTS voice: {speaker}"
                voice_info = VoiceInfo(
                    id=speaker,
                    name=speaker,
                    language=languages[0] if languages else "Auto",
                    description=description,
                )
                voices.append(voice_info.model_dump())
        else:
            voices = [v.model_dump() for v in default_voices]
        
        # OpenAI aliases map to built-in speakers; skip them on Base models
        if backend.get_model_type() != "base":
            voices += [v.model_dump() for v in openai_voices]

        return {
            "voices": voices + clone_voices,
            "languages": languages if languages else default_languages,
        }
        
    except Exception as e:
        logger.warning(f"Could not get voices from backend: {e}")
        # Return default voices if backend is not loaded
        return {
            "voices": (
                [v.model_dump() for v in default_voices]
                + [v.model_dump() for v in openai_voices]
                + clone_voices
            ),
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
        async with _generation_semaphore:
            audio, sample_rate = await backend.generate_voice_clone(
                text=normalized_text,
                ref_audio=ref_audio,
                ref_audio_sr=ref_sr,
                ref_text=request.ref_text,
                language=request.language or "Auto",
                x_vector_only_mode=request.x_vector_only_mode,
                speed=request.speed,
            )

        # Encode audio to requested format (offloaded – pydub MP3 encoding is CPU-heavy)
        audio_bytes = await asyncio.to_thread(encode_audio, audio, request.response_format, sample_rate)

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
