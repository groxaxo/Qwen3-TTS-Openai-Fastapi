# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Audio encoding helpers for the TTS API.

The public API returns exactly the format requested by the caller. Compressed
formats therefore fail clearly when the required encoder is unavailable rather
than returning WAV bytes under an MP3/Opus/AAC/FLAC content type.
"""

from __future__ import annotations

import io
import logging
import struct
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

AudioFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
DEFAULT_SAMPLE_RATE = 24000
_SUPPORTED_FORMATS = {"mp3", "opus", "aac", "flac", "wav", "pcm"}


class AudioEncodingError(RuntimeError):
    """Raised when audio cannot be encoded in the requested format."""


def get_content_type(audio_format: AudioFormat) -> str:
    """Return the MIME content type for a supported output format."""
    content_types = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        # OpenAI-compatible PCM output is signed 16-bit little-endian mono.
        "pcm": "audio/pcm",
    }
    return content_types.get(audio_format, f"audio/{audio_format}")


def _prepare_audio(audio: np.ndarray) -> np.ndarray:
    """Validate, convert, and safely normalize a mono/stereo audio array."""
    array = np.asarray(audio)
    if array.size == 0:
        raise ValueError("Audio array is empty")
    if array.ndim not in (1, 2):
        raise ValueError(
            f"Audio must be a 1-D mono or 2-D frame/channel array; got shape {array.shape}"
        )
    if array.ndim == 2 and array.shape[1] < 1:
        raise ValueError("Audio has no channels")

    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("Audio contains NaN or infinite samples")

    peak = float(np.max(np.abs(array)))
    if peak > 1.0:
        array = array / peak

    # Clipping also protects against tiny floating-point overshoots.
    return np.clip(array, -1.0, 1.0)


def _to_pcm_array(audio: np.ndarray, bits_per_sample: int) -> np.ndarray:
    """Convert normalized float audio to an integer PCM numpy array."""
    if bits_per_sample == 16:
        # Use 32768 for negative full scale, then clip to int16 bounds.
        return np.clip(np.rint(audio * 32768.0), -32768, 32767).astype("<i2")
    if bits_per_sample == 8:
        # WAV 8-bit PCM is unsigned with silence at 128.
        return np.clip(np.rint((audio + 1.0) * 127.5), 0, 255).astype(np.uint8)
    raise ValueError("bits_per_sample must be 8 or 16")


def convert_to_wav(
    audio: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    num_channels: Optional[int] = None,
    bits_per_sample: int = 16,
) -> bytes:
    """Convert a mono or frame-by-channel numpy array to PCM WAV bytes."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")

    prepared = _prepare_audio(audio)
    inferred_channels = 1 if prepared.ndim == 1 else int(prepared.shape[1])
    channels = inferred_channels if num_channels is None else int(num_channels)
    if channels != inferred_channels:
        raise ValueError(
            f"num_channels={channels} does not match audio shape {prepared.shape} "
            f"({inferred_channels} channel(s))"
        )

    pcm = _to_pcm_array(prepared, bits_per_sample)
    pcm_bytes = pcm.tobytes(order="C")
    bytes_per_sample = bits_per_sample // 8
    byte_rate = sample_rate * channels * bytes_per_sample
    block_align = channels * bytes_per_sample
    data_size = len(pcm_bytes)

    buffer = io.BytesIO()
    buffer.write(b"RIFF")
    buffer.write(struct.pack("<I", 36 + data_size))
    buffer.write(b"WAVE")
    buffer.write(b"fmt ")
    buffer.write(struct.pack("<I", 16))
    buffer.write(struct.pack("<H", 1))
    buffer.write(struct.pack("<H", channels))
    buffer.write(struct.pack("<I", sample_rate))
    buffer.write(struct.pack("<I", byte_rate))
    buffer.write(struct.pack("<H", block_align))
    buffer.write(struct.pack("<H", bits_per_sample))
    buffer.write(b"data")
    buffer.write(struct.pack("<I", data_size))
    buffer.write(pcm_bytes)
    return buffer.getvalue()


def convert_to_pcm(audio: np.ndarray, bits_per_sample: int = 16) -> bytes:
    """Convert audio to headerless integer PCM bytes.

    The API's ``pcm`` response format uses signed 16-bit little-endian PCM by
    default, matching OpenAI-compatible clients. Eight-bit output remains
    available for internal callers and is unsigned PCM.
    """
    prepared = _prepare_audio(audio)
    return _to_pcm_array(prepared, bits_per_sample).tobytes(order="C")


def encode_audio(
    audio: np.ndarray,
    format: AudioFormat = "mp3",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> bytes:
    """Encode audio to the requested format without silent format fallback."""
    if format not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported audio format '{format}'. Supported: {sorted(_SUPPORTED_FORMATS)}"
        )

    if format == "wav":
        return convert_to_wav(audio, sample_rate)
    if format == "pcm":
        return convert_to_pcm(audio)

    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise AudioEncodingError(
            f"Cannot encode {format}: pydub is not installed. "
            "Install the API extra with `pip install -e \".[api]\"`."
        ) from exc

    wav_bytes = convert_to_wav(audio, sample_rate)
    format_params = {
        "mp3": {"format": "mp3", "bitrate": "128k"},
        "opus": {"format": "opus", "bitrate": "48k"},
        "aac": {"format": "adts", "bitrate": "192k"},
        "flac": {"format": "flac"},
    }
    params = dict(format_params[format])
    export_format = params.pop("format")

    try:
        segment = AudioSegment.from_wav(io.BytesIO(wav_bytes))
        output = io.BytesIO()
        segment.export(output, format=export_format, **params)
        encoded = output.getvalue()
    except Exception as exc:
        raise AudioEncodingError(
            f"Failed to encode audio as {format}. Ensure FFmpeg is installed "
            f"and supports the requested codec: {exc}"
        ) from exc

    if not encoded:
        raise AudioEncodingError(f"Encoder produced an empty {format} payload")
    return encoded


async def encode_audio_streaming(
    audio_generator,
    format: AudioFormat = "mp3",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
):
    """Encode chunks from an async audio generator.

    This helper is safe for independently decodable formats/chunks. The API's
    real-time path uses raw PCM; compressed streaming should encode one complete
    stream and then split its bytes rather than concatenate separate files.
    """
    async for audio_chunk in audio_generator:
        if audio_chunk is not None and len(audio_chunk) > 0:
            yield encode_audio(audio_chunk, format, sample_rate)
