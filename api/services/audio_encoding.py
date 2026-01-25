# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Audio encoding service for TTS API.
Handles conversion of raw audio to various formats (mp3, opus, aac, flac, wav, pcm).
"""

import io
import logging
import struct
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

AudioFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]

# Default sample rate for Qwen3-TTS output
DEFAULT_SAMPLE_RATE = 24000


def get_content_type(audio_format: AudioFormat) -> str:
    """Get MIME content type for audio format."""
    content_types = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }
    return content_types.get(audio_format, f"audio/{audio_format}")


def convert_to_wav(
    audio: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    num_channels: int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    """
    Convert numpy audio array to WAV format bytes.
    
    Args:
        audio: Audio data as numpy array (float32 normalized to [-1, 1])
        sample_rate: Sample rate in Hz
        num_channels: Number of audio channels
        bits_per_sample: Bits per sample (8 or 16)
    
    Returns:
        WAV file bytes
    """
    # Ensure audio is float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Normalize if needed
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val
    
    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Create WAV header
    num_samples = len(audio_int16)
    bytes_per_sample = bits_per_sample // 8
    byte_rate = sample_rate * num_channels * bytes_per_sample
    block_align = num_channels * bytes_per_sample
    data_size = num_samples * bytes_per_sample
    
    buffer = io.BytesIO()
    
    # RIFF header
    buffer.write(b'RIFF')
    buffer.write(struct.pack('<I', 36 + data_size))  # File size - 8
    buffer.write(b'WAVE')
    
    # Format chunk
    buffer.write(b'fmt ')
    buffer.write(struct.pack('<I', 16))  # Chunk size
    buffer.write(struct.pack('<H', 1))  # Audio format (PCM)
    buffer.write(struct.pack('<H', num_channels))
    buffer.write(struct.pack('<I', sample_rate))
    buffer.write(struct.pack('<I', byte_rate))
    buffer.write(struct.pack('<H', block_align))
    buffer.write(struct.pack('<H', bits_per_sample))
    
    # Data chunk
    buffer.write(b'data')
    buffer.write(struct.pack('<I', data_size))
    buffer.write(audio_int16.tobytes())
    
    return buffer.getvalue()


def create_wav_streaming_header(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    num_channels: int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    """
    Create a WAV header for streaming with unknown final size.

    Uses 0xFFFFFFFF as placeholder for file and data sizes, which is a
    common convention for streaming WAV. Most players handle this correctly.

    Args:
        sample_rate: Sample rate in Hz
        num_channels: Number of audio channels
        bits_per_sample: Bits per sample (typically 16)

    Returns:
        WAV header bytes (44 bytes)
    """
    buffer = io.BytesIO()

    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    # RIFF header with placeholder file size
    buffer.write(b'RIFF')
    buffer.write(struct.pack('<I', 0xFFFFFFFF))  # Placeholder: will be filled by client
    buffer.write(b'WAVE')

    # Format chunk
    buffer.write(b'fmt ')
    buffer.write(struct.pack('<I', 16))  # Chunk size
    buffer.write(struct.pack('<H', 1))  # Audio format (PCM)
    buffer.write(struct.pack('<H', num_channels))
    buffer.write(struct.pack('<I', sample_rate))
    buffer.write(struct.pack('<I', byte_rate))
    buffer.write(struct.pack('<H', block_align))
    buffer.write(struct.pack('<H', bits_per_sample))

    # Data chunk with placeholder size
    buffer.write(b'data')
    buffer.write(struct.pack('<I', 0xFFFFFFFF))  # Placeholder: will be filled by client

    return buffer.getvalue()


def convert_to_pcm(
    audio: np.ndarray,
    bits_per_sample: int = 16,
) -> bytes:
    """
    Convert numpy audio array to raw PCM bytes.
    
    Args:
        audio: Audio data as numpy array (float32 normalized to [-1, 1])
        bits_per_sample: Bits per sample (8 or 16)
    
    Returns:
        Raw PCM bytes
    """
    # Ensure audio is float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Normalize if needed
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val
    
    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    
    return audio_int16.tobytes()


def encode_audio(
    audio: np.ndarray,
    format: AudioFormat = "mp3",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> bytes:
    """
    Encode audio to the specified format.
    
    Args:
        audio: Audio data as numpy array (float32 normalized to [-1, 1])
        format: Target audio format
        sample_rate: Sample rate in Hz
    
    Returns:
        Encoded audio bytes
    """
    if format == "wav":
        return convert_to_wav(audio, sample_rate)
    
    if format == "pcm":
        return convert_to_pcm(audio)
    
    # For compressed formats, use pydub if available, otherwise fall back to wav
    try:
        from pydub import AudioSegment
        
        # Convert to WAV first
        wav_bytes = convert_to_wav(audio, sample_rate)
        
        # Load into pydub
        segment = AudioSegment.from_wav(io.BytesIO(wav_bytes))
        
        # Export to target format
        output = io.BytesIO()
        
        format_params = {
            "mp3": {"format": "mp3", "bitrate": "192k"},
            "opus": {"format": "opus", "bitrate": "128k"},
            "aac": {"format": "adts", "bitrate": "192k"},  # AAC in ADTS container
            "flac": {"format": "flac"},
        }
        
        params = format_params.get(format, {"format": format})
        export_format = params.pop("format", format)
        
        segment.export(output, format=export_format, **params)
        return output.getvalue()
        
    except ImportError:
        # Fall back to WAV if pydub not available
        logger.warning(f"pydub not available, returning WAV instead of {format}")
        return convert_to_wav(audio, sample_rate)
    except Exception as e:
        # Fall back to WAV on any encoding error
        logger.warning(f"Failed to encode to {format} ({e}), returning WAV")
        return convert_to_wav(audio, sample_rate)


async def encode_audio_streaming(
    audio_generator,
    format: AudioFormat = "mp3",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
):
    """
    Async generator that encodes audio chunks to the specified format.
    
    Args:
        audio_generator: Async generator yielding audio chunks as numpy arrays
        format: Target audio format
        sample_rate: Sample rate in Hz
    
    Yields:
        Encoded audio chunks
    """
    async for audio_chunk in audio_generator:
        if audio_chunk is not None and len(audio_chunk) > 0:
            yield encode_audio(audio_chunk, format, sample_rate)
