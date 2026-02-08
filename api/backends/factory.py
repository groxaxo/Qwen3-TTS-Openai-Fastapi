# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Factory for creating TTS backend instances.
"""

import os
import logging
from pathlib import Path
from typing import Optional

from .base import TTSBackend
from .official_qwen3_tts import OfficialQwen3TTSBackend
from .vllm_omni_qwen3_tts import VLLMOmniQwen3TTSBackend
from .pytorch_backend import PyTorchCPUBackend
from .openvino_backend import OpenVINOBackend

logger = logging.getLogger(__name__)

# Global backend instance
_backend_instance: Optional[TTSBackend] = None


def get_backend() -> TTSBackend:
    """
    Get or create the global TTS backend instance.
    
    The backend is selected based on the TTS_BACKEND environment variable:
    - "official" (default): Use official Qwen3-TTS implementation (GPU/CPU auto-detect)
    - "vllm_omni": Use vLLM-Omni for faster inference
    - "pytorch": CPU-optimized PyTorch backend
    - "openvino": Experimental OpenVINO backend for Intel CPUs
    
    Returns:
        TTSBackend instance
    """
    global _backend_instance
    
    if _backend_instance is not None:
        return _backend_instance
    
    # Read configuration from environment variables directly to support testing
    backend_type = os.getenv("TTS_BACKEND", "official").lower()
    model_name = os.getenv("TTS_MODEL_NAME", os.getenv("TTS_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"))
    
    # Device and dtype settings
    device = os.getenv("TTS_DEVICE", "auto")
    dtype = os.getenv("TTS_DTYPE", "auto")
    attn = os.getenv("TTS_ATTN", "auto")
    
    # CPU settings
    cpu_threads = int(os.getenv("CPU_THREADS", "12"))
    cpu_interop = int(os.getenv("CPU_INTEROP", "2"))
    use_ipex = os.getenv("USE_IPEX", "false").lower() == "true"
    
    # OpenVINO settings
    ov_device = os.getenv("OV_DEVICE", "CPU")
    ov_cache_dir = os.getenv("OV_CACHE_DIR", "./.ov_cache")
    ov_model_dir = os.getenv("OV_MODEL_DIR", "./.ov_models")
    
    logger.info(f"Initializing TTS backend: {backend_type}")
    
    if backend_type == "official":
        # Official backend (GPU/CPU auto-detect)
        if model_name:
            _backend_instance = OfficialQwen3TTSBackend(model_name=model_name)
        else:
            # Use default CustomVoice model
            _backend_instance = OfficialQwen3TTSBackend()
        
        logger.info(f"Using official Qwen3-TTS backend with model: {_backend_instance.get_model_id()}")
    
    elif backend_type == "vllm_omni" or backend_type == "vllm-omni" or backend_type == "vllm":
        # vLLM-Omni backend
        if model_name:
            _backend_instance = VLLMOmniQwen3TTSBackend(model_name=model_name)
        else:
            # Use 1.7B model for best quality/speed tradeoff
            _backend_instance = VLLMOmniQwen3TTSBackend(
                model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
            )
        
        logger.info(f"Using vLLM-Omni backend with model: {_backend_instance.get_model_id()}")
    
    elif backend_type == "pytorch":
        # CPU-optimized PyTorch backend
        device_val = device if device != "auto" else "cpu"
        dtype_val = dtype if dtype != "auto" else "float32"
        attn_val = attn if attn != "auto" else "sdpa"
        
        _backend_instance = PyTorchCPUBackend(
            model_id=model_name,
            device=device_val,
            dtype=dtype_val,
            attn_implementation=attn_val,
            cpu_threads=cpu_threads,
            cpu_interop_threads=cpu_interop,
            use_ipex=use_ipex,
        )
        
        logger.info(f"Using CPU-optimized PyTorch backend with model: {_backend_instance.get_model_id()}")
        logger.info(f"Device: {device_val}, Dtype: {dtype_val}, Attention: {attn_val}")
        logger.info(f"CPU Threads: {cpu_threads}, Interop: {cpu_interop}, IPEX: {use_ipex}")
    
    elif backend_type == "openvino":
        # Experimental OpenVINO backend
        _backend_instance = OpenVINOBackend(
            ov_model_dir=ov_model_dir,
            ov_device=ov_device,
            ov_cache_dir=ov_cache_dir,
        )
        
        logger.info(f"Using experimental OpenVINO backend")
        logger.info(f"Model dir: {ov_model_dir}, Device: {ov_device}")
        logger.warning(
            "OpenVINO backend is experimental and requires manual model export. "
            "For reliable CPU inference, use TTS_BACKEND=pytorch instead."
        )
    
    else:
        logger.error(f"Unknown backend type: {backend_type}")
        raise ValueError(
            f"Unknown TTS_BACKEND: {backend_type}. "
            f"Supported values: 'official', 'vllm_omni', 'pytorch', 'openvino'"
        )
    
    return _backend_instance


async def initialize_backend(warmup: bool = False) -> TTSBackend:
    """
    Initialize the backend and optionally perform warmup.
    
    Args:
        warmup: Whether to run a warmup inference
    
    Returns:
        Initialized TTSBackend instance
    """
    backend = get_backend()

    # Initialize the backend
    await backend.initialize()

    # Load custom voices
    custom_voices_dir = os.getenv(
        "TTS_CUSTOM_VOICES",
        str(Path(__file__).resolve().parent.parent.parent / "custom_voices"),
    )
    try:
        await backend.load_custom_voices(custom_voices_dir)
    except Exception as e:
        logger.warning(f"Custom voice loading failed (non-critical): {e}")

    # Perform warmup if requested
    if warmup:
        warmup_enabled = os.getenv("TTS_WARMUP_ON_START", "false").lower() == "true"
        if warmup_enabled:
            logger.info("Performing backend warmup...")
            try:
                custom_names = backend.get_custom_voice_names()
                if backend.get_model_type() == "base" and custom_names:
                    # Base models only support voice cloning, warm up with a custom voice
                    await backend.generate_speech_with_custom_voice(
                        text="Hello, this is a warmup test.",
                        voice=custom_names[0],
                        language="English",
                    )
                elif backend.get_model_type() == "base":
                    logger.info("Skipping warmup: Base model has no custom voices to warm up with")
                else:
                    await backend.generate_speech(
                        text="Hello, this is a warmup test.",
                        voice="Vivian",
                        language="English",
                    )
                logger.info("Backend warmup completed successfully")
            except Exception as e:
                logger.warning(f"Backend warmup failed (non-critical): {e}")
    
    return backend


def reset_backend() -> None:
    """Reset the global backend instance (useful for testing)."""
    global _backend_instance
    _backend_instance = None
