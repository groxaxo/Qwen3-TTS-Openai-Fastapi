# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Factory for creating TTS backend instances.
"""

import os
import logging
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
    
    # Import config after module is loaded to allow environment variable setup
    from ..config import (
        TTS_BACKEND, TTS_MODEL_ID, TTS_DEVICE, TTS_DTYPE, TTS_ATTN,
        CPU_THREADS, CPU_INTEROP, USE_IPEX,
        OV_DEVICE, OV_CACHE_DIR, OV_MODEL_DIR
    )
    
    # Get backend type from environment
    backend_type = TTS_BACKEND.lower()
    
    # Get model name from environment (optional override)
    model_name = os.getenv("TTS_MODEL_NAME", TTS_MODEL_ID)
    
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
        device = TTS_DEVICE if TTS_DEVICE != "auto" else "cpu"
        dtype = TTS_DTYPE if TTS_DTYPE != "auto" else "float32"
        attn = TTS_ATTN if TTS_ATTN != "auto" else "sdpa"
        
        _backend_instance = PyTorchCPUBackend(
            model_id=model_name,
            device=device,
            dtype=dtype,
            attn_implementation=attn,
            cpu_threads=CPU_THREADS,
            cpu_interop_threads=CPU_INTEROP,
            use_ipex=USE_IPEX,
        )
        
        logger.info(f"Using CPU-optimized PyTorch backend with model: {_backend_instance.get_model_id()}")
        logger.info(f"Device: {device}, Dtype: {dtype}, Attention: {attn}")
        logger.info(f"CPU Threads: {CPU_THREADS}, Interop: {CPU_INTEROP}, IPEX: {USE_IPEX}")
    
    elif backend_type == "openvino":
        # Experimental OpenVINO backend
        _backend_instance = OpenVINOBackend(
            ov_model_dir=OV_MODEL_DIR,
            ov_device=OV_DEVICE,
            ov_cache_dir=OV_CACHE_DIR,
        )
        
        logger.info(f"Using experimental OpenVINO backend")
        logger.info(f"Model dir: {OV_MODEL_DIR}, Device: {OV_DEVICE}")
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
    
    # Perform warmup if requested
    if warmup:
        warmup_enabled = os.getenv("TTS_WARMUP_ON_START", "false").lower() == "true"
        if warmup_enabled:
            logger.info("Performing backend warmup...")
            try:
                # Run a simple warmup generation
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
