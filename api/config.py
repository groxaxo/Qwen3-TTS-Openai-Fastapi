# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Configuration module for TTS backend settings.

This module centralizes all configuration options for the TTS API,
including backend selection, device settings, CPU tuning, and OpenVINO options.
"""

import os

# ============================================================================
# Backend Selection
# ============================================================================

TTS_BACKEND = os.getenv("TTS_BACKEND", "official")
"""
TTS backend to use.
Options: 'official', 'vllm', 'pytorch', 'openvino'
- 'official': Official Qwen3-TTS implementation (default, GPU/CPU auto-detect)
- 'vllm': vLLM-Omni backend for optimized inference
- 'pytorch': CPU-optimized PyTorch backend
- 'openvino': Experimental OpenVINO backend for Intel CPUs/NPUs
"""

TTS_MODEL_ID = os.getenv("TTS_MODEL_ID", os.getenv("TTS_MODEL_NAME", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"))
"""
Model identifier for HuggingFace.
Examples: 
- Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice (default, voice design)
- Qwen/Qwen3-TTS-12Hz-1.7B-Base (voice cloning support)
- Qwen/Qwen3-TTS-12Hz-0.6B-Base (smaller model for CPU)
"""

# ============================================================================
# Device and Precision Settings
# ============================================================================

TTS_DEVICE = os.getenv("TTS_DEVICE", "auto")
"""
Device to run inference on.
Options: 'auto', 'cpu', 'cuda', 'cuda:0', etc.
- 'auto': Automatically detect (GPU if available, otherwise CPU)
- 'cpu': Force CPU inference
- 'cuda' or 'cuda:0': Use specific GPU
"""

TTS_DTYPE = os.getenv("TTS_DTYPE", "auto")
"""
Data type for model weights and computation.
Options: 'auto', 'float32', 'float16', 'bfloat16'
- 'auto': bfloat16 on GPU, float32 on CPU
- 'float32': Full precision (recommended for CPU, slower but stable)
- 'float16': Half precision (GPU only, may cause issues on some models)
- 'bfloat16': BFloat16 precision (Ampere+ GPUs, best for GPU)
"""

TTS_ATTN = os.getenv("TTS_ATTN", "auto")
"""
Attention implementation to use.
Options: 'auto', 'flash_attention_2', 'sdpa', 'eager'
- 'auto': Try flash_attention_2, fall back to sdpa, then eager
- 'flash_attention_2': Flash Attention 2 (fastest, Ampere+ GPUs)
- 'sdpa': Scaled Dot Product Attention (PyTorch native, good for CPU/GPU)
- 'eager': Standard attention (slowest, most compatible, good for CPU)
"""

# ============================================================================
# CPU Performance Tuning
# ============================================================================

CPU_THREADS = int(os.getenv("CPU_THREADS", str(os.cpu_count() or 4)))
"""
Number of threads for PyTorch CPU operations.
Recommended: Set to number of physical cores (not logical cores).
Default: Auto-detect available cores, fallback to 4.
For i5-1240P: 12 threads (4 P-cores + 8 E-cores)
"""

CPU_INTEROP = int(os.getenv("CPU_INTEROP", "2"))
"""
Number of threads for inter-op parallelism.
Recommended: 1-2 for most cases.
"""

# Optional: Set OpenMP/MKL threads (applied at import time)
if TTS_DEVICE == "cpu" or (TTS_DEVICE == "auto" and not os.getenv("CUDA_VISIBLE_DEVICES")):
    os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))

# ============================================================================
# OpenVINO Settings (Experimental)
# ============================================================================

OV_DEVICE = os.getenv("OV_DEVICE", "CPU")
"""
OpenVINO device target.
Options: 'CPU', 'GPU', 'AUTO'
- 'CPU': Intel CPU (most compatible)
- 'GPU': Intel GPU (Iris Xe, Arc, etc., if supported)
- 'AUTO': Let OpenVINO choose the best device
"""

OV_CACHE_DIR = os.getenv("OV_CACHE_DIR", "./.ov_cache")
"""
Directory for OpenVINO compilation cache.
Speeds up model loading on subsequent runs.
"""

OV_MODEL_DIR = os.getenv("OV_MODEL_DIR", "./.ov_models")
"""
Directory containing exported OpenVINO IR models.
The model.xml and model.bin files should be in this directory.
"""

# ============================================================================
# Warmup and Optimization Settings
# ============================================================================

TTS_WARMUP_ON_START = os.getenv("TTS_WARMUP_ON_START", "false").lower() == "true"
"""
Whether to run a warmup inference on server startup.
Recommended: true for production to initialize torch.compile() and cuDNN.
"""

# ============================================================================
# Intel Extension for PyTorch (IPEX) - Optional
# ============================================================================

USE_IPEX = os.getenv("USE_IPEX", "false").lower() == "true"
"""
Whether to use Intel Extension for PyTorch (IPEX).
Only applicable for CPU inference on Intel processors.
Requires: pip install intel-extension-for-pytorch
"""

if USE_IPEX and TTS_DEVICE in ("cpu", "auto"):
    try:
        import intel_extension_for_pytorch as ipex
        IPEX_AVAILABLE = True
    except ImportError:
        IPEX_AVAILABLE = False
        import logging
        logging.getLogger(__name__).warning(
            "USE_IPEX=true but intel-extension-for-pytorch is not installed. "
            "Install with: pip install intel-extension-for-pytorch"
        )
else:
    IPEX_AVAILABLE = False
