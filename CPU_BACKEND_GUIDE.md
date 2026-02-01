# CPU Backend Guide for Intel i5-1240P and Similar Processors

This guide explains how to run Qwen3-TTS efficiently on CPU-only systems, particularly Intel processors like the i5-1240P.

## Table of Contents
- [Overview](#overview)
- [CPU Backend Options](#cpu-backend-options)
- [Quick Start: PyTorch CPU Backend](#quick-start-pytorch-cpu-backend)
- [Advanced: Intel Extension for PyTorch (IPEX)](#advanced-intel-extension-for-pytorch-ipex)
- [Experimental: OpenVINO Backend](#experimental-openvino-backend)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

## Overview

The Qwen3-TTS API server now supports optimized CPU inference for systems without a GPU. The implementation includes:

- **CPU-optimized PyTorch backend** - Recommended for most CPU users
- **Intel Extension for PyTorch (IPEX)** - Optional performance boost on Intel CPUs
- **OpenVINO backend** - Experimental, requires manual model export

### Reality Check for i5-1240P (CPU-only)

* ✅ **It will run**, but expect slower inference compared to GPU
* 🎯 **Best strategy**: Keep it simple, reduce overhead
* 📊 **Recommended model**: `Qwen3-TTS-12Hz-0.6B-Base` (smaller, faster on CPU)
* ⚠️ **Not a drop-in win**: OpenVINO requires manual export and may not accelerate all components

## CPU Backend Options

The API server supports three backend options for CPU inference:

| Backend | Setup | Performance | Stability | Recommended For |
|---------|-------|-------------|-----------|-----------------|
| **PyTorch CPU** | ✅ Simple | ⭐⭐⭐ Good | ✅ Stable | **All CPU users** |
| **PyTorch + IPEX** | ⚠️ Extra dependency | ⭐⭐⭐⭐ Better | ✅ Stable | Intel CPUs (Linux) |
| **OpenVINO** | ⚠️ Complex | ⭐⭐⭐⭐⭐ Best* | ⚠️ Experimental | Advanced users |

*OpenVINO may only accelerate parts of the pipeline

## Quick Start: PyTorch CPU Backend

### 1. Set Environment Variables

Create a `.env` file or export these variables:

```bash
# Backend selection
export TTS_BACKEND=pytorch           # Use CPU-optimized PyTorch backend

# Model configuration
export TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-0.6B-Base  # Smaller model for CPU

# Device and precision
export TTS_DEVICE=cpu                # Force CPU device
export TTS_DTYPE=float32             # Recommended for CPU (stable, fast)
export TTS_ATTN=sdpa                 # Scaled Dot Product Attention (CPU-friendly)

# CPU threading (adjust for your CPU)
export CPU_THREADS=12                # Physical cores (i5-1240P: 4 P-cores + 8 E-cores)
export CPU_INTEROP=2                 # Inter-op parallelism

# Optional: Set OpenMP/MKL threads
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
```

### 2. Start the Server

```bash
python -m api.main
# or
qwen-tts-api
```

### 3. Test the API

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed")

response = client.audio.speech.create(
    model="qwen3-tts",
    voice="Vivian",
    input="Hello! This is Qwen3-TTS running on CPU."
)
response.stream_to_file("output.mp3")
```

## Advanced: Intel Extension for PyTorch (IPEX)

IPEX can provide significant speedup on Intel CPUs by optimizing matrix operations.

### Installation

```bash
# Linux (recommended)
pip install intel-extension-for-pytorch

# Windows/macOS support may vary
```

### Enable IPEX

```bash
export USE_IPEX=true
export TTS_BACKEND=pytorch
export TTS_DEVICE=cpu
```

### Performance Impact

* **Expected speedup**: 20-40% on Intel CPUs
* **Best for**: Matmul/linear operations (the model's main compute)
* **Compatibility**: Works with PyTorch CPU backend automatically

## Experimental: OpenVINO Backend

⚠️ **Warning**: This backend is experimental and not fully implemented. Use PyTorch CPU for reliable inference.

### Why OpenVINO is Experimental

Qwen3-TTS includes components that may not export cleanly:
- Codec/tokenizer decode to waveform
- Generation loop with dynamic behavior
- Custom audio processing

**You might end up accelerating only part of the pipeline.**

### Requirements

1. Export the Qwen3-TTS model to OpenVINO IR format
2. Place `model.xml` and `model.bin` in the model directory
3. Set up the OpenVINO backend

### Setup (if you have a converted model)

```bash
export TTS_BACKEND=openvino
export OV_DEVICE=CPU                 # CPU, GPU, or AUTO
export OV_MODEL_DIR=./.ov_models     # Directory with model.xml/model.bin
export OV_CACHE_DIR=./.ov_cache      # Compilation cache
```

### Model Export (Advanced)

If you want to attempt OpenVINO export:

```python
# This is a conceptual example - actual export may require significant work
from optimum.intel import OVModelForCausalLM

# Export only the text/token model part
# Keep audio decode in PyTorch
# This is NOT a working recipe for Qwen3-TTS - it's a starting point
```

**Recommendation**: Don't spend time fighting export issues. Use PyTorch CPU + IPEX instead.

## Performance Tuning

### Threading Configuration

The most important tuning for CPU inference is thread configuration:

```bash
# For i5-1240P (4 P-cores + 8 E-cores = 12 cores)
export CPU_THREADS=12                # Total cores
export CPU_INTEROP=2                 # Keep low (1-2)

# For other CPUs, use physical core count
# Check with: lscpu | grep "^CPU(s):"
```

### Model Selection

Choose a model based on your CPU performance:

| Model | Parameters | Speed on CPU | Quality | Best For |
|-------|-----------|--------------|---------|----------|
| 0.6B-Base | 600M | ⚡⚡⚡ Fast | ⭐⭐⭐ Good | **CPU inference** |
| 1.7B-Base | 1.7B | ⚡⚡ Moderate | ⭐⭐⭐⭐ Excellent | GPU/fast CPU |
| 1.7B-CustomVoice | 1.7B | ⚡⚡ Moderate | ⭐⭐⭐⭐ Excellent | GPU/fast CPU |

**Recommendation**: Use `Qwen3-TTS-12Hz-0.6B-Base` for CPU inference.

### Attention Implementation

```bash
# Recommended for CPU (in order of preference)
export TTS_ATTN=sdpa     # Best for CPU (PyTorch native, optimized)
export TTS_ATTN=eager    # Fallback if sdpa has issues

# NOT recommended for CPU
export TTS_ATTN=flash_attention_2  # GPU-only, will auto-fallback to sdpa
```

### Data Type Selection

```bash
# Recommended for CPU
export TTS_DTYPE=float32   # Most stable and often fastest on CPU

# NOT recommended for CPU
export TTS_DTYPE=float16   # May have precision issues on CPU
export TTS_DTYPE=bfloat16  # GPU-optimized, not recommended for CPU
```

## Configuration Examples

### Example 1: i5-1240P (Default Recommended)

```bash
export TTS_BACKEND=pytorch
export TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-0.6B-Base
export TTS_DEVICE=cpu
export TTS_DTYPE=float32
export TTS_ATTN=sdpa
export CPU_THREADS=12
export CPU_INTEROP=2
```

### Example 2: i5-1240P with IPEX (Maximum Performance)

```bash
export TTS_BACKEND=pytorch
export TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-0.6B-Base
export TTS_DEVICE=cpu
export TTS_DTYPE=float32
export TTS_ATTN=sdpa
export CPU_THREADS=12
export CPU_INTEROP=2
export USE_IPEX=true
```

### Example 3: Generic CPU (4 cores)

```bash
export TTS_BACKEND=pytorch
export TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-0.6B-Base
export TTS_DEVICE=cpu
export TTS_DTYPE=float32
export TTS_ATTN=sdpa
export CPU_THREADS=4
export CPU_INTEROP=1
```

## Troubleshooting

### Issue: Slow inference

**Solutions:**
1. Use smaller model: `Qwen3-TTS-12Hz-0.6B-Base`
2. Optimize thread count: Match your CPU core count
3. Enable IPEX: `USE_IPEX=true` (Intel CPUs only)
4. Check CPU load: Ensure other processes aren't competing

### Issue: Out of memory

**Solutions:**
1. Use smaller model: 0.6B instead of 1.7B
2. Close other applications
3. Reduce batch size (if processing multiple requests)

### Issue: Thread count warnings

If you see "cannot set number of interop threads" warnings:
- **This is non-critical** - The backend will use existing thread settings
- Usually happens when creating multiple backend instances in tests
- In production (single backend), this won't occur

### Issue: IPEX not found

**Solutions:**
```bash
# Install IPEX
pip install intel-extension-for-pytorch

# Verify installation
python -c "import intel_extension_for_pytorch as ipex; print(ipex.__version__)"
```

### Issue: OpenVINO model not found

**Expected behavior:**
```
RuntimeError: OpenVINO IR model not found at ./.ov_models/model.xml

To use the OpenVINO backend, you need to:
1. Export the Qwen3-TTS model to OpenVINO IR format
2. Place the model.xml and model.bin files in ./.ov_models
```

**Solution:**
- OpenVINO backend requires manual model export
- For reliable CPU inference, use `TTS_BACKEND=pytorch` instead

## Performance Expectations

### i5-1240P (12 threads, 0.6B model, float32)

| Configuration | RTF* | First Request | Subsequent Requests |
|--------------|------|---------------|---------------------|
| PyTorch CPU | ~2.5-3.0 | ~30-45s (model load) | ~2-3s per request |
| PyTorch + IPEX | ~2.0-2.5 | ~30-45s (model load) | ~1.5-2.5s per request |

*RTF = Real-Time Factor (lower is better, <1.0 means faster than real-time)

### Notes on Performance

- **First request is slow**: Model loading takes 30-45 seconds
- **Warmup recommended**: Set `TTS_WARMUP_ON_START=true` for production
- **CPU inference is slower than GPU**: This is expected, CPU is ~10-50x slower
- **Quality is identical**: CPU and GPU produce the same audio quality

## Best Practices

### For Development

```bash
export TTS_BACKEND=pytorch
export TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-0.6B-Base
export TTS_DEVICE=cpu
export TTS_DTYPE=float32
export TTS_WARMUP_ON_START=false  # Skip warmup for faster startup
```

### For Production

```bash
export TTS_BACKEND=pytorch
export TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-0.6B-Base
export TTS_DEVICE=cpu
export TTS_DTYPE=float32
export TTS_WARMUP_ON_START=true   # Warmup on startup
export USE_IPEX=true              # Enable IPEX if available
export CPU_THREADS=12             # Match your CPU
export CPU_INTEROP=2
```

## Summary

**Recommended approach for i5-1240P:**

1. ✅ **Use PyTorch CPU backend** - Simple, reliable, fast enough
2. ✅ **Enable IPEX** (optional) - 20-40% speedup on Intel CPUs
3. ✅ **Use 0.6B model** - Faster inference on CPU
4. ✅ **Tune thread count** - Match your CPU cores
5. ❌ **Skip OpenVINO** (for now) - Experimental, not worth the complexity

**Expected result**: Reliable CPU inference with reasonable performance for development and low-volume production use.

For high-throughput production, consider GPU deployment or a faster CPU.
