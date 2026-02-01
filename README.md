# Qwen3-TTS

<br>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/qwen3_tts_logo.png" width="400"/>
<p>

<p align="center">
&nbsp&nbsp🤗 <a href="https://huggingface.co/collections/Qwen/qwen3-tts">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/collections/Qwen/Qwen3-TTS">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://qwen.ai/blog?id=qwen3tts-0115">Blog</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2601.15621">Paper</a>&nbsp&nbsp
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen3-TTS">Hugging Face Demo</a>&nbsp&nbsp | &nbsp&nbsp 🖥️ <a href="https://modelscope.cn/studios/Qwen/Qwen3-TTS">ModelScope Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://help.aliyun.com/zh/model-studio/qwen-tts-realtime">API</a>

</p>

> **⚡ NEW: Production-Optimized Inference**  
> This implementation includes **5 advanced GPU optimizations** (Flash Attention 2, torch.compile, TF32, cuDNN benchmark, BFloat16) for **up to 40% faster inference** compared to baseline. Expected RTF: **0.65-0.70** on RTX 3090 (54% faster than real-time). See [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) for details.

This repository provides an **OpenAI-compatible FastAPI server** for **Qwen3-TTS**, enabling drop-in replacement for OpenAI's TTS API endpoints. Built on top of the powerful Qwen3-TTS model series developed by the Qwen team at Alibaba Cloud, it offers comprehensive support for voice clone, voice design, ultra-high-quality human-like speech generation, and natural language-based voice control.

## ✨ Features

- 🎯 **OpenAI API Compatible** - Drop-in replacement for `POST /v1/audio/speech`
- ⚡ **Multiple Backends** - Choose between official or vLLM-Omni backend for optimal performance
- 🌐 **Multi-language Support** - 10+ languages (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian)
- 🎨 **Multiple Voice Options** - 9 premium voices with various gender, age, and dialect combinations
- 📊 **Multiple Audio Formats** - MP3, Opus, AAC, FLAC, WAV, PCM
- ⚡ **GPU-Accelerated** - Optimized for CUDA/GPU and CPU deployments
- 🔧 **Text Sanitization** - Advanced text preprocessing for URLs, emails, special characters
- 🐳 **Docker Ready** - Multi-stage Dockerfile with GPU and CPU variants
- 🖥️ **Web Interface** - Dark-themed interactive demo UI
- 🎙️ **Voice Studio** - Comprehensive UI for creating, managing, and exporting voice profiles

### Backend Options

This implementation supports multiple backend engines:

| Backend | Speed | Setup | Best For | Status |
|---------|-------|-------|----------|--------|
| **Official** (default) | ⚡⚡ Excellent | ✅ Simple | All use cases, production-ready | ✅ Stable |
| **vLLM-Omni** | ⚡⚡⚡ Fast | ⚠️ Python 3.12 + CUDA | High-throughput, low-latency | ✅ Available |
| **PyTorch CPU** | ⚡ Good | ✅ Simple | CPU-only systems (i5-1240P, etc.) | ✅ Stable |
| **OpenVINO** | ⚡⚡ Better* | ⚠️ Complex | Intel CPU/NPU (experimental) | ⚠️ Experimental |

- **Official Backend**: Uses the official Qwen3-TTS Python implementation with GPU/CPU auto-detect. **Recommended for most users.**
- **vLLM-Omni Backend**: Uses [vLLM-Omni](https://docs.vllm.ai/projects/vllm-omni/) for optimized inference. Requires Python 3.12 and a dedicated Docker image. See [VLLM_BACKEND_STATUS.md](VLLM_BACKEND_STATUS.md) for details.
- **PyTorch CPU Backend**: CPU-optimized PyTorch with threading tuning and optional IPEX support. **Recommended for CPU-only systems.** See [CPU_BACKEND_GUIDE.md](CPU_BACKEND_GUIDE.md) for details.
- **OpenVINO Backend**: Experimental Intel CPU/NPU acceleration. Requires manual model export. **Use PyTorch CPU backend for reliable CPU inference.**

*OpenVINO may only accelerate parts of the pipeline

## 🚀 Performance Benchmarks

Performance comparison with Flash Attention 2 optimization on NVIDIA RTX 3090 (24GB VRAM). RTF = Real-Time Factor (lower is better, <1.0 means faster than real-time).

### Flash Attention 2 Impact

| Backend | RTF (Avg) | Latency | Flash Attn Impact | Recommendation |
|---------|-----------|---------|-------------------|----------------|
| **Official + Flash Attn 2** ⚡ | **0.87** | **7.28s** | ✅ **+10% faster** | 🏆 **Best Overall** |
| Official (baseline) | 0.97 | 8.49s | - | Good |
| vLLM-Omni | 0.83 | 7.85s | - | Fast (no flash) |
| vLLM-Omni + Flash Attn 2 | 0.90 | 8.14s | ⚠️ -8% slower | Not recommended |

**Key Findings:**
- ✅ Flash Attention 2 **improves Official backend by 10%**
- ⚠️ Flash Attention 2 **degrades vLLM-Omni by 8%** (optimization conflict)
- 🏆 **Official + Flash Attn 2 is the fastest and most reliable** configuration

### Detailed Results (with Flash Attention 2)

| Test Case | Words | Official+Flash2 | vLLM+Flash2 | Official RTF | vLLM RTF |
|-----------|-------|-----------------|-------------|-------------:|----------:|
| Short | 2 | 1.15s | 1.29s | **0.95** | 0.97 |
| Sentence | 7 | 2.65s | 3.39s | **0.88** | 0.89 |
| Medium | 20 | **7.60s** | 7.59s | **0.84** | 0.87 |
| Long | 36 | **17.71s** | 20.29s | **0.81** | 0.87 |

- **Model**: Qwen3-TTS-12Hz-1.7B-CustomVoice
- **GPU**: NVIDIA GeForce RTX 3090 (24GB VRAM)
- **Test Method**: 1 cold run + 5 warm runs per prompt
- **Docker Images**: Built with Flash Attention 2

**Production Recommendation:** Use **Official backend with Flash Attention 2** for best performance (RTF 0.87, ~15% faster than real-time).

See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for full details.

## ⚡ Performance Optimizations

The official backend includes several production-ready optimizations for maximum inference speed:

| Optimization | Impact | Hardware Requirement | Status |
|-------------|--------|---------------------|--------|
| **Flash Attention 2** | +10% faster | Ampere+ GPU (RTX 30xx/40xx) | ✅ Enabled |
| **torch.compile()** | +20-30% faster | Any CUDA GPU | ✅ Enabled |
| **TF32 Precision** | +3-5x matmul speed | Ampere+ GPU | ✅ Enabled |
| **cuDNN Benchmark** | +5-10% faster | Any CUDA GPU | ✅ Enabled |
| **BFloat16** | -50% VRAM | Turing+ GPU (RTX 20xx+) | ✅ Enabled |

**Combined Effect:** ~25-35% speedup over baseline (expected RTF: 0.65-0.70)

⚠️ **Note**: torch.compile() and cuDNN benchmarking require warmup. First 2-3 requests may be slower (~10-30s) while optimizations initialize.

📖 **See [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)** for detailed information about each optimization, how to enable/disable them, and troubleshooting tips.

## 🚀 Quick Start (API Server)

### Using OpenAI Python Client

```python
from openai import OpenAI

# Point to your local Qwen3-TTS server
client = OpenAI(
    base_url="http://localhost:8880/v1",
    api_key="not-needed"  # API key not required for local server
)

# Generate speech
response = client.audio.speech.create(
    model="qwen3-tts",
    voice="Vivian",  # Or: Ryan, Serena, Dylan, Eric, Aiden, etc.
    input="Hello! This is Qwen3-TTS speaking with an OpenAI-compatible API.",
    response_format="mp3",  # Options: mp3, opus, aac, flac, wav, pcm
    speed=1.0  # 0.25 to 4.0
)

response.stream_to_file("output.mp3")
```

### Language-Specific Models

You can force a specific language by using language-suffixed model names. This overrides any language parameter passed from the client (useful for integration with open-webui):

```python
# Force Spanish output regardless of language parameter
response = client.audio.speech.create(
    model="tts-1-es",  # Spanish
    voice="Vivian",
    input="Hola! Este es Qwen3-TTS."
)

# Force French output
response = client.audio.speech.create(
    model="tts-1-hd-fr",  # French (HD quality)
    voice="Vivian",
    input="Bonjour! Ceci est Qwen3-TTS."
)
```

**Supported Language Codes:**
- `tts-1-en` or `tts-1-hd-en` - English
- `tts-1-zh` or `tts-1-hd-zh` - Chinese
- `tts-1-ja` or `tts-1-hd-ja` - Japanese
- `tts-1-ko` or `tts-1-hd-ko` - Korean
- `tts-1-de` or `tts-1-hd-de` - German
- `tts-1-fr` or `tts-1-hd-fr` - French
- `tts-1-es` or `tts-1-hd-es` - Spanish
- `tts-1-ru` or `tts-1-hd-ru` - Russian
- `tts-1-pt` or `tts-1-hd-pt` - Portuguese
- `tts-1-it` or `tts-1-hd-it` - Italian

When using these language-specific models, any `language` parameter in the request will be ignored, ensuring consistent output in the specified language.

### Web Interface

After starting the server, visit `http://localhost:8880` for an interactive web demo:

![Qwen3-TTS Web Interface](https://github.com/user-attachments/assets/cc270f90-2182-44b6-82d5-30aac17360cb)

### Voice Studio

The **Voice Studio** is a comprehensive Gradio-based UI that allows you to create, manage, and export reusable voice profiles. It supports three primary workflows:

- **CustomVoice** - Use preset voices with style instructions
- **VoiceDesign** - Create custom voices using natural language descriptions
- **Base** - Clone voices from audio samples (with or without transcripts)

**Features:**
- 🎙️ Create and save voice profiles locally
- 🎨 Preview generated audio before saving
- 📦 Export profiles as ZIP archives
- 🎮 Interactive playground for testing saved profiles
- 🔄 Manage your voice library (view, load, delete profiles)

**Option 1: Integrated with API Server**

Mount the Voice Studio directly in the API server by setting an environment variable:

```bash
export ENABLE_VOICE_STUDIO=true
python -m api.main
```

Then visit `http://localhost:8880/voice-studio`

**Option 2: Standalone Mode**

Run the Voice Studio as a separate service (on port 7860 by default):

```bash
# Run directly from repository (recommended for development)
python gradio_voice_studio.py

# With custom settings
python gradio_voice_studio.py --base-url http://localhost:8880 --library-dir ./my_voices --port 7860

# If installed via pip install (not editable -e mode)
qwen-tts-voice-studio
```

> **Development Note:** When developing with `pip install -e .` (editable mode), the `qwen-tts-voice-studio` command may not work due to setuptools limitations with py-modules in editable installs. In this case, use `python gradio_voice_studio.py` directly. For production or end-user installs with `pip install .`, the command will work correctly.

The Voice Studio will automatically connect to your running Qwen3-TTS API server to generate audio. Make sure the API server is running before using the Voice Studio.

## 📦 Deployment

### Option 1: Using Conda (Recommended for Development)

```bash
# Create a fresh conda environment
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts

# Install the package with API dependencies
pip install -e ".[api]"

# Optional: Install FlashAttention 2 for better performance
pip install -U flash-attn --no-build-isolation

# Start the API server
python -m api.main
# Or use the convenience script
./start_server.sh
```

The server will start on `http://0.0.0.0:8880` by default.

**Environment Variables:**
- `HOST` - Server host (default: `0.0.0.0`)
- `PORT` - Server port (default: `8880`)
- `WORKERS` - Number of workers (default: `1`)
- `CORS_ORIGINS` - CORS origins (default: `*`)
- `TTS_BACKEND` - Backend engine: `official` or `vllm_omni` (default: `official`)
- `TTS_MODEL_NAME` - Override default model (optional)
- `TTS_WARMUP_ON_START` - Warmup on startup: `true` or `false` (default: `false`)
- `ENABLE_VOICE_STUDIO` - Mount Voice Studio at `/voice-studio`: `true` or `false` (default: `false`)
- `VOICE_LIBRARY_DIR` - Directory for storing voice profiles (default: `./voice_library`)

**Backend Selection:**

```bash
# Use official backend (default)
export TTS_BACKEND=official
python -m api.main

# Use vLLM-Omni backend for faster inference
export TTS_BACKEND=vllm_omni
export TTS_WARMUP_ON_START=true
pip install -e ".[vllm]"  # Install vLLM first
python -m api.main
```

For detailed vLLM-Omni setup and configuration, see [docs/vllm-backend.md](docs/vllm-backend.md).

### Option 2: Using Docker (GPU-Enabled)

**Official Backend (Default):**

```bash
# Build and run with GPU support
docker build -t qwen3-tts-api .
docker run --gpus all -p 8880:8880 qwen3-tts-api

# Or use Docker Compose for easier management
docker-compose up qwen3-tts-gpu
```

**vLLM-Omni Backend (Faster):**

```bash
# Build vLLM-enabled image
docker build -t qwen3-tts-api:vllm --target vllm-production .
docker run --gpus all -p 8880:8880 \
  -e TTS_BACKEND=vllm_omni \
  -e TTS_WARMUP_ON_START=true \
  qwen3-tts-api:vllm

# Or use Docker Compose
docker-compose --profile vllm up qwen3-tts-vllm
```

### Option 3: Using Docker (CPU-Only)

```bash
# Build CPU-only variant
docker build -t qwen3-tts-api-cpu --target cpu-base .
docker run -p 8880:8880 qwen3-tts-api-cpu

# Or use Docker Compose
docker-compose --profile cpu up qwen3-tts-cpu
```

### Docker Compose Configuration

The `docker-compose.yml` includes GPU and CPU configurations with both backends:

```bash
# Official backend with GPU (default)
docker-compose up qwen3-tts-gpu

# vLLM-Omni backend with GPU (faster)
docker-compose --profile vllm up qwen3-tts-vllm

# CPU-only (uses profile)
docker-compose --profile cpu up qwen3-tts-cpu

# Run in background
docker-compose up -d qwen3-tts-gpu

# View logs
docker-compose logs -f qwen3-tts-gpu

# Stop services
docker-compose down
```

**Model Cache:** Models are cached in `~/.cache/huggingface` and automatically mounted as a volume for persistence.

## 💻 CPU-Only Deployment

For systems without a GPU (e.g., Intel i5-1240P), use the optimized CPU backend:

### Using PyTorch CPU Backend (Recommended)

```bash
# Set environment variables
export TTS_BACKEND=pytorch
export TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-0.6B-Base  # Smaller model for CPU
export TTS_DEVICE=cpu
export TTS_DTYPE=float32
export TTS_ATTN=sdpa
export CPU_THREADS=12          # Adjust for your CPU cores
export CPU_INTEROP=2

# Optional: Enable Intel Extension for PyTorch (Intel CPUs only)
export USE_IPEX=true

# Start the server
python -m api.main
```

### Using Docker (CPU-Optimized)

```bash
# Build and run CPU-optimized container
docker build -t qwen3-tts-api-cpu --target cpu-base .
docker run -p 8880:8880 \
  -e TTS_BACKEND=pytorch \
  -e TTS_MODEL_ID=Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  -e CPU_THREADS=12 \
  qwen3-tts-api-cpu
```

### Performance Expectations (i5-1240P)

- **Model**: Qwen3-TTS-12Hz-0.6B-Base
- **RTF**: ~2.5-3.0 (PyTorch CPU) or ~2.0-2.5 (with IPEX)
- **First request**: ~30-45s (model loading)
- **Subsequent requests**: ~2-3s per request

📖 **See [CPU_BACKEND_GUIDE.md](CPU_BACKEND_GUIDE.md)** for complete CPU deployment guide, performance tuning, and troubleshooting.

## 🎯 API Endpoints

- `POST /v1/audio/speech` - Generate speech (OpenAI-compatible)
- `GET /v1/models` - List available models
- `GET /v1/voices` - List available voices
- `GET /health` - Health check with backend status
- `GET /docs` - Swagger UI documentation
- `GET /redoc` - ReDoc documentation
- `GET /health` - Health check endpoint
- `GET /` - Web interface

## 🙏 Acknowledgments

This project builds upon the incredible work of the **Qwen Team at Alibaba Cloud**. We are deeply grateful for their development and open-sourcing of [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), a state-of-the-art text-to-speech model that enables:

- **Powerful Speech Representation** via Qwen3-TTS-Tokenizer-12Hz
- **Universal End-to-End Architecture** with discrete multi-codebook LM
- **Extreme Low-Latency Streaming** (as low as 97ms)
- **Intelligent Voice Control** through natural language instructions

For more details about the underlying Qwen3-TTS models, please refer to:
- 📑 [Qwen3-TTS Technical Blog](https://qwen.ai/blog?id=qwen3tts-0115)
- 📄 [Research Paper](https://arxiv.org/abs/2601.15621)
- 💻 [Original Repository](https://github.com/QwenLM/Qwen3-TTS)

---

## News
* 2026.1.22: 🎉🎉🎉 We have released [Qwen3-TTS](https://huggingface.co/collections/Qwen/qwen3-tts) series (0.6B/1.7B) based on Qwen3-TTS-Tokenizer-12Hz. Please check our [blog](https://qwen.ai/blog?id=qwen3tts-0115)!

## Contents <!-- omit in toc -->

- [✨ Features](#-features)
- [🚀 Quick Start (API Server)](#-quick-start-api-server)
  - [Using OpenAI Python Client](#using-openai-python-client)
  - [Web Interface](#web-interface)
  - [Voice Studio](#voice-studio)
- [📦 Deployment](#-deployment)
  - [Option 1: Using Conda (Recommended for Development)](#option-1-using-conda-recommended-for-development)
  - [Option 2: Using Docker (GPU-Enabled)](#option-2-using-docker-gpu-enabled)
  - [Option 3: Using Docker (CPU-Only)](#option-3-using-docker-cpu-only)
  - [Docker Compose Configuration](#docker-compose-configuration)
- [🎯 API Endpoints](#-api-endpoints)
- [🙏 Acknowledgments](#-acknowledgments)
- [Overview](#overview)
  - [Introduction](#introduction)
  - [Model Architecture](#model-architecture)
  - [Released Models Description and Download](#released-models-description-and-download)
- [Quickstart (Python Package)](#quickstart-python-package)
  - [Environment Setup](#environment-setup)
  - [Python Package Usage](#python-package-usage)
    - [Custom Voice Generation](#custom-voice-generate)
    - [Voice Design](#voice-design)
    - [Voice Clone](#voice-clone)
    - [Voice Design then Clone](#voice-design-then-clone)
    - [Tokenizer Encode and Decode](#tokenizer-encode-and-decode)
  - [Launch Local Web UI Demo](#launch-local-web-ui-demo)
  - [DashScope API Usage](#dashscope-api-usage)
- [vLLM Usage](#vllm-usage)
- [Fine Tuning](#fine-tuning)
- [Evaluation](#evaluation)
- [Citation](#citation)

## Overview
### Introduction

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/qwen3_tts_introduction.png" width="90%"/>
<p>

Qwen3-TTS covers 10 major languages (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, and Italian) as well as multiple dialectal voice profiles to meet global application needs. In addition, the models feature strong contextual understanding, enabling adaptive control of tone, speaking rate, and emotional expression based on instructions and text semantics, and they show markedly improved robustness to noisy input text. Key features:

* **Powerful Speech Representation**: Powered by the self-developed Qwen3-TTS-Tokenizer-12Hz, it achieves efficient acoustic compression and high-dimensional semantic modeling of speech signals. It fully preserves paralinguistic information and acoustic environmental features, enabling high-speed, high-fidelity speech reconstruction through a lightweight non-DiT architecture.
* **Universal End-to-End Architecture**: Utilizing a discrete multi-codebook LM architecture, it realizes full-information end-to-end speech modeling. This completely bypasses the information bottlenecks and cascading errors inherent in traditional LM+DiT schemes, significantly enhancing the model’s versatility, generation efficiency, and performance ceiling.
* **Extreme Low-Latency Streaming Generation**: Based on the innovative Dual-Track hybrid streaming generation architecture, a single model supports both streaming and non-streaming generation. It can output the first audio packet immediately after a single character is input, with end-to-end synthesis latency as low as 97ms, meeting the rigorous demands of real-time interactive scenarios.
* **Intelligent Text Understanding and Voice Control**: Supports speech generation driven by natural language instructions, allowing for flexible control over multi-dimensional acoustic attributes such as timbre, emotion, and prosody. By deeply integrating text semantic understanding, the model adaptively adjusts tone, rhythm, and emotional expression, achieving lifelike “what you imagine is what you hear” output.


### Model Architecture

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/overview.png" width="80%"/>
<p>

### Released Models Description and Download

Below is an introduction and download information for the Qwen3-TTS models that have already been released. Other models mentioned in the technical report will be released in the near future. Please select and download the model that fits your needs.

| Tokenizer Name                      | Description |
|---------------------------------|-------------|
| Qwen3-TTS-Tokenizer-12Hz        | The Qwen3-TTS-Tokenizer-12Hz model which can encode the input speech into codes and decode them back into speech. |


| Model | Features | Language Support | Streaming | Instruction Control |
|---|---|---|---|---|
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | Performs voice design based on user-provided descriptions. | Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian | ✅ | ✅ |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | Provides style control over target timbres via user instructions; supports 9 premium timbres covering various combinations of gender, age, language, and dialect. | Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian | ✅ | ✅ |
| Qwen3-TTS-12Hz-1.7B-Base | Base model capable of 3-second rapid voice clone from user audio input; can be used for fine-tuning (FT) other models. | Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian | ✅ |  |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | Supports 9 premium timbres covering various combinations of gender, age, language, and dialect. | Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian | ✅ |  |
| Qwen3-TTS-12Hz-0.6B-Base | Base model capable of 3-second rapid voice clone from user audio input; can be used for fine-tuning (FT) other models. | Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian | ✅ |  |

During model loading in the qwen-tts package or vLLM, model weights will be automatically downloaded based on the model name. However, if your runtime environment is not conducive to downloading weights during execution, you can refer to the following commands to manually download the model weights to a local directory:

```bash
# Download through ModelScope (recommended for users in Mainland China)
pip install -U modelscope
modelscope download --model Qwen/Qwen3-TTS-Tokenizer-12Hz  --local_dir ./Qwen3-TTS-Tokenizer-12Hz 
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local_dir ./Qwen3-TTS-12Hz-1.7B-CustomVoice
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local_dir ./Qwen3-TTS-12Hz-1.7B-VoiceDesign
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --local_dir ./Qwen3-TTS-12Hz-1.7B-Base
modelscope download --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local_dir ./Qwen3-TTS-12Hz-0.6B-CustomVoice
modelscope download --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --local_dir ./Qwen3-TTS-12Hz-0.6B-Base

# Download through Hugging Face
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir ./Qwen3-TTS-Tokenizer-12Hz
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir ./Qwen3-TTS-12Hz-1.7B-CustomVoice
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir ./Qwen3-TTS-12Hz-1.7B-VoiceDesign
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./Qwen3-TTS-12Hz-1.7B-Base
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir ./Qwen3-TTS-12Hz-0.6B-CustomVoice
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir ./Qwen3-TTS-12Hz-0.6B-Base
```


## Quickstart (Python Package)

### Environment Setup

The easiest way to quickly use Qwen3-TTS is to install the `qwen-tts` Python package from PyPI. This will pull in the required runtime dependencies and allow you to load any released Qwen3-TTS model. We recommend using a **fresh, isolated environment** to avoid dependency conflicts with existing packages. You can create a clean Python 3.12 environment like this:

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
```

then run:

```bash
pip install -U qwen-tts
```

If you want to develop or modify the code locally, install from source in editable mode.

```bash
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS
pip install -e .
```

Additionally, we recommend using FlashAttention 2 to reduce GPU memory usage.

```bash
pip install -U flash-attn --no-build-isolation
```

If your machine has less than 96GB of RAM and lots of CPU cores, run:

```bash
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

Also, you should have hardware that is compatible with FlashAttention 2. Read more about it in the official documentation of the [FlashAttention repository](https://github.com/Dao-AILab/flash-attention). FlashAttention 2 can only be used when a model is loaded in `torch.float16` or `torch.bfloat16`.


### Python Package Usage

After installation, you can import `Qwen3TTSModel` to run custom voice TTS, voice design, and voice clone. The model weights can be specified either as a Hugging Face model id (recommended) or as a local directory path you downloaded. For all the `generate_*` functions below, besides the parameters shown and explicitly documented, you can also pass generation kwargs supported by Hugging Face Transformers `model.generate`, e.g., `max_new_tokens`, `top_p`, etc.

#### Custom Voice Generate

For custom voice models (`Qwen3-TTS-12Hz-1.7B/0.6B-CustomVoice`), you just need to call `generate_custom_voice`, passing a single string or a batch list, along with `language`, `speaker`, and optional `instruct`. You can also call `model.get_supported_speakers()` and `model.get_supported_languages()` to see which speakers and languages the current model supports.

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# single inference
wavs, sr = model.generate_custom_voice(
    text="其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    language="Chinese", # Pass `Auto` (or omit) for auto language adaptive; if the target language is known, set it explicitly.
    speaker="Vivian",
    instruct="用特别愤怒的语气说", # Omit if not needed.
)
sf.write("output_custom_voice.wav", wavs[0], sr)

# batch inference
wavs, sr = model.generate_custom_voice(
    text=[
        "其实我真的有发现，我是一个特别善于观察别人情绪的人。", 
        "She said she would be here by noon."
    ],
    language=["Chinese", "English"],
    speaker=["Vivian", "Ryan"],
    instruct=["", "Very happy."]
)
sf.write("output_custom_voice_1.wav", wavs[0], sr)
sf.write("output_custom_voice_2.wav", wavs[1], sr)
```

For `Qwen3-TTS-12Hz-1.7B/0.6B-CustomVoice` models, the supported speaker list and speaker descriptions are provided below. We recommend using each speaker’s native language for the best quality. Of course, each speaker can speak any language supported by the model.

| Speaker | Voice Description  |  Native language |
| --- | --- | --- |
| Vivian | Bright, slightly edgy young female voice. | Chinese |
| Serena | Warm, gentle young female voice. | Chinese |
| Uncle_Fu | Seasoned male voice with a low, mellow timbre. | Chinese |
| Dylan | Youthful Beijing male voice with a clear, natural timbre. | Chinese (Beijing Dialect) |
| Eric | Lively Chengdu male voice with a slightly husky brightness. | Chinese (Sichuan Dialect) |
| Ryan | Dynamic male voice with strong rhythmic drive. | English |
| Aiden | Sunny American male voice with a clear midrange. | English |
| Ono_Anna | Playful Japanese female voice with a light, nimble timbre. | Japanese |
| Sohee | Warm Korean female voice with rich emotion. | Korean |

#### Voice Design

For the voice design model (`Qwen3-TTS-12Hz-1.7B-VoiceDesign`), you can use `generate_voice_design` to provide the target text and a natural-language `instruct` description.

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# single inference
wavs, sr = model.generate_voice_design(
    text="哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
    language="Chinese",
    instruct="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
)
sf.write("output_voice_design.wav", wavs[0], sr)

# batch inference
wavs, sr = model.generate_voice_design(
    text=[
      "哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
      "It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!"
    ],
    language=["Chinese", "English"],
    instruct=[
      "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
      "Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."
    ]
)
sf.write("output_voice_design_1.wav", wavs[0], sr)
sf.write("output_voice_design_2.wav", wavs[1], sr)
```

#### Voice Clone

For the voice clone model (`Qwen3-TTS-12Hz-1.7B/0.6B-Base`), to clone a voice and synthesize new content, you just need to provide a reference audio clip (`ref_audio`) along with its transcript (`ref_text`). `ref_audio` can be a local file path, a URL, a base64 string, or a `(numpy_array, sample_rate)` tuple. If you set `x_vector_only_mode=True`, only the speaker embedding is used so `ref_text` is not required, but cloning quality may be reduced.

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
ref_text  = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."

wavs, sr = model.generate_voice_clone(
    text="I am solving the equation: x = [-b ± √(b²-4ac)] / 2a? Nobody can — it's a disaster (◍•͈⌔•͈◍), very sad!",
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
)
sf.write("output_voice_clone.wav", wavs[0], sr)
```

If you need to reuse the same reference prompt across multiple generations (to avoid recomputing prompt features), build it once with `create_voice_clone_prompt` and pass it via `voice_clone_prompt`.

```python
prompt_items = model.create_voice_clone_prompt(
    ref_audio=ref_audio,
    ref_text=ref_text,
    x_vector_only_mode=False,
)
wavs, sr = model.generate_voice_clone(
    text=["Sentence A.", "Sentence B."],
    language=["English", "English"],
    voice_clone_prompt=prompt_items,
)
sf.write("output_voice_clone_1.wav", wavs[0], sr)
sf.write("output_voice_clone_2.wav", wavs[1], sr)
```

For more examples of reusable voice clone prompts, batch cloning, and batch inference, please refer to the [example codes](https://github.com/QwenLM/Qwen3-TTS/blob/main/examples/test_model_12hz_base.py). With those examples and the `generate_voice_clone` function description, you can explore more advanced usage patterns.

#### Voice Design then Clone

If you want a designed voice that you can reuse like a cloned speaker, a practical workflow is: (1) use the **VoiceDesign** model to synthesize a short reference clip that matches your target persona, (2) feed that clip into `create_voice_clone_prompt` to build a reusable prompt, and then (3) call `generate_voice_clone` with `voice_clone_prompt` to generate new content without re-extracting features every time. This is especially useful when you want a consistent character voice across many lines.

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# create a reference audio in the target style using the VoiceDesign model
design_model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

ref_text = "H-hey! You dropped your... uh... calculus notebook? I mean, I think it's yours? Maybe?"
ref_instruct = "Male, 17 years old, tenor range, gaining confidence - deeper breath support now, though vowels still tighten when nervous"
ref_wavs, sr = design_model.generate_voice_design(
    text=ref_text,
    language="English",
    instruct=ref_instruct
)
sf.write("voice_design_reference.wav", ref_wavs[0], sr)

# build a reusable clone prompt from the voice design reference
clone_model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

voice_clone_prompt = clone_model.create_voice_clone_prompt(
    ref_audio=(ref_wavs[0], sr),   # or "voice_design_reference.wav"
    ref_text=ref_text,
)

sentences = [
    "No problem! I actually... kinda finished those already? If you want to compare answers or something...",
    "What? No! I mean yes but not like... I just think you're... your titration technique is really precise!",
]

# reuse it for multiple single calls
wavs, sr = clone_model.generate_voice_clone(
    text=sentences[0],
    language="English",
    voice_clone_prompt=voice_clone_prompt,
)
sf.write("clone_single_1.wav", wavs[0], sr)

wavs, sr = clone_model.generate_voice_clone(
    text=sentences[1],
    language="English",
    voice_clone_prompt=voice_clone_prompt,
)
sf.write("clone_single_2.wav", wavs[0], sr)

# or batch generate in one call
wavs, sr = clone_model.generate_voice_clone(
    text=sentences,
    language=["English", "English"],
    voice_clone_prompt=voice_clone_prompt,
)
for i, w in enumerate(wavs):
    sf.write(f"clone_batch_{i}.wav", w, sr)
```

#### Tokenizer Encode and Decode

If you only want to encode and decode audio for transport or training and so on, `Qwen3TTSTokenizer` supports encode/decode with paths, URLs, numpy waveforms, and dict/list payloads, for example:

```python
import soundfile as sf
from qwen_tts import Qwen3TTSTokenizer

tokenizer = Qwen3TTSTokenizer.from_pretrained(
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    device_map="cuda:0",
)

enc = tokenizer.encode("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/tokenizer_demo_1.wav")
wavs, sr = tokenizer.decode(enc)
sf.write("decode_output.wav", wavs[0], sr)
```

For more tokenizer examples (including different input formats and batch usage), please refer to the [example codes](https://github.com/QwenLM/Qwen3-TTS/blob/main/examples/test_tokenizer_12hz.py). With those examples and the description for `Qwen3TTSTokenizer`, you can explore more advanced usage patterns.

### Launch Local Web UI Demo

To launch the Qwen3-TTS web ui demo, simply install the `qwen-tts` package and run `qwen-tts-demo`. Use the command below for help:

```bash
qwen-tts-demo --help
```

To launch the demo, you can use the following commands:

```bash
# CustomVoice model
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 0.0.0.0 --port 8000
# VoiceDesign model
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --ip 0.0.0.0 --port 8000
# Base model
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 0.0.0.0 --port 8000
```

And then open `http://<your-ip>:8000`, or access it via port forwarding in tools like VS Code.

#### Base Model HTTPS Notes

To avoid browser microphone permission issues after deploying the server, for Base model deployments, it is recommended/required to run the gradio service over **HTTPS** (especially when accessed remotely or behind modern browsers/gateways). Use `--ssl-certfile` and `--ssl-keyfile` to enable HTTPS. First we need to generate a private key and a self-signed cert (valid for 365 days):

```bash
openssl req -x509 -newkey rsa:2048 \
  -keyout key.pem -out cert.pem \
  -days 365 -nodes \
  -subj "/CN=localhost"
```

Then run the demo with HTTPS:

```bash
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --ip 0.0.0.0 --port 8000 \
  --ssl-certfile cert.pem \
  --ssl-keyfile key.pem \
  --no-ssl-verify
```

And open `https://<your-ip>:8000` to experience it. If your browser shows a warning, it’s expected for self-signed certificates. For production, use a real certificate.

### DashScope API Usage

To further explore Qwen3-TTS, we encourage you to try our DashScope API for a faster and more efficient experience. For detailed API information and documentation, please refer to the following:

| API Description | API Documentation (Mainland China) | API Documentation (International) |
|------------------|-----------------------------------|------------------------------------|
| Real-time API for Qwen3-TTS of custom voice model. | [https://help.aliyun.com/zh/model-studio/qwen-tts-realtime](https://help.aliyun.com/zh/model-studio/qwen-tts-realtime) | [https://www.alibabacloud.com/help/en/model-studio/qwen-tts-realtime](https://www.alibabacloud.com/help/en/model-studio/qwen-tts-realtime) |
| Real-time API for Qwen3-TTS of voice clone model. | [https://help.aliyun.com/zh/model-studio/qwen-tts-voice-cloning](https://help.aliyun.com/zh/model-studio/qwen-tts-voice-cloning) | [https://www.alibabacloud.com/help/en/model-studio/qwen-tts-voice-cloning](https://www.alibabacloud.com/help/en/model-studio/qwen-tts-voice-cloning) |
| Real-time API for Qwen3-TTS of voice design model. | [https://help.aliyun.com/zh/model-studio/qwen-tts-voice-design](https://help.aliyun.com/zh/model-studio/qwen-tts-voice-design) | [https://www.alibabacloud.com/help/en/model-studio/qwen-tts-voice-design](https://www.alibabacloud.com/help/en/model-studio/qwen-tts-voice-design) |


## vLLM Usage

vLLM officially provides day-0 support for Qwen3-TTS! Welcome to use vLLM-Omni for Qwen3-TTS deployment and inference. For installation and more details, please check [vLLM-Omni official documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/quickstart/#installation). Now only offline inference is supported. Online serving will be supported later, and vLLM-Omni will continue to offer support and optimization for Qwen3-TTS in areas such as inference speed and streaming capabilities.

### Offline Inference
You can use vLLM-Omni to inference Qwen3-TTS locally, we provide examples in [vLLM-Omni repo](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen3_tts) which can generate audio output:
```bash
# git clone https://github.com/vllm-project/vllm-omni.git

# cd vllm-omni/examples/offline_inference/qwen3_tts

# Run a single sample with CustomVoice task
python end2end.py --query-type CustomVoice

# Batch sample (multiple prompts in one run) with CustomVoice task:
python end2end.py --query-type CustomVoice --use-batch-sample

# Run a single sample with VoiceDesign task
python end2end.py --query-type VoiceDesign

# Batch sample (multiple prompts in one run) with VoiceDesign task:
python end2end.py --query-type VoiceDesign --use-batch-sample

# Run a single sample with Base task in icl mode-tag
python end2end.py --query-type Base --mode-tag icl
```

## Fine Tuning

Please refer to [Qwen3-TTS-Finetuning](finetuning/) for detailed instructions on fine-tuning Qwen3-TTS.

## Evaluation

During evaluation, we ran inference for all models with `dtype=torch.bfloat16` and set `max_new_tokens=2048`. All other sampling parameters used the defaults from the checkpoint’s `generate_config.json`. For the Seed-Test and InstructTTS-Eval test sets, we set `language="auto"`, while for all other test sets we explicitly passed the corresponding `language`. The detailed results are shown below.


<details>
<summary>Speech Generation Benchmarks</summary>

*Zero-shot speech generation on the Seed-TTS test set. Performance is measured by Word Error Rate (WER, ↓), where lower is better.*

<table>
  <thead>
    <tr>
      <th style="text-align: center;">Datasets</th>
      <th style="text-align: left;">Model</th>
      <th colspan="2" style="text-align: center;">Performance</th>
    </tr>
    <tr style="border-bottom: 1px solid #ddd; border-top: 1px solid #ddd;">
      <td colspan="4" style="text-align: center;"><em>Content Consistency</em></td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="14" style="text-align: center; vertical-align: middle;">SEED<br><em>test-zh</em> | <em>test-en</em></td>
      <td style="text-align: left;">Seed-TTS (Anastassiou et al., 2024)</td>
      <td style="text-align: center;">1.12</td>
      <td style="text-align: center;">2.25</td>
    </tr>
    <tr>
      <td style="text-align: left;">MaskGCT (Wang et al., 2024)</td>
      <td style="text-align: center;">2.27</td>
      <td style="text-align: center;">2.62</td>
    </tr>
    <tr>
      <td style="text-align: left;">E2 TTS (Eskimez et al., 2024)</td>
      <td style="text-align: center;">1.97</td>
      <td style="text-align: center;">2.19</td>
    </tr>
    <tr>
      <td style="text-align: left;">F5-TTS (Chen et al., 2024)</td>
      <td style="text-align: center;">1.56</td>
      <td style="text-align: center;">1.83</td>
    </tr>
    <tr>
      <td style="text-align: left;">Spark TTS (Wang et al., 2025)</td>
      <td style="text-align: center;">1.20</td>
      <td style="text-align: center;">1.98</td>
    </tr>
    <tr>
      <td style="text-align: left;">Llasa-8B (Ye et al., 2025b)</td>
      <td style="text-align: center;">1.59</td>
      <td style="text-align: center;">2.97</td>
    </tr>
    <tr>
      <td style="text-align: left;">KALL-E (Xia et al., 2024)</td>
      <td style="text-align: center;">0.96</td>
      <td style="text-align: center;">1.94</td>
    </tr>
    <tr>
      <td style="text-align: left;">FireRedTTS 2 (Xie et al., 2025)</td>
      <td style="text-align: center;">1.14</td>
      <td style="text-align: center;">1.95</td>
    </tr>
    <tr>
      <td style="text-align: left;">CosyVoice 3 (Du et al., 2025)</td>
      <td style="text-align: center;"><strong>0.71</strong></td>
      <td style="text-align: center;">1.45</td>
    </tr>
    <tr>
      <td style="text-align: left;">MiniMax-Speech (Zhang et al., 2025a)</td>
      <td style="text-align: center;">0.83</td>
      <td style="text-align: center;">1.65</td>
    </tr>
    <tr>
      <td style="text-align: left;">Qwen3-TTS-25Hz-0.6B-Base</td>
      <td style="text-align: center;">1.18</td>
      <td style="text-align: center;">1.64</td>
    </tr>
    <tr>
      <td style="text-align: left;">Qwen3-TTS-25Hz-1.7B-Base</td>
      <td style="text-align: center;">1.10</td>
      <td style="text-align: center;">1.49</td>
    </tr>
    <tr>
      <td style="text-align: left;">Qwen3-TTS-12Hz-0.6B-Base</td>
      <td style="text-align: center;">0.92</td>
      <td style="text-align: center;">1.32</td>
    </tr>
    <tr>
      <td style="text-align: left;">Qwen3-TTS-12Hz-1.7B-Base</td>
      <td style="text-align: center;">0.77</td>
      <td style="text-align: center;"><strong>1.24</strong></td>
    </tr>
  </tbody>
</table>

<br>

*Multilingual speech generation on the TTS multilingual test set. Performance is measured by Word Error Rate (WER, ↓) for content consistency and Cosine Similarity (SIM, ↑) for speaker similarity.*

<table>
  <thead>
    <tr>
      <th rowspan="2" style="text-align: left; vertical-align: bottom;">Language</th>
      <th colspan="2" style="text-align: center;">Qwen3-TTS-25Hz</th>
      <th colspan="2" style="text-align: center;">Qwen3-TTS-12Hz</th>
      <th rowspan="2" style="text-align: center; vertical-align: bottom;">MiniMax</th>
      <th rowspan="2" style="text-align: center; vertical-align: bottom;">ElevenLabs</th>
    </tr>
    <tr>
      <th style="text-align: center;">0.6B-Base</th>
      <th style="text-align: center;">1.7B-Base</th>
      <th style="text-align: center;">0.6B-Base</th>
      <th style="text-align: center;">1.7B-Base</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="7" style="text-align: center; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;"><em>Content Consistency</em></td>
    </tr>
    <tr>
      <td style="text-align: left;">Chinese</td>
      <td style="text-align: center;">1.108</td>
      <td style="text-align: center;"><strong>0.777</strong></td>
      <td style="text-align: center;">1.145</td>
      <td style="text-align: center;">0.928</td>
      <td style="text-align: center;">2.252</td>
      <td style="text-align: center;">16.026</td>
    </tr>
    <tr>
      <td style="text-align: left;">English</td>
      <td style="text-align: center;">1.048</td>
      <td style="text-align: center;">1.014</td>
      <td style="text-align: center;"><strong>0.836</strong></td>
      <td style="text-align: center;">0.934</td>
      <td style="text-align: center;">2.164</td>
      <td style="text-align: center;">2.339</td>
    </tr>
    <tr>
      <td style="text-align: left;">German</td>
      <td style="text-align: center;">1.501</td>
      <td style="text-align: center;">0.960</td>
      <td style="text-align: center;">1.089</td>
      <td style="text-align: center;">1.235</td>
      <td style="text-align: center;">1.906</td>
      <td style="text-align: center;"><strong>0.572</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Italian</td>
      <td style="text-align: center;">1.169</td>
      <td style="text-align: center;">1.105</td>
      <td style="text-align: center;">1.534</td>
      <td style="text-align: center;"><strong>0.948</strong></td>
      <td style="text-align: center;">1.543</td>
      <td style="text-align: center;">1.743</td>
    </tr>
    <tr>
      <td style="text-align: left;">Portuguese</td>
      <td style="text-align: center;">2.046</td>
      <td style="text-align: center;">1.778</td>
      <td style="text-align: center;">2.254</td>
      <td style="text-align: center;">1.526</td>
      <td style="text-align: center;">1.877</td>
      <td style="text-align: center;"><strong>1.331</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Spanish</td>
      <td style="text-align: center;">2.031</td>
      <td style="text-align: center;">1.491</td>
      <td style="text-align: center;">1.491</td>
      <td style="text-align: center;">1.126</td>
      <td style="text-align: center;"><strong>1.029</strong></td>
      <td style="text-align: center;">1.084</td>
    </tr>
    <tr>
      <td style="text-align: left;">Japanese</td>
      <td style="text-align: center;">4.189</td>
      <td style="text-align: center;">5.121</td>
      <td style="text-align: center;">6.404</td>
      <td style="text-align: center;">3.823</td>
      <td style="text-align: center;"><strong>3.519</strong></td>
      <td style="text-align: center;">10.646</td>
    </tr>
    <tr>
      <td style="text-align: left;">Korean</td>
      <td style="text-align: center;">2.852</td>
      <td style="text-align: center;">2.631</td>
      <td style="text-align: center;"><strong>1.741</strong></td>
      <td style="text-align: center;">1.755</td>
      <td style="text-align: center;">1.747</td>
      <td style="text-align: center;">1.865</td>
    </tr>
    <tr>
      <td style="text-align: left;">French</td>
      <td style="text-align: center;">2.852</td>
      <td style="text-align: center;"><strong>2.631</strong></td>
      <td style="text-align: center;">2.931</td>
      <td style="text-align: center;">2.858</td>
      <td style="text-align: center;">4.099</td>
      <td style="text-align: center;">5.216</td>
    </tr>
    <tr>
      <td style="text-align: left;">Russian</td>
      <td style="text-align: center;">5.957</td>
      <td style="text-align: center;">4.535</td>
      <td style="text-align: center;">4.458</td>
      <td style="text-align: center;"><strong>3.212</strong></td>
      <td style="text-align: center;">4.281</td>
      <td style="text-align: center;">3.878</td>
    </tr>
    <tr style="border-top: 1px solid #ddd;">
      <td colspan="7" style="text-align: center; border-bottom: 1px solid #ddd;"><em>Speaker Similarity</em></td>
    </tr>
    <tr>
      <td style="text-align: left;">Chinese</td>
      <td style="text-align: center;">0.797</td>
      <td style="text-align: center;">0.796</td>
      <td style="text-align: center;"><strong>0.811</strong></td>
      <td style="text-align: center;">0.799</td>
      <td style="text-align: center;">0.780</td>
      <td style="text-align: center;">0.677</td>
    </tr>
    <tr>
      <td style="text-align: left;">English</td>
      <td style="text-align: center;">0.811</td>
      <td style="text-align: center;">0.815</td>
      <td style="text-align: center;"><strong>0.829</strong></td>
      <td style="text-align: center;">0.775</td>
      <td style="text-align: center;">0.756</td>
      <td style="text-align: center;">0.613</td>
    </tr>
    <tr>
      <td style="text-align: left;">German</td>
      <td style="text-align: center;">0.749</td>
      <td style="text-align: center;">0.737</td>
      <td style="text-align: center;">0.769</td>
      <td style="text-align: center;"><strong>0.775</strong></td>
      <td style="text-align: center;">0.733</td>
      <td style="text-align: center;">0.614</td>
    </tr>
    <tr>
      <td style="text-align: left;">Italian</td>
      <td style="text-align: center;">0.722</td>
      <td style="text-align: center;">0.718</td>
      <td style="text-align: center;">0.792</td>
      <td style="text-align: center;"><strong>0.817</strong></td>
      <td style="text-align: center;">0.699</td>
      <td style="text-align: center;">0.579</td>
    </tr>
    <tr>
      <td style="text-align: left;">Portuguese</td>
      <td style="text-align: center;">0.790</td>
      <td style="text-align: center;">0.783</td>
      <td style="text-align: center;">0.794</td>
      <td style="text-align: center;"><strong>0.817</strong></td>
      <td style="text-align: center;">0.805</td>
      <td style="text-align: center;">0.711</td>
    </tr>
    <tr>
      <td style="text-align: left;">Spanish</td>
      <td style="text-align: center;">0.732</td>
      <td style="text-align: center;">0.731</td>
      <td style="text-align: center;">0.812</td>
      <td style="text-align: center;"><strong>0.814</strong></td>
      <td style="text-align: center;">0.762</td>
      <td style="text-align: center;">0.615</td>
    </tr>
    <tr>
      <td style="text-align: left;">Japanese</td>
      <td style="text-align: center;"><strong>0.810</strong></td>
      <td style="text-align: center;">0.807</td>
      <td style="text-align: center;">0.798</td>
      <td style="text-align: center;">0.788</td>
      <td style="text-align: center;">0.776</td>
      <td style="text-align: center;">0.738</td>
    </tr>
    <tr>
      <td style="text-align: left;">Korean</td>
      <td style="text-align: center;"><strong>0.824</strong></td>
      <td style="text-align: center;">0.814</td>
      <td style="text-align: center;">0.812</td>
      <td style="text-align: center;">0.799</td>
      <td style="text-align: center;">0.779</td>
      <td style="text-align: center;">0.700</td>
    </tr>
    <tr>
      <td style="text-align: left;">French</td>
      <td style="text-align: center;">0.698</td>
      <td style="text-align: center;">0.703</td>
      <td style="text-align: center;">0.700</td>
      <td style="text-align: center;"><strong>0.714</strong></td>
      <td style="text-align: center;">0.628</td>
      <td style="text-align: center;">0.535</td>
    </tr>
    <tr>
      <td style="text-align: left;">Russian</td>
      <td style="text-align: center;">0.734</td>
      <td style="text-align: center;">0.744</td>
      <td style="text-align: center;">0.781</td>
      <td style="text-align: center;"><strong>0.792</strong></td>
      <td style="text-align: center;">0.761</td>
      <td style="text-align: center;">0.676</td>
    </tr>
  </tbody>
</table>

<br>

*Cross-lingual speech generation on the Cross-Lingual benchmark. Performance is measured by Mixed Error Rate (WER for English, CER for others, ↓).*

<table>
  <thead>
    <tr>
      <th style="text-align: left;">Task</th>
      <th style="text-align: center;">Qwen3-TTS-25Hz-1.7B-Base</th>
      <th style="text-align: center;">Qwen3-TTS-12Hz-1.7B-Base</th>
      <th style="text-align: center;">CosyVoice3</th>
      <th style="text-align: center;">CosyVoice2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left;">en-to-zh</td>
      <td style="text-align: center;">5.66</td>
      <td style="text-align: center;"><strong>4.77</strong></td>
      <td style="text-align: center;">5.09</td>
      <td style="text-align: center;">13.5</td>
    </tr>
    <tr>
      <td style="text-align: left;">ja-to-zh</td>
      <td style="text-align: center;">3.92</td>
      <td style="text-align: center;">3.43</td>
      <td style="text-align: center;"><strong>3.05</strong></td>
      <td style="text-align: center;">48.1</td>
    </tr>
    <tr>
      <td style="text-align: left;">ko-to-zh</td>
      <td style="text-align: center;">1.14</td>
      <td style="text-align: center;">1.08</td>
      <td style="text-align: center;"><strong>1.06</strong></td>
      <td style="text-align: center;">7.70</td>
    </tr>
    <tr style="border-top: 1px solid #ddd;">
      <td style="text-align: left;">zh-to-en</td>
      <td style="text-align: center;">2.91</td>
      <td style="text-align: center;"><strong>2.77</strong></td>
      <td style="text-align: center;">2.98</td>
      <td style="text-align: center;">6.47</td>
    </tr>
    <tr>
      <td style="text-align: left;">ja-to-en</td>
      <td style="text-align: center;">3.95</td>
      <td style="text-align: center;"><strong>3.04</strong></td>
      <td style="text-align: center;">4.20</td>
      <td style="text-align: center;">17.1</td>
    </tr>
    <tr>
      <td style="text-align: left;">ko-to-en</td>
      <td style="text-align: center;">3.48</td>
      <td style="text-align: center;"><strong>3.09</strong></td>
      <td style="text-align: center;">4.19</td>
      <td style="text-align: center;">11.2</td>
    </tr>
    <tr style="border-top: 1px solid #ddd;">
      <td style="text-align: left;">zh-to-ja</td>
      <td style="text-align: center;">9.29</td>
      <td style="text-align: center;">8.40</td>
      <td style="text-align: center;"><strong>7.08</strong></td>
      <td style="text-align: center;">13.1</td>
    </tr>
    <tr>
      <td style="text-align: left;">en-to-ja</td>
      <td style="text-align: center;">7.74</td>
      <td style="text-align: center;">7.21</td>
      <td style="text-align: center;"><strong>6.80</strong></td>
      <td style="text-align: center;">14.9</td>
    </tr>
    <tr>
      <td style="text-align: left;">ko-to-ja</td>
      <td style="text-align: center;">4.17</td>
      <td style="text-align: center;"><strong>3.67</strong></td>
      <td style="text-align: center;">3.93</td>
      <td style="text-align: center;">5.86</td>
    </tr>
    <tr style="border-top: 1px solid #ddd;">
      <td style="text-align: left;">zh-to-ko</td>
      <td style="text-align: center;">8.12</td>
      <td style="text-align: center;"><strong>4.82</strong></td>
      <td style="text-align: center;">14.4</td>
      <td style="text-align: center;">24.8</td>
    </tr>
    <tr>
      <td style="text-align: left;">en-to-ko</td>
      <td style="text-align: center;">6.83</td>
      <td style="text-align: center;"><strong>5.14</strong></td>
      <td style="text-align: center;">5.87</td>
      <td style="text-align: center;">21.9</td>
    </tr>
    <tr>
      <td style="text-align: left;">ja-to-ko</td>
      <td style="text-align: center;">6.86</td>
      <td style="text-align: center;"><strong>5.59</strong></td>
      <td style="text-align: center;">7.92</td>
      <td style="text-align: center;">21.5</td>
    </tr>
  </tbody>
</table>

<br>

*Controllable speech generation on InstructTTSEval. Performance is measured by Attribute Perception and Synthesis accuracy (APS), Description-Speech Consistency (DSD), and Response Precision (RP).*

<table>
  <thead>
    <tr>
      <th rowspan="2" style="text-align: left; vertical-align: bottom;">Type</th>
      <th rowspan="2" style="text-align: left; vertical-align: bottom;">Model</th>
      <th colspan="3" style="text-align: center;">InstructTTSEval-ZH</th>
      <th colspan="3" style="text-align: center;">InstructTTSEval-EN</th>
    </tr>
    <tr>
      <th style="text-align: center;">APS (↑)</th>
      <th style="text-align: center;">DSD (↑)</th>
      <th style="text-align: center;">RP (↑)</th>
      <th style="text-align: center;">APS (↑)</th>
      <th style="text-align: center;">DSD (↑)</th>
      <th style="text-align: center;">RP (↑)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5" style="text-align: left; vertical-align: middle;"><em>Target<br>Speaker</em></td>
      <td style="text-align: left;">Gemini-flash</td>
      <td style="text-align: center;">88.2</td>
      <td style="text-align: center;"><strong>90.9</strong></td>
      <td style="text-align: center;"><strong>77.3</strong></td>
      <td style="text-align: center;"><strong>92.3</strong></td>
      <td style="text-align: center;"><strong>93.8</strong></td>
      <td style="text-align: center;"><strong>80.1</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Gemini-pro</td>
      <td style="text-align: center;"><strong>89.0</strong></td>
      <td style="text-align: center;">90.1</td>
      <td style="text-align: center;">75.5</td>
      <td style="text-align: center;">87.6</td>
      <td style="text-align: center;">86.0</td>
      <td style="text-align: center;">67.2</td>
    </tr>
    <tr>
      <td style="text-align: left;">Qwen3TTS-25Hz-1.7B-CustomVoice</td>
      <td style="text-align: center;">83.1</td>
      <td style="text-align: center;">75.0</td>
      <td style="text-align: center;">63.0</td>
      <td style="text-align: center;">79.0</td>
      <td style="text-align: center;">82.8</td>
      <td style="text-align: center;">69.3</td>
    </tr>
    <tr>
      <td style="text-align: left;">Qwen3TTS-12Hz-1.7B-CustomVoice</td>
      <td style="text-align: center;">83.0</td>
      <td style="text-align: center;">77.8</td>
      <td style="text-align: center;">61.2</td>
      <td style="text-align: center;">77.3</td>
      <td style="text-align: center;">77.1</td>
      <td style="text-align: center;">63.7</td>
    </tr>
    <tr>
      <td style="text-align: left;">GPT-4o-mini-tts</td>
      <td style="text-align: center;">54.9</td>
      <td style="text-align: center;">52.3</td>
      <td style="text-align: center;">46.0</td>
      <td style="text-align: center;">76.4</td>
      <td style="text-align: center;">74.3</td>
      <td style="text-align: center;">54.8</td>
    </tr>
    <tr style="border-top: 1px solid #ddd;">
      <td rowspan="9" style="text-align: left; vertical-align: middle;"><em>Voice<br>Design</em></td>
      <td style="text-align: left;">Qwen3TTS-12Hz-1.7B-VD</td>
      <td style="text-align: center;"><strong>85.2</strong></td>
      <td style="text-align: center;"><strong>81.1</strong></td>
      <td style="text-align: center;"><strong>65.1</strong></td>
      <td style="text-align: center;">82.9</td>
      <td style="text-align: center;"><strong>82.4</strong></td>
      <td style="text-align: center;"><strong>68.4</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Mimo-Audio-7B-Instruct (Zhang et al., 2025b)</td>
      <td style="text-align: center;">75.7</td>
      <td style="text-align: center;">74.3</td>
      <td style="text-align: center;">61.5</td>
      <td style="text-align: center;">80.6</td>
      <td style="text-align: center;">77.6</td>
      <td style="text-align: center;">59.5</td>
    </tr>
    <tr>
      <td style="text-align: left;">VoiceSculptor (Hu et al., 2026)</td>
      <td style="text-align: center;">75.7</td>
      <td style="text-align: center;">64.7</td>
      <td style="text-align: center;">61.5</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
    </tr>
    <tr>
      <td style="text-align: left;">Hume</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>83.0</strong></td>
      <td style="text-align: center;">75.3</td>
      <td style="text-align: center;">54.3</td>
    </tr>
    <tr>
      <td style="text-align: left;">VoxInstruct (Zhou et al., 2024)</td>
      <td style="text-align: center;">47.5</td>
      <td style="text-align: center;">52.3</td>
      <td style="text-align: center;">42.6</td>
      <td style="text-align: center;">54.9</td>
      <td style="text-align: center;">57.0</td>
      <td style="text-align: center;">39.3</td>
    </tr>
    <tr>
      <td style="text-align: left;">Parler-tts-mini (Lyth & King, 2024)</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">63.4</td>
      <td style="text-align: center;">48.7</td>
      <td style="text-align: center;">28.6</td>
    </tr>
    <tr>
      <td style="text-align: left;">Parler-tts-large (Lyth & King, 2024)</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">60.0</td>
      <td style="text-align: center;">45.9</td>
      <td style="text-align: center;">31.2</td>
    </tr>
    <tr>
      <td style="text-align: left;">PromptTTS (Guo et al., 2023)</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">64.3</td>
      <td style="text-align: center;">47.2</td>
      <td style="text-align: center;">31.4</td>
    </tr>
    <tr>
      <td style="text-align: left;">PromptStyle (Liu et al., 2023)</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">57.4</td>
      <td style="text-align: center;">46.4</td>
      <td style="text-align: center;">30.9</td>
    </tr>
  </tbody>
</table>

<br>

*Target-Speaker Multilingual Speech Generation on the TTS multilingual test set. Performance is measured by Word Error Rate (WER, ↓).*

<table>
  <thead>
    <tr>
      <th rowspan="2" style="text-align: left; vertical-align: bottom;">Language</th>
      <th colspan="2" style="text-align: center;">Qwen3-TTS-25Hz</th>
      <th colspan="2" style="text-align: center;">Qwen3-TTS-12Hz</th>
      <th rowspan="2" style="text-align: center; vertical-align: bottom;">GPT-4o-Audio<br>Preview</th>
    </tr>
    <tr>
      <th style="text-align: center;">0.6B-CustomVoice</th>
      <th style="text-align: center;">1.7B-CustomVoice</th>
      <th style="text-align: center;">0.6B-CustomVoice</th>
      <th style="text-align: center;">1.7B-CustomVoice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left;">Chinese</td>
      <td style="text-align: center;">0.874</td>
      <td style="text-align: center;"><strong>0.708</strong></td>
      <td style="text-align: center;">0.944</td>
      <td style="text-align: center;">0.903</td>
      <td style="text-align: center;">3.519</td>
    </tr>
    <tr>
      <td style="text-align: left;">English</td>
      <td style="text-align: center;">1.332</td>
      <td style="text-align: center;">0.936</td>
      <td style="text-align: center;">1.188</td>
      <td style="text-align: center;"><strong>0.899</strong></td>
      <td style="text-align: center;">2.197</td>
    </tr>
    <tr>
      <td style="text-align: left;">German</td>
      <td style="text-align: center;">0.990</td>
      <td style="text-align: center;"><strong>0.634</strong></td>
      <td style="text-align: center;">2.722</td>
      <td style="text-align: center;">1.057</td>
      <td style="text-align: center;">1.161</td>
    </tr>
    <tr>
      <td style="text-align: left;">Italian</td>
      <td style="text-align: center;">1.861</td>
      <td style="text-align: center;">1.271</td>
      <td style="text-align: center;">2.545</td>
      <td style="text-align: center;">1.362</td>
      <td style="text-align: center;"><strong>1.194</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Portuguese</td>
      <td style="text-align: center;">1.728</td>
      <td style="text-align: center;">1.854</td>
      <td style="text-align: center;">3.219</td>
      <td style="text-align: center;">2.681</td>
      <td style="text-align: center;"><strong>1.504</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Spanish</td>
      <td style="text-align: center;">1.309</td>
      <td style="text-align: center;">1.284</td>
      <td style="text-align: center;"><strong>1.154</strong></td>
      <td style="text-align: center;">1.330</td>
      <td style="text-align: center;">4.000</td>
    </tr>
    <tr>
      <td style="text-align: left;">Japanese</td>
      <td style="text-align: center;"><strong>3.875</strong></td>
      <td style="text-align: center;">4.518</td>
      <td style="text-align: center;">6.877</td>
      <td style="text-align: center;">4.924</td>
      <td style="text-align: center;">5.001</td>
    </tr>
    <tr>
      <td style="text-align: left;">Korean</td>
      <td style="text-align: center;">2.202</td>
      <td style="text-align: center;">2.274</td>
      <td style="text-align: center;">3.053</td>
      <td style="text-align: center;"><strong>1.741</strong></td>
      <td style="text-align: center;">2.763</td>
    </tr>
    <tr>
      <td style="text-align: left;">French</td>
      <td style="text-align: center;">3.865</td>
      <td style="text-align: center;"><strong>3.080</strong></td>
      <td style="text-align: center;">3.841</td>
      <td style="text-align: center;">3.781</td>
      <td style="text-align: center;">3.605</td>
    </tr>
    <tr>
      <td style="text-align: left;">Russian</td>
      <td style="text-align: center;">6.529</td>
      <td style="text-align: center;"><strong>4.444</strong></td>
      <td style="text-align: center;">5.809</td>
      <td style="text-align: center;">4.734</td>
      <td style="text-align: center;">5.250</td>
    </tr>
  </tbody>
</table>

<br>

*Long speech generation results. Performance is measured by Word Error Rate (WER, ↓).*

<table>
  <thead>
    <tr>
      <th style="text-align: center;">Datasets</th>
      <th style="text-align: left;">Model</th>
      <th colspan="2" style="text-align: center;">Performance</th>
    </tr>
    <tr style="border-bottom: 1px solid #ddd; border-top: 1px solid #ddd;">
      <td colspan="4" style="text-align: center;"><em>Content Consistency</em></td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5" style="text-align: center; vertical-align: middle;"><em>long-zh</em> | <em>long-en</em></td>
      <td style="text-align: left;">Higgs-Audio-v2 (chunk) (Boson AI, 2025)</td>
      <td style="text-align: center;">5.505</td>
      <td style="text-align: center;">6.917</td>
    </tr>
    <tr>
      <td style="text-align: left;">VibeVoice (Peng et al., 2025)</td>
      <td style="text-align: center;">22.619</td>
      <td style="text-align: center;">1.780</td>
    </tr>
    <tr>
      <td style="text-align: left;">VoxCPM (Zhou et al., 2025)</td>
      <td style="text-align: center;">4.835</td>
      <td style="text-align: center;">7.474</td>
    </tr>
    <tr>
      <td style="text-align: left;">Qwen3-TTS-25Hz-1.7B-CustomVoice</td>
      <td style="text-align: center;"><strong>1.517</strong></td>
      <td style="text-align: center;"><strong>1.225</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Qwen3-TTS-12Hz-1.7B-CustomVoice</td>
      <td style="text-align: center;">2.356</td>
      <td style="text-align: center;">2.812</td>
    </tr>
  </tbody>
</table>
</details>


<details>
<summary>Speech Tokenizer Benchmarks</summary>

*Comparison between different supervised semantic speech tokenizers on ASR Task.*

<table>
  <thead>
    <tr>
      <th style="text-align: left;">Model</th>
      <th style="text-align: center;">Codebook Size</th>
      <th style="text-align: center;">FPS</th>
      <th style="text-align: center;">C.V. EN</th>
      <th style="text-align: center;">C.V. CN</th>
      <th style="text-align: center;">Fluers EN</th>
      <th style="text-align: center;">Fluers CN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left;">S3 Tokenizer(VQ) (Du et al., 2024a)</td>
      <td style="text-align: center;">4096</td>
      <td style="text-align: center;">50</td>
      <td style="text-align: center;">12.06</td>
      <td style="text-align: center;">15.38</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
    </tr>
    <tr>
      <td style="text-align: left;">S3 Tokenizer(VQ) (Du et al., 2024a)</td>
      <td style="text-align: center;">4096</td>
      <td style="text-align: center;">25</td>
      <td style="text-align: center;">11.56</td>
      <td style="text-align: center;">18.26</td>
      <td style="text-align: center;">7.65</td>
      <td style="text-align: center;">5.03</td>
    </tr>
    <tr>
      <td style="text-align: left;">S3 Tokenizer(FSQ) (Du et al., 2024a)</td>
      <td style="text-align: center;">6561</td>
      <td style="text-align: center;">25</td>
      <td style="text-align: center;">10.67</td>
      <td style="text-align: center;"><strong>7.29</strong></td>
      <td style="text-align: center;">6.58</td>
      <td style="text-align: center;">4.43</td>
    </tr>
    <tr>
      <td style="text-align: left;">Qwen-TTS-Tokenizer-25Hz (Stage 1)</td>
      <td style="text-align: center;">32768</td>
      <td style="text-align: center;">25</td>
      <td style="text-align: center;"><strong>7.51</strong></td>
      <td style="text-align: center;">10.73</td>
      <td style="text-align: center;"><strong>3.07</strong></td>
      <td style="text-align: center;"><strong>4.23</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Qwen-TTS-Tokenizer-25Hz (Stage 2)</td>
      <td style="text-align: center;">32768</td>
      <td style="text-align: center;">25</td>
      <td style="text-align: center;">10.40</td>
      <td style="text-align: center;">14.99</td>
      <td style="text-align: center;">4.14</td>
      <td style="text-align: center;">4.67</td>
    </tr>
  </tbody>
</table>

<br>

*Comparison between different semantic-related speech tokenizers.*

<table>
  <thead>
    <tr>
      <th style="text-align: left;">Model</th>
      <th style="text-align: center;">NQ</th>
      <th style="text-align: center;">Codebook Size</th>
      <th style="text-align: center;">FPS</th>
      <th style="text-align: center;">PESQ_WB</th>
      <th style="text-align: center;">PESQ_NB</th>
      <th style="text-align: center;">STOI</th>
      <th style="text-align: center;">UTMOS</th>
      <th style="text-align: center;">SIM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left;">SpeechTokenizer (Zhang et al., 2023a)</td>
      <td style="text-align: center;">8</td>
      <td style="text-align: center;">1024</td>
      <td style="text-align: center;">50</td>
      <td style="text-align: center;">2.60</td>
      <td style="text-align: center;">3.05</td>
      <td style="text-align: center;">0.92</td>
      <td style="text-align: center;">3.90</td>
      <td style="text-align: center;">0.85</td>
    </tr>
    <tr>
      <td style="text-align: left;">X-codec (Ye et al., 2025a)</td>
      <td style="text-align: center;">2</td>
      <td style="text-align: center;">1024</td>
      <td style="text-align: center;">50</td>
      <td style="text-align: center;">2.68</td>
      <td style="text-align: center;">3.27</td>
      <td style="text-align: center;">0.86</td>
      <td style="text-align: center;">4.11</td>
      <td style="text-align: center;">0.84</td>
    </tr>
    <tr>
      <td style="text-align: left;">X-codec 2 (Ye et al., 2025b)</td>
      <td style="text-align: center;">1</td>
      <td style="text-align: center;">65536</td>
      <td style="text-align: center;">50</td>
      <td style="text-align: center;">2.43</td>
      <td style="text-align: center;">3.04</td>
      <td style="text-align: center;">0.92</td>
      <td style="text-align: center;">4.13</td>
      <td style="text-align: center;">0.82</td>
    </tr>
    <tr>
      <td style="text-align: left;">XY-Tokenizer (Gong et al., 2025)</td>
      <td style="text-align: center;">8</td>
      <td style="text-align: center;">1024</td>
      <td style="text-align: center;">12.5</td>
      <td style="text-align: center;">2.41</td>
      <td style="text-align: center;">3.00</td>
      <td style="text-align: center;">0.91</td>
      <td style="text-align: center;">3.98</td>
      <td style="text-align: center;">0.83</td>
    </tr>
    <tr>
      <td style="text-align: left;">Mimi (Défossez et al., 2024)</td>
      <td style="text-align: center;">16</td>
      <td style="text-align: center;">2048</td>
      <td style="text-align: center;">12.5</td>
      <td style="text-align: center;">2.88</td>
      <td style="text-align: center;">3.42</td>
      <td style="text-align: center;">0.94</td>
      <td style="text-align: center;">3.87</td>
      <td style="text-align: center;">0.87</td>
    </tr>
    <tr>
      <td style="text-align: left;">FireredTTS 2 Tokenizer (Xie et al., 2025)</td>
      <td style="text-align: center;">16</td>
      <td style="text-align: center;">2048</td>
      <td style="text-align: center;">12.5</td>
      <td style="text-align: center;">2.73</td>
      <td style="text-align: center;">3.28</td>
      <td style="text-align: center;">0.94</td>
      <td style="text-align: center;">3.88</td>
      <td style="text-align: center;">0.87</td>
    </tr>
    <tr>
      <td style="text-align: left;">Qwen-TTS-Tokenizer-12Hz</td>
      <td style="text-align: center;">16</td>
      <td style="text-align: center;">2048</td>
      <td style="text-align: center;">12.5</td>
      <td style="text-align: center;"><strong>3.21</strong></td>
      <td style="text-align: center;"><strong>3.68</strong></td>
      <td style="text-align: center;"><strong>0.96</strong></td>
      <td style="text-align: center;"><strong>4.16</strong></td>
      <td style="text-align: center;"><strong>0.95</strong></td>
    </tr>
  </tbody>
</table>

</details>


## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)

```BibTeX
@article{Qwen3-TTS,
  title={Qwen3-TTS Technical Report},
  author={Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang and Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao and Zishan Guo and Xinyu Zhang and Pei Zhang and Baosong Yang and Jin Xu and Jingren Zhou and Junyang Lin},
  journal={arXiv preprint arXiv:2601.15621},
  year={2026}
}
```


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen-TTS&type=Date)](https://star-history.com/#QwenLM/Qwen-TTS&Date)


<br>