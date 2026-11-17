# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""Factory and lifecycle management for Qwen3-TTS backends."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from .base import TTSBackend

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
_DEFAULT_CPU_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
_backend_instance: Optional[TTSBackend] = None
_initialization_lock: Optional[asyncio.Lock] = None


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %d", name, raw, default)
        return default
    if value < minimum:
        logger.warning("%s must be >= %d; using %d", name, minimum, default)
        return default
    return value


def _env_float(name: str, default: float, minimum: float = 0.001) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %.3f", name, raw, default)
        return default
    if value < minimum:
        logger.warning("%s must be >= %.3f; using %.3f", name, minimum, default)
        return default
    return value


def _get_initialization_lock() -> asyncio.Lock:
    global _initialization_lock
    if _initialization_lock is None:
        _initialization_lock = asyncio.Lock()
    return _initialization_lock


def get_backend() -> TTSBackend:
    """Return the process-wide backend instance, creating it lazily."""
    global _backend_instance
    if _backend_instance is not None:
        return _backend_instance

    backend_type = os.getenv("TTS_BACKEND", "cuda_graphs").strip().lower()
    configured_model = os.getenv("TTS_MODEL_NAME") or os.getenv("TTS_MODEL_ID")
    model_name = (configured_model or _DEFAULT_MODEL).strip()

    device = os.getenv("TTS_DEVICE", "auto").strip() or "auto"
    dtype = os.getenv("TTS_DTYPE", "auto").strip() or "auto"
    attn = os.getenv("TTS_ATTN", "auto").strip() or "auto"
    cpu_threads = _env_int("CPU_THREADS", 12)
    cpu_interop = _env_int("CPU_INTEROP", 2)
    use_ipex = os.getenv("USE_IPEX", "false").strip().lower() == "true"
    ov_device = os.getenv("OV_DEVICE", "CPU").strip() or "CPU"
    ov_cache_dir = os.getenv("OV_CACHE_DIR", "./.ov_cache")
    ov_model_dir = os.getenv("OV_MODEL_DIR", "./.ov_models")

    logger.info("Creating TTS backend: %s", backend_type)

    # Imports are deliberately local. Optional backends must not make the
    # default installation fail merely because their dependencies are absent.
    if backend_type == "cuda_graphs":
        from .cuda_graphs import CUDAGraphBackend

        _backend_instance = CUDAGraphBackend(model_name=model_name)
    elif backend_type == "optimized":
        from .optimized_backend import OptimizedQwen3TTSBackend

        _backend_instance = OptimizedQwen3TTSBackend()
    elif backend_type == "official":
        from .official_qwen3_tts import OfficialQwen3TTSBackend

        _backend_instance = OfficialQwen3TTSBackend(model_name=model_name)
    elif backend_type in {"vllm_omni", "vllm-omni", "vllm"}:
        from .vllm_omni_qwen3_tts import VLLMOmniQwen3TTSBackend

        _backend_instance = VLLMOmniQwen3TTSBackend(model_name=model_name)
    elif backend_type == "pytorch":
        from .pytorch_backend import PyTorchCPUBackend

        device_val = device if device != "auto" else "cpu"
        dtype_val = dtype if dtype != "auto" else "float32"
        attn_val = attn if attn != "auto" else "sdpa"
        cpu_model_name = (configured_model or _DEFAULT_CPU_MODEL).strip()
        _backend_instance = PyTorchCPUBackend(
            model_id=cpu_model_name,
            device=device_val,
            dtype=dtype_val,
            attn_implementation=attn_val,
            cpu_threads=cpu_threads,
            cpu_interop_threads=cpu_interop,
            use_ipex=use_ipex,
        )
        logger.info(
            "PyTorch backend: device=%s dtype=%s attention=%s threads=%d interop=%d ipex=%s",
            device_val,
            dtype_val,
            attn_val,
            cpu_threads,
            cpu_interop,
            use_ipex,
        )
    elif backend_type == "openvino":
        from .openvino_backend import OpenVINOBackend

        _backend_instance = OpenVINOBackend(
            ov_model_dir=ov_model_dir,
            ov_device=ov_device,
            ov_cache_dir=ov_cache_dir,
        )
        logger.warning(
            "OpenVINO is experimental and requires a manually exported model; "
            "use TTS_BACKEND=pytorch for the reliable CPU path."
        )
    elif backend_type == "mlx":
        from .mlx_qwen3_tts import DEFAULT_MLX_MODEL, MLXQwen3TTSBackend

        mlx_model_name = os.getenv("MLX_MODEL_ID", DEFAULT_MLX_MODEL).strip()
        _backend_instance = MLXQwen3TTSBackend(
            model_name=mlx_model_name or DEFAULT_MLX_MODEL
        )
    else:
        raise ValueError(
            f"Unknown TTS_BACKEND: {backend_type!r}. Supported values: "
            "'cuda_graphs', 'optimized', 'official', 'vllm_omni', 'pytorch', 'openvino', 'mlx'"
        )

    logger.info(
        "Using %s backend with model %s",
        _backend_instance.get_backend_name(),
        _backend_instance.get_model_id(),
    )
    return _backend_instance


async def _run_warmup_request(backend: TTSBackend, text: str) -> None:
    custom_names = backend.get_custom_voice_names()
    if backend.get_model_type() == "base" and custom_names:
        await backend.generate_speech_with_custom_voice(
            text=text,
            voice=custom_names[0],
            language="English",
        )
    elif backend.get_model_type() == "base":
        raise LookupError("Base model has no custom voice available for warmup")
    else:
        await backend.generate_speech(
            text=text,
            voice="Vivian",
            language="English",
        )


async def _warmup_backend(backend: TTSBackend) -> None:
    """Warm both regular and streaming paths with real wall-clock timeouts."""
    import time

    max_seconds = _env_float("TTS_WARMUP_MAX_SECONDS", 10.0)
    texts = [
        "Hello.",
        "Hello, this is a warmup test.",
        "Hello, this is a longer warmup test to exercise the full decode pipeline.",
    ]

    logger.info(
        "Performing backend warmup (%d requests, %.1fs timeout each)",
        len(texts),
        max_seconds,
    )
    for index, text in enumerate(texts, 1):
        started = time.monotonic()
        try:
            await asyncio.wait_for(
                _run_warmup_request(backend, text), timeout=max_seconds
            )
        except LookupError as exc:
            logger.info("Skipping warmup: %s", exc)
            return
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"Warmup request {index} exceeded {max_seconds:.1f}s"
            ) from exc
        logger.info(
            "Warmup request %d/%d completed in %.2fs",
            index,
            len(texts),
            time.monotonic() - started,
        )

    streaming_method = getattr(backend, "generate_speech_streaming", None)
    if streaming_method is None:
        return

    async def _drain_stream() -> None:
        async for _chunk, _sample_rate in streaming_method(
            text="Streaming warmup.",
            voice="Vivian",
            language="English",
        ):
            pass

    try:
        await asyncio.wait_for(_drain_stream(), timeout=max_seconds)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            f"Streaming warmup exceeded {max_seconds:.1f}s"
        ) from exc


async def initialize_backend(warmup: bool = False) -> TTSBackend:
    """Initialize the global backend exactly once, even under concurrency."""
    global _backend_instance

    async with _get_initialization_lock():
        backend = get_backend()
        if backend.is_ready():
            return backend

        try:
            await backend.initialize()

            custom_voices_dir = os.getenv(
                "TTS_CUSTOM_VOICES",
                str(Path(__file__).resolve().parent.parent.parent / "custom_voices"),
            )
            try:
                await backend.load_custom_voices(custom_voices_dir)
            except Exception as exc:
                logger.warning("Custom voice loading failed (non-critical): %s", exc)

            if warmup and os.getenv("TTS_WARMUP_ON_START", "false").lower() == "true":
                try:
                    await _warmup_backend(backend)
                except Exception as exc:
                    logger.error("Backend warmup failed: %s", exc)

            return backend
        except Exception:
            # A partially initialized model is unsafe to reuse. The next request
            # constructs a fresh instance rather than retrying corrupted state.
            _backend_instance = None
            raise


def reset_backend() -> None:
    """Reset process globals (primarily for tests)."""
    global _backend_instance, _initialization_lock
    _backend_instance = None
    _initialization_lock = None
