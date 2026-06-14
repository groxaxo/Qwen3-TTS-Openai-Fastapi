# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Tests for backend selection and initialization.
"""

import sys
import types
import asyncio
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from api.backends.factory import get_backend, reset_backend
from api.backends.base import TTSBackend
from api.backends.official_qwen3_tts import OfficialQwen3TTSBackend
from api.backends.vllm_omni_qwen3_tts import VLLMOmniQwen3TTSBackend
from api.backends.pytorch_backend import PyTorchCPUBackend
from api.backends.openvino_backend import OpenVINOBackend


class TestBackendSelection:
    """Test backend selection via environment variables."""
    
    def teardown_method(self):
        """Reset backend after each test."""
        reset_backend()
    
    def test_default_backend_is_official(self, monkeypatch):
        """Test that official backend is selected by default."""
        # Ensure TTS_BACKEND is not set
        monkeypatch.delenv("TTS_BACKEND", raising=False)
        
        backend = get_backend()
        assert isinstance(backend, OfficialQwen3TTSBackend)
        assert backend.get_backend_name() == "official"
    
    def test_official_backend_via_env(self, monkeypatch):
        """Test selecting official backend via environment variable."""
        monkeypatch.setenv("TTS_BACKEND", "official")
        
        backend = get_backend()
        assert isinstance(backend, OfficialQwen3TTSBackend)
        assert backend.get_backend_name() == "official"
    
    def test_vllm_backend_via_env(self, monkeypatch):
        """Test selecting vLLM-Omni backend via environment variable."""
        monkeypatch.setenv("TTS_BACKEND", "vllm_omni")
        
        backend = get_backend()
        assert isinstance(backend, VLLMOmniQwen3TTSBackend)
        assert backend.get_backend_name() == "vllm_omni"
    
    def test_vllm_backend_alternate_name(self, monkeypatch):
        """Test vLLM backend with alternate name format."""
        monkeypatch.setenv("TTS_BACKEND", "vllm-omni")
        
        backend = get_backend()
        assert isinstance(backend, VLLMOmniQwen3TTSBackend)
        assert backend.get_backend_name() == "vllm_omni"
    
    def test_invalid_backend_raises_error(self, monkeypatch):
        """Test that invalid backend name raises ValueError."""
        monkeypatch.setenv("TTS_BACKEND", "invalid_backend")
        
        with pytest.raises(ValueError, match="Unknown TTS_BACKEND"):
            get_backend()
    
    def test_custom_model_name_via_env(self, monkeypatch):
        """Test overriding model name via environment variable."""
        monkeypatch.setenv("TTS_BACKEND", "official")
        monkeypatch.setenv("TTS_MODEL_NAME", "custom/model")
        
        backend = get_backend()
        assert backend.get_model_id() == "custom/model"
    
    def test_backend_singleton(self, monkeypatch):
        """Test that get_backend returns the same instance."""
        monkeypatch.setenv("TTS_BACKEND", "official")
        
        backend1 = get_backend()
        backend2 = get_backend()
        
        assert backend1 is backend2


class TestBackendInterface:
    """Test that all backends implement the required interface."""
    
    def test_official_backend_implements_interface(self):
        """Test official backend implements TTSBackend interface."""
        backend = OfficialQwen3TTSBackend()
        
        assert isinstance(backend, TTSBackend)
        assert hasattr(backend, 'initialize')
        assert hasattr(backend, 'generate_speech')
        assert hasattr(backend, 'get_backend_name')
        assert hasattr(backend, 'get_model_id')
        assert hasattr(backend, 'get_supported_voices')
        assert hasattr(backend, 'get_supported_languages')
        assert hasattr(backend, 'is_ready')
        assert hasattr(backend, 'get_device_info')
    
    def test_vllm_backend_implements_interface(self):
        """Test vLLM backend implements TTSBackend interface."""
        backend = VLLMOmniQwen3TTSBackend()
        
        assert isinstance(backend, TTSBackend)
        assert hasattr(backend, 'initialize')
        assert hasattr(backend, 'generate_speech')
        assert hasattr(backend, 'get_backend_name')
        assert hasattr(backend, 'get_model_id')
        assert hasattr(backend, 'get_supported_voices')
        assert hasattr(backend, 'get_supported_languages')
        assert hasattr(backend, 'is_ready')
        assert hasattr(backend, 'get_device_info')
    
    def test_backend_names_are_correct(self):
        """Test that backends return correct names."""
        official = OfficialQwen3TTSBackend()
        vllm = VLLMOmniQwen3TTSBackend()
        
        assert official.get_backend_name() == "official"
        assert vllm.get_backend_name() == "vllm_omni"
    
    def test_backends_return_voices(self):
        """Test that backends return voice lists."""
        official = OfficialQwen3TTSBackend()
        vllm = VLLMOmniQwen3TTSBackend()
        
        # Both backends should return a list of voices
        assert isinstance(official.get_supported_voices(), list)
        assert isinstance(vllm.get_supported_voices(), list)
        assert len(official.get_supported_voices()) > 0
        assert len(vllm.get_supported_voices()) > 0
    
    def test_backends_return_languages(self):
        """Test that backends return language lists."""
        official = OfficialQwen3TTSBackend()
        vllm = VLLMOmniQwen3TTSBackend()
        
        # Both backends should return a list of languages
        assert isinstance(official.get_supported_languages(), list)
        assert isinstance(vllm.get_supported_languages(), list)
        assert len(official.get_supported_languages()) > 0
        assert len(vllm.get_supported_languages()) > 0
    
    def test_backends_initially_not_ready(self):
        """Test that backends are not ready before initialization."""
        official = OfficialQwen3TTSBackend()
        vllm = VLLMOmniQwen3TTSBackend()
        
        assert not official.is_ready()
        assert not vllm.is_ready()
    
    def test_backends_return_device_info(self):
        """Test that backends return device info dict."""
        official = OfficialQwen3TTSBackend()
        vllm = VLLMOmniQwen3TTSBackend()
        
        info1 = official.get_device_info()
        info2 = vllm.get_device_info()
        
        # Check required keys
        assert "device" in info1
        assert "gpu_available" in info1
        assert "device" in info2
        assert "gpu_available" in info2


class TestVoiceCloningInterface:
    """Tests for voice cloning interface across all backends."""

    def test_official_backend_has_voice_cloning_methods(self):
        """Test that official backend has voice cloning methods."""
        backend = OfficialQwen3TTSBackend()
        
        assert hasattr(backend, 'supports_voice_cloning')
        assert hasattr(backend, 'get_model_type')
        assert hasattr(backend, 'generate_voice_clone')

    def test_vllm_backend_has_voice_cloning_methods(self):
        """Test that vLLM backend has voice cloning methods."""
        backend = VLLMOmniQwen3TTSBackend()
        
        assert hasattr(backend, 'supports_voice_cloning')
        assert hasattr(backend, 'get_model_type')

    def test_customvoice_model_does_not_support_cloning(self):
        """Test that CustomVoice models don't support voice cloning."""
        official = OfficialQwen3TTSBackend(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        vllm = VLLMOmniQwen3TTSBackend(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        
        assert not official.supports_voice_cloning()
        assert not vllm.supports_voice_cloning()
        assert official.get_model_type() == "customvoice"
        assert vllm.get_model_type() == "customvoice"

    def test_base_model_supports_cloning(self):
        """Test that Base models support voice cloning."""
        official = OfficialQwen3TTSBackend(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        vllm = VLLMOmniQwen3TTSBackend(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        
        assert official.supports_voice_cloning()
        assert vllm.supports_voice_cloning()
        assert official.get_model_type() == "base"
        assert vllm.get_model_type() == "base"

    def test_voicedesign_model_does_not_support_cloning(self):
        """Test that VoiceDesign models don't support voice cloning."""
        official = OfficialQwen3TTSBackend(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
        vllm = VLLMOmniQwen3TTSBackend(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
        
        assert not official.supports_voice_cloning()
        assert not vllm.supports_voice_cloning()

    def test_vllm_backend_voicedesign_model_type(self):
        """Test vLLM backend returns correct model type for VoiceDesign."""
        vllm = VLLMOmniQwen3TTSBackend(model_name="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
        
        assert vllm.get_model_type() == "voicedesign"

    def test_model_type_defaults_to_customvoice(self):
        """Test that default model type is customvoice."""
        official = OfficialQwen3TTSBackend()
        vllm = VLLMOmniQwen3TTSBackend()
        
        assert official.get_model_type() == "customvoice"
        assert vllm.get_model_type() == "customvoice"


class TestCPUBackendSelection:
    """Test CPU-optimized PyTorch backend selection."""
    
    def teardown_method(self):
        """Reset backend after each test."""
        reset_backend()
    
    def test_pytorch_backend_via_env(self, monkeypatch):
        """Test selecting PyTorch CPU backend via environment variable."""
        monkeypatch.setenv("TTS_BACKEND", "pytorch")
        
        backend = get_backend()
        assert isinstance(backend, PyTorchCPUBackend)
        assert backend.get_backend_name() == "pytorch_cpu"
    
    def test_pytorch_backend_with_config(self, monkeypatch):
        """Test PyTorch CPU backend with configuration options."""
        monkeypatch.setenv("TTS_BACKEND", "pytorch")
        monkeypatch.setenv("TTS_DEVICE", "cpu")
        monkeypatch.setenv("TTS_DTYPE", "float32")
        monkeypatch.setenv("TTS_ATTN", "sdpa")
        monkeypatch.setenv("CPU_THREADS", "8")
        monkeypatch.setenv("CPU_INTEROP", "2")
        
        backend = get_backend()
        assert isinstance(backend, PyTorchCPUBackend)
        
        device_info = backend.get_device_info()
        assert device_info["cpu_threads"] == 8
        assert device_info["cpu_interop_threads"] == 2
    
    def test_pytorch_backend_implements_interface(self):
        """Test PyTorch CPU backend implements TTSBackend interface."""
        backend = PyTorchCPUBackend()
        
        assert isinstance(backend, TTSBackend)
        assert hasattr(backend, 'initialize')
        assert hasattr(backend, 'generate_speech')
        assert hasattr(backend, 'get_backend_name')
        assert hasattr(backend, 'get_model_id')
        assert hasattr(backend, 'get_supported_voices')
        assert hasattr(backend, 'get_supported_languages')
        assert hasattr(backend, 'is_ready')
        assert hasattr(backend, 'get_device_info')
        assert hasattr(backend, 'supports_voice_cloning')
    
    def test_pytorch_backend_not_ready_initially(self):
        """Test that PyTorch CPU backend is not ready before initialization."""
        backend = PyTorchCPUBackend()
        assert not backend.is_ready()
    
    def test_pytorch_backend_device_info(self):
        """Test that PyTorch CPU backend returns device info."""
        backend = PyTorchCPUBackend()
        
        info = backend.get_device_info()
        assert "device" in info
        assert "cpu_threads" in info
        assert "cpu_interop_threads" in info
        assert "ipex_enabled" in info
        assert info["device"] == "cpu"
    
    def test_pytorch_backend_supports_cloning_with_base_model(self):
        """Test that PyTorch CPU backend supports cloning with Base model."""
        backend = PyTorchCPUBackend(model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
        
        assert backend.supports_voice_cloning()
        assert backend.get_model_type() == "base"
    
    def test_pytorch_backend_no_cloning_with_customvoice(self):
        """Test that PyTorch CPU backend doesn't support cloning with CustomVoice model."""
        backend = PyTorchCPUBackend(model_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        
        assert not backend.supports_voice_cloning()
        assert backend.get_model_type() == "customvoice"


class TestOpenVINOBackendSelection:
    """Test OpenVINO backend selection."""
    
    def teardown_method(self):
        """Reset backend after each test."""
        reset_backend()
    
    def test_openvino_backend_via_env(self, monkeypatch):
        """Test selecting OpenVINO backend via environment variable."""
        monkeypatch.setenv("TTS_BACKEND", "openvino")
        
        backend = get_backend()
        assert isinstance(backend, OpenVINOBackend)
        assert backend.get_backend_name() == "openvino"
    
    def test_openvino_backend_implements_interface(self):
        """Test OpenVINO backend implements TTSBackend interface."""
        backend = OpenVINOBackend()
        
        assert isinstance(backend, TTSBackend)
        assert hasattr(backend, 'initialize')
        assert hasattr(backend, 'generate_speech')
        assert hasattr(backend, 'get_backend_name')
        assert hasattr(backend, 'get_model_id')
        assert hasattr(backend, 'get_supported_voices')
        assert hasattr(backend, 'get_supported_languages')
        assert hasattr(backend, 'is_ready')
        assert hasattr(backend, 'get_device_info')
    
    def test_openvino_backend_not_ready_initially(self):
        """Test that OpenVINO backend is not ready before initialization."""
        backend = OpenVINOBackend()
        assert not backend.is_ready()
    
    def test_openvino_backend_device_info(self):
        """Test that OpenVINO backend returns device info."""
        backend = OpenVINOBackend()
        
        info = backend.get_device_info()
        assert "device" in info
        assert "backend" in info
        assert info["backend"] == "OpenVINO"
    
    def test_openvino_backend_does_not_support_cloning(self):
        """Test that OpenVINO backend does not support voice cloning."""
        backend = OpenVINOBackend()
        
        # OpenVINO backend is experimental and doesn't support cloning
        assert not backend.supports_voice_cloning()
        assert backend.get_model_type() == "openvino_experimental"


class TestBackendErrorHandling:
    """Test backend error handling and validation."""
    
    def teardown_method(self):
        """Reset backend after each test."""
        reset_backend()
    
    def test_invalid_backend_raises_error(self, monkeypatch):
        """Test that invalid backend name raises ValueError."""
        monkeypatch.setenv("TTS_BACKEND", "invalid_backend")
        
        with pytest.raises(ValueError, match="Unknown TTS_BACKEND"):
            get_backend()
    
    def test_custom_model_name_with_pytorch_backend(self, monkeypatch):
        """Test overriding model name with PyTorch backend."""
        monkeypatch.setenv("TTS_BACKEND", "pytorch")
        monkeypatch.setenv("TTS_MODEL_ID", "custom/model")

        backend = get_backend()
        assert backend.get_model_id() == "custom/model"


# ---------------------------------------------------------------------------
# MLX backend tests
# ---------------------------------------------------------------------------
#
# The MLX backend lazily imports ``mlx_audio``. These tests stub the
# ``mlx_audio.tts.utils`` module so the suite can run on a host that does
# not have mlx-audio installed (Linux, CI, etc.).


def _install_fake_mlx_audio(monkeypatch, speakers=None, languages=None, sample_rate=24000):
    """Install a stub ``mlx_audio`` package into ``sys.modules``.

    Returns the fake ``load_model`` function, which returns a fake model
    object whose ``generate_custom_voice`` returns an empty list by default
    (callers can replace it on the returned model).
    """
    fake_pkg = types.ModuleType("mlx_audio")
    fake_tts = types.ModuleType("mlx_audio.tts")
    fake_utils = types.ModuleType("mlx_audio.tts.utils")
    fake_core = types.ModuleType("mlx_audio.core")  # not used, but keeps the surface real

    fake_model = MagicMock(name="FakeMLXQwen3Model")
    fake_model.get_supported_speakers = MagicMock(return_value=list(speakers or []))
    fake_model.get_supported_languages = MagicMock(return_value=list(languages or []))
    fake_model.sample_rate = sample_rate
    fake_model.generate_custom_voice = MagicMock(return_value=[])

    def fake_load_model(name):
        fake_model.name = name
        return fake_model

    fake_utils.load_model = fake_load_model
    # mlx.core is referenced inside _result_to_numpy; provide a minimal shim
    fake_core.eval = MagicMock(name="mlx.core.eval")

    monkeypatch.setitem(sys.modules, "mlx_audio", fake_pkg)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts", fake_tts)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "mlx_audio.core", fake_core)

    return fake_model, fake_load_model


class TestMLXBackendSelection:
    """Test Apple Silicon MLX backend selection."""

    def teardown_method(self):
        reset_backend()

    def test_mlx_backend_via_env(self, monkeypatch):
        """``TTS_BACKEND=mlx`` should instantiate the MLX backend."""
        _install_fake_mlx_audio(monkeypatch)
        monkeypatch.setenv("TTS_BACKEND", "mlx")

        backend = get_backend()
        assert backend.get_backend_name() == "mlx"
        from api.backends.mlx_qwen3_tts import MLXQwen3TTSBackend
        assert isinstance(backend, MLXQwen3TTSBackend)

    def test_mlx_default_model_id(self, monkeypatch):
        """Default ``MLX_MODEL_ID`` is the 0.6B CustomVoice checkpoint."""
        _install_fake_mlx_audio(monkeypatch)
        monkeypatch.setenv("TTS_BACKEND", "mlx")
        monkeypatch.delenv("MLX_MODEL_ID", raising=False)

        backend = get_backend()
        assert backend.get_model_id() == "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"

    def test_mlx_model_id_override(self, monkeypatch):
        """``MLX_MODEL_ID`` env var overrides the default checkpoint."""
        _install_fake_mlx_audio(monkeypatch)
        monkeypatch.setenv("TTS_BACKEND", "mlx")
        monkeypatch.setenv("MLX_MODEL_ID", "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit")

        backend = get_backend()
        assert backend.get_model_id() == "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"

    def test_mlx_backend_in_error_message(self, monkeypatch):
        """The 'unknown backend' error must list mlx as a valid option."""
        _install_fake_mlx_audio(monkeypatch)
        monkeypatch.setenv("TTS_BACKEND", "definitely_not_a_real_backend")

        with pytest.raises(ValueError) as exc_info:
            get_backend()
        assert "'mlx'" in str(exc_info.value)

    def test_mlx_does_not_eagerly_import(self):
        """``factory`` must not import mlx-audio at module load time.

        The MLX backend is only available on Apple Silicon. Linux and CUDA
        users must be able to import ``api.backends.factory`` without
        mlx-audio being installed, so the import must live inside the
        ``elif backend_type == "mlx":`` branch, not at the top of the file.
        """
        import inspect
        from api.backends import factory

        source = inspect.getsource(factory)

        # Top-level (column-0) statements must not import mlx-audio.
        for line in source.splitlines():
            if not line.strip():
                continue
            # Top-level: column 0 (no leading whitespace) and not inside a
            # multi-line string we opened earlier. A safe check: lines
            # that begin a top-level statement start with no leading
            # whitespace and are not continuation lines.
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if indent == 0 and (
                stripped.startswith("import mlx")
                or stripped.startswith("from mlx")
                or stripped.startswith("from .mlx_qwen3_tts")
            ):
                raise AssertionError(
                    f"factory.py imports mlx-audio at module scope: {line!r}"
                )

        # The lazy import lives inside the mlx branch — verify it's there.
        assert "from .mlx_qwen3_tts import" in source

    def test_mlx_supports_voice_cloning_is_false(self, monkeypatch):
        """CustomVoice checkpoint must not advertise voice cloning."""
        _install_fake_mlx_audio(monkeypatch)
        from api.backends.mlx_qwen3_tts import MLXQwen3TTSBackend
        backend = MLXQwen3TTSBackend()
        assert backend.supports_voice_cloning() is False

    def test_mlx_get_model_type(self, monkeypatch):
        """The default MLX checkpoint is a CustomVoice model."""
        _install_fake_mlx_audio(monkeypatch)
        from api.backends.mlx_qwen3_tts import MLXQwen3TTSBackend
        backend = MLXQwen3TTSBackend()
        assert backend.get_model_type() == "customvoice"

    def test_mlx_device_info_advertises_metal(self, monkeypatch):
        """``get_device_info`` should report Metal as the device."""
        _install_fake_mlx_audio(monkeypatch)
        from api.backends.mlx_qwen3_tts import MLXQwen3TTSBackend
        backend = MLXQwen3TTSBackend()
        info = backend.get_device_info()
        assert info["device"] == "metal"
        assert info["gpu_available"] is True
        # Apple Silicon uses unified memory, not dedicated VRAM
        assert info["vram_total"] is None
        assert info["vram_used"] is None

    def test_mlx_falls_back_to_default_voices(self, monkeypatch):
        """When the model reports no speakers, fall back to the baked-in list."""
        _install_fake_mlx_audio(monkeypatch, speakers=[], languages=[])
        from api.backends.mlx_qwen3_tts import (
            MLXQwen3TTSBackend,
            FALLBACK_VOICES,
            FALLBACK_LANGUAGES,
        )
        backend = MLXQwen3TTSBackend()
        assert backend.get_supported_voices() == FALLBACK_VOICES
        assert backend.get_supported_languages() == FALLBACK_LANGUAGES


class TestMLXBackendInitialize:
    """Test that ``initialize()`` guards and loads correctly."""

    def teardown_method(self):
        reset_backend()

    def test_initialize_requires_apple_silicon(self, monkeypatch):
        """``initialize()`` must raise on non-darwin / non-arm64 hosts."""
        _install_fake_mlx_audio(monkeypatch)
        from api.backends.mlx_qwen3_tts import MLXQwen3TTSBackend
        backend = MLXQwen3TTSBackend()

        with patch("sys.platform", "linux"), patch("platform.machine", return_value="x86_64"):
            with pytest.raises(RuntimeError, match="Apple Silicon"):
                asyncio.get_event_loop().run_until_complete(backend.initialize())
        assert backend.is_ready() is False

    def test_initialize_calls_load_model(self, monkeypatch):
        """On Apple Silicon, ``initialize()`` must call ``mlx_audio.tts.utils.load_model``."""
        _install_fake_mlx_audio(monkeypatch, speakers=["Vivian", "Ryan"], languages=["English"])
        from api.backends.mlx_qwen3_tts import MLXQwen3TTSBackend

        backend = MLXQwen3TTSBackend(model_name="custom/mlx-checkpoint")

        with patch("sys.platform", "darwin"), patch("platform.machine", return_value="arm64"):
            asyncio.get_event_loop().run_until_complete(backend.initialize())

        assert backend.is_ready() is True
        assert backend.model is not None
        assert backend.get_model_id() == "custom/mlx-checkpoint"
        assert set(backend.get_supported_voices()) == {"Vivian", "Ryan"}
        assert backend.get_supported_languages() == ["English"]

    def test_initialize_raises_when_mlx_audio_missing(self, monkeypatch):
        """``initialize()`` should give a clear error when mlx-audio is not installed."""
        # Make sure no fake mlx_audio is on sys.modules
        for mod in list(sys.modules):
            if mod.startswith("mlx_audio"):
                monkeypatch.delitem(sys.modules, mod, raising=False)

        from api.backends.mlx_qwen3_tts import MLXQwen3TTSBackend
        backend = MLXQwen3TTSBackend()

        # Block the import inside the executor as well
        import builtins
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "mlx_audio.tts.utils" or name == "mlx_audio":
                raise ImportError("No module named 'mlx_audio'")
            return real_import(name, *args, **kwargs)

        with patch("sys.platform", "darwin"), patch("platform.machine", return_value="arm64"), \
             patch.object(builtins, "__import__", side_effect=guarded_import):
            with pytest.raises(RuntimeError, match=r"\[api,mlx\]"):
                asyncio.get_event_loop().run_until_complete(backend.initialize())


class TestMLXVoiceResolution:
    """Test that voice name resolution works as documented."""

    def teardown_method(self):
        reset_backend()

    def test_known_voice_passes_through(self, monkeypatch):
        """A voice present in the model's speaker list is accepted as-is."""
        _install_fake_mlx_audio(monkeypatch, speakers=["vivian", "ryan", "serena"])
        from api.backends.mlx_qwen3_tts import MLXQwen3TTSBackend
        backend = MLXQwen3TTSBackend()
        assert backend._resolve_voice("Ryan") == "ryan"
        assert backend._resolve_voice("ryan") == "ryan"  # case-insensitive
        assert backend._resolve_voice("RYAN") == "ryan"

    def test_compatibility_aliases_map_to_supported_speakers(self, monkeypatch):
        """OpenAI aliases (sophia, lily, evan, isabella) map to MLX speakers."""
        _install_fake_mlx_audio(monkeypatch, speakers=["vivian", "serena", "aiden"])
        from api.backends.mlx_qwen3_tts import MLXQwen3TTSBackend
        backend = MLXQwen3TTSBackend()
        assert backend._resolve_voice("sophia") == "serena"
        assert backend._resolve_voice("Sophia") == "serena"
        assert backend._resolve_voice("lily") == "serena"
        assert backend._resolve_voice("isabella") == "vivian"
        assert backend._resolve_voice("evan") == "aiden"

    def test_unknown_voice_raises(self, monkeypatch):
        """Unknown voices raise ``ValueError`` with a helpful message."""
        _install_fake_mlx_audio(monkeypatch, speakers=["vivian", "ryan"])
        from api.backends.mlx_qwen3_tts import MLXQwen3TTSBackend
        backend = MLXQwen3TTSBackend()
        with pytest.raises(ValueError, match="Unsupported MLX voice"):
            backend._resolve_voice("NotARealSpeaker")


class TestMLXGenerateSpeech:
    """Test ``generate_speech`` against a fake MLX model."""

    def teardown_method(self):
        reset_backend()

    def test_generate_speech_returns_numpy_waveform(self, monkeypatch):
        """End-to-end: backend -> MLX stub -> numpy float32 waveform + sr."""
        fake_model, _ = _install_fake_mlx_audio(
            monkeypatch,
            speakers=["Vivian", "Ryan", "Serena"],
            languages=["English"],
            sample_rate=24000,
        )

        # Build two stub result objects, each carrying an .audio array-like.
        # We rely on _result_to_numpy using np.asarray(...).reshape(-1).
        results = []
        for arr in (np.zeros(1200, dtype=np.float32), np.ones(800, dtype=np.float32) * 0.5):
            r = MagicMock()
            r.audio = arr
            results.append(r)
        fake_model.generate_custom_voice = MagicMock(return_value=results)

        from api.backends.mlx_qwen3_tts import MLXQwen3TTSBackend
        backend = MLXQwen3TTSBackend()

        with patch("sys.platform", "darwin"), patch("platform.machine", return_value="arm64"):
            audio, sr = asyncio.get_event_loop().run_until_complete(
                backend.generate_speech(
                    text="Hello MLX.",
                    voice="Ryan",
                    language="English",
                )
            )

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert audio.shape == (2000,)
        # First chunk is zeros, second is 0.5
        np.testing.assert_array_equal(audio[:1200], 0.0)
        np.testing.assert_array_equal(audio[1200:], 0.5)
        assert sr == 24000
        # And the model was called with the right speaker
        fake_model.generate_custom_voice.assert_called_once()
        kwargs = fake_model.generate_custom_voice.call_args.kwargs
        assert kwargs["speaker"] == "Ryan"
        assert kwargs["language"] == "English"
        assert kwargs["text"] == "Hello MLX."

    def test_generate_speech_releases_unused_mlx_cache(self, monkeypatch):
        """Completed synthesis should release reusable Metal cache buffers."""
        fake_model, _ = _install_fake_mlx_audio(
            monkeypatch,
            speakers=["Vivian"],
            languages=["English"],
            sample_rate=24000,
        )
        result = MagicMock()
        result.audio = np.zeros(1200, dtype=np.float32)
        fake_model.generate_custom_voice = MagicMock(return_value=[result])

        fake_mlx = types.ModuleType("mlx")
        fake_core = types.ModuleType("mlx.core")
        fake_core.eval = MagicMock()
        fake_core.clear_cache = MagicMock()
        fake_mlx.core = fake_core
        monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
        monkeypatch.setitem(sys.modules, "mlx.core", fake_core)

        from api.backends.mlx_qwen3_tts import MLXQwen3TTSBackend
        backend = MLXQwen3TTSBackend()

        with patch("sys.platform", "darwin"), patch(
            "platform.machine",
            return_value="arm64",
        ):
            asyncio.get_event_loop().run_until_complete(
                backend.generate_speech(
                    text="Release the cache.",
                    voice="Vivian",
                    language="English",
                )
            )

        fake_core.clear_cache.assert_called_once_with()

    def test_generate_speech_routes_alias_to_speaker(self, monkeypatch):
        """The OpenAI alias ``lily`` should be resolved to ``Serena`` before calling the model."""
        fake_model, _ = _install_fake_mlx_audio(
            monkeypatch,
            speakers=["Vivian", "Serena", "Aiden"],
            languages=["English"],
        )
        fake_model.generate_custom_voice = MagicMock(return_value=[])

        from api.backends.mlx_qwen3_tts import MLXQwen3TTSBackend
        backend = MLXQwen3TTSBackend()

        with patch("sys.platform", "darwin"), patch("platform.machine", return_value="arm64"):
            with pytest.raises(RuntimeError, match="no audio samples"):
                asyncio.get_event_loop().run_until_complete(
                    backend.generate_speech(
                        text="alias check",
                        voice="lily",
                        language="English",
                    )
                )

        kwargs = fake_model.generate_custom_voice.call_args.kwargs
        assert kwargs["speaker"] == "Serena"

    def test_generate_speech_passes_instruct(self, monkeypatch):
        """``instruct`` should be forwarded to the model when provided."""
        fake_model, _ = _install_fake_mlx_audio(
            monkeypatch,
            speakers=["Vivian"],
            languages=["English"],
        )
        fake_model.generate_custom_voice = MagicMock(return_value=[])

        from api.backends.mlx_qwen3_tts import MLXQwen3TTSBackend
        backend = MLXQwen3TTSBackend()

        with patch("sys.platform", "darwin"), patch("platform.machine", return_value="arm64"):
            with pytest.raises(RuntimeError):
                asyncio.get_event_loop().run_until_complete(
                    backend.generate_speech(
                        text="hi",
                        voice="Vivian",
                        language="English",
                        instruct="Warm, calm.",
                    )
                )

        kwargs = fake_model.generate_custom_voice.call_args.kwargs
        assert kwargs.get("instruct") == "Warm, calm."

    def test_generate_speech_no_audio_raises(self, monkeypatch):
        """Empty audio list -> ``RuntimeError``."""
        fake_model, _ = _install_fake_mlx_audio(
            monkeypatch,
            speakers=["Vivian"],
            languages=["English"],
        )
        fake_model.generate_custom_voice = MagicMock(return_value=[])

        from api.backends.mlx_qwen3_tts import MLXQwen3TTSBackend
        backend = MLXQwen3TTSBackend()

        with patch("sys.platform", "darwin"), patch("platform.machine", return_value="arm64"):
            with pytest.raises(RuntimeError, match="no audio samples"):
                asyncio.get_event_loop().run_until_complete(
                    backend.generate_speech(
                        text="hi",
                        voice="Vivian",
                        language="English",
                    )
                )


class TestMLXGenerationSafetyCaps:
    """Test the wall-clock and audio-length safety caps.

    mlx-audio 0.3.x can hang in the model's graph compile, either
    producing looping audio (non-streaming) or never finishing
    (streaming). The backend must fail loudly instead of wedging
    the server.
    """

    def teardown_method(self):
        reset_backend()

    def test_generate_speech_aborts_on_audio_length_cap(self, monkeypatch):
        """If the model produces more than MAX_GENERATED_AUDIO_SECONDS,
        ``generate_speech`` must raise rather than return garbage."""
        from api.backends.mlx_qwen3_tts import MLXQwen3TTSBackend

        fake_model, _ = _install_fake_mlx_audio(
            monkeypatch,
            speakers=["vivian"],
            languages=["english"],
            sample_rate=24000,
        )

        # Make the fake generator produce 1 second of audio per
        # yield — way more than MAX_GENERATED_AUDIO_SECONDS so the
        # cap fires.
        def infinite_yields(**kwargs):
            chunk = MagicMock()
            chunk.audio = np.zeros(24000, dtype=np.float32)
            while True:
                yield chunk

        fake_model.generate_custom_voice = MagicMock(side_effect=infinite_yields)

        backend = MLXQwen3TTSBackend()
        with patch("sys.platform", "darwin"), patch("platform.machine", return_value="arm64"):
            with pytest.raises(RuntimeError, match="more than"):
                asyncio.get_event_loop().run_until_complete(
                    backend.generate_speech(
                        text="trigger the cap",
                        voice="vivian",
                        language="english",
                    )
                )

    def test_generate_speech_aborts_on_wall_clock_cap(self, monkeypatch):
        """If the model takes longer than MAX_GENERATION_WALL_SECONDS
        wall-clock, ``generate_speech`` must raise rather than hang."""
        from api.backends.mlx_qwen3_tts import (
            MLXQwen3TTSBackend,
            MAX_GENERATION_WALL_SECONDS,
        )

        fake_model, _ = _install_fake_mlx_audio(
            monkeypatch,
            speakers=["vivian"],
            languages=["english"],
            sample_rate=24000,
        )

        # Patch the time module used by the backend so we can trip
        # the wall-clock check without sleeping for real minutes.
        import api.backends.mlx_qwen3_tts as mlx_module
        import time as real_time
        fake_now = [real_time.monotonic()]

        def fake_monotonic():
            fake_now[0] += 1.0
            return fake_now[0]

        def slow_yields(**kwargs):
            # Yield tiny chunks very slowly. Each ``next()`` advances
            # fake_now by 1s of wall clock, so we'll trip the cap
            # after a few iterations.
            for i in range(int(MAX_GENERATION_WALL_SECONDS) + 10):
                chunk = MagicMock()
                chunk.audio = np.zeros(100, dtype=np.float32)  # tiny
                yield chunk

        fake_model.generate_custom_voice = MagicMock(side_effect=slow_yields)

        backend = MLXQwen3TTSBackend()
        with patch("sys.platform", "darwin"), patch("platform.machine", return_value="arm64"), \
             patch.object(mlx_module.time, "monotonic", side_effect=fake_monotonic):
            with pytest.raises(RuntimeError, match="exceeded|wedged"):
                asyncio.get_event_loop().run_until_complete(
                    backend.generate_speech(
                        text="trigger the wall clock cap",
                        voice="vivian",
                        language="english",
                    )
                )

        # And the backend should now be marked as broken.
        assert backend._broken is True

    def test_generate_speech_streaming_aborts_on_wall_clock(self, monkeypatch):
        """Same cap must apply to the streaming path."""
        from api.backends.mlx_qwen3_tts import (
            MLXQwen3TTSBackend,
            MAX_GENERATION_WALL_SECONDS,
        )
        import api.backends.mlx_qwen3_tts as mlx_module
        import time as real_time

        fake_model, _ = _install_fake_mlx_audio(
            monkeypatch,
            speakers=["vivian"],
            languages=["english"],
            sample_rate=24000,
        )

        fake_now = [real_time.monotonic()]

        def fake_monotonic():
            fake_now[0] += 1.0
            return fake_now[0]

        def slow_stream(**kwargs):
            for i in range(int(MAX_GENERATION_WALL_SECONDS) + 10):
                r = MagicMock()
                r.audio = np.zeros(100, dtype=np.float32)
                yield r

        fake_model.generate = MagicMock(side_effect=slow_stream)

        backend = MLXQwen3TTSBackend()

        async def run():
            agen = backend.generate_speech_streaming(
                text="trigger the streaming wall clock cap",
                voice="vivian",
                language="english",
            )
            async for _ in agen:
                pass

        with patch("sys.platform", "darwin"), patch("platform.machine", return_value="arm64"), \
             patch.object(mlx_module.time, "monotonic", side_effect=fake_monotonic):
            with pytest.raises(RuntimeError, match="exceeded|wedged"):
                asyncio.get_event_loop().run_until_complete(run())


def _collect_async_gen(agen):
    """Drain an async generator into a list using a fresh event loop.

    Always uses ``asyncio.new_event_loop()`` to avoid the 3.12
    deprecation around implicit event loops, and to make the
    generator's queue/thread interactions deterministic.
    """
    loop = asyncio.new_event_loop()
    try:
        out = []
        while True:
            try:
                out.append(loop.run_until_complete(agen.__anext__()))
            except StopAsyncIteration:
                return out
    finally:
        try:
            loop.run_until_complete(agen.aclose())
        except Exception:
            pass
        loop.close()


class TestMLXGenerateSpeechStreaming:
    """Test ``generate_speech_streaming`` against a fake MLX model."""

    def teardown_method(self):
        reset_backend()

    def test_streaming_yields_chunks(self, monkeypatch):
        """``generate_speech_streaming`` yields (pcm, sr) for each chunk."""
        fake_model, _ = _install_fake_mlx_audio(
            monkeypatch,
            speakers=["vivian", "ryan", "serena"],
            languages=["english"],
            sample_rate=24000,
        )

        # Build three stub result objects; their .audio is a real numpy
        # array (which is what _result_to_numpy expects).
        results = []
        for arr in (
            np.zeros(4800, dtype=np.float32),
            np.ones(4800, dtype=np.float32) * 0.25,
            np.ones(2400, dtype=np.float32) * -0.5,
        ):
            r = MagicMock()
            r.audio = arr
            results.append(r)

        # model.generate(..., stream=True) must be a *generator* (not a
        # list), so the worker can iterate it.
        def fake_generate_stream(**kwargs):
            for r in results:
                yield r

        fake_model.generate = MagicMock(side_effect=fake_generate_stream)

        from api.backends.mlx_qwen3_tts import MLXQwen3TTSBackend
        backend = MLXQwen3TTSBackend()

        with patch("sys.platform", "darwin"), patch("platform.machine", return_value="arm64"):
            agen = backend.generate_speech_streaming(
                text="streaming test",
                voice="Ryan",
                language="English",
            )
            chunks = _collect_async_gen(agen)

        # All three chunks were delivered
        assert len(chunks) == 3
        for chunk, sr in chunks:
            assert isinstance(chunk, np.ndarray)
            assert chunk.dtype == np.float32
            assert sr == 24000

        # Reassemble the full audio and confirm it matches the input
        full = np.concatenate([c for c, _ in chunks])
        assert full.shape == (12000,)
        np.testing.assert_array_equal(full[:4800], 0.0)
        np.testing.assert_array_equal(full[4800:9600], 0.25)
        np.testing.assert_array_equal(full[9600:], -0.5)

        # The unified generate API was called with stream=True and
        # the resolved lowercase voice
        fake_model.generate.assert_called_once()
        kwargs = fake_model.generate.call_args.kwargs
        assert kwargs["stream"] is True
        assert kwargs["voice"] == "ryan"
        assert kwargs["lang_code"] == "english"
        assert "streaming_interval" in kwargs

    def test_streaming_routes_alias_to_speaker(self, monkeypatch):
        """Voice alias ``lily`` should be resolved to ``serena`` before
        the model is called."""
        fake_model, _ = _install_fake_mlx_audio(
            monkeypatch,
            speakers=["vivian", "serena", "aiden"],
            languages=["english"],
        )
        fake_model.generate = MagicMock(return_value=iter(()))

        from api.backends.mlx_qwen3_tts import MLXQwen3TTSBackend
        backend = MLXQwen3TTSBackend()

        with patch("sys.platform", "darwin"), patch("platform.machine", return_value="arm64"):
            agen = backend.generate_speech_streaming(
                text="alias check",
                voice="lily",
                language="English",
            )
            _collect_async_gen(agen)

        kwargs = fake_model.generate.call_args.kwargs
        assert kwargs["voice"] == "serena"

    def test_streaming_propagates_worker_exception(self, monkeypatch):
        """If the worker thread raises, the async generator must
        surface that exception to the caller."""
        fake_model, _ = _install_fake_mlx_audio(
            monkeypatch,
            speakers=["vivian"],
            languages=["english"],
        )

        def boom(**kwargs):
            raise RuntimeError("mlx blew up")
            yield  # pragma: no cover  - make this a generator

        fake_model.generate = MagicMock(side_effect=boom)

        from api.backends.mlx_qwen3_tts import MLXQwen3TTSBackend
        backend = MLXQwen3TTSBackend()

        with patch("sys.platform", "darwin"), patch("platform.machine", return_value="arm64"):
            agen = backend.generate_speech_streaming(
                text="hi",
                voice="vivian",
                language="english",
            )
            with pytest.raises(RuntimeError, match="mlx blew up"):
                _collect_async_gen(agen)

    def test_streaming_falls_back_when_stream_rejected(self, monkeypatch):
        """If ``model.generate`` rejects ``stream=True`` (older API),
        the backend should fall back to the non-streaming
        ``generate_custom_voice`` and still produce chunks."""
        fake_model, _ = _install_fake_mlx_audio(
            monkeypatch,
            speakers=["vivian"],
            languages=["english"],
        )

        # First call: stream=True rejected. Second call (fallback):
        # generate_custom_voice returns one chunk.
        def generate_rejects_stream(**kwargs):
            if kwargs.get("stream"):
                raise TypeError(
                    "generate() got an unexpected keyword argument 'stream'"
                )
            # Should not be called
            yield  # pragma: no cover

        def generate_custom_voice_ok(**kwargs):
            r = MagicMock()
            r.audio = np.ones(4800, dtype=np.float32) * 0.3
            yield r

        fake_model.generate = MagicMock(side_effect=generate_rejects_stream)
        fake_model.generate_custom_voice = MagicMock(side_effect=generate_custom_voice_ok)

        from api.backends.mlx_qwen3_tts import MLXQwen3TTSBackend
        backend = MLXQwen3TTSBackend()

        with patch("sys.platform", "darwin"), patch("platform.machine", return_value="arm64"):
            agen = backend.generate_speech_streaming(
                text="fallback check",
                voice="vivian",
                language="english",
            )
            chunks = _collect_async_gen(agen)

        assert len(chunks) == 1
        chunk, sr = chunks[0]
        assert sr == 24000
        np.testing.assert_allclose(chunk, 0.3, atol=1e-6)
        fake_model.generate_custom_voice.assert_called_once()


class TestMLXLanguageResolution:
    """Test the language-name normalizer."""

    def teardown_method(self):
        reset_backend()

    def test_resolve_known_language(self, monkeypatch):
        """Capitalized ``English`` resolves to the lowercase model form."""
        _install_fake_mlx_audio(monkeypatch, speakers=["vivian"], languages=["english"])
        from api.backends.mlx_qwen3_tts import MLXQwen3TTSBackend
        backend = MLXQwen3TTSBackend()
        assert backend._resolve_language("English") == "english"
        assert backend._resolve_language("english") == "english"
        assert backend._resolve_language("ENGLISH") == "english"

    def test_resolve_auto(self, monkeypatch):
        """``Auto`` maps to the model's ``auto`` form."""
        _install_fake_mlx_audio(monkeypatch, speakers=["vivian"], languages=["auto"])
        from api.backends.mlx_qwen3_tts import MLXQwen3TTSBackend
        backend = MLXQwen3TTSBackend()
        assert backend._resolve_language("Auto") == "auto"

    def test_resolve_empty(self, monkeypatch):
        """Empty / None-like inputs become ``auto``."""
        _install_fake_mlx_audio(monkeypatch, speakers=["vivian"], languages=["auto"])
        from api.backends.mlx_qwen3_tts import MLXQwen3TTSBackend
        backend = MLXQwen3TTSBackend()
        assert backend._resolve_language("") == "auto"

    def test_resolve_unknown_passes_through_lowercased(self, monkeypatch):
        """Unknown languages are lowercased but not raised — the model
        will be told to do its best."""
        _install_fake_mlx_audio(monkeypatch, speakers=["vivian"], languages=["english"])
        from api.backends.mlx_qwen3_tts import MLXQwen3TTSBackend
        backend = MLXQwen3TTSBackend()
        assert backend._resolve_language("Klingon") == "klingon"
