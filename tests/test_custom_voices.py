# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Tests for custom voice directory support.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import numpy as np

from api.backends.base import TTSBackend
from api.backends.official_qwen3_tts import OfficialQwen3TTSBackend
from api.backends.vllm_omni_qwen3_tts import VLLMOmniQwen3TTSBackend
from api.backends.factory import reset_backend


@pytest.fixture(autouse=True)
def reset_backend_after_test():
    """Reset backend after each test."""
    yield
    reset_backend()


# ---------------------------------------------------------------------------
# Base class defaults
# ---------------------------------------------------------------------------

class TestBaseCustomVoiceInterface:
    """Tests for custom voice methods on the base TTSBackend."""

    def test_custom_voices_dict_initialized_empty(self):
        """_custom_voices should be an empty dict on init."""
        backend = OfficialQwen3TTSBackend()
        assert backend._custom_voices == {}

    def test_is_custom_voice_false_by_default(self):
        backend = OfficialQwen3TTSBackend()
        assert backend.is_custom_voice("anything") is False

    def test_get_custom_voice_names_empty_by_default(self):
        backend = OfficialQwen3TTSBackend()
        assert backend.get_custom_voice_names() == []

    @pytest.mark.asyncio
    async def test_generate_speech_with_custom_voice_raises_on_base(self):
        """Base class should raise NotImplementedError."""
        # Use vLLM backend since it doesn't override the method
        backend = VLLMOmniQwen3TTSBackend()
        with pytest.raises(NotImplementedError):
            await backend.generate_speech_with_custom_voice(
                text="hello", voice="test", language="Auto", speed=1.0
            )


# ---------------------------------------------------------------------------
# Official backend — load_custom_voices
# ---------------------------------------------------------------------------

class TestOfficialLoadCustomVoices:
    """Tests for OfficialQwen3TTSBackend.load_custom_voices."""

    @pytest.mark.asyncio
    async def test_nonexistent_directory_is_noop(self, tmp_path):
        """Non-existent directory should load nothing."""
        backend = OfficialQwen3TTSBackend(
            model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )
        await backend.load_custom_voices(str(tmp_path / "does_not_exist"))
        assert backend.get_custom_voice_names() == []

    @pytest.mark.asyncio
    async def test_custom_voice_model_skips_loading(self, tmp_path):
        """CustomVoice model (no cloning support) should skip loading."""
        voice_dir = tmp_path / "MyVoice"
        voice_dir.mkdir()
        (voice_dir / "reference.wav").write_bytes(b"fake")

        backend = OfficialQwen3TTSBackend(
            model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        )
        await backend.load_custom_voices(str(tmp_path))
        assert backend.get_custom_voice_names() == []

    @pytest.mark.asyncio
    async def test_collision_with_builtin_voice_is_skipped(self, tmp_path):
        """A folder named after a built-in voice should be skipped."""
        # "Vivian" is a built-in voice
        voice_dir = tmp_path / "Vivian"
        voice_dir.mkdir()
        (voice_dir / "reference.wav").write_bytes(b"fake")

        backend = OfficialQwen3TTSBackend(
            model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )
        # Mock model to avoid real init
        backend._ready = True
        backend.model = MagicMock()

        await backend.load_custom_voices(str(tmp_path))
        assert "Vivian" not in backend.get_custom_voice_names()

    @pytest.mark.asyncio
    async def test_case_insensitive_collision(self, tmp_path):
        """Collision check should be case-insensitive."""
        voice_dir = tmp_path / "vivian"
        voice_dir.mkdir()
        (voice_dir / "reference.wav").write_bytes(b"fake")

        backend = OfficialQwen3TTSBackend(
            model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )
        backend._ready = True
        backend.model = MagicMock()

        await backend.load_custom_voices(str(tmp_path))
        assert "vivian" not in backend.get_custom_voice_names()

    @pytest.mark.asyncio
    async def test_missing_reference_audio_is_skipped(self, tmp_path):
        """Folder without reference audio should be skipped."""
        voice_dir = tmp_path / "NoAudio"
        voice_dir.mkdir()
        (voice_dir / "readme.txt").write_text("not an audio file")

        backend = OfficialQwen3TTSBackend(
            model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )
        backend._ready = True
        backend.model = MagicMock()

        await backend.load_custom_voices(str(tmp_path))
        assert backend.get_custom_voice_names() == []

    @pytest.mark.asyncio
    async def test_hidden_directories_are_skipped(self, tmp_path):
        """Directories starting with '.' should be skipped."""
        voice_dir = tmp_path / ".hidden"
        voice_dir.mkdir()
        (voice_dir / "reference.wav").write_bytes(b"fake")

        backend = OfficialQwen3TTSBackend(
            model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )
        backend._ready = True
        backend.model = MagicMock()

        await backend.load_custom_voices(str(tmp_path))
        assert backend.get_custom_voice_names() == []

    @pytest.mark.asyncio
    async def test_cache_miss_calls_create_and_saves(self, tmp_path):
        """On cache miss, should call create_voice_clone_prompt and torch.save."""
        voice_dir = tmp_path / "TestVoice"
        voice_dir.mkdir()
        (voice_dir / "reference.wav").write_bytes(b"fake")

        backend = OfficialQwen3TTSBackend(
            model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )
        backend._ready = True
        backend.model = MagicMock()

        fake_prompt = [MagicMock()]
        backend.model.create_voice_clone_prompt.return_value = fake_prompt

        with patch("torch.save") as mock_save:
            await backend.load_custom_voices(str(tmp_path))

        backend.model.create_voice_clone_prompt.assert_called_once_with(
            ref_audio=str(voice_dir / "reference.wav"),
            ref_text=None,
            x_vector_only_mode=True,
        )
        mock_save.assert_called_once_with(fake_prompt, voice_dir / ".cached_prompt.pt")
        assert "TestVoice" in backend.get_custom_voice_names()
        assert backend._custom_voices["TestVoice"] is fake_prompt

    @pytest.mark.asyncio
    async def test_icl_mode_when_ref_text_exists(self, tmp_path):
        """reference.txt present → x_vector_only_mode=False, ref_text passed."""
        voice_dir = tmp_path / "ICLVoice"
        voice_dir.mkdir()
        (voice_dir / "reference.wav").write_bytes(b"fake")
        (voice_dir / "reference.txt").write_text("Hello this is my voice.")

        backend = OfficialQwen3TTSBackend(
            model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )
        backend._ready = True
        backend.model = MagicMock()
        backend.model.create_voice_clone_prompt.return_value = [MagicMock()]

        with patch("torch.save"):
            await backend.load_custom_voices(str(tmp_path))

        backend.model.create_voice_clone_prompt.assert_called_once_with(
            ref_audio=str(voice_dir / "reference.wav"),
            ref_text="Hello this is my voice.",
            x_vector_only_mode=False,
        )

    @pytest.mark.asyncio
    async def test_empty_ref_text_treated_as_absent(self, tmp_path):
        """Empty reference.txt should behave like no reference.txt."""
        voice_dir = tmp_path / "EmptyTxt"
        voice_dir.mkdir()
        (voice_dir / "reference.wav").write_bytes(b"fake")
        (voice_dir / "reference.txt").write_text("   ")

        backend = OfficialQwen3TTSBackend(
            model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )
        backend._ready = True
        backend.model = MagicMock()
        backend.model.create_voice_clone_prompt.return_value = [MagicMock()]

        with patch("torch.save"):
            await backend.load_custom_voices(str(tmp_path))

        backend.model.create_voice_clone_prompt.assert_called_once_with(
            ref_audio=str(voice_dir / "reference.wav"),
            ref_text=None,
            x_vector_only_mode=True,
        )

    @pytest.mark.asyncio
    async def test_cache_hit_loads_from_disk(self, tmp_path):
        """When .cached_prompt.pt exists, should torch.load and skip extraction."""
        voice_dir = tmp_path / "CachedVoice"
        voice_dir.mkdir()
        (voice_dir / "reference.wav").write_bytes(b"fake")
        (voice_dir / ".cached_prompt.pt").write_bytes(b"fake_cache")

        backend = OfficialQwen3TTSBackend(
            model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )
        backend._ready = True
        backend.model = MagicMock()
        backend.device = "cpu"

        fake_prompt = [MagicMock()]

        with patch("torch.load", return_value=fake_prompt) as mock_load:
            await backend.load_custom_voices(str(tmp_path))

        mock_load.assert_called_once_with(
            voice_dir / ".cached_prompt.pt",
            map_location="cpu",
            weights_only=False,
        )
        # create_voice_clone_prompt should NOT have been called
        backend.model.create_voice_clone_prompt.assert_not_called()
        assert backend._custom_voices["CachedVoice"] is fake_prompt

    @pytest.mark.asyncio
    async def test_multiple_audio_extensions_detected(self, tmp_path):
        """Should find reference.mp3, .flac, .ogg, etc."""
        for ext in ("mp3", "flac", "ogg", "m4a"):
            voice_dir = tmp_path / f"Voice_{ext}"
            voice_dir.mkdir()
            (voice_dir / f"reference.{ext}").write_bytes(b"fake")

        backend = OfficialQwen3TTSBackend(
            model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )
        backend._ready = True
        backend.model = MagicMock()
        backend.model.create_voice_clone_prompt.return_value = [MagicMock()]

        with patch("torch.save"):
            await backend.load_custom_voices(str(tmp_path))

        assert len(backend.get_custom_voice_names()) == 4

    @pytest.mark.asyncio
    async def test_multiple_voices_loaded(self, tmp_path):
        """Multiple valid voice directories should all load."""
        for name in ("Alice", "Bob", "Charlie"):
            voice_dir = tmp_path / name
            voice_dir.mkdir()
            (voice_dir / "reference.wav").write_bytes(b"fake")

        backend = OfficialQwen3TTSBackend(
            model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )
        backend._ready = True
        backend.model = MagicMock()
        backend.model.create_voice_clone_prompt.return_value = [MagicMock()]

        with patch("torch.save"):
            await backend.load_custom_voices(str(tmp_path))

        names = backend.get_custom_voice_names()
        assert "Alice" in names
        assert "Bob" in names
        assert "Charlie" in names

    @pytest.mark.asyncio
    async def test_extraction_failure_skips_voice(self, tmp_path):
        """If create_voice_clone_prompt raises, the voice is skipped."""
        voice_dir = tmp_path / "BadVoice"
        voice_dir.mkdir()
        (voice_dir / "reference.wav").write_bytes(b"fake")

        backend = OfficialQwen3TTSBackend(
            model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )
        backend._ready = True
        backend.model = MagicMock()
        backend.model.create_voice_clone_prompt.side_effect = RuntimeError("extraction failed")

        with patch("torch.save"):
            await backend.load_custom_voices(str(tmp_path))

        assert "BadVoice" not in backend.get_custom_voice_names()


# ---------------------------------------------------------------------------
# Official backend — generate_speech_with_custom_voice
# ---------------------------------------------------------------------------

class TestOfficialGenerateCustomVoice:
    """Tests for OfficialQwen3TTSBackend.generate_speech_with_custom_voice."""

    @pytest.mark.asyncio
    async def test_routes_through_model_voice_clone(self):
        """Should call model.generate_voice_clone with cached prompt."""
        backend = OfficialQwen3TTSBackend(
            model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )
        backend._ready = True
        backend.model = MagicMock()

        fake_prompt = [MagicMock()]
        backend._custom_voices["TestVoice"] = fake_prompt

        fake_audio = np.zeros(16000, dtype=np.float32)
        backend.model.generate_voice_clone.return_value = ([fake_audio], 24000)

        audio, sr = await backend.generate_speech_with_custom_voice(
            text="Hello", voice="TestVoice", language="English", speed=1.0
        )

        backend.model.generate_voice_clone.assert_called_once_with(
            text="Hello",
            language="English",
            voice_clone_prompt=fake_prompt,
        )
        assert sr == 24000
        assert isinstance(audio, np.ndarray)

    @pytest.mark.asyncio
    async def test_unknown_custom_voice_raises(self):
        """Requesting a non-existent custom voice should raise RuntimeError."""
        backend = OfficialQwen3TTSBackend(
            model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )
        backend._ready = True
        backend.model = MagicMock()

        with pytest.raises(RuntimeError, match="not found"):
            await backend.generate_speech_with_custom_voice(
                text="Hello", voice="NoSuchVoice", language="Auto", speed=1.0
            )


# ---------------------------------------------------------------------------
# get_supported_voices includes custom voices
# ---------------------------------------------------------------------------

class TestGetSupportedVoicesIncludesCustom:

    def test_custom_voices_appended(self):
        backend = OfficialQwen3TTSBackend()
        backend._custom_voices = {"MyClone": "fake_prompt"}

        voices = backend.get_supported_voices()
        # Built-in voices should still be there
        assert "Vivian" in voices
        # Custom voice should be appended
        assert "MyClone" in voices


# ---------------------------------------------------------------------------
# vLLM backend — load_custom_voices warns only
# ---------------------------------------------------------------------------

class TestVllmCustomVoices:

    @pytest.mark.asyncio
    async def test_warns_when_voice_dirs_exist(self, tmp_path, caplog):
        """vLLM backend should warn if custom voice folders are present."""
        voice_dir = tmp_path / "SomeVoice"
        voice_dir.mkdir()

        backend = VLLMOmniQwen3TTSBackend()

        import logging
        with caplog.at_level(logging.WARNING):
            await backend.load_custom_voices(str(tmp_path))

        assert "does not support voice cloning" in caplog.text
        assert backend.get_custom_voice_names() == []

    @pytest.mark.asyncio
    async def test_no_warn_for_empty_dir(self, tmp_path, caplog):
        """No warning if the custom voices directory has no subdirs."""
        backend = VLLMOmniQwen3TTSBackend()

        import logging
        with caplog.at_level(logging.WARNING):
            await backend.load_custom_voices(str(tmp_path))

        assert "does not support voice cloning" not in caplog.text

    @pytest.mark.asyncio
    async def test_nonexistent_dir_is_noop(self, tmp_path):
        """Non-existent directory should silently do nothing."""
        backend = VLLMOmniQwen3TTSBackend()
        await backend.load_custom_voices(str(tmp_path / "nope"))
        assert backend.get_custom_voice_names() == []


# ---------------------------------------------------------------------------
# Factory — initialize_backend calls load_custom_voices
# ---------------------------------------------------------------------------

class TestFactoryCustomVoiceIntegration:

    @pytest.mark.asyncio
    async def test_initialize_backend_calls_load_custom_voices(self, monkeypatch):
        """initialize_backend should invoke load_custom_voices on the backend."""
        from api.backends import factory

        mock_backend = MagicMock()
        mock_backend.initialize = AsyncMock()
        mock_backend.load_custom_voices = AsyncMock()
        mock_backend.is_ready.return_value = True

        monkeypatch.setattr(factory, "_backend_instance", mock_backend)

        # Disable warmup
        monkeypatch.setenv("TTS_WARMUP_ON_START", "false")

        await factory.initialize_backend(warmup=False)

        mock_backend.load_custom_voices.assert_called_once()

    @pytest.mark.asyncio
    async def test_custom_voices_dir_from_env(self, monkeypatch, tmp_path):
        """TTS_CUSTOM_VOICES env var should be forwarded to load_custom_voices."""
        from api.backends import factory

        custom_dir = str(tmp_path / "my_voices")
        monkeypatch.setenv("TTS_CUSTOM_VOICES", custom_dir)

        mock_backend = MagicMock()
        mock_backend.initialize = AsyncMock()
        mock_backend.load_custom_voices = AsyncMock()

        monkeypatch.setattr(factory, "_backend_instance", mock_backend)

        await factory.initialize_backend(warmup=False)

        mock_backend.load_custom_voices.assert_called_once_with(custom_dir)

    @pytest.mark.asyncio
    async def test_load_custom_voices_failure_is_non_fatal(self, monkeypatch):
        """If load_custom_voices raises, initialize_backend should still succeed."""
        from api.backends import factory

        mock_backend = MagicMock()
        mock_backend.initialize = AsyncMock()
        mock_backend.load_custom_voices = AsyncMock(
            side_effect=RuntimeError("boom")
        )

        monkeypatch.setattr(factory, "_backend_instance", mock_backend)

        # Should not raise
        await factory.initialize_backend(warmup=False)


# ---------------------------------------------------------------------------
# Router — speech generation routes through custom voice
# ---------------------------------------------------------------------------

class TestRouterCustomVoiceRouting:

    def test_speech_with_custom_voice_routes_correctly(self, monkeypatch):
        """POST /v1/audio/speech with a custom voice should use custom path."""
        from fastapi.testclient import TestClient
        from api.main import app
        from api.backends import factory

        mock_backend = MagicMock()
        mock_backend.is_ready.return_value = True
        mock_backend.is_custom_voice.return_value = True
        mock_backend.generate_speech_with_custom_voice = AsyncMock(
            return_value=(np.zeros(16000, dtype=np.float32), 24000)
        )
        mock_backend.generate_speech = AsyncMock()

        factory._backend_instance = mock_backend

        client = TestClient(app)
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "qwen3-tts",
                "input": "Hello",
                "voice": "MyCustomVoice",
            },
        )

        assert response.status_code == 200
        mock_backend.generate_speech_with_custom_voice.assert_called_once()
        # Regular generate_speech should NOT have been called
        mock_backend.generate_speech.assert_not_called()

    def test_speech_with_builtin_voice_skips_custom_path(self, monkeypatch):
        """POST /v1/audio/speech with a built-in voice should NOT use custom path."""
        from fastapi.testclient import TestClient
        from api.main import app
        from api.backends import factory

        mock_backend = MagicMock()
        mock_backend.is_ready.return_value = True
        mock_backend.is_custom_voice.return_value = False
        mock_backend.generate_speech = AsyncMock(
            return_value=(np.zeros(16000, dtype=np.float32), 24000)
        )
        mock_backend.generate_speech_with_custom_voice = AsyncMock()

        factory._backend_instance = mock_backend

        client = TestClient(app)
        response = client.post(
            "/v1/audio/speech",
            json={
                "model": "qwen3-tts",
                "input": "Hello",
                "voice": "Vivian",
            },
        )

        assert response.status_code == 200
        mock_backend.generate_speech.assert_called_once()
        mock_backend.generate_speech_with_custom_voice.assert_not_called()


class TestVoicesEndpointCustomVoiceLabels:

    def test_custom_voices_get_distinct_description(self):
        """Custom voices in /v1/voices should have 'Custom cloned voice' description."""
        from fastapi.testclient import TestClient
        from api.main import app
        from api.backends import factory

        mock_backend = MagicMock()
        mock_backend.is_ready.return_value = True
        mock_backend.get_supported_voices.return_value = ["Vivian", "MyClone"]
        mock_backend.get_supported_languages.return_value = ["English"]
        mock_backend.is_custom_voice.side_effect = lambda v: v == "MyClone"

        factory._backend_instance = mock_backend

        client = TestClient(app)
        response = client.get("/v1/voices")
        assert response.status_code == 200

        data = response.json()
        voices_by_id = {v["id"]: v for v in data["voices"]}

        assert "MyClone" in voices_by_id
        assert "Custom cloned voice" in voices_by_id["MyClone"]["description"]

        assert "Vivian" in voices_by_id
        assert "Custom cloned voice" not in voices_by_id["Vivian"]["description"]
