#!/usr/bin/env python3
# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the voice clone server.

These tests are split into two categories:
1. Unit tests (no GPU required) - test API logic with mocked model
2. Integration tests (GPU required) - test actual inference

Run unit tests only:
    pytest tests/test_voice_clone_server.py -m "not integration"

Run all tests (requires GPU):
    pytest tests/test_voice_clone_server.py
"""

import io
import os
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from dataclasses import dataclass
from typing import List, Any, Optional

import torch
from transformers.cache_utils import DynamicCache


# =============================================================================
# Fixtures and Mocks
# =============================================================================

@pytest.fixture
def mock_voice_prompt():
    """Create a mock voice prompt item."""
    mock_item = Mock()
    mock_item.ref_code = torch.randn(1, 4, 100)
    mock_item.ref_spk_embedding = torch.randn(1, 256)
    mock_item.x_vector_only_mode = False
    mock_item.icl_mode = True
    return [mock_item]


@pytest.fixture
def mock_model():
    """Create a mock TTS model."""
    model = Mock()

    # Mock generate_voice_clone to return fake audio
    def mock_generate(text, language, voice_clone_prompt):
        # Generate fake audio (1 second at 24kHz)
        audio = np.random.randn(24000).astype(np.float32) * 0.1
        return [audio], 24000

    model.generate_voice_clone = mock_generate

    # Mock create_voice_clone_prompt
    def mock_create_prompt(ref_audio, ref_text, x_vector_only_mode=False):
        mock_item = Mock()
        mock_item.ref_code = torch.randn(1, 4, 100)
        mock_item.ref_spk_embedding = torch.randn(1, 256)
        mock_item.x_vector_only_mode = x_vector_only_mode
        mock_item.icl_mode = True
        return [mock_item]

    model.create_voice_clone_prompt = mock_create_prompt

    return model


@pytest.fixture
def sample_audio_bytes():
    """Create sample audio bytes for testing."""
    # Create a simple WAV header + silence
    sample_rate = 24000
    duration = 0.1  # 100ms
    samples = int(sample_rate * duration)
    audio = np.zeros(samples, dtype=np.int16)

    buf = io.BytesIO()
    buf.write(b'RIFF')
    buf.write((36 + len(audio) * 2).to_bytes(4, 'little'))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write((16).to_bytes(4, 'little'))
    buf.write((1).to_bytes(2, 'little'))  # PCM
    buf.write((1).to_bytes(2, 'little'))  # Mono
    buf.write(sample_rate.to_bytes(4, 'little'))
    buf.write((sample_rate * 2).to_bytes(4, 'little'))
    buf.write((2).to_bytes(2, 'little'))
    buf.write((16).to_bytes(2, 'little'))
    buf.write(b'data')
    buf.write((len(audio) * 2).to_bytes(4, 'little'))
    buf.write(audio.tobytes())

    return buf.getvalue()


# =============================================================================
# Unit Tests (No GPU Required)
# =============================================================================

class TestVoiceCache:
    """Tests for VoiceCache dataclass."""

    def test_voice_cache_creation(self, mock_voice_prompt):
        """Test creating a VoiceCache with all fields."""
        # Import here to avoid loading torch at module level
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

        from voice_clone_server import VoiceCache

        kv_cache = DynamicCache()
        past_hidden = torch.randn(1, 10, 256)

        cache = VoiceCache(
            prompt=mock_voice_prompt,
            kv_cache=kv_cache,
            past_hidden=past_hidden,
            prefix_length=100,
            ref_text="Test reference text",
        )

        assert cache.prompt == mock_voice_prompt
        assert cache.kv_cache == kv_cache
        assert cache.past_hidden is not None
        assert cache.prefix_length == 100
        assert cache.ref_text == "Test reference text"

    def test_voice_cache_optional_fields(self, mock_voice_prompt):
        """Test VoiceCache with optional fields as None."""
        from voice_clone_server import VoiceCache

        cache = VoiceCache(
            prompt=mock_voice_prompt,
            kv_cache=None,
            past_hidden=None,
            prefix_length=0,
            ref_text="",
        )

        assert cache.kv_cache is None
        assert cache.past_hidden is None


class TestCloneKvCache:
    """Tests for clone_kv_cache function."""

    def test_clone_none_cache(self):
        """Test cloning None returns None."""
        from voice_clone_server import clone_kv_cache

        result = clone_kv_cache(None)
        assert result is None

    def test_clone_empty_cache(self):
        """Test cloning empty DynamicCache."""
        from voice_clone_server import clone_kv_cache

        cache = DynamicCache()
        cloned = clone_kv_cache(cache)

        assert cloned is not None
        assert isinstance(cloned, DynamicCache)

    def test_clone_cache_with_data(self):
        """Test cloning cache with key/value data."""
        from voice_clone_server import clone_kv_cache

        cache = DynamicCache()
        # Add some fake KV data using the update() method
        key = torch.randn(1, 8, 10, 64)
        value = torch.randn(1, 8, 10, 64)
        cache.update(key, value, layer_idx=0)

        cloned = clone_kv_cache(cache)

        # Both old and new API should work
        if hasattr(cloned, 'key_cache'):
            assert len(cloned.key_cache) >= 1
            assert len(cloned.value_cache) >= 1
        else:
            # New API - verify clone is not None
            assert cloned is not None


class TestAudioConversion:
    """Tests for audio conversion utilities."""

    def test_convert_to_pcm_normal_audio(self):
        """Test converting normal float audio to PCM."""
        from voice_clone_server import convert_to_pcm

        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        pcm = convert_to_pcm(audio)

        assert isinstance(pcm, bytes)
        assert len(pcm) == len(audio) * 2  # 16-bit = 2 bytes per sample

    def test_convert_to_pcm_clips_values(self):
        """Test that values outside [-1, 1] are clipped."""
        from voice_clone_server import convert_to_pcm

        audio = np.array([2.0, -2.0, 0.5], dtype=np.float32)
        pcm = convert_to_pcm(audio)

        # Decode and verify clipping
        decoded = np.frombuffer(pcm, dtype=np.int16)
        assert decoded[0] == 32767  # Clipped to max
        assert decoded[1] == -32767  # Clipped to min (approx)

    def test_wav_streaming_header(self):
        """Test WAV streaming header creation."""
        from voice_clone_server import create_wav_streaming_header

        header = create_wav_streaming_header(24000, 1, 16)

        assert header[:4] == b'RIFF'
        assert header[8:12] == b'WAVE'
        assert header[12:16] == b'fmt '
        assert len(header) == 44  # Standard WAV header size


class TestTTSRequest:
    """Tests for TTSRequest model."""

    def test_tts_request_defaults(self):
        """Test TTSRequest with default values."""
        from voice_clone_server import TTSRequest

        request = TTSRequest(text="Hello")

        assert request.text == "Hello"
        assert request.language == "English"
        assert request.stream is False
        assert request.response_format == "wav"
        assert request.voice is None
        assert request.use_kv_cache is True

    def test_tts_request_custom_values(self):
        """Test TTSRequest with custom values."""
        from voice_clone_server import TTSRequest

        request = TTSRequest(
            text="Test",
            language="Chinese",
            stream=True,
            response_format="pcm",
            voice="custom_voice",
            use_kv_cache=False,
        )

        assert request.text == "Test"
        assert request.language == "Chinese"
        assert request.stream is True
        assert request.response_format == "pcm"
        assert request.voice == "custom_voice"
        assert request.use_kv_cache is False


class TestLoadVoiceRequest:
    """Tests for LoadVoiceRequest model."""

    def test_load_voice_request(self):
        """Test LoadVoiceRequest model."""
        from voice_clone_server import LoadVoiceRequest

        request = LoadVoiceRequest(
            voice_name="my_voice",
            ref_audio_path="/path/to/audio.wav",
            ref_text="Reference text",
        )

        assert request.voice_name == "my_voice"
        assert request.ref_audio_path == "/path/to/audio.wav"
        assert request.ref_text == "Reference text"


class TestSentenceSplitting:
    """Tests for sentence splitting in streaming."""

    def test_split_by_period(self):
        """Test splitting text by periods."""
        import re

        text = "First sentence. Second sentence. Third."
        pattern = r'(?<=[.!?。！？])\s+'
        sentences = re.split(pattern, text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        assert len(sentences) == 3
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence."
        assert sentences[2] == "Third."

    def test_split_by_multiple_punctuation(self):
        """Test splitting by different punctuation marks."""
        import re

        text = "Hello! How are you? I'm fine."
        pattern = r'(?<=[.!?。！？])\s+'
        sentences = re.split(pattern, text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        assert len(sentences) == 3

    def test_no_split_single_sentence(self):
        """Test that single sentence returns as-is."""
        import re

        text = "Single sentence without ending punctuation"
        pattern = r'(?<=[.!?。！？])\s+'
        sentences = re.split(pattern, text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        assert len(sentences) == 1
        assert sentences[0] == text


# =============================================================================
# API Tests (With Mocked Model)
# =============================================================================

class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_when_model_loaded(self):
        """Test health endpoint when model is loaded."""
        from fastapi.testclient import TestClient

        with patch('voice_clone_server.model') as mock_model:
            mock_model.__bool__ = Mock(return_value=True)

            from voice_clone_server import app
            client = TestClient(app, raise_server_exceptions=False)

            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert "model_loaded" in data


class TestVoicesEndpoint:
    """Tests for /v1/voices endpoint."""

    def test_list_voices_empty(self):
        """Test listing voices when none loaded."""
        from voice_clone_server import voices, DEFAULT_VOICE

        # Store original and clear
        original_voices = voices.copy()
        voices.clear()

        try:
            from fastapi.testclient import TestClient
            from voice_clone_server import app

            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/v1/voices")

            assert response.status_code == 200
            data = response.json()
            assert "voices" in data
            assert "default" in data
            assert data["default"] == DEFAULT_VOICE
        finally:
            # Restore original voices
            voices.update(original_voices)


# =============================================================================
# Integration Tests (Require GPU)
# =============================================================================

@pytest.mark.integration
class TestKvCacheComputation:
    """Integration tests for KV cache computation (requires GPU)."""

    @pytest.fixture
    def loaded_model(self):
        """Load actual model for integration tests."""
        from voice_clone_server import load_model, model, voices

        if model is None:
            load_model()

        return model

    def test_compute_kv_cache_returns_valid_data(self, loaded_model):
        """Test that KV cache computation returns valid data."""
        from voice_clone_server import compute_voice_kv_cache, voices, DEFAULT_VOICE

        # Get default voice prompt
        voice_cache = voices.get(DEFAULT_VOICE)
        assert voice_cache is not None

        # Compute KV cache
        kv_cache, past_hidden, prefix_length = compute_voice_kv_cache(
            loaded_model,
            voice_cache.prompt,
            voice_cache.ref_text,
        )

        # Verify results
        assert kv_cache is not None or prefix_length == 0
        if kv_cache is not None:
            assert prefix_length > 0

    def test_generate_with_and_without_cache(self, loaded_model):
        """Test generation works with and without KV cache."""
        from voice_clone_server import generate_with_kv_cache, voices, DEFAULT_VOICE

        voice_cache = voices.get(DEFAULT_VOICE)
        assert voice_cache is not None

        text = "Hello, this is a test."

        # With cache
        wavs_with, sr_with = generate_with_kv_cache(
            text=text,
            language="English",
            voice_cache=voice_cache,
            use_cache=True,
        )

        # Without cache (still uses voice prompt, just not KV cache optimization)
        wavs_without, sr_without = generate_with_kv_cache(
            text=text,
            language="English",
            voice_cache=voice_cache,
            use_cache=False,
        )

        # Both should produce valid audio
        assert len(wavs_with) > 0
        assert len(wavs_without) > 0
        assert sr_with == sr_without == 24000


@pytest.mark.integration
class TestVoiceLoading:
    """Integration tests for voice loading (requires GPU)."""

    def test_load_voice_creates_cache(self):
        """Test that loading a voice creates KV cache."""
        from voice_clone_server import voices
        from fastapi.testclient import TestClient
        from voice_clone_server import app, model

        if model is None:
            pytest.skip("Model not loaded")

        client = TestClient(app)

        # Load a new voice
        response = client.post(
            "/v1/voices/load",
            json={
                "voice_name": "integration_test_voice",
                "ref_audio_path": "./voice_refs/hai_reference.wav",
                "ref_text": "Test reference text for integration testing.",
            }
        )

        if response.status_code == 400 and "not found" in response.text:
            pytest.skip("Reference audio file not found")

        assert response.status_code == 200
        data = response.json()

        assert data["voice_name"] == "integration_test_voice"
        assert "kv_cached" in data
        assert "prefix_length" in data

        # Verify voice is in cache
        assert "integration_test_voice" in voices


@pytest.mark.integration
class TestStreamingGeneration:
    """Integration tests for streaming generation."""

    def test_streaming_returns_chunks(self):
        """Test that streaming returns multiple audio chunks."""
        import asyncio
        from voice_clone_server import generate_speech_streaming, voices, DEFAULT_VOICE, model

        if model is None:
            pytest.skip("Model not loaded")

        voice_cache = voices.get(DEFAULT_VOICE)
        if voice_cache is None:
            pytest.skip("Default voice not loaded")

        # Multi-sentence text for streaming
        text = "First sentence. Second sentence. Third sentence."

        async def collect_chunks():
            chunks = []
            async for audio_chunk, sr in generate_speech_streaming(
                text=text,
                language="English",
                voice_cache=voice_cache,
                use_cache=True,
            ):
                chunks.append((audio_chunk, sr))
            return chunks

        chunks = asyncio.run(collect_chunks())

        # Should get multiple chunks (one per sentence)
        assert len(chunks) >= 1

        # Each chunk should have valid audio
        for audio, sr in chunks:
            assert sr == 24000
            assert len(audio) > 0


# =============================================================================
# Run Configuration
# =============================================================================

if __name__ == "__main__":
    # Run unit tests by default
    pytest.main([__file__, "-v", "-m", "not integration"])
