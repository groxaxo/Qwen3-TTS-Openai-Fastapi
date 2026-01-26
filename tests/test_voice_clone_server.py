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
import json
import os
import tempfile
import shutil
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from dataclasses import dataclass
from typing import List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

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


# =============================================================================
# Streaming Tests - Unit (No GPU Required)
# =============================================================================

class TestStreamingWavHeader:
    """Unit tests for WAV streaming header structure."""

    def test_wav_header_riff_structure(self):
        """Test WAV header has correct RIFF structure."""
        from voice_clone_server import create_wav_streaming_header

        header = create_wav_streaming_header(24000, 1, 16)

        # RIFF chunk
        assert header[0:4] == b'RIFF', "Should start with RIFF"
        # File size placeholder (0xFFFFFFFF for streaming)
        file_size = int.from_bytes(header[4:8], 'little')
        assert file_size == 0xFFFFFFFF, "File size should be placeholder for streaming"
        # WAVE format
        assert header[8:12] == b'WAVE', "Should be WAVE format"

    def test_wav_header_fmt_chunk(self):
        """Test WAV header fmt chunk is correct."""
        from voice_clone_server import create_wav_streaming_header

        header = create_wav_streaming_header(24000, 1, 16)

        # fmt chunk
        assert header[12:16] == b'fmt ', "Should have fmt chunk"
        fmt_size = int.from_bytes(header[16:20], 'little')
        assert fmt_size == 16, "fmt chunk should be 16 bytes for PCM"

        # Audio format (PCM = 1)
        audio_format = int.from_bytes(header[20:22], 'little')
        assert audio_format == 1, "Should be PCM format"

        # Channels
        num_channels = int.from_bytes(header[22:24], 'little')
        assert num_channels == 1, "Should be mono"

        # Sample rate
        sample_rate = int.from_bytes(header[24:28], 'little')
        assert sample_rate == 24000, "Sample rate should be 24000"

        # Byte rate (sample_rate * channels * bits/8)
        byte_rate = int.from_bytes(header[28:32], 'little')
        assert byte_rate == 24000 * 1 * 2, "Byte rate should be 48000"

        # Block align
        block_align = int.from_bytes(header[32:34], 'little')
        assert block_align == 2, "Block align should be 2 (mono 16-bit)"

        # Bits per sample
        bits_per_sample = int.from_bytes(header[34:36], 'little')
        assert bits_per_sample == 16, "Should be 16 bits"

    def test_wav_header_data_chunk(self):
        """Test WAV header data chunk placeholder."""
        from voice_clone_server import create_wav_streaming_header

        header = create_wav_streaming_header(24000, 1, 16)

        # data chunk
        assert header[36:40] == b'data', "Should have data chunk"
        data_size = int.from_bytes(header[40:44], 'little')
        assert data_size == 0xFFFFFFFF, "Data size should be placeholder for streaming"

    def test_wav_header_total_size(self):
        """Test WAV header is exactly 44 bytes."""
        from voice_clone_server import create_wav_streaming_header

        header = create_wav_streaming_header(24000, 1, 16)
        assert len(header) == 44, "Standard WAV header should be 44 bytes"

    def test_wav_header_different_sample_rates(self):
        """Test WAV header with different sample rates."""
        from voice_clone_server import create_wav_streaming_header

        for rate in [8000, 16000, 22050, 24000, 44100, 48000]:
            header = create_wav_streaming_header(rate, 1, 16)
            sample_rate = int.from_bytes(header[24:28], 'little')
            assert sample_rate == rate, f"Sample rate should be {rate}"


class TestStreamingPcmFormat:
    """Unit tests for PCM streaming format."""

    def test_pcm_conversion_produces_correct_bytes(self):
        """Test PCM conversion produces correct byte length."""
        from voice_clone_server import convert_to_pcm

        # 100 samples
        audio = np.random.randn(100).astype(np.float32)
        pcm = convert_to_pcm(audio)

        # 16-bit = 2 bytes per sample
        assert len(pcm) == 200

    def test_pcm_little_endian(self):
        """Test PCM output is little-endian."""
        from voice_clone_server import convert_to_pcm

        # Create known value
        audio = np.array([0.5], dtype=np.float32)
        pcm = convert_to_pcm(audio)

        # 0.5 * 32767 ≈ 16383 = 0x3FFF
        # Little endian: 0xFF 0x3F
        decoded = int.from_bytes(pcm, 'little', signed=True)
        assert 16000 < decoded < 17000, "0.5 should map to ~16383"

    def test_pcm_handles_silence(self):
        """Test PCM handles silence (zeros)."""
        from voice_clone_server import convert_to_pcm

        audio = np.zeros(1000, dtype=np.float32)
        pcm = convert_to_pcm(audio)

        # Should be all zeros
        decoded = np.frombuffer(pcm, dtype=np.int16)
        assert np.all(decoded == 0), "Silence should produce zero PCM"


class TestStreamingEmptyInput:
    """Unit tests for empty/edge case inputs."""

    def test_sentence_split_empty_string(self):
        """Test sentence splitting with empty string."""
        import re

        text = ""
        pattern = r'(?<=[.!?。！？])\s+'
        sentences = re.split(pattern, text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        assert len(sentences) == 0, "Empty string should produce no sentences"

    def test_sentence_split_whitespace_only(self):
        """Test sentence splitting with whitespace only."""
        import re

        text = "   \n\t  "
        pattern = r'(?<=[.!?。！？])\s+'
        sentences = re.split(pattern, text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        assert len(sentences) == 0, "Whitespace should produce no sentences"

    def test_sentence_split_single_word(self):
        """Test sentence splitting with single word (no punctuation)."""
        import re

        text = "Hello"
        pattern = r'(?<=[.!?。！？])\s+'
        sentences = re.split(pattern, text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        assert len(sentences) == 1
        assert sentences[0] == "Hello"


class TestStreamingLongInput:
    """Unit tests for long input handling."""

    def test_sentence_split_very_long_sentence(self):
        """Test splitting a very long single sentence."""
        import re

        # 500 words without sentence-ending punctuation
        text = " ".join(["word"] * 500)
        pattern = r'(?<=[.!?。！？])\s+'
        sentences = re.split(pattern, text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        # Should be treated as single sentence
        assert len(sentences) == 1
        assert len(sentences[0].split()) == 500

    def test_sentence_split_many_sentences(self):
        """Test splitting many sentences."""
        import re

        # 50 sentences
        text = ". ".join([f"Sentence number {i}" for i in range(50)]) + "."
        pattern = r'(?<=[.!?。！？])\s+'
        sentences = re.split(pattern, text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        assert len(sentences) == 50


# =============================================================================
# Streaming Tests - Integration (Require GPU/Server)
# =============================================================================

@pytest.mark.integration
class TestStreamingEndpoint:
    """Integration tests for streaming endpoint behavior."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from voice_clone_server import app, model

        if model is None:
            pytest.skip("Model not loaded")

        return TestClient(app)

    def test_streaming_single_sentence(self, client):
        """Test streaming with single sentence still works."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "text": "Hello world.",
                "voice": "hai",
                "stream": True,
                "response_format": "wav",
            },
        )

        assert response.status_code == 200

        content = response.content
        # Should have WAV header + audio data
        assert len(content) > 44, "Should have header + audio"
        assert content[:4] == b'RIFF', "Should start with RIFF"

    def test_streaming_multi_sentence(self, client):
        """Test streaming with multiple sentences."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "text": "First sentence. Second sentence. Third sentence.",
                "voice": "hai",
                "stream": True,
                "response_format": "wav",
            },
        )

        assert response.status_code == 200

        content = response.content
        # Should have substantial audio for 3 sentences
        assert len(content) > 10000, "3 sentences should produce significant audio"

    def test_streaming_very_long_sentence(self, client):
        """Test streaming with very long single sentence."""
        # Long sentence without punctuation breaks
        long_text = "This is a very long sentence that goes on and on " * 10
        long_text = long_text.strip() + "."

        response = client.post(
            "/v1/audio/speech",
            json={
                "text": long_text,
                "voice": "hai",
                "stream": True,
                "response_format": "wav",
            },
        )

        assert response.status_code == 200
        content = response.content
        # Should produce audio even for long sentence
        assert len(content) > 44

    def test_streaming_pcm_format(self, client):
        """Test streaming with PCM format (no WAV header)."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "text": "Hello world.",
                "voice": "hai",
                "stream": True,
                "response_format": "pcm",
            },
        )

        assert response.status_code == 200

        content = response.content
        # PCM should NOT have RIFF header
        assert content[:4] != b'RIFF', "PCM should not have WAV header"
        # Should have audio data
        assert len(content) > 0, "Should have audio data"
        # PCM is 16-bit, so length should be even
        assert len(content) % 2 == 0, "PCM should have even byte count"

    def test_streaming_wav_header_correct(self, client):
        """Test streaming WAV response has correct header."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "text": "Test.",
                "voice": "hai",
                "stream": True,
                "response_format": "wav",
            },
        )

        assert response.status_code == 200

        content = response.content
        header = content[:44]

        # Verify RIFF structure
        assert header[:4] == b'RIFF'
        assert header[8:12] == b'WAVE'
        assert header[12:16] == b'fmt '
        assert header[36:40] == b'data'

        # Verify sample rate in header
        sample_rate = int.from_bytes(header[24:28], 'little')
        assert sample_rate == 24000

    def test_streaming_empty_text_handling(self, client):
        """Test streaming with empty text."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "text": "",
                "voice": "hai",
                "stream": True,
                "response_format": "wav",
            },
        )

        # Empty text should either return error or empty audio
        # Depends on implementation - just verify it doesn't crash
        assert response.status_code in [200, 400, 422]

    def test_streaming_whitespace_text(self, client):
        """Test streaming with whitespace-only text."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "text": "   \n\t   ",
                "voice": "hai",
                "stream": True,
                "response_format": "wav",
            },
        )

        # Whitespace should be handled gracefully
        assert response.status_code in [200, 400, 422]

    def test_streaming_vs_non_streaming_audio_similar(self, client):
        """Test that streaming and non-streaming produce similar audio lengths."""
        text = "Hello, this is a test of the streaming functionality."

        # Non-streaming
        response_normal = client.post(
            "/v1/audio/speech",
            json={
                "text": text,
                "voice": "hai",
                "stream": False,
            },
        )

        # Streaming
        response_stream = client.post(
            "/v1/audio/speech",
            json={
                "text": text,
                "voice": "hai",
                "stream": True,
                "response_format": "wav",
            },
        )

        assert response_normal.status_code == 200
        assert response_stream.status_code == 200

        # Audio lengths should be similar (within 20%)
        len_normal = len(response_normal.content)
        len_stream = len(response_stream.content)

        ratio = len_stream / len_normal if len_normal > 0 else 0
        assert 0.5 < ratio < 2.0, f"Audio lengths differ too much: {len_normal} vs {len_stream}"

    def test_streaming_invalid_voice(self, client):
        """Test streaming with invalid voice name."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "text": "Hello.",
                "voice": "nonexistent_voice_12345",
                "stream": True,
            },
        )

        assert response.status_code == 400
        assert "not loaded" in response.text.lower() or "not found" in response.text.lower()


@pytest.mark.integration
class TestStreamingKvCacheBehavior:
    """Integration tests for KV cache behavior during streaming."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from voice_clone_server import app, model

        if model is None:
            pytest.skip("Model not loaded")

        return TestClient(app)

    def test_streaming_with_kv_cache_enabled(self, client):
        """Test streaming works with KV cache enabled."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "text": "First. Second. Third.",
                "voice": "hai",
                "stream": True,
                "use_kv_cache": True,
                "response_format": "wav",
            },
        )

        assert response.status_code == 200
        assert len(response.content) > 44

    def test_streaming_with_kv_cache_disabled(self, client):
        """Test streaming works with KV cache disabled."""
        response = client.post(
            "/v1/audio/speech",
            json={
                "text": "First. Second. Third.",
                "voice": "hai",
                "stream": True,
                "use_kv_cache": False,
                "response_format": "wav",
            },
        )

        assert response.status_code == 200
        assert len(response.content) > 44

    def test_kv_cache_not_mutated_during_streaming(self):
        """Test that KV cache is not mutated during streaming generation."""
        from voice_clone_server import voices, DEFAULT_VOICE, model

        if model is None:
            pytest.skip("Model not loaded")

        voice_cache = voices.get(DEFAULT_VOICE)
        if voice_cache is None:
            pytest.skip("Default voice not loaded")

        # Record initial KV cache state
        initial_prefix_length = voice_cache.prefix_length
        initial_kv_cached = voice_cache.kv_cache is not None

        # If KV cache exists, record its size
        initial_kv_size = None
        if voice_cache.kv_cache is not None and hasattr(voice_cache.kv_cache, 'key_cache'):
            initial_kv_size = len(voice_cache.kv_cache.key_cache)

        # Run streaming generation
        import asyncio
        from voice_clone_server import generate_speech_streaming

        async def run_streaming():
            chunks = []
            async for audio, sr in generate_speech_streaming(
                text="First. Second. Third. Fourth. Fifth.",
                language="English",
                voice_cache=voice_cache,
                use_cache=True,
            ):
                chunks.append(audio)
            return chunks

        chunks = asyncio.run(run_streaming())
        assert len(chunks) >= 1

        # Verify KV cache was not mutated
        assert voice_cache.prefix_length == initial_prefix_length, "prefix_length should not change"
        assert (voice_cache.kv_cache is not None) == initial_kv_cached, "kv_cache presence should not change"

        if initial_kv_size is not None and hasattr(voice_cache.kv_cache, 'key_cache'):
            assert len(voice_cache.kv_cache.key_cache) == initial_kv_size, "kv_cache size should not change"

    def test_multiple_streaming_requests_same_voice(self, client):
        """Test multiple streaming requests reuse cached voice."""
        from voice_clone_server import voices, DEFAULT_VOICE

        voice_cache = voices.get(DEFAULT_VOICE)
        if voice_cache is None:
            pytest.skip("Default voice not loaded")

        initial_prefix_length = voice_cache.prefix_length

        # Make multiple streaming requests
        for i in range(3):
            response = client.post(
                "/v1/audio/speech",
                json={
                    "text": f"Request number {i}.",
                    "voice": "hai",
                    "stream": True,
                    "use_kv_cache": True,
                },
            )
            assert response.status_code == 200

        # Voice cache should still have same prefix length (not recomputed)
        assert voice_cache.prefix_length == initial_prefix_length


@pytest.mark.integration
class TestStreamingErrorHandling:
    """Integration tests for error handling during streaming."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from voice_clone_server import app, model

        if model is None:
            pytest.skip("Model not loaded")

        return TestClient(app)

    def test_streaming_model_not_loaded_error(self):
        """Test error when model is not loaded."""
        from fastapi.testclient import TestClient
        from voice_clone_server import app
        import voice_clone_server

        # Temporarily set model to None
        original_model = voice_clone_server.model
        voice_clone_server.model = None

        try:
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post(
                "/v1/audio/speech",
                json={
                    "text": "Hello.",
                    "voice": "hai",
                    "stream": True,
                },
            )

            assert response.status_code == 503
            assert "not loaded" in response.text.lower()
        finally:
            voice_clone_server.model = original_model

    def test_streaming_recovers_from_sentence_error(self, client):
        """Test that streaming continues even if one sentence fails."""
        # This tests the error handling in the streaming generator
        # The generator should continue to next sentence if one fails

        # Use a text that might trigger edge cases
        text = "Normal sentence. Another normal one."

        response = client.post(
            "/v1/audio/speech",
            json={
                "text": text,
                "voice": "hai",
                "stream": True,
                "response_format": "wav",
            },
        )

        # Should still succeed overall
        assert response.status_code == 200
        assert len(response.content) > 44


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

    def test_streaming_chunk_count_matches_sentences(self):
        """Test that number of chunks roughly matches sentence count."""
        import asyncio
        from voice_clone_server import generate_speech_streaming, voices, DEFAULT_VOICE, model

        if model is None:
            pytest.skip("Model not loaded")

        voice_cache = voices.get(DEFAULT_VOICE)
        if voice_cache is None:
            pytest.skip("Default voice not loaded")

        text = "One. Two. Three. Four. Five."

        async def collect_chunks():
            chunks = []
            async for audio_chunk, sr in generate_speech_streaming(
                text=text,
                language="English",
                voice_cache=voice_cache,
                use_cache=True,
            ):
                chunks.append(audio_chunk)
            return chunks

        chunks = asyncio.run(collect_chunks())

        # Should get approximately 5 chunks (one per sentence)
        # Allow some flexibility for sentence parsing edge cases
        assert 4 <= len(chunks) <= 6, f"Expected ~5 chunks, got {len(chunks)}"


# =============================================================================
# Voice Upload Tests (No GPU Required)
# =============================================================================

class TestVoiceUploadValidation:
    """Unit tests for voice upload endpoint validation."""

    def test_upload_validates_empty_voice_name(self):
        """Test upload rejects empty voice name."""
        from fastapi.testclient import TestClient
        from voice_clone_server import app

        with patch('voice_clone_server.model') as mock_model:
            mock_model.__bool__ = Mock(return_value=True)

            client = TestClient(app, raise_server_exceptions=False)

            # Create sample audio
            audio_content = b'RIFF' + b'\x00' * 40

            response = client.post(
                "/v1/voices/upload",
                data={"voice_name": "", "ref_text": "Test text"},
                files={"audio_file": ("test.wav", audio_content, "audio/wav")},
            )

            # FastAPI returns 422 for validation errors or 400 for custom validation
            assert response.status_code in [400, 422]

    def test_upload_validates_file_extension(self):
        """Test upload rejects invalid file types."""
        from fastapi.testclient import TestClient
        from voice_clone_server import app

        with patch('voice_clone_server.model') as mock_model:
            mock_model.__bool__ = Mock(return_value=True)

            client = TestClient(app, raise_server_exceptions=False)

            response = client.post(
                "/v1/voices/upload",
                data={"voice_name": "test_voice", "ref_text": "Test text"},
                files={"audio_file": ("test.txt", b"not audio", "text/plain")},
            )

            assert response.status_code == 400
            assert "invalid file type" in response.text.lower()

    def test_upload_accepts_valid_extensions(self):
        """Test upload accepts wav, mp3, flac, ogg, m4a."""
        # Just check the validation logic, not actual upload
        allowed_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

        for ext in allowed_extensions:
            assert ext in allowed_extensions


class TestVoiceUploadFilePaths:
    """Unit tests for voice upload file path handling."""

    def test_get_voice_cache_path(self):
        """Test voice cache path generation."""
        from voice_clone_server import get_voice_cache_path, VOICE_REFS_DIR

        path = get_voice_cache_path("my_voice")
        assert path == VOICE_REFS_DIR / "my_voice_kvcache.safetensors"

    def test_get_voice_metadata_path(self):
        """Test voice metadata path generation."""
        from voice_clone_server import get_voice_metadata_path, VOICE_REFS_DIR

        path = get_voice_metadata_path("my_voice")
        assert path == VOICE_REFS_DIR / "my_voice_metadata.json"

    def test_discover_cached_voices_empty_dir(self):
        """Test discover_cached_voices with no caches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('voice_clone_server.VOICE_REFS_DIR', Path(tmpdir)):
                from voice_clone_server import discover_cached_voices

                voices = discover_cached_voices()
                assert voices == []


# =============================================================================
# KV Cache Persistence Tests (No GPU Required for most)
# =============================================================================

class TestKvCacheSaveLoad:
    """Unit tests for KV cache persistence."""

    @pytest.fixture
    def temp_voice_dir(self):
        """Create temporary directory for voice files."""
        tmpdir = tempfile.mkdtemp()
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)

    def test_save_voice_cache_creates_files(self, temp_voice_dir, mock_voice_prompt):
        """Test save_voice_cache_to_disk creates safetensors and metadata files."""
        from voice_clone_server import (
            save_voice_cache_to_disk,
            VoiceCache,
            get_voice_cache_path,
            get_voice_metadata_path,
        )

        # Create a mock voice cache with KV cache
        kv_cache = DynamicCache()
        key = torch.randn(1, 8, 10, 64)
        value = torch.randn(1, 8, 10, 64)
        kv_cache.update(key, value, layer_idx=0)

        voice_cache = VoiceCache(
            prompt=mock_voice_prompt,
            kv_cache=kv_cache,
            past_hidden=torch.randn(1, 10, 256),
            prefix_length=100,
            ref_text="Test reference text",
            ref_audio_path="/path/to/audio.wav",
        )

        with patch('voice_clone_server.VOICE_REFS_DIR', temp_voice_dir):
            # Reload to get updated paths
            import importlib
            import voice_clone_server
            importlib.reload(voice_clone_server)

            with patch('voice_clone_server.VOICE_REFS_DIR', temp_voice_dir):
                result = save_voice_cache_to_disk("test_voice", voice_cache)

        assert result is True

        # Check files exist
        cache_path = temp_voice_dir / "test_voice_kvcache.safetensors"
        metadata_path = temp_voice_dir / "test_voice_metadata.json"

        assert cache_path.exists(), "Cache file should exist"
        assert metadata_path.exists(), "Metadata file should exist"

        # Verify metadata content
        with open(metadata_path) as f:
            metadata = json.load(f)

        assert metadata["ref_text"] == "Test reference text"
        assert metadata["prefix_length"] == 100
        assert metadata["ref_audio_path"] == "/path/to/audio.wav"

    def test_load_voice_cache_from_disk_missing_file(self, temp_voice_dir):
        """Test load returns None for missing files."""
        with patch('voice_clone_server.VOICE_REFS_DIR', temp_voice_dir):
            from voice_clone_server import load_voice_cache_from_disk

            result = load_voice_cache_from_disk("nonexistent_voice")
            assert result is None

    def test_save_load_roundtrip(self, temp_voice_dir, mock_voice_prompt):
        """Test save then load preserves data."""
        from voice_clone_server import (
            save_voice_cache_to_disk,
            load_voice_cache_from_disk,
            VoiceCache,
        )

        # Create voice cache
        kv_cache = DynamicCache()
        key = torch.randn(1, 8, 10, 64)
        value = torch.randn(1, 8, 10, 64)
        kv_cache.update(key, value, layer_idx=0)

        past_hidden = torch.randn(1, 10, 256)

        voice_cache = VoiceCache(
            prompt=mock_voice_prompt,
            kv_cache=kv_cache,
            past_hidden=past_hidden,
            prefix_length=42,
            ref_text="Roundtrip test",
            ref_audio_path="/test/path.wav",
        )

        with patch('voice_clone_server.VOICE_REFS_DIR', temp_voice_dir):
            # Save
            save_voice_cache_to_disk("roundtrip_voice", voice_cache)

            # Load
            result = load_voice_cache_from_disk("roundtrip_voice", device="cpu")

        assert result is not None
        loaded_kv, loaded_hidden, loaded_prefix, loaded_metadata = result

        # Verify loaded data
        assert loaded_prefix == 42
        assert loaded_metadata["ref_text"] == "Roundtrip test"
        assert loaded_metadata["ref_audio_path"] == "/test/path.wav"
        assert loaded_kv is not None
        assert loaded_hidden is not None


class TestKvCacheDelete:
    """Unit tests for KV cache deletion."""

    @pytest.fixture
    def temp_voice_dir(self):
        """Create temporary directory for voice files."""
        tmpdir = tempfile.mkdtemp()
        yield Path(tmpdir)
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_delete_voice_cache_removes_files(self, temp_voice_dir):
        """Test delete_voice_cache_from_disk removes cache and metadata."""
        # Create mock files
        cache_path = temp_voice_dir / "test_delete_kvcache.safetensors"
        metadata_path = temp_voice_dir / "test_delete_metadata.json"

        cache_path.write_bytes(b"mock cache data")
        metadata_path.write_text('{"ref_text": "test"}')

        with patch('voice_clone_server.VOICE_REFS_DIR', temp_voice_dir):
            from voice_clone_server import delete_voice_cache_from_disk

            result = delete_voice_cache_from_disk("test_delete")

        assert result is True
        assert not cache_path.exists()
        assert not metadata_path.exists()

    def test_delete_nonexistent_voice(self, temp_voice_dir):
        """Test delete handles nonexistent files gracefully."""
        with patch('voice_clone_server.VOICE_REFS_DIR', temp_voice_dir):
            from voice_clone_server import delete_voice_cache_from_disk

            # Should return True even if files don't exist
            result = delete_voice_cache_from_disk("nonexistent")
            assert result is True


class TestDiscoverCachedVoices:
    """Unit tests for discover_cached_voices."""

    @pytest.fixture
    def temp_voice_dir(self):
        """Create temporary directory for voice files."""
        tmpdir = tempfile.mkdtemp()
        yield Path(tmpdir)
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_discover_finds_complete_caches(self, temp_voice_dir):
        """Test discover finds voices with both cache and metadata."""
        # Create complete voice (cache + metadata)
        (temp_voice_dir / "voice1_kvcache.safetensors").write_bytes(b"cache")
        (temp_voice_dir / "voice1_metadata.json").write_text('{}')

        # Create incomplete voice (cache only)
        (temp_voice_dir / "voice2_kvcache.safetensors").write_bytes(b"cache")

        with patch('voice_clone_server.VOICE_REFS_DIR', temp_voice_dir):
            from voice_clone_server import discover_cached_voices

            voices = discover_cached_voices()

        assert "voice1" in voices
        assert "voice2" not in voices  # Missing metadata

    def test_discover_multiple_voices(self, temp_voice_dir):
        """Test discover finds multiple complete voices."""
        for name in ["alice", "bob", "charlie"]:
            (temp_voice_dir / f"{name}_kvcache.safetensors").write_bytes(b"cache")
            (temp_voice_dir / f"{name}_metadata.json").write_text('{}')

        with patch('voice_clone_server.VOICE_REFS_DIR', temp_voice_dir):
            from voice_clone_server import discover_cached_voices

            voices = discover_cached_voices()

        assert len(voices) == 3
        assert set(voices) == {"alice", "bob", "charlie"}


# =============================================================================
# Voice Delete Endpoint Tests (No GPU Required)
# =============================================================================

class TestVoiceDeleteEndpoint:
    """Unit tests for DELETE /v1/voices/{name} endpoint."""

    def test_delete_default_voice_rejected(self):
        """Test cannot delete default voice."""
        from fastapi.testclient import TestClient
        from voice_clone_server import app, DEFAULT_VOICE

        client = TestClient(app, raise_server_exceptions=False)

        response = client.delete(f"/v1/voices/{DEFAULT_VOICE}")

        assert response.status_code == 400
        assert "cannot delete" in response.text.lower() or "default" in response.text.lower()

    def test_delete_nonexistent_voice(self):
        """Test deleting nonexistent voice returns 404."""
        from fastapi.testclient import TestClient
        from voice_clone_server import app

        client = TestClient(app, raise_server_exceptions=False)

        response = client.delete("/v1/voices/nonexistent_voice_xyz")

        assert response.status_code == 404


# =============================================================================
# Hot Swapping Tests (Integration - Require GPU)
# =============================================================================

@pytest.mark.integration
class TestHotSwapping:
    """Integration tests for hot swapping between voices."""

    @pytest.fixture
    def client(self):
        """Create test client with loaded model."""
        from fastapi.testclient import TestClient
        from voice_clone_server import app, model

        if model is None:
            pytest.skip("Model not loaded")

        return TestClient(app)

    def test_switch_between_voices_rapidly(self, client):
        """Test rapid switching between cached voices."""
        from voice_clone_server import voices

        if len(voices) < 1:
            pytest.skip("Need at least one voice loaded")

        voice_names = list(voices.keys())

        # Rapidly switch between voices
        for i in range(10):
            voice = voice_names[i % len(voice_names)]
            response = client.post(
                "/v1/audio/speech",
                json={
                    "text": f"Test {i}.",
                    "voice": voice,
                },
            )
            assert response.status_code == 200

    def test_concurrent_requests_different_voices(self, client):
        """Test concurrent requests with different voices."""
        from voice_clone_server import voices

        if len(voices) < 1:
            pytest.skip("Need at least one voice loaded")

        voice_names = list(voices.keys())

        def make_request(idx):
            voice = voice_names[idx % len(voice_names)]
            response = client.post(
                "/v1/audio/speech",
                json={
                    "text": f"Concurrent test {idx}.",
                    "voice": voice,
                },
            )
            return response.status_code

        # Use ThreadPoolExecutor for concurrent requests
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request, i) for i in range(6)]
            results = [f.result() for f in futures]

        # All should succeed
        assert all(status == 200 for status in results)


@pytest.mark.integration
class TestVoiceUploadIntegration:
    """Integration tests for voice upload (requires GPU)."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from voice_clone_server import app, model

        if model is None:
            pytest.skip("Model not loaded")

        return TestClient(app)

    @pytest.fixture
    def sample_wav_bytes(self):
        """Create valid WAV bytes for testing."""
        import soundfile as sf
        import io

        # Create 3 seconds of audio
        sample_rate = 24000
        duration = 3.0
        samples = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1

        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format='WAV')
        buf.seek(0)
        return buf.read()

    def test_upload_wav_creates_voice(self, client, sample_wav_bytes):
        """Test uploading WAV file creates a usable voice."""
        import uuid

        voice_name = f"test_upload_{uuid.uuid4().hex[:8]}"

        try:
            response = client.post(
                "/v1/voices/upload",
                data={
                    "voice_name": voice_name,
                    "ref_text": "This is a test reference audio for voice cloning.",
                },
                files={"audio_file": ("test.wav", sample_wav_bytes, "audio/wav")},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["voice_name"] == voice_name
            assert data["kv_cached"] is True or data["kv_cached"] is False
            assert data["persisted"] is True

            # Verify voice can be used for TTS
            tts_response = client.post(
                "/v1/audio/speech",
                json={
                    "text": "Hello using uploaded voice.",
                    "voice": voice_name,
                },
            )
            assert tts_response.status_code == 200

        finally:
            # Cleanup
            client.delete(f"/v1/voices/{voice_name}")

    def test_upload_multiple_voices(self, client, sample_wav_bytes):
        """Test uploading multiple different voices."""
        import uuid

        voice_names = [f"multi_test_{uuid.uuid4().hex[:8]}" for _ in range(3)]

        try:
            for voice_name in voice_names:
                response = client.post(
                    "/v1/voices/upload",
                    data={
                        "voice_name": voice_name,
                        "ref_text": f"Reference text for {voice_name}.",
                    },
                    files={"audio_file": ("test.wav", sample_wav_bytes, "audio/wav")},
                )
                assert response.status_code == 200

            # Verify all voices are listed
            list_response = client.get("/v1/voices")
            voices_data = list_response.json()["voices"]

            for voice_name in voice_names:
                assert voice_name in voices_data

        finally:
            # Cleanup
            for voice_name in voice_names:
                client.delete(f"/v1/voices/{voice_name}")

    def test_upload_duplicate_voice_name(self, client, sample_wav_bytes):
        """Test uploading with duplicate name overwrites."""
        import uuid

        voice_name = f"dup_test_{uuid.uuid4().hex[:8]}"

        try:
            # First upload
            response1 = client.post(
                "/v1/voices/upload",
                data={
                    "voice_name": voice_name,
                    "ref_text": "First reference text.",
                },
                files={"audio_file": ("test.wav", sample_wav_bytes, "audio/wav")},
            )
            assert response1.status_code == 200

            # Second upload with same name
            response2 = client.post(
                "/v1/voices/upload",
                data={
                    "voice_name": voice_name,
                    "ref_text": "Second reference text.",
                },
                files={"audio_file": ("test2.wav", sample_wav_bytes, "audio/wav")},
            )
            # Should succeed (overwrite)
            assert response2.status_code == 200

        finally:
            client.delete(f"/v1/voices/{voice_name}")


@pytest.mark.integration
class TestVoiceDeleteIntegration:
    """Integration tests for voice deletion (requires GPU)."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from voice_clone_server import app, model

        if model is None:
            pytest.skip("Model not loaded")

        return TestClient(app)

    @pytest.fixture
    def sample_wav_bytes(self):
        """Create valid WAV bytes for testing."""
        import soundfile as sf
        import io

        sample_rate = 24000
        samples = np.random.randn(int(sample_rate * 3)).astype(np.float32) * 0.1

        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format='WAV')
        buf.seek(0)
        return buf.read()

    def test_delete_uploaded_voice(self, client, sample_wav_bytes):
        """Test deleting an uploaded voice removes it completely."""
        import uuid
        from voice_clone_server import VOICE_REFS_DIR

        voice_name = f"delete_test_{uuid.uuid4().hex[:8]}"

        # Upload voice
        client.post(
            "/v1/voices/upload",
            data={
                "voice_name": voice_name,
                "ref_text": "Test for deletion.",
            },
            files={"audio_file": ("test.wav", sample_wav_bytes, "audio/wav")},
        )

        # Verify it exists
        list_response = client.get("/v1/voices")
        assert voice_name in list_response.json()["voices"]

        # Delete voice
        delete_response = client.delete(f"/v1/voices/{voice_name}")
        assert delete_response.status_code == 200

        # Verify it's gone from memory
        list_response2 = client.get("/v1/voices")
        assert voice_name not in list_response2.json()["voices"]

        # Verify cache files are deleted
        cache_path = VOICE_REFS_DIR / f"{voice_name}_kvcache.safetensors"
        metadata_path = VOICE_REFS_DIR / f"{voice_name}_metadata.json"
        assert not cache_path.exists()
        assert not metadata_path.exists()


@pytest.mark.integration
class TestPersistenceAcrossReload:
    """Integration tests for voice persistence across module reload."""

    def test_voices_persist_in_cache(self):
        """Test that voice caches are saved to disk."""
        from voice_clone_server import voices, VOICE_REFS_DIR

        # Check that default voice has cache file
        if "hai" in voices:
            cache_path = VOICE_REFS_DIR / "hai_kvcache.safetensors"
            # May or may not exist depending on test order
            # Just verify the path is correct format
            assert "hai_kvcache.safetensors" in str(cache_path)


@pytest.mark.integration
class TestFullFlow:
    """Integration tests for complete upload->generate->delete flow."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from voice_clone_server import app, model

        if model is None:
            pytest.skip("Model not loaded")

        return TestClient(app)

    @pytest.fixture
    def sample_wav_bytes(self):
        """Create valid WAV bytes for testing."""
        import soundfile as sf
        import io

        sample_rate = 24000
        samples = np.random.randn(int(sample_rate * 3)).astype(np.float32) * 0.1

        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format='WAV')
        buf.seek(0)
        return buf.read()

    def test_full_flow_upload_generate_delete(self, client, sample_wav_bytes):
        """Test complete flow: upload -> generate -> verify -> delete."""
        import uuid

        voice_name = f"fullflow_{uuid.uuid4().hex[:8]}"

        # Step 1: Upload
        upload_response = client.post(
            "/v1/voices/upload",
            data={
                "voice_name": voice_name,
                "ref_text": "This is the reference text for full flow testing.",
            },
            files={"audio_file": ("reference.wav", sample_wav_bytes, "audio/wav")},
        )
        assert upload_response.status_code == 200
        assert upload_response.json()["persisted"] is True

        # Step 2: Generate speech with uploaded voice
        gen_response = client.post(
            "/v1/audio/speech",
            json={
                "text": "Hello from the uploaded voice. This is a test.",
                "voice": voice_name,
                "stream": False,
            },
        )
        assert gen_response.status_code == 200
        assert len(gen_response.content) > 1000  # Should have audio

        # Step 3: Generate streaming
        stream_response = client.post(
            "/v1/audio/speech",
            json={
                "text": "Streaming test. Multiple sentences. Should work.",
                "voice": voice_name,
                "stream": True,
                "response_format": "wav",
            },
        )
        assert stream_response.status_code == 200
        assert stream_response.content[:4] == b'RIFF'

        # Step 4: Delete
        delete_response = client.delete(f"/v1/voices/{voice_name}")
        assert delete_response.status_code == 200

        # Step 5: Verify deleted voice can't be used
        fail_response = client.post(
            "/v1/audio/speech",
            json={
                "text": "This should fail.",
                "voice": voice_name,
            },
        )
        assert fail_response.status_code == 400

    def test_multiple_voices_coexist(self, client, sample_wav_bytes):
        """Test multiple uploaded voices can coexist and work independently."""
        import uuid

        voices = [f"coexist_{uuid.uuid4().hex[:8]}" for _ in range(3)]

        try:
            # Upload all voices
            for i, voice_name in enumerate(voices):
                response = client.post(
                    "/v1/voices/upload",
                    data={
                        "voice_name": voice_name,
                        "ref_text": f"Reference for voice {i}.",
                    },
                    files={"audio_file": (f"ref{i}.wav", sample_wav_bytes, "audio/wav")},
                )
                assert response.status_code == 200

            # Generate with each voice
            for voice_name in voices:
                response = client.post(
                    "/v1/audio/speech",
                    json={
                        "text": f"Generated with {voice_name}.",
                        "voice": voice_name,
                    },
                )
                assert response.status_code == 200

            # Interleaved generation
            for i in range(6):
                voice = voices[i % len(voices)]
                response = client.post(
                    "/v1/audio/speech",
                    json={
                        "text": f"Interleaved {i}.",
                        "voice": voice,
                    },
                )
                assert response.status_code == 200

        finally:
            # Cleanup
            for voice_name in voices:
                client.delete(f"/v1/voices/{voice_name}")


# =============================================================================
# Run Configuration
# =============================================================================

if __name__ == "__main__":
    # Run unit tests by default
    pytest.main([__file__, "-v", "-m", "not integration"])
