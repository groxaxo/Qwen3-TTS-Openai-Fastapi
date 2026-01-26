#!/usr/bin/env python3
"""Test script for KV cache implementation in voice clone server."""

import time
import requests
import json

BASE_URL = "http://localhost:8882"


def test_voice_load():
    """Test loading a new voice."""
    print("\n=== Testing voice load ===")
    response = requests.post(
        f"{BASE_URL}/v1/voices/load",
        json={
            "voice_name": "test_voice",
            "ref_audio_path": "./voice_refs/hai_reference.wav",
            "ref_text": "Yeah so basically I checked the system logs and found a couple of errors. Nothing critical, but you should probably take a look when you get a chance.",
        },
    )
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    return data


def test_list_voices():
    """Test listing voices."""
    print("\n=== Testing list voices ===")
    response = requests.get(f"{BASE_URL}/v1/voices")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")
    return data


def test_speech_generation(voice: str = "hai", text: str = "Hello, this is a test.", use_kv_cache: bool = True):
    """Test speech generation with timing."""
    print(f"\n=== Testing speech generation (voice='{voice}', kv_cache={use_kv_cache}) ===")
    print(f"Text: {text}")

    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/v1/audio/speech",
        json={
            "text": text,
            "language": "English",
            "voice": voice,
            "use_kv_cache": use_kv_cache,
        },
    )
    elapsed = time.time() - start_time

    print(f"Status: {response.status_code}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Audio size: {len(response.content)} bytes")
    return elapsed, response


def benchmark_kv_cache():
    """Benchmark KV cache vs no KV cache performance."""
    print("\n" + "=" * 60)
    print("KV CACHE BENCHMARK")
    print("=" * 60)

    test_texts = [
        "Hello, how are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the way we work and live.",
    ]

    # First, make sure voices are loaded
    voices = test_list_voices()
    if not voices.get("voices"):
        print("No voices loaded!")
        return

    print("\n--- Warmup ---")
    test_speech_generation(text="Warmup request.")

    print("\n--- With KV cache ---")
    with_cache_times = []
    for i, text in enumerate(test_texts):
        elapsed, _ = test_speech_generation(text=text, use_kv_cache=True)
        with_cache_times.append(elapsed)

    print("\n--- Without KV cache ---")
    without_cache_times = []
    for i, text in enumerate(test_texts):
        elapsed, _ = test_speech_generation(text=text, use_kv_cache=False)
        without_cache_times.append(elapsed)

    print("\n--- Results ---")
    print(f"With KV cache:    {[f'{t:.3f}s' for t in with_cache_times]} (avg: {sum(with_cache_times) / len(with_cache_times):.3f}s)")
    print(f"Without KV cache: {[f'{t:.3f}s' for t in without_cache_times]} (avg: {sum(without_cache_times) / len(without_cache_times):.3f}s)")

    avg_with = sum(with_cache_times) / len(with_cache_times)
    avg_without = sum(without_cache_times) / len(without_cache_times)
    if avg_with > 0:
        speedup = avg_without / avg_with
        print(f"Speedup: {speedup:.2f}x")


def main():
    # Check server health
    print("=== Checking server health ===")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("ERROR: Server not running. Start with: python voice_clone_server.py")
        return

    # Run tests
    test_list_voices()
    test_speech_generation()
    benchmark_kv_cache()


if __name__ == "__main__":
    main()
