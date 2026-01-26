#!/usr/bin/env python3
"""Benchmark KV cache performance for voice clone server."""

import time
import requests
import json
import statistics

BASE_URL = "http://localhost:8882"


def generate_speech(voice_id: str, text: str, use_cache: bool = True) -> tuple[float, int]:
    """Generate speech and return (time_elapsed, audio_bytes)."""
    start = time.perf_counter()
    response = requests.post(
        f"{BASE_URL}/v1/audio/speech",
        json={
            "text": text,
            "language": "English",
            "voice": voice_id,
            "use_kv_cache": use_cache,
        },
    )
    elapsed = time.perf_counter() - start
    return elapsed, len(response.content)


def benchmark():
    print("=" * 70)
    print("KV CACHE BENCHMARK - Voice Clone Server")
    print("=" * 70)

    # Check server health
    health = requests.get(f"{BASE_URL}/health").json()
    print(f"\nServer status: {health}")

    # List current voices
    voices = requests.get(f"{BASE_URL}/v1/voices").json()
    print(f"Loaded voices: {list(voices.get('voices', {}).keys())}")

    # Get default voice name
    default_voice_name = voices.get("default", "hai")
    default_voice_info = voices.get("voices", {}).get(default_voice_name, {})
    if default_voice_info:
        print(f"Default voice '{default_voice_name}' KV cache: prefix_length={default_voice_info.get('prefix_length', 'N/A')}")

    # Test texts of varying lengths
    test_texts = [
        "Hello, how are you?",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the way we work and live every single day.",
    ]

    print("\n" + "-" * 70)
    print("Warmup (one request with KV cache)")
    print("-" * 70)
    elapsed, size = generate_speech(default_voice_name, "Warmup test.", use_cache=True)
    print(f"Warmup: {elapsed:.3f}s, {size} bytes")

    # Benchmark WITH KV cache
    print("\n" + "-" * 70)
    print("WITH KV CACHE (voice prefix pre-computed)")
    print("-" * 70)

    with_cache_times = []
    for i, text in enumerate(test_texts):
        times = []
        for run in range(3):
            elapsed, size = generate_speech(default_voice_name, text, use_cache=True)
            times.append(elapsed)

        avg = statistics.mean(times)
        with_cache_times.append(avg)
        print(f"Text {i+1} ({len(text):3d} chars): {avg:.3f}s (runs: {', '.join(f'{t:.3f}' for t in times)})")

    # Benchmark WITHOUT KV cache
    print("\n" + "-" * 70)
    print("WITHOUT KV CACHE (baseline - recomputes prefix each time)")
    print("-" * 70)

    without_cache_times = []
    for i, text in enumerate(test_texts):
        times = []
        for run in range(3):
            elapsed, size = generate_speech(default_voice_name, text, use_cache=False)
            times.append(elapsed)

        avg = statistics.mean(times)
        without_cache_times.append(avg)
        print(f"Text {i+1} ({len(text):3d} chars): {avg:.3f}s (runs: {', '.join(f'{t:.3f}' for t in times)})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Text':<8} {'With KV':<12} {'Without KV':<12} {'Speedup':<12} {'Saved':<12}")
    print("-" * 56)

    total_with = 0
    total_without = 0
    for i, (with_t, without_t) in enumerate(zip(with_cache_times, without_cache_times)):
        speedup = without_t / with_t if with_t > 0 else 0
        saved = without_t - with_t
        print(f"Text {i+1:<3} {with_t:<12.3f} {without_t:<12.3f} {speedup:<12.2f}x {saved:<12.3f}s")
        total_with += with_t
        total_without += without_t

    print("-" * 56)
    overall_speedup = total_without / total_with if total_with > 0 else 0
    overall_saved = total_without - total_with
    print(f"{'TOTAL':<8} {total_with:<12.3f} {total_without:<12.3f} {overall_speedup:<12.2f}x {overall_saved:<12.3f}s")

    print(f"\nOverall improvement: {overall_speedup:.2f}x faster with KV cache")
    print(f"Time saved per request (avg): {overall_saved / len(test_texts):.3f}s")


if __name__ == "__main__":
    benchmark()
