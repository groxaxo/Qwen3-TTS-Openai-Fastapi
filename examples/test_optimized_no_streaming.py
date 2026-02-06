"""
Test TTS with optimizations (non-streaming mode).

Uses the same optimizations as streaming for ~4x speedup:
1. generate_fast() - bypasses HuggingFace generate() overhead (2-3x speedup)
2. torch.compile for decoder with max-autotune mode
3. Compiled codebook predictor

Usage:
    cd Qwen3-TTS
    python examples/test_optimized_no_streaming.py
"""

import time
import numpy as np
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Enable TensorFloat32 for better performance on Ampere+ GPUs
torch.set_float32_matmul_precision('high')


def log_time(start, operation):
    elapsed = time.time() - start
    print(f"[{elapsed:.2f}s] {operation}")
    return time.time()


def run_generation(
    model,
    text: str,
    language: str,
    voice_clone_prompt,
    label: str = "generation",
):
    """Run non-streaming generation and return timing stats."""
    start = time.time()

    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=voice_clone_prompt,
    )

    total_time = time.time() - start
    audio = wavs[0] if wavs else np.array([])

    audio_duration = len(audio) / sr if sr > 0 else 0

    return {
        "label": label,
        "total_time": total_time,
        "audio": audio,
        "sample_rate": sr,
        "audio_duration": audio_duration,
    }


def main():
    total_start = time.time()

    print("=" * 60)
    print("Loading model...")
    print("=" * 60)

    start = time.time()
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    log_time(start, "Model loaded")

    # Reference audio setup
    ref_audio_path = "kuklina-1.wav"
    ref_text = (
        "Это брат Кэти, моей одноклассницы. А что у тебя с рукой? И почему ты голая? У него ведь куча наград по "
        "боевым искусствам. Кэти рассказывала, правда, Лео? Понимаешь кого ты побила, Лая? "
        "Только потрогай эти мышцы... Не знала, что у тебя такой классный котик. Рожденная луной. "
        "Лай всегда откопает что-нибудь этакое. Да, жаль только, что занимает почти всё её время. "
        "Не понимаю, почему эта рухлядь не может подождать, пока ты проведешь время с сестрой."
    )

    start = time.time()
    voice_clone_prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        ref_text=ref_text,
    )
    log_time(start, "Voice clone prompt created")

    # Test text
    test_text = "Всем привет! Это тестовый текст для озвучки! Стриминг звучит нормально только через несколько секунд."

    results = []

    # ============== Test 1: Standard generation (baseline) ==============
    print("\n" + "=" * 60)
    print("Test 1: Standard generation (baseline)")
    print("=" * 60)

    result = run_generation(
        model, test_text, "Russian", voice_clone_prompt,
        label="baseline",
    )
    results.append(result)
    sf.write("output_baseline.wav", result["audio"], result["sample_rate"])
    rtf = result['total_time'] / result['audio_duration'] if result['audio_duration'] > 0 else 0
    print(f"Total: {result['total_time']:.2f}s, Audio: {result['audio_duration']:.2f}s, RTF: {rtf:.2f}")

    # ============== Test 2: With optimizations ==============
    print("\n" + "=" * 60)
    print("Test 2: With optimizations (fast codebook + compiled decoder)")
    print("=" * 60)

    # Enable optimizations - using the same method as streaming but tuned for batch
    print("\nEnabling optimizations...")
    model.enable_streaming_optimizations(
        decode_window_frames=300,  # Larger window for non-streaming
        use_compile=True,
        use_cuda_graphs=False,  # Not needed for non-streaming (variable sizes)
        compile_mode="max-autotune",  # Better for batch processing than reduce-overhead
        use_fast_codebook=True,  # KEY: 2-3x speedup by bypassing HF generate()
        compile_codebook_predictor=True,  # Compile the codebook predictor too
    )

    # Warmup run (first run after compile is slower due to compilation)
    print("\nWarmup run (first run after compile)...")
    warmup_result = run_generation(
        model, "Тест один два три четыре пять.", "Russian", voice_clone_prompt,
        label="warmup",
    )
    warmup_rtf = warmup_result['total_time'] / warmup_result['audio_duration'] if warmup_result['audio_duration'] > 0 else 0
    print(f"Warmup: Total: {warmup_result['total_time']:.2f}s, Audio: {warmup_result['audio_duration']:.2f}s, RTF: {warmup_rtf:.2f}")

    # Actual test run
    print("\nOptimized test run...")
    result = run_generation(
        model, test_text, "Russian", voice_clone_prompt,
        label="optimized",
    )
    results.append(result)
    sf.write("output_optimized.wav", result["audio"], result["sample_rate"])
    opt_rtf = result['total_time'] / result['audio_duration'] if result['audio_duration'] > 0 else 0
    print(f"Total: {result['total_time']:.2f}s, Audio: {result['audio_duration']:.2f}s, RTF: {opt_rtf:.2f}")

    # Second optimized run to show stable performance
    print("\nSecond optimized run...")
    result2 = run_generation(
        model, test_text, "Russian", voice_clone_prompt,
        label="optimized_2",
    )
    results.append(result2)
    opt2_rtf = result2['total_time'] / result2['audio_duration'] if result2['audio_duration'] > 0 else 0
    print(f"Total: {result2['total_time']:.2f}s, Audio: {result2['audio_duration']:.2f}s, RTF: {opt2_rtf:.2f}")

    # Third run for stability check
    print("\nThird optimized run...")
    result3 = run_generation(
        model, test_text, "Russian", voice_clone_prompt,
        label="optimized_3",
    )
    results.append(result3)
    opt3_rtf = result3['total_time'] / result3['audio_duration'] if result3['audio_duration'] > 0 else 0
    print(f"Total: {result3['total_time']:.2f}s, Audio: {result3['audio_duration']:.2f}s, RTF: {opt3_rtf:.2f}")

    # ============== Summary ==============
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    baseline_total = results[0]["total_time"]

    print(f"\n{'Method':<20} {'Total':>10} {'Audio':>10} {'RTF':>8} {'Speedup':>10}")
    print("-" * 60)

    for r in results:
        total = r["total_time"]
        audio_dur = r.get("audio_duration", 0)
        rtf = total / audio_dur if audio_dur > 0 else 0
        speedup = baseline_total / total if total > 0 else 0
        print(f"{r['label']:<20} {total:>9.2f}s {audio_dur:>9.2f}s {rtf:>8.2f} {speedup:>9.2f}x")

    print(f"\n[{time.time() - total_start:.2f}s] TOTAL SCRIPT TIME")

    # Tips
    print("\n" + "=" * 60)
    print("OPTIMIZATIONS APPLIED")
    print("=" * 60)
    print("""
1. torch.set_float32_matmul_precision('high') - TensorFloat32 on Ampere+ GPUs
2. bfloat16 dtype - faster computation with minimal quality loss
3. flash_attention_2 - efficient attention computation
4. use_fast_codebook=True - bypasses HuggingFace generate() for 2-3x speedup
5. torch.compile with max-autotune mode for decoder
6. compile_codebook_predictor=True - compiled code predictor

Expected speedup: 2.5-4x over baseline
""")


if __name__ == "__main__":
    main()
