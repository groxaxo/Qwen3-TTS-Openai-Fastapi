# Apple M5: Qwen3-TTS 1.7B CustomVoice MLX 8-bit

Tested on 2026-11-17 with an Apple M5 Mac, 24 GB unified memory, AC power, and Low Power Mode disabled. The 1.7B MLX8 model runs faster than playback and delivers throughput close to the tested RTX 3060 CUDA-graph deployment.

## Measured results

Two warm Mac batches each contained four WAV requests and three incremental PCM requests. Warmup was excluded. The same short English and conversation fixtures, preset voices, and 200–600-character API chunk settings were used for the NVIDIA comparison. Request payloads and generated audio remain private; this document publishes aggregate timings only.

| Metric | M5 MLX8, first batch | M5 MLX8, after cleanup fix | RTX 3060 CUDA graphs |
|---|---:|---:|---:|
| Median WAV RTF | 0.545 | 0.639 | 0.563 |
| Median PCM RTF | 0.508 | 0.625 | 0.598 |
| Median first PCM audio | 0.208 s | 0.248 s | 0.460 s |

RTF = wall seconds / generated audio seconds; lower is faster. The Mac delivered about **1.6–1.8 seconds of audio per wall second**, with roughly **0.21–0.25-second first audio**. The two batches show normal variation rather than evidence that the Mac is consistently faster than the GPU deployment. Four WAV samples per batch are insufficient for a statistically strong hardware ranking.

These are complete runtime comparisons: Mac requests used the local API, while the NVIDIA reference included a reverse proxy to another machine. The Mac checkpoint uses 8-bit affine quantization with group size 64; the NVIDIA model used BF16. Hardware, quantization, and transport differ. Desktop load and memory pressure were not controlled; the Mac already had swap in use before testing, and usage increased during the session. Listening quality and transcription accuracy were not evaluated.

Audio checks confirmed nonempty, nonsilent, finite 24 kHz mono output, native speaker discovery, and incremental PCM delivery. After the cleanup fix, three real disconnect/recovery cycles completed successfully; the following WAV requests took 1.33, 0.83, and 0.94 seconds.

## Tested runtime

- Model: `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`
- Checkpoint revision: `41d3337e8b7f2843a75841595fc14e4b9a7a4b96`
- Python 3.11; MLX 0.32.4; MLX-Audio 0.6.2; MLX-LM 0.32.2
- Transformers 5.21.0; NumPy 2.4.6
- FastAPI 0.163.1; Uvicorn 0.54.0; HTTPX 0.28.1

`requirements/mlx.txt` pins the tested runtime. Use an existing Conda installation to create a dedicated environment. The root `qwen-tts` package pins Transformers 4, so installing `.[api,mlx]` into the current Transformers 5 runtime creates incompatible requirements. Run the API directly from this checkout instead.

```bash
brew install ffmpeg
conda create -y -n qwen3-tts-mlx -c conda-forge python=3.11 pip
conda activate qwen3-tts-mlx
python -m pip install -r requirements/mlx.txt
python -m pip check

TTS_BACKEND=mlx \
MLX_MODEL_ID=mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit \
HOST=0.0.0.0 PORT=18882 WORKERS=1 TTS_MAX_CONCURRENT=1 \
TTS_LAZY_LOAD=false TTS_WARMUP_ON_START=true TTS_WARMUP_MAX_SECONDS=60 \
TTS_MIN_CHUNK_CHARS=200 TTS_MAX_CHUNK_CHARS=600 \
python -m api.main
```

Run from the repository root. Keep this process running for warm latency; Ctrl+C exits it and releases its model. Other Mac speech engines and their defaults do not need to change. The Linux CUDA-graph installer/launcher is a separate deployment path.

To reproduce the exact weights, use `hf download` with the revision above, then set `MLX_MODEL_ID` to the returned absolute snapshot directory. Once the checkpoint is fully cached, `HF_HUB_OFFLINE=1` supports offline startup.

```bash
curl --fail http://127.0.0.1:18882/health
curl --fail http://127.0.0.1:18882/v1/voices
python scripts/benchmark_api.py --base-url http://127.0.0.1:18882/v1 --mode wav --requests 4
python scripts/benchmark_api.py --base-url http://127.0.0.1:18882/v1 --mode pcm --requests 3
```

These commands use a generic benchmark sentence. Use your own `--text` to measure representative workloads; generated length and latency vary with text and sampling. Keep one loaded model and one generation at a time when comparing results.

## Stream disconnect fix

The original bridge could block its single inference thread while publishing to a full queue after the client disconnected. Subsequent synthesis then timed out despite a healthy model. Publishing now observes a cancellation event, cancels pending queue puts, and closes the model generator when the consumer exits.

`tests/test_mlx_stream_cleanup.py` reproduces a full queue and early consumer close without Apple hardware. It timed out before the fix and now verifies both generator cleanup and successful subsequent synthesis. The fix also passed the real Mac HTTP disconnect/recovery check.
