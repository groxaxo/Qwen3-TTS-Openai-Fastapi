# Mac Voice Studio

Install on Apple Silicon from the checkout using an existing Conda installation.

```bash
conda create -n qwen3-tts-mlx -c conda-forge python=3.11 pip
conda activate qwen3-tts-mlx
python -m pip install -r requirements/mlx.txt
python scripts/install_mac_service.py
```

Reuse an existing `qwen3-tts-mlx` environment if it already contains the pinned requirements. Install FFmpeg with `brew install ffmpeg` for studio previews and compressed speech. The installer downloads the two pinned 1.7B 8-bit MLX checkpoints and starts a launchd login service. The service uses its cached model snapshots offline. Allow the preset model to finish warming before opening [Voice Studio](http://127.0.0.1:18882/studio).

The studio and speech API share the same process on `127.0.0.1:18882`. The CustomVoice checkpoint serves the built-in speakers and the separate Base checkpoint loads on the first reference voice request. Both models remain resident until the service exits. The earlier Mac speed benchmark used CustomVoice alone. Reference cloning has its own cold load and synthesis latency.

Choose **New voice**, upload a 1–30 second reference clip, and enter its exact transcript. Use **Speaker-only reference** when you want to supply audio without a transcript. Only use recordings you have permission to use. **Preview** synthesizes from the current draft. **Save voice** writes its reference and settings to the backend library. Select the saved voice to edit its name, language, transcript, or recording. Preview speed applies to the audition. Set the `speed` field in speech requests to choose a playback speed from another client.

Use the voice ID displayed after saving in the existing speech endpoint.

```bash
curl --fail http://127.0.0.1:18882/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","voice":"clone:My voice","input":"Here is a test.","response_format":"opus"}' \
  --output sample.ogg
```

Studio previews use Ogg Opus at a target of 48 kbps, or 128 kbps MP3 if the browser lacks Ogg Opus playback. The speech API defaults to 128 kbps MP3 and accepts `response_format: "opus"` for Ogg Opus. These are complete audio files. For incremental generation use the existing PCM streaming format.

The service stores voices under `~/Library/Application Support/Qwen3-TTS/voices`. Its optional `--library-dir` argument can point to an existing library. A replacement recording gets a new filename before the metadata changes, so in-flight readers can finish with the previous reference. Back up the entire directory to retain the published profiles and reference revisions.

For Talk on this Mac, the Qwen provider uses the existing HQ route at port 18882. Set `TALK_TTS_ENGINE=qwen` in that client's configuration or invocation. Use `QWEN_TTS_VOICE=vivian` for a preset or `QWEN_TTS_VOICE='clone:My voice'` for a published studio profile. Explicit Talk engine and voice overrides remain available.

Inspect the service and logs using these commands.

```bash
launchctl print "gui/$(id -u)/com.local.qwen3-tts-mlx"
curl --fail http://127.0.0.1:18882/health
tail -f "$HOME/Library/Logs/Qwen3-TTS/api.log"
```

After pulling a code update, run the installer again or use `launchctl kickstart -k "gui/$(id -u)/com.local.qwen3-tts-mlx"`. To stop it, use `launchctl bootout "gui/$(id -u)/com.local.qwen3-tts-mlx"`. The launch agent file is `~/Library/LaunchAgents/com.local.qwen3-tts-mlx.plist`.
