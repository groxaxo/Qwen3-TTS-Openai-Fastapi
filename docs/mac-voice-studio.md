# Mac Voice Studio

Install on Apple Silicon from the checkout using an existing Conda installation.

```bash
conda create -n qwen3-tts-mlx -c conda-forge python=3.11 pip
conda activate qwen3-tts-mlx
python -m pip install -r requirements/mlx.txt
python scripts/install_mac_service.py
```

Reuse an existing `qwen3-tts-mlx` environment if it already contains the pinned requirements. Install FFmpeg with `brew install ffmpeg` for upload conversion, studio previews, and compressed speech. The installer downloads the two pinned 1.7B 8-bit MLX checkpoints and starts a launchd login service. The service uses its cached model snapshots offline. Allow the preset model to finish warming before opening [Voice Studio](http://127.0.0.1:18882/studio).

The studio and speech API share the same process on `127.0.0.1:18882`. The CustomVoice checkpoint serves the built-in speakers and the separate Base checkpoint loads on the first reference voice request. Both models remain resident until the service exits. The earlier Mac speed benchmark used CustomVoice alone. Reference cloning has its own cold load and synthesis latency.

## Supported reference uploads

| Upload type | Common formats |
| --- | --- |
| Video with audio | MP4, AVI, MKV, MOV, WebM, MPEG, MPEG-TS, FLV, WMV, 3GP |
| Audio | WAV, MP3, M4A, FLAC, Ogg Vorbis, Ogg Opus, AAC, AIFF, CAF, WMA |

Choose **New voice** and select or drop a file up to **100 MiB**. FFmpeg reads the first audio track and the studio converts it to **24 kHz mono, 16-bit PCM WAV** for playback and cloning. Container and codec support depend on the installed FFmpeg build. Files with no readable audio track, silence, or less than one second of audio show an upload error.

If the audio exceeds 30 seconds, the studio keeps the first 30 seconds and labels the clip. Listen to the prepared reference and enter its exact transcript, or select **Speaker-only reference** to use audio without a transcript. Only upload recordings you have permission to use.

**Preview** synthesizes from the current draft. **Save voice** writes its reference and settings to the backend library. Select the saved voice to edit its name, language, transcript, or recording. Preview speed applies to the audition. Set the `speed` field in speech requests to choose a playback speed from another client.

Use the voice ID displayed after saving in the existing speech endpoint.

```bash
curl --fail http://127.0.0.1:18882/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","voice":"clone:My voice","input":"Here is a test.","response_format":"opus"}' \
  --output sample.ogg
```

Studio previews use Ogg Opus at a target of 48 kbps, or 128 kbps MP3 if the browser lacks Ogg Opus playback. The speech API defaults to 128 kbps MP3 and accepts `response_format: "opus"` for Ogg Opus. These are complete audio files. For incremental generation use the existing PCM streaming format.

The service stores voices under `~/Library/Application Support/Qwen3-TTS/voices`. Its optional `--library-dir` argument can point to an existing library. A replacement recording gets a new filename before the metadata changes, so in-flight readers can finish with the previous reference. Back up the entire directory to retain the published profiles and reference revisions.

## Open Voice Studio over Tailscale

1. On the Mac, connect to your tailnet and pick an unused HTTPS port for the studio, such as `18884`. Publish the loopback service through Tailscale Serve. Enable HTTPS for the tailnet if the CLI prompts you.

   ```bash
   tailscale serve --bg --https=18884 http://127.0.0.1:18882
   tailscale serve status
   ```

2. Use the HTTPS origin printed by Serve and the exact Tailscale login of the account that will use the studio. Run the installer from the activated MLX environment. If you previously chose a custom library folder or port, pass the same `--library-dir` or `--port`.

   ```bash
   python scripts/install_mac_service.py \
     --tailscale-origin 'https://your-mac.your-tailnet.ts.net:18884' \
     --tailscale-login 'you@example.com'
   ```

3. From another device connected as that account, open the Serve URL with `/studio` appended. If the URL fails to resolve, check that the client uses the tailnet DNS settings. If it returns 403, check the exact origin, port, and Tailscale login. Tagged devices do not supply a user login to Serve. [Tailscale Serve and identity headers](https://tailscale.com/docs/features/tailscale-serve#identity-headers)

Keep the FastAPI listener on `127.0.0.1` so Serve is the only network path into it. Serve supplies the authenticated login used by the owner-only studio routes. Use Serve for these ports and keep Funnel disabled. The synthesis API is available to devices permitted by your tailnet policy, so use tailnet access rules to restrict who can invoke it.

### Route saved voices through an existing TTS gateway

Point the gateway at the Mac's Serve URL, forward `GET /v1/voices` and `POST /v1/audio/speech`, and use upstream model `tts-1`. Use `GET /health` to check whether the Mac backend is ready. The gateway should keep the saved `clone:Voice name` value and forward `response_format: "opus"` for Ogg Opus.

An alternative for a gateway host without tailnet DNS is a dedicated tailnet-only HTTP Serve port. Configure the gateway to connect to the Mac's Tailscale IP on that port and send the DNS name and port printed by Serve as the HTTP `Host` header, including on health checks.

   ```bash
   tailscale serve --bg --http=18885 http://127.0.0.1:18882
   ```

If your gateway publishes the Mac as `qwen3-tts-mac-mlx8`, replace `gateway` with its hostname in these examples.

   ```bash
   curl --fail 'http://gateway:12434/v1/voices?model=qwen3-tts-mac-mlx8'
   curl --fail http://gateway:12434/v1/audio/speech \
     -H 'Content-Type: application/json' \
     -d '{"model":"qwen3-tts-mac-mlx8","voice":"clone:My voice","input":"Here is a test.","response_format":"opus"}' \
     --output sample.ogg
   ```

Set the Mac route to default to Opus if you want clients to receive Ogg when they omit the format. Keep the studio on the HTTPS Serve URL for browser use. To remove just the example endpoints, run `tailscale serve --https=18884 off` and, if you enabled it, `tailscale serve --http=18885 off`. [Tailscale Serve CLI reference](https://tailscale.com/docs/reference/tailscale-cli/serve)

## Service management

For Talk on this Mac, the Qwen provider uses the existing HQ route at port 18882. Set `TALK_TTS_ENGINE=qwen` in that client's configuration or invocation. Use `QWEN_TTS_VOICE=vivian` for a preset or `QWEN_TTS_VOICE='clone:My voice'` for a published studio profile. Explicit Talk engine and voice overrides remain available.

Inspect the service and logs using these commands.

```bash
launchctl print "gui/$(id -u)/com.local.qwen3-tts-mlx"
curl --fail http://127.0.0.1:18882/health
tail -f "$HOME/Library/Logs/Qwen3-TTS/api.log"
```

After pulling a code update, use `launchctl kickstart -k "gui/$(id -u)/com.local.qwen3-tts-mlx"` to keep the current settings. When rerunning the installer, pass the same Tailscale options if you want to keep remote studio access. To stop the service, use `launchctl bootout "gui/$(id -u)/com.local.qwen3-tts-mlx"`. The launch agent file is `~/Library/LaunchAgents/com.local.qwen3-tts-mlx.plist`.
