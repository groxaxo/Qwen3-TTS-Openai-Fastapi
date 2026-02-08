# Persistent Custom Voices

> **Status:** Experimental

Load your own cloned voices at server startup. Each voice is a folder inside `custom_voices/` containing a reference audio file and an optional transcript.

## Requirements

- **Official backend only** (`TTS_BACKEND=official`). vLLM does not support custom voices.
- **Base model** (`Qwen/Qwen3-TTS-12Hz-1.7B-Base` or the 0.6B equivalent). The default CustomVoice model will skip loading.

## Directory layout

```
custom_voices/
├── Alice/
│   ├── reference.wav        # required (or .mp3, .flac, .ogg, .m4a)
│   ├── reference.txt        # optional — transcript of the audio
│   └── .cached_prompt.pt    # auto-generated on first load
└── Bob/
    └── reference.wav
```

- **Voice name** = folder name (case-sensitive, but must not collide case-insensitively with built-in voices like Vivian, Ryan, etc.).
- Folders starting with `.` are ignored.
- If `reference.txt` is provided, ICL (in-context learning) mode is used for higher quality. Without it, x-vector-only mode is used.
- `.cached_prompt.pt` is created automatically on first startup and reused on subsequent starts.

## Setup — Host

```bash
# 1. Create the directory at the project root
mkdir -p custom_voices/Alice

# 2. Add a reference audio clip (5–15 s recommended)
cp ~/my_sample.wav custom_voices/Alice/reference.wav

# 3. Optionally add a transcript of that clip
echo "This is Alice speaking clearly." > custom_voices/Alice/reference.txt

# 4. Start the server with the Base model
TTS_MODEL_NAME=Qwen/Qwen3-TTS-12Hz-1.7B-Base python -m api.main
```

The `TTS_CUSTOM_VOICES` env var overrides the default path:

```bash
TTS_CUSTOM_VOICES=/path/to/my/voices python -m api.main
```

## Setup — Docker

Mount your `custom_voices/` directory into the container. In `docker-compose.yml`, add a volume to the `qwen3-tts-gpu` service:

```yaml
qwen3-tts-gpu:
  environment:
    - TTS_MODEL_NAME=Qwen/Qwen3-TTS-12Hz-1.7B-Base
  volumes:
    - ./custom_voices:/app/custom_voices          # <-- add this
```

The container default path is `/app/custom_voices`. To use a different path, set `TTS_CUSTOM_VOICES`:

```yaml
  environment:
    - TTS_CUSTOM_VOICES=/data/voices
  volumes:
    - /host/path/to/voices:/data/voices
```

## Adding or removing voices

Custom voices are scanned **only at startup**. After adding, removing, or updating voice folders, restart the server for changes to take effect:

```bash
# Host
# (Ctrl-C the running server, then start again)
python -m api.main

# Docker
docker-compose restart qwen3-tts-gpu
```

To force re-extraction of a cached voice, delete its `.cached_prompt.pt` before restarting.

## Usage

Once loaded, custom voices appear in the voice list and can be used like any built-in voice:

```bash
# List voices (custom voices included)
curl http://localhost:8880/v1/audio/voices

# Generate speech with a custom voice
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts","voice":"Alice","input":"Hello!"}' \
  -o output.mp3
```
