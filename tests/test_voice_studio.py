import io
import subprocess

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import openai_compatible, voice_studio


class ReferenceBackend:
    def get_supported_voices(self):
        return ["Ryan"]

    def supports_voice_cloning(self):
        return True

    async def generate_voice_clone(self, text, ref_audio, ref_audio_sr, **kwargs):
        return ref_audio, ref_audio_sr


def recording(amplitude):
    file = io.BytesIO()
    sf.write(file, np.full(24000, amplitude, dtype=np.float32), 24000, format="WAV")
    return {"reference": ("reference.wav", file.getvalue(), "audio/wav")}


def test_tailnet_studio_requires_the_configured_login_and_origin(monkeypatch):
    origin = "https://mac.example.ts.net:18884"
    login = "owner@example.com"
    monkeypatch.setenv("VOICE_STUDIO_TAILSCALE_ORIGIN", origin)
    monkeypatch.setenv("VOICE_STUDIO_TAILSCALE_LOGIN", login)
    app = FastAPI()
    app.include_router(voice_studio.router)

    with TestClient(app, base_url=origin, client=("100.64.0.2", 20000)) as client:
        identity = {"Tailscale-User-Login": login}
        assert client.get("/studio", headers=identity).status_code == 200
        assert client.get("/studio").status_code == 403
        assert client.get("/studio", headers={
            "Tailscale-User-Login": "other@example.com",
        }).status_code == 403
        assert client.get("/studio", headers={
            **identity, "Origin": "https://elsewhere.example",
        }).status_code == 403
        assert client.get("/studio", headers={
            **identity, "Host": "other.example.ts.net:18884",
        }).status_code == 403

    with TestClient(app, base_url="http://localhost", client=("127.0.0.1", 20000)) as client:
        assert client.get("/studio").status_code == 200


@pytest.mark.parametrize(("extension", "codecs"), [
    ("mp4", ["-c:v", "mpeg4", "-c:a", "aac"]),
    ("avi", ["-c:v", "mpeg4", "-c:a", "pcm_s16le"]),
    ("mkv", ["-c:v", "mpeg4", "-c:a", "flac"]),
    ("wav", ["-vn", "-c:a", "pcm_s16le"]),
    ("m4a", ["-vn", "-c:a", "aac"]),
])
def test_upload_normalizes_audio_and_video(tmp_path, extension, codecs):
    source = tmp_path / f"reference.{extension}"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=32x32:r=10",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
        "-ac", "2", "-t", "2", *codecs, str(source),
    ], check=True)
    app = FastAPI()
    app.include_router(voice_studio.router)
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1", 20000)) as client:
        with source.open("rb") as recording_file:
            response = client.post("/studio/api/reference/normalize",
                                   files={"reference": (source.name, recording_file)})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["x-reference-trimmed"] == "false"
    with sf.SoundFile(io.BytesIO(response.content)) as audio:
        assert (audio.samplerate, audio.channels, audio.subtype) == (24000, 1, "PCM_16")
        assert 1.9 <= audio.frames / audio.samplerate <= 2.1
        assert np.max(np.abs(audio.read())) > 0.01


def test_reference_conversion_reports_trimming_and_unreadable_input():
    app = FastAPI()
    app.include_router(voice_studio.router)
    source = io.BytesIO()
    sf.write(source, np.full(24000 * 31, 0.1), 24000, format="WAV")
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1", 20000)) as client:
        response = client.post("/studio/api/reference/normalize",
                               files={"reference": ("long.wav", source.getvalue())})
        invalid = client.post("/studio/api/reference/normalize",
                              files={"reference": ("bad.mp4", b"not a media file")})
    assert response.status_code == 200, response.text
    assert response.headers["x-reference-trimmed"] == "true"
    assert sf.info(io.BytesIO(response.content)).duration == 30
    assert invalid.status_code == 422


def test_preview_is_private_and_saved_replacement_reaches_speech_api(tmp_path, monkeypatch):
    async def get_backend():
        return ReferenceBackend()

    for module in (voice_studio, openai_compatible):
        monkeypatch.setattr(module, "get_tts_backend", get_backend)
        monkeypatch.setattr(module, "VOICE_LIBRARY_DIR", tmp_path)
    monkeypatch.setattr(openai_compatible, "_ref_audio_cache", {})
    app = FastAPI()
    app.include_router(voice_studio.router)
    app.include_router(openai_compatible.router, prefix="/v1")
    with TestClient(app, base_url="http://localhost", client=("127.0.0.1", 20000)) as client:
        settings = {"name": "My voice", "language": "English", "speaker_only": "true"}
        denied = client.post("/studio/api/voices", data=settings, files=recording(0.1),
                             headers={"Origin": "https://elsewhere.example"})
        assert denied.status_code == 403

        preview = client.post("/studio/api/preview", data={**settings, "text": "Test"},
                              files=recording(0.1))
        assert preview.status_code == 200
        assert preview.headers["content-type"] == "audio/ogg"
        assert preview.content.startswith(b"OggS") and b"OpusHead" in preview.content[:128]
        assert client.get("/studio/api/voices").json()["profiles"] == []

        saved = client.post("/studio/api/voices", data=settings, files=recording(0.1))
        assert saved.status_code == 200
        saved = saved.json()
        payload = {"input": "Test", "model": "tts-1", "voice": saved["voice"],
                   "response_format": "wav"}
        response = client.post("/v1/audio/speech", json=payload)
        assert response.status_code == 200, response.text
        original, _ = sf.read(io.BytesIO(response.content))

        updated = client.post("/studio/api/voices", files=recording(0.3), data={
            **settings, "profile_id": saved["id"], "revision": saved["revision"],
        })
        assert updated.status_code == 200
        response = client.post("/v1/audio/speech", json=payload)
        assert response.status_code == 200, response.text
        replacement, _ = sf.read(io.BytesIO(response.content))
        assert np.mean(original) < np.mean(replacement)
        stale = client.post("/studio/api/voices", data={
            **settings, "profile_id": saved["id"], "revision": saved["revision"],
        })
        assert stale.status_code == 409
