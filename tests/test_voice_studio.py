import io

import numpy as np
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
