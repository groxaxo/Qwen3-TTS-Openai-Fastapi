"""CPU frontend for the separately isolated CustomVoice CUDA-graph worker."""
import asyncio
import io
import os
import time

import httpx
import numpy as np
import soundfile as sf

from .base import TTSBackend


class CUDAGraphBackend(TTSBackend):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.url = os.getenv("TTS_GRAPH_WORKER_URL", "http://127.0.0.1:18881")
        self._health = {}

    async def initialize(self):
        deadline = time.monotonic() + 240
        async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
            while True:
                try:
                    response = await client.get(self.url + "/health")
                    response.raise_for_status()
                    self._health = response.json()
                    if self._health.get("ready"):
                        return
                except (httpx.HTTPError, ValueError):
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError("CUDA-graph worker did not become ready. Use ./start_server.sh "
                                       "to start both processes, or check TTS_GRAPH_WORKER_URL.")
                await asyncio.sleep(1)

    async def generate_speech(self, text, voice, language="Auto", instruct=None, speed=1.0):
        async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
            response = await client.post(self.url + "/synthesize", json={
                "text": text, "speaker": voice, "language": language,
                "instruct": instruct, "stream": False,
            })
            response.raise_for_status()
        audio, rate = sf.read(io.BytesIO(response.content), dtype="float32")
        if speed != 1.0:
            import librosa
            audio = await asyncio.to_thread(librosa.effects.time_stretch, audio, rate=speed)
        return audio, rate

    async def generate_speech_streaming(self, text, voice, language="Auto", instruct=None, model="tts-1"):
        # The private worker emits little-endian float32 samples; the public
        # API's encode_audio performs the existing signed-16-bit PCM conversion.
        async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
            async with client.stream("POST", self.url + "/synthesize", json={
                "text": text, "speaker": voice, "language": language,
                "instruct": instruct, "stream": True,
            }) as response:
                response.raise_for_status()
                pending = b""
                async for data in response.aiter_bytes():
                    pending += data
                    size = len(pending) // 4 * 4
                    if size:
                        yield np.frombuffer(pending[:size], dtype="<f4").copy(), 24000
                        pending = pending[size:]
                if pending:
                    raise RuntimeError("Worker returned truncated float32 PCM")

    def get_backend_name(self):
        return "cuda_graphs"

    def get_model_id(self):
        return self.model_name

    def get_supported_voices(self):
        return list(self._health.get("voices", []))

    def get_supported_languages(self):
        return list(self._health.get("languages", []))

    def is_ready(self):
        return bool(self._health.get("ready"))

    def get_device_info(self):
        info = dict(self._health.get("device", {}))
        if "type" in info:
            info.setdefault("device", info["type"])
        return info

    def get_model_type(self):
        return "customvoice"
