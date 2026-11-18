"""Private CustomVoice CUDA-graph worker; launch by absolute path from a neutral cwd."""
import io
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

model = None
lock = threading.Lock()
MODEL = os.getenv("TTS_MODEL_NAME") or os.getenv("TTS_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
VOICES = ["serena", "vivian", "uncle_fu", "ryan", "aiden", "ono_anna", "sohee", "eric", "dylan"]
LANGUAGES = ["auto", "chinese", "english", "german", "italian", "portuguese", "spanish", "japanese", "korean", "french", "russian"]


class Speech(BaseModel):
    text: str = Field(min_length=1, max_length=12000)
    speaker: str
    language: str = "Auto"
    instruct: str | None = None
    stream: bool = False


@asynccontextmanager
async def lifespan(app):
    global model
    from faster_qwen3_tts import FasterQwen3TTS
    model_path = MODEL
    if os.getenv("TTS_MODEL_REVISION") and not Path(MODEL).is_dir():
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(MODEL, revision=os.environ["TTS_MODEL_REVISION"])
    model = FasterQwen3TTS.from_pretrained(
        model_path, device="cuda:0", dtype=torch.bfloat16,
        attn_implementation="sdpa", max_seq_len=4096, backend="torch",
    )
    model.warmup()
    model.generate_custom_voice(text="Ready.", speaker="ryan", language="English", max_new_tokens=128)
    for _ in model.generate_custom_voice_streaming(
        text="Ready.", speaker="ryan", language="English", chunk_size=8, max_new_tokens=128
    ):
        pass
    yield
    model = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "ready": model is not None, "model": MODEL, "voices": VOICES, "languages": LANGUAGES,
        "device": {"type": "cuda:0", "gpu_name": torch.cuda.get_device_name(0), "gpu_available": True},
    }


def kwargs(p):
    return dict(text=p.text, speaker=p.speaker.lower(), language=p.language,
                instruct=p.instruct, max_new_tokens=2048, non_streaming_mode=False)


def waveform(p):
    with lock:
        wavs, rate = model.generate_custom_voice(**kwargs(p))
        buffer = io.BytesIO()
        sf.write(buffer, wavs[0], rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue()


def chunks(p):
    with lock:
        for audio, rate, timing in model.generate_custom_voice_streaming(**kwargs(p), chunk_size=8):
            if rate != 24000:
                raise RuntimeError("Unexpected sample rate")
            yield np.asarray(audio, dtype="<f4").tobytes()


def next_chunk(iterator):
    return next(iterator, None)


@app.post("/synthesize")
async def synthesize(p: Speech):
    if p.speaker.lower() not in VOICES:
        raise HTTPException(400, "Unknown CustomVoice speaker")
    if p.language.lower() not in LANGUAGES:
        raise HTTPException(400, "Unknown language")
    if not p.stream:
        return Response(await run_in_threadpool(waveform, p), media_type="audio/wav")

    async def stream():
        iterator = chunks(p)
        try:
            while True:
                data = await run_in_threadpool(next_chunk, iterator)
                if data is None:
                    break
                yield data
        finally:
            with anyio.CancelScope(shield=True):
                await run_in_threadpool(iterator.close)

    return StreamingResponse(stream(), media_type="application/octet-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("TTS_GRAPH_PORT", "18881")))
