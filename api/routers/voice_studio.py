"""Local voice library editor and audition endpoints."""
import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response

from .openai_compatible import VOICE_LIBRARY_DIR, _generation_semaphore, get_tts_backend
from ..services.audio_encoding import convert_to_wav, encode_audio

MAX_UPLOAD = 100 * 1024 * 1024
REFERENCE_RATE = 24000
MAX_REFERENCE_SAMPLES = 30 * REFERENCE_RATE
REFERENCE_FORMATS = (
    "aac,ac3,aiff,alac,amr,ape,asf,au,avi,caf,eac3,flac,flv,matroska,webm,"
    "mov,mp4,m4a,3gp,3g2,mj2,mp3,mpeg,mpegts,ogg,tta,w64,wav,wv"
)
reference_conversions = asyncio.Semaphore(2)
LANGUAGES = ("Auto", "English", "Chinese", "Japanese", "Korean", "German",
             "French", "Spanish", "Russian", "Portuguese", "Italian")


def local_browser(request: Request):
    host = request.headers.get("host", "").lower()
    origin = request.headers.get("origin")
    login = request.headers.get("tailscale-user-login")
    tailscale_origin = os.getenv("VOICE_STUDIO_TAILSCALE_ORIGIN", "")
    tailscale_login = os.getenv("VOICE_STUDIO_TAILSCALE_LOGIN", "")
    local = (
        request.client is not None
        and request.client.host in {"127.0.0.1", "::1"}
        and re.fullmatch(r"(?:127\.0\.0\.1|localhost|\[::1\])(?::\d+)?", host)
        and not login
        and (not origin or origin in {f"http://{host}", f"https://{host}"})
    )
    # Deploy only behind Tailscale Serve on a loopback-bound listener. Serve
    # strips supplied identity headers and injects the authenticated login.
    tailnet = (
        tailscale_origin == f"https://{host}" and bool(tailscale_login)
        and login == tailscale_login and (not origin or origin == tailscale_origin)
    )
    if (
        not (local or tailnet)
        or request.headers.get("sec-fetch-site") == "cross-site"
    ):
        raise HTTPException(403, "Use localhost or the configured Tailscale account.")


router = APIRouter(dependencies=[Depends(local_browser)])


def profile_path(profile_id: str) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", profile_id):
        raise HTTPException(404, "Voice not found.")
    root = VOICE_LIBRARY_DIR / "profiles"
    path = root / profile_id
    if path.is_symlink():
        raise HTTPException(404, "Voice not found.")
    return path


def read_profile(profile_id: str):
    path = profile_path(profile_id) / "meta.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        raise HTTPException(404, "Voice not found.") from None


def reference_path(profile_id: str, profile):
    filename = profile.get("ref_audio_filename", "")
    path = profile_path(profile_id) / filename
    if not filename or Path(filename).name != filename or path.is_symlink() or not path.is_file():
        raise HTTPException(404, "Reference audio not found.")
    return path


def profiles():
    for directory in (VOICE_LIBRARY_DIR / "profiles").glob("*"):
        try:
            profile = read_profile(directory.name)
            reference_path(directory.name, profile)
        except HTTPException:
            continue
        yield {
            "id": directory.name, "name": profile["name"],
            "language": profile.get("language", "Auto"),
            "transcript": profile.get("ref_text", ""),
            "speaker_only": profile.get("x_vector_only_mode", False),
            "revision": profile.get("studio_revision", ""),
        }


def decode_reference(source):
    try:
        with sf.SoundFile(source) as audio:
            if not 1 <= audio.frames / audio.samplerate <= 30 or audio.channels > 2:
                raise ValueError("Use a mono or stereo recording between 1 and 30 seconds.")
            samples = audio.read(dtype="float32", always_2d=True).mean(axis=1)
            if not np.isfinite(samples).all() or np.max(np.abs(samples)) < 0.0001:
                raise ValueError("Reference audio must be finite and audible.")
            return samples, audio.samplerate
    except (ValueError, RuntimeError) as error:
        raise HTTPException(422, f"Invalid reference recording. {error}") from None


def normalize_upload(source):
    if source.seek(0, os.SEEK_END) > MAX_UPLOAD:
        raise HTTPException(413, "Reference file must be 100 MiB or smaller.")
    source.seek(0)
    # Reuse a normalized WAV sent back by the browser on preview or save.
    try:
        with sf.SoundFile(source) as audio:
            if (audio.format == "WAV" and audio.subtype == "PCM_16"
                    and audio.samplerate == REFERENCE_RATE and audio.channels == 1
                    and REFERENCE_RATE <= audio.frames <= MAX_REFERENCE_SAMPLES):
                samples = audio.read(dtype="float32")
                if samples.size >= REFERENCE_RATE and np.max(np.abs(samples)) >= 0.0001:
                    return samples, False
    except (ValueError, RuntimeError):
        pass
    source.seek(0)

    # A seekable, private input also supports MP4s with metadata at the end.
    # Only decode standalone media containers and keep network protocols disabled.
    with tempfile.NamedTemporaryFile(prefix="qwen-reference-") as media:
        shutil.copyfileobj(source, media)
        media.flush()
        try:
            result = subprocess.run([
                "ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe",
                "-format_whitelist", REFERENCE_FORMATS, "-threads", "2", "-i", media.name,
                "-map", "0:a:0", "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(REFERENCE_RATE),
                # Decode slightly past the limit to detect that the clip was shortened.
                "-t", "30.01", "-f", "s16le", "-c:a", "pcm_s16le", "pipe:1",
            ], stdin=subprocess.DEVNULL, capture_output=True, check=True, timeout=30)
        except FileNotFoundError:
            raise HTTPException(503, "Reference conversion requires FFmpeg.") from None
        except subprocess.TimeoutExpired:
            raise HTTPException(422, "Reference conversion timed out. Try a shorter file.") from None
        except subprocess.CalledProcessError:
            raise HTTPException(
                422, "Could not read an audio track. Choose a supported file with sound."
            ) from None
    samples = np.frombuffer(result.stdout, dtype="<i2")
    if samples.size < REFERENCE_RATE:
        raise HTTPException(422, "The reference must contain at least one second of audio.")
    trimmed = samples.size > MAX_REFERENCE_SAMPLES
    samples = samples[:MAX_REFERENCE_SAMPLES].astype(np.float32) / 32768
    if np.max(np.abs(samples)) < 0.0001:
        raise HTTPException(422, "The reference recording is silent.")
    return samples, trimmed


async def normalize_reference(reference: UploadFile):
    async with reference_conversions:
        return await asyncio.to_thread(normalize_upload, reference.file)


async def get_reference(reference, profile_id=None):
    if reference is not None:
        samples, _ = await normalize_reference(reference)
        return samples, REFERENCE_RATE
    if profile_id:
        path = reference_path(profile_id, read_profile(profile_id))
        return await asyncio.to_thread(decode_reference, path)
    raise HTTPException(422, "Choose a reference recording first.")


def validate_settings(language, transcript, speaker_only):
    if language not in LANGUAGES:
        raise HTTPException(422, "Choose a supported language.")
    if not speaker_only and not transcript.strip():
        raise HTTPException(422, "Enter the recording transcript or enable speaker-only mode.")


@router.get("/studio", include_in_schema=False)
async def studio():
    return FileResponse(Path(__file__).parents[1] / "static" / "voice-studio.html")


@router.post("/studio/api/reference/normalize")
async def upload_reference(reference: UploadFile = File(...)):
    samples, trimmed = await normalize_reference(reference)
    wav = await asyncio.to_thread(convert_to_wav, samples, REFERENCE_RATE)
    return Response(wav, media_type="audio/wav", headers={
        "Cache-Control": "no-store", "X-Reference-Trimmed": str(trimmed).lower(),
    })


@router.get("/studio/api/voices")
async def voices():
    backend = await get_tts_backend()
    return {"presets": backend.get_supported_voices(), "profiles": list(profiles()),
            "can_clone": backend.supports_voice_cloning()}


@router.get("/studio/api/reference/{profile_id}")
async def reference_audio(profile_id: str):
    path = reference_path(profile_id, read_profile(profile_id))
    return FileResponse(path, headers={"Cache-Control": "no-store"})


@router.post("/studio/api/preview")
async def preview(
    text: str = Form(min_length=1, max_length=1000),
    language: str = Form("Auto"),
    transcript: str = Form("", max_length=4096),
    speaker_only: bool = Form(False),
    speed: float = Form(1.0, ge=0.5, le=2.0),
    preset: str = Form(""),
    profile_id: str = Form(""),
    instruction: str = Form("", max_length=500),
    response_format: Literal["opus", "mp3"] = Form("opus"),
    reference: UploadFile | None = File(None),
):
    backend = await get_tts_backend()
    text = text.strip()
    if not text:
        raise HTTPException(422, "Enter text to preview.")
    if preset:
        if preset not in backend.get_supported_voices() or language not in LANGUAGES:
            raise HTTPException(422, "Choose a supported speaker and language.")
        async with _generation_semaphore:
            audio, rate = await backend.generate_speech(text, preset, language, instruction or None, speed)
    else:
        if not backend.supports_voice_cloning():
            raise HTTPException(409, "Configure a Base checkpoint to preview reference voices.")
        validate_settings(language, transcript, speaker_only)
        samples, reference_rate = await get_reference(reference, profile_id)
        async with _generation_semaphore:
            audio, rate = await backend.generate_voice_clone(
                text=text, ref_audio=samples, ref_audio_sr=reference_rate,
                ref_text=transcript.strip() or None, language=language,
                x_vector_only_mode=speaker_only, speed=speed,
            )
    data = await asyncio.to_thread(encode_audio, audio, response_format, rate)
    media_type = "audio/ogg" if response_format == "opus" else "audio/mpeg"
    return Response(data, media_type=media_type, headers={"Cache-Control": "no-store"})


@router.post("/studio/api/voices")
async def save_voice(
    name: str = Form(min_length=1, max_length=80),
    language: str = Form("Auto"),
    transcript: str = Form("", max_length=4096),
    speaker_only: bool = Form(False),
    profile_id: str = Form(""),
    revision: str = Form(""),
    reference: UploadFile | None = File(None),
):
    name = name.strip()
    if not name or any(ord(character) < 32 for character in name):
        raise HTTPException(422, "Enter a valid voice name.")
    validate_settings(language, transcript, speaker_only)
    if reference is None and not profile_id:
        raise HTTPException(422, "Choose a reference recording first.")
    audio = await get_reference(reference) if reference is not None else None
    previous = read_profile(profile_id) if profile_id else {}
    if profile_id and previous.get("studio_revision", "") != revision:
        raise HTTPException(409, "This voice changed in another tab. Reload it before saving.")
    profile_id = profile_id or uuid4().hex
    if any(
        voice["id"] != profile_id and
        name.casefold() in {voice["name"].casefold(), voice["id"].casefold()}
        for voice in profiles()
    ):
        raise HTTPException(409, "That voice name is already in use.")
    directory = profile_path(profile_id)
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    new_revision = uuid4().hex
    if audio is not None:
        filename = f"reference-{new_revision}.wav"
        samples, sample_rate = audio
        sf.write(directory / filename, samples, sample_rate, subtype="PCM_16")
    else:
        filename = reference_path(profile_id, previous).name
    metadata = dict(previous, profile_id=profile_id, name=name, language=language,
                    ref_text=transcript.strip(), x_vector_only_mode=speaker_only,
                    ref_audio_filename=filename, studio_revision=new_revision)
    temporary_meta = directory / f"{new_revision}.json"
    temporary_meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    os.replace(temporary_meta, directory / "meta.json")
    return {"id": profile_id, "revision": new_revision, "voice": f"clone:{name}"}
