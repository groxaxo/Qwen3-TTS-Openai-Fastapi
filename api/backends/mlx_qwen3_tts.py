# coding=utf-8
# SPDX-License-Identifier: Apache-2.0

"""Apple Silicon MLX backend for Qwen3-TTS.

This backend uses mlx-audio to run Qwen3-TTS natively on Apple Silicon.
The default checkpoint is a CustomVoice model, so the backend uses
``generate_custom_voice`` for non-streaming and the unified ``generate()``
with ``stream=True`` for incremental chunks. The backend reports
``supports_voice_cloning() == False``; the ``clone:ProfileName``
voice-library route therefore falls through to the existing
"voice cloning not supported" error path.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np

from .base import TTSBackend

logger = logging.getLogger(__name__)

DEFAULT_MLX_MODEL = (
    "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"
)

# Default chunk interval (seconds) for streaming. The unified
# ``model.generate(stream=True)`` API accepts this; 0.32s ~ 4 tokens at
# the 12.5 Hz Qwen3-TTS frame rate. MLX-Audio 0.3.x currently returns
# the full audio in one chunk regardless of this setting (upstream
# issue #720), but the parameter is accepted and forward-compatible
# with newer versions that do honor it.
DEFAULT_STREAMING_INTERVAL = 0.32

# Safety caps. MLX-Audio 0.3.x has a known issue where the first call
# after a fresh process start can hang in the model's graph compile,
# either producing looping audio (non-streaming) or never yielding
# (streaming). These caps prevent a stuck request from wedging the
# whole server (which is single-concurrent by design).
#
# Realistic maxima for a single TTS request on Qwen3-TTS:
# - 4096 max_tokens at 12.5 Hz = ~327s of audio
# - A typical request produces <30s of audio
#
# 300s of audio is a generous ceiling for legitimate use. The wall
# cap is 30s to fail fast (matches typical HTTP client timeouts) so
# the user sees a clear error within their client's wait window
# rather than the model silently hanging for 3 minutes.
MAX_GENERATED_AUDIO_SECONDS = 300.0
MAX_GENERATION_WALL_SECONDS = 30.0

# Lowercase names — the real model returns lowercase, and casefold()
# is used for lookups, so the fallback list matches that convention.
FALLBACK_VOICES = [
    "vivian",
    "serena",
    "uncle_fu",
    "dylan",
    "eric",
    "ryan",
    "aiden",
    "ono_anna",
    "sohee",
]

FALLBACK_LANGUAGES = [
    "auto",
    "chinese",
    "english",
    "japanese",
    "korean",
    "german",
    "french",
    "russian",
    "portuguese",
    "spanish",
    "italian",
]

# The global OpenAI mapping currently converts some aliases into
# official-backend speaker names that the MLX checkpoint does not
# contain.
MLX_COMPATIBILITY_ALIASES = {
    "sophia": "serena",
    "isabella": "vivian",
    "evan": "aiden",
    "lily": "serena",
}


class MLXQwen3TTSBackend(TTSBackend):
    """Qwen3-TTS backend using MLX-Audio on Apple Silicon."""

    def __init__(self, model_name: str = DEFAULT_MLX_MODEL):
        super().__init__()
        self.model_name = model_name
        self.device = "metal"
        self.dtype = "8-bit"
        self._ready = False
        self._broken = False  # set when generation hangs past the cap
        self._voices: List[str] = []
        self._languages: List[str] = []

        # Keep loading and generation on one dedicated worker. This
        # prevents MLX inference from blocking FastAPI's event loop and
        # avoids running the same model concurrently on arbitrary
        # asyncio worker threads.
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mlx-qwen3-tts",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        if self._ready:
            return

        machine = platform.machine().lower()
        if sys.platform != "darwin" or machine not in {"arm64", "aarch64"}:
            raise RuntimeError(
                "The MLX backend requires an Apple Silicon Mac. "
                f"Detected platform={sys.platform!r}, machine={machine!r}."
            )

        def load():
            try:
                from mlx_audio.tts.utils import load_model
            except ImportError as exc:
                raise RuntimeError(
                    "MLX dependencies are missing. Install with: "
                    'pip install -e ".[api,mlx]"'
                ) from exc

            # transformers 5.0.0rc3 logs a one-time "incorrect regex
            # pattern ... set fix_mistral_regex=True" advisory when the
            # Qwen3-TTS text tokenizer is loaded (lazily, inside the
            # model's post_load_hook on the first generate() call).
            #
            # We deliberately do NOT set that flag:
            #  * It is a false positive here -- the checkpoint ships a
            #    Qwen2 tokenizer, not a Mistral one. transformers trips
            #    the warning via a coarse vocab-size heuristic on the
            #    local-load path, not because this is a Mistral model.
            #  * It is un-settable in this transformers version anyway:
            #    AutoTokenizer.from_pretrained(..., fix_mistral_regex=True)
            #    raises "TokenizersBackend._patch_mistral_regex() got
            #    multiple values for keyword argument 'fix_mistral_regex'"
            #    (upstream forwards the kwarg both explicitly and via
            #    **kwargs at tokenization_utils_tokenizers.py:373-379),
            #    which breaks tokenizer load entirely and 500s the
            #    /v1/audio/speech route.
            #
            # So we silence just that one advisory log line. The filter
            # is installed once and left in place, because the tokenizer
            # is loaded later -- during the first generate() call, not
            # during load_model() below.
            import logging as _logging

            class _DropMistralRegexWarning(_logging.Filter):
                def filter(self, record: _logging.LogRecord) -> bool:
                    return "fix_mistral_regex" not in record.getMessage()

            _tok_logger = _logging.getLogger(
                "transformers.tokenization_utils_tokenizers"
            )
            if not any(
                isinstance(f, _DropMistralRegexWarning)
                for f in _tok_logger.filters
            ):
                _tok_logger.addFilter(_DropMistralRegexWarning())

            logger.info("Loading MLX model: %s", self.model_name)
            model = load_model(self.model_name)

            speaker_getter = getattr(model, "get_supported_speakers", None)
            language_getter = getattr(model, "get_supported_languages", None)

            voices = (
                list(speaker_getter())
                if callable(speaker_getter)
                else list(getattr(model, "supported_speakers", []))
            )
            languages = (
                list(language_getter())
                if callable(language_getter)
                else list(getattr(model, "supported_languages", []))
            )

            return model, voices, languages

        loop = asyncio.get_running_loop()
        self.model, self._voices, self._languages = await loop.run_in_executor(
            self._executor,
            load,
        )

        self._ready = True
        logger.info(
            "MLX backend loaded: model=%s voices=%s",
            self.model_name,
            self.get_supported_voices(),
        )

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    async def generate_speech(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        if not self._ready:
            await self.initialize()

        if self._broken:
            raise RuntimeError(
                "MLX backend previously wedged on a generation request "
                "and is now marked as broken. Restart the server "
                "(the model is in an unrecoverable state until the "
                "process is fresh). This is an upstream mlx-audio 0.3.x "
                "issue — see TTS_WARMUP_ON_START=true to absorb the "
                "cold-start graph compile at boot."
            )

        resolved_voice = self._resolve_voice(voice)
        resolved_language = self._resolve_language(language)
        sample_rate = int(getattr(self.model, "sample_rate", 24000))
        max_samples = int(MAX_GENERATED_AUDIO_SECONDS * sample_rate)
        start = time.monotonic()

        def generate() -> Tuple[np.ndarray, int]:
            kwargs: Dict[str, Any] = {
                "text": text,
                "speaker": resolved_voice,
                "language": resolved_language,
            }
            if instruct:
                kwargs["instruct"] = instruct

            # mlx-audio 0.3.x's ``generate_custom_voice`` is a generator
            # that yields chunks until the model decides it's done. The
            # cold-start graph compile is broken (upstream issue: the
            # model can loop forever producing 30+ minutes of garbage).
            # We consume the generator ourselves, bounding both the
            # accumulated sample count and the wall clock, so a stuck
            # call fails fast instead of wedging the server.
            chunks: List[np.ndarray] = []
            total_samples = 0
            for result in self.model.generate_custom_voice(**kwargs):
                audio = getattr(result, "audio", None)
                if audio is None:
                    continue
                chunk = self._result_to_numpy(result)
                chunks.append(chunk)
                total_samples += len(chunk)

                if total_samples > max_samples:
                    raise RuntimeError(
                        f"MLX-Audio generated more than "
                        f"{MAX_GENERATED_AUDIO_SECONDS:.0f}s of audio "
                        f"for a single request — aborting. This usually "
                        f"indicates mlx-audio is in a bad state (the "
                        f"underlying model looped). Try restarting the "
                        f"server with TTS_WARMUP_ON_START=true to "
                        f"absorb the cold-start graph compile at boot."
                    )

                elapsed = time.monotonic() - start
                if elapsed > MAX_GENERATION_WALL_SECONDS:
                    raise RuntimeError(
                        f"MLX-Audio generation exceeded "
                        f"{MAX_GENERATION_WALL_SECONDS:.0f}s wall-clock "
                        f"for a single request — aborting. This usually "
                        f"indicates mlx-audio is in a bad state (the "
                        f"underlying model is not finishing). Try "
                        f"restarting the server with "
                        f"TTS_WARMUP_ON_START=true."
                    )

            if not chunks:
                raise RuntimeError("MLX-Audio returned no audio samples")

            audio = np.concatenate(chunks).astype(np.float32, copy=False)

            if speed != 1.0:
                from librosa.effects import time_stretch

                audio = time_stretch(
                    audio,
                    rate=speed,
                ).astype(np.float32, copy=False)

            return audio, sample_rate

        loop = asyncio.get_running_loop()

        # Run the blocking call in the worker thread, but enforce a
        # hard wall-clock deadline on the asyncio side. If the worker
        # exceeds the deadline, the caller's coroutine raises
        # TimeoutError; the worker thread is *not* killed (Python
        # cannot interrupt a C-extension holding the GIL), so we mark
        # the backend as broken — subsequent requests fail fast with a
        # clear error pointing at the upstream bug and the warmup
        # workaround.
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(self._executor, generate),
                timeout=MAX_GENERATION_WALL_SECONDS,
            )
        except asyncio.TimeoutError:
            self._broken = True
            logger.error(
                "MLX generation exceeded %.0fs wall-clock and the "
                "worker thread is wedged inside mlx-audio. Marking "
                "the backend as broken; subsequent requests will fail "
                "fast until the process is restarted. The root cause "
                "is upstream mlx-audio 0.3.x graph-compile hangs — "
                "set TTS_WARMUP_ON_START=true to absorb the bad cold "
                "call at boot.",
                MAX_GENERATION_WALL_SECONDS,
            )
            raise RuntimeError(
                f"MLX-Audio generation timed out after "
                f"{MAX_GENERATION_WALL_SECONDS:.0f}s wall-clock. The "
                f"model is in a wedged state (upstream mlx-audio 0.3.x "
                f"bug). The backend is now marked broken; restart the "
                f"server. Using TTS_WARMUP_ON_START=true at boot will "
                f"absorb the bad cold call before users hit it."
            ) from None
        finally:
            try:
                import mlx.core as mx  # type: ignore

                # Release reusable Metal scratch buffers after the waveform has
                # been copied to NumPy. Model parameters remain resident.
                mx.clear_cache()
            except (ImportError, AttributeError):
                pass

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def generate_speech_streaming(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
        model: str = "tts-1",
        streaming_interval: float = DEFAULT_STREAMING_INTERVAL,
    ) -> AsyncGenerator[Tuple[np.ndarray, int], None]:
        """Stream speech as ``(pcm_chunk, sample_rate)`` tuples.

        The MLX-Audio unified ``model.generate(stream=True)`` API is a
        synchronous generator. We bridge it to FastAPI by running the
        generator in our single-worker thread executor and forwarding
        each chunk through an ``asyncio.Queue`` to the async caller.

        In mlx-audio 0.3.x the generator typically yields a single
        full-audio chunk (the ``streaming_interval`` parameter is
        accepted but the chunking is not yet honored — upstream
        issue #720). The transport path is still real: the first chunk
        reaches the client as soon as MLX-Audio finishes the
        generation, which is faster than the non-streaming path
        because the FastAPI side can start sending bytes immediately.

        On newer mlx-audio versions that *do* honor
        ``streaming_interval``, the same code emits multiple chunks
        incrementally with no further changes.
        """
        if not self._ready:
            await self.initialize()

        if self._broken:
            raise RuntimeError(
                "MLX backend previously wedged on a generation request "
                "and is now marked as broken. Restart the server "
                "(the model is in an unrecoverable state until the "
                "process is fresh). This is an upstream mlx-audio 0.3.x "
                "issue — see TTS_WARMUP_ON_START=true to absorb the "
                "cold-start graph compile at boot."
            )

        resolved_voice = self._resolve_voice(voice)
        resolved_language = self._resolve_language(language)
        start = time.monotonic()

        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=4)
        loop = asyncio.get_running_loop()

        def _run_generator() -> None:
            """Worker thread: pull chunks from the sync generator and
            push them into the asyncio queue. Sentinel ``None`` marks
            the end; exceptions are forwarded as ``("error", exc)``.
            """
            try:
                kwargs: Dict[str, Any] = {
                    "text": text,
                    "voice": resolved_voice,
                    "lang_code": resolved_language,
                    "stream": True,
                    "streaming_interval": streaming_interval,
                    "verbose": False,
                }
                if instruct:
                    kwargs["instruct"] = instruct

                # ``model.generate`` accepts a stream=True kwarg.
                # Defensive: if the loaded model variant rejects it
                # (raised either at call time or when the generator
                # is first iterated, since Python generator functions
                # defer body execution to iteration), fall back to
                # the non-streaming ``generate_custom_voice`` and
                # post the result as a single chunk. We probe with
                # ``next()`` to surface iteration-time errors, then
                # re-iterate the same iterator for the rest of the
                # stream.
                def _build_non_streaming_iter():
                    return iter(
                        self.model.generate_custom_voice(
                            text=text,
                            speaker=resolved_voice,
                            language=resolved_language,
                            instruct=instruct,
                        )
                    )

                source_iter = None
                try:
                    gen = self.model.generate(**kwargs)
                    # Probe: yields the first chunk or raises.
                    first = next(iter(gen))
                except TypeError as exc:
                    if "stream" in str(exc):
                        logger.warning(
                            "MLX model rejected stream=True; "
                            "falling back to non-streaming: %s",
                            exc,
                        )
                        source_iter = _build_non_streaming_iter()
                    else:
                        raise
                except StopIteration:
                    # Stream was empty; nothing to send.
                    return
                else:
                    # Yield the probed chunk then chain the rest of
                    # the original generator.
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("chunk", first)), loop
                    ).result()
                    source_iter = gen

                # mlx-audio 0.3.x can also hang in the streaming path
                # (cold graph compile, model never finishes). Bound the
                # wall clock so a stuck call fails fast instead of
                # wedging the server.
                for result in source_iter:
                    if (
                        time.monotonic() - start
                        > MAX_GENERATION_WALL_SECONDS
                    ):
                        raise RuntimeError(
                            f"MLX-Audio streaming generation "
                            f"exceeded "
                            f"{MAX_GENERATION_WALL_SECONDS:.0f}s "
                            f"wall-clock for a single request — "
                            f"aborting. Try restarting the server "
                            f"with TTS_WARMUP_ON_START=true to "
                            f"absorb the cold-start graph compile at "
                            f"boot."
                        )
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("chunk", result)), loop
                    ).result()
            except BaseException as exc:  # noqa: BLE001
                asyncio.run_coroutine_threadsafe(
                    queue.put(("error", exc)), loop
                ).result()
            finally:
                asyncio.run_coroutine_threadsafe(
                    queue.put(None), loop
                ).result()

        # Run the blocking generator in our dedicated thread so it
        # shares the same MLX model state as ``generate_speech``.
        future = loop.run_in_executor(self._executor, _run_generator)

        sample_rate = int(getattr(self.model, "sample_rate", 24000))

        try:
            while True:
                # Independent wall-clock watchdog: the worker thread
                # may be wedged inside mlx-audio without ever putting
                # anything on the queue, in which case ``queue.get()``
                # would block forever. ``asyncio.wait_for`` on the
                # queue with a hard deadline surfaces that to us so we
                # can mark the backend as broken and bail out.
                try:
                    item = await asyncio.wait_for(
                        queue.get(),
                        timeout=MAX_GENERATION_WALL_SECONDS,
                    )
                except asyncio.TimeoutError:
                    self._broken = True
                    logger.error(
                        "MLX streaming generation waited more than "
                        "%.0fs for the first chunk from a wedged "
                        "worker. Marking the backend as broken; "
                        "subsequent requests will fail fast until the "
                        "process is restarted.",
                        MAX_GENERATION_WALL_SECONDS,
                    )
                    raise RuntimeError(
                        f"MLX-Audio streaming generation timed out "
                        f"after {MAX_GENERATION_WALL_SECONDS:.0f}s "
                        f"without yielding any audio. The model is in "
                        f"a wedged state (upstream mlx-audio 0.3.x "
                        f"bug). The backend is now marked broken; "
                        f"restart the server."
                    ) from None

                if item is None:
                    # Generator finished cleanly
                    break
                kind, payload = item
                if kind == "error":
                    raise payload  # type: ignore[misc]

                chunk = self._result_to_numpy(payload)

                if speed != 1.0:
                    from librosa.effects import time_stretch

                    chunk = time_stretch(
                        chunk,
                        rate=speed,
                    ).astype(np.float32, copy=False)

                yield chunk, sample_rate

                # Yield to the event loop so the StreamingResponse can
                # actually flush the bytes to the client.
                await asyncio.sleep(0)
        finally:
            # Make sure the worker thread exits even if the consumer
            # bails out early.
            try:
                await asyncio.wrap_future(future)
            except Exception:
                # The exception (if any) is already queued; suppress
                # here because the consumer either saw it via the
                # ``error`` tuple or has already given up.
                pass

    # ------------------------------------------------------------------
    # Voice / language helpers
    # ------------------------------------------------------------------

    def _resolve_voice(self, voice: str) -> str:
        voices = self.get_supported_voices()
        lookup = {item.casefold(): item for item in voices}

        requested = voice.casefold()
        compatibility_voice = MLX_COMPATIBILITY_ALIASES.get(requested)
        if compatibility_voice:
            requested = compatibility_voice.casefold()

        if requested not in lookup:
            raise ValueError(
                f"Unsupported MLX voice {voice!r}. "
                f"Supported voices: {voices}"
            )

        return lookup[requested]

    def _resolve_language(self, language: str) -> str:
        """Normalize a language name to the lowercase form the model
        expects. ``Auto`` maps to ``auto``."""
        if not language:
            return "auto"
        languages = self.get_supported_languages()
        lookup = {item.casefold(): item for item in languages}
        resolved = lookup.get(language.casefold())
        if resolved is None:
            # Don't hard-fail on unknown languages; let the model pick.
            return language.casefold()
        return resolved

    @staticmethod
    def _result_to_numpy(result: Any) -> np.ndarray:
        audio = result.audio

        # MLX-Audio returns results as ``mlx.core`` arrays. We need
        # ``mx.eval`` to make sure the array is materialized before
        # copying it to NumPy. The import is best-effort: if
        # ``mlx.core`` is not on the path (e.g. unit tests using a
        # stub model) we fall back to a plain NumPy conversion.
        try:
            import mlx.core as mx  # type: ignore

            mx.eval(audio)
        except ImportError:
            pass

        return np.asarray(audio, dtype=np.float32).reshape(-1)

    # ------------------------------------------------------------------
    # TTSBackend interface
    # ------------------------------------------------------------------

    def get_backend_name(self) -> str:
        return "mlx"

    def get_model_id(self) -> str:
        return self.model_name

    def get_supported_voices(self) -> List[str]:
        return self._voices or FALLBACK_VOICES.copy()

    def get_supported_languages(self) -> List[str]:
        return self._languages or FALLBACK_LANGUAGES.copy()

    def is_ready(self) -> bool:
        return self._ready

    def get_device_info(self) -> Dict[str, Any]:
        return {
            "device": "metal",
            "gpu_available": True,
            "gpu_name": "Apple Silicon GPU via MLX",
            # Apple Silicon uses unified memory rather than dedicated
            # VRAM.
            "vram_total": None,
            "vram_used": None,
        }

    def supports_voice_cloning(self) -> bool:
        return False

    def get_model_type(self) -> str:
        return "customvoice"
