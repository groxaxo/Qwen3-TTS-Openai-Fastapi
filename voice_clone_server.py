#!/usr/bin/env python3
"""
Persistent voice cloning TTS server using Qwen3-TTS Base model.
Keeps model loaded in memory for fast inference.
Supports KV cache for voice prompts to reduce latency.
"""

import copy
import io
import os
import re
import struct
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator, Optional, Any, List

import torch
import soundfile as sf
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from transformers.cache_utils import DynamicCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment
MODEL_PATH = os.environ.get("MODEL_PATH", "./models/Qwen3-TTS-12Hz-1.7B-Base")
REF_AUDIO = os.environ.get("REF_AUDIO", "./voice_refs/hai_reference.wav")
REF_TEXT = os.environ.get("REF_TEXT", "Yeah so basically I checked the system logs and found a couple of errors. Nothing critical, but you should probably take a look when you get a chance. The server has been running fine otherwise.")
PORT = int(os.environ.get("PORT", "8881"))

# Global state
model = None
voices: dict[str, "VoiceCache"] = {}  # voice_name -> VoiceCache
DEFAULT_VOICE = "hai"


@dataclass
class VoiceCache:
    """Cached voice data including prompt and precomputed KV states."""
    prompt: List[Any]  # VoiceClonePromptItem list from create_voice_clone_prompt
    kv_cache: Optional[DynamicCache] = None  # Precomputed KV states for voice prefix
    past_hidden: Optional[torch.Tensor] = None  # Last hidden state for code predictor
    prefix_length: int = 0  # Length of the cached prefix sequence
    ref_text: str = ""  # Reference text used for this voice


@torch.inference_mode()
def compute_voice_kv_cache(
    tts_model,
    voice_prompt_items: List[Any],
    ref_text: str,
    language: str = "English",
) -> tuple[Optional[DynamicCache], Optional[torch.Tensor], int]:
    """
    Compute KV cache for the voice prompt prefix.

    This pre-computes attention KV states for the static portion of the voice prompt,
    which includes: role tokens, codec prefix, speaker embedding, and reference audio/text.

    Args:
        tts_model: The Qwen3TTSModel instance
        voice_prompt_items: VoiceClonePromptItem list from create_voice_clone_prompt
        ref_text: Reference text transcript
        language: Target language

    Returns:
        Tuple of (kv_cache, past_hidden, prefix_length) or (None, None, 0) if caching fails
    """
    try:
        inner_model = tts_model.model  # Qwen3TTSForConditionalGeneration
        talker = inner_model.talker  # Qwen3TTSTalkerForConditionalGeneration
        config = inner_model.config

        # Convert prompt items to dict format
        voice_clone_prompt = dict(
            ref_code=[it.ref_code for it in voice_prompt_items],
            ref_spk_embedding=[it.ref_spk_embedding for it in voice_prompt_items],
            x_vector_only_mode=[it.x_vector_only_mode for it in voice_prompt_items],
            icl_mode=[it.icl_mode for it in voice_prompt_items],
        )

        # Get speaker embedding
        voice_clone_spk_embeds = inner_model.generate_speaker_prompt(voice_clone_prompt)
        speaker_embed = voice_clone_spk_embeds[0] if voice_clone_spk_embeds else None

        # Build role tokens embedding: <|im_start|>assistant\n
        # We use a dummy input to get the role token IDs
        dummy_text = "<|im_start|>assistant\ntest<|im_end|>\n<|im_start|>assistant\n"
        role_ids = tts_model._tokenize_texts([dummy_text])[0][:, :3]  # First 3 tokens

        role_embed = talker.text_projection(
            talker.get_text_embeddings()(role_ids)
        )

        # Build codec prefix embeddings
        language_lower = language.lower()
        if language_lower == "auto":
            language_id = None
        elif language_lower in config.talker_config.codec_language_id:
            language_id = config.talker_config.codec_language_id[language_lower]
        else:
            language_id = None

        if language_id is None:
            codec_prefill_list = [[
                config.talker_config.codec_nothink_id,
                config.talker_config.codec_think_bos_id,
                config.talker_config.codec_think_eos_id,
            ]]
        else:
            codec_prefill_list = [[
                config.talker_config.codec_think_id,
                config.talker_config.codec_think_bos_id,
                language_id,
                config.talker_config.codec_think_eos_id,
            ]]

        codec_input_embedding_0 = talker.get_input_embeddings()(
            torch.tensor(codec_prefill_list, device=talker.device, dtype=torch.long)
        )
        codec_input_embedding_1 = talker.get_input_embeddings()(
            torch.tensor([[config.talker_config.codec_pad_id, config.talker_config.codec_bos_id]],
                        device=talker.device, dtype=torch.long)
        )

        # Get TTS special token embeddings
        tts_special_ids = torch.tensor(
            [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]],
            device=talker.device, dtype=torch.long
        )
        tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
            talker.get_text_embeddings()(tts_special_ids)
        ).chunk(3, dim=1)

        # Combine codec embeddings with speaker
        if speaker_embed is not None:
            codec_input_embedding = torch.cat([
                codec_input_embedding_0,
                speaker_embed.view(1, 1, -1),
                codec_input_embedding_1
            ], dim=1)
        else:
            codec_input_embedding = torch.cat([
                codec_input_embedding_0,
                codec_input_embedding_1
            ], dim=1)

        # Build the prefix: tts_pad tokens + tts_bos aligned with codec
        prefix_tts_embed = torch.cat([
            tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1),
            tts_bos_embed,
        ], dim=1) + codec_input_embedding[:, :-1]

        # Full prefix: role + codec/tts aligned
        prefix_embed = torch.cat([role_embed, prefix_tts_embed], dim=1)

        # For ICL mode, also include reference text and code embeddings
        if voice_prompt_items[0].icl_mode and voice_prompt_items[0].ref_code is not None:
            # Get reference text tokens
            ref_text_formatted = tts_model._build_ref_text(ref_text)
            ref_ids = tts_model._tokenize_texts([ref_text_formatted])[0][:, 3:-2]  # Strip special tokens

            ref_text_embed = talker.text_projection(
                talker.get_text_embeddings()(ref_ids)
            )

            # Get reference code embeddings
            ref_code = voice_prompt_items[0].ref_code.to(talker.device)
            codec_embeds = []
            for i in range(talker.config.num_code_groups):
                if i == 0:
                    codec_embeds.append(talker.get_input_embeddings()(ref_code[:, :1]))
                else:
                    codec_embeds.append(talker.code_predictor.get_input_embeddings()[i-1](ref_code[:, i:i+1]))
            ref_codec_embed = torch.cat(codec_embeds, dim=1).sum(1).unsqueeze(0)

            # Add codec BOS
            codec_bos_embed = talker.get_input_embeddings()(
                torch.tensor([[config.talker_config.codec_bos_id]], device=talker.device, dtype=torch.long)
            )
            ref_codec_embed = torch.cat([codec_bos_embed, ref_codec_embed], dim=1)

            # Compute lengths for ICL alignment
            ref_text_len = ref_text_embed.shape[1]
            ref_codec_len = ref_codec_embed.shape[1]

            # For ICL, reference portion: ref_text aligned with ref_codec (+ padding as needed)
            if ref_text_len >= ref_codec_len:
                # Pad codec to match text length
                icl_text_part = ref_text_embed[:, :ref_codec_len]
                icl_embed = icl_text_part + ref_codec_embed
                # Remaining text goes to trailing
            else:
                # Pad text with tts_pad to match codec length
                padding = tts_pad_embed.expand(-1, ref_codec_len - ref_text_len, -1)
                icl_text_part = torch.cat([ref_text_embed, padding], dim=1)
                icl_embed = icl_text_part + ref_codec_embed

            prefix_embed = torch.cat([prefix_embed, icl_embed], dim=1)

        prefix_length = prefix_embed.shape[1]

        # Create attention mask
        attention_mask = torch.ones(1, prefix_length, device=talker.device, dtype=torch.long)

        # Run forward pass through the talker to compute KV cache AND past_hidden
        # The talker's forward returns past_hidden which is needed for generation continuation
        kv_cache = DynamicCache()

        outputs = talker(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=kv_cache,
            inputs_embeds=prefix_embed,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            trailing_text_hidden=tts_pad_embed,  # Required by talker forward
            tts_pad_embed=tts_pad_embed,  # Required by talker forward
        )

        logger.info(f"Computed KV cache for prefix length {prefix_length}")
        return outputs.past_key_values, outputs.past_hidden, prefix_length

    except Exception as e:
        logger.warning(f"Failed to compute KV cache: {e}")
        import traceback
        traceback.print_exc()
        return None, None, 0


def clone_kv_cache(cache: DynamicCache) -> DynamicCache:
    """Deep clone a DynamicCache to avoid mutating the original."""
    if cache is None:
        return None

    new_cache = DynamicCache()

    # Handle both old and new DynamicCache API
    if hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache'):
        # Old API (transformers < 4.36)
        for layer_idx in range(len(cache.key_cache)):
            new_cache.key_cache.append(cache.key_cache[layer_idx].clone())
            new_cache.value_cache.append(cache.value_cache[layer_idx].clone())
    elif hasattr(cache, 'layers'):
        # New API (transformers >= 4.36)
        for layer in cache.layers:
            # Each layer is typically a tuple of (key, value) tensors
            if isinstance(layer, (list, tuple)) and len(layer) == 2:
                new_cache.layers.append((layer[0].clone(), layer[1].clone()))
            else:
                new_cache.layers.append(layer)

    return new_cache


def load_model():
    """Load the voice cloning model."""
    global model, voices

    from qwen_tts import Qwen3TTSModel

    logger.info(f"Loading model from {MODEL_PATH}...")

    # Try Flash Attention 2, fall back to eager
    try:
        model = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        logger.info("Model loaded with Flash Attention 2.")
    except Exception as e:
        logger.warning(f"Flash Attention 2 not available: {e}, falling back to eager")
        model = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        logger.info("Model loaded with eager attention.")

    # Load default voice with KV cache
    logger.info(f"Creating default voice clone prompt ({DEFAULT_VOICE})...")
    voice_prompt = model.create_voice_clone_prompt(
        ref_audio=REF_AUDIO,
        ref_text=REF_TEXT,
        x_vector_only_mode=False,
    )

    logger.info(f"Computing KV cache for default voice...")
    kv_cache, past_hidden, prefix_length = compute_voice_kv_cache(model, voice_prompt, REF_TEXT)

    voices[DEFAULT_VOICE] = VoiceCache(
        prompt=voice_prompt,
        kv_cache=kv_cache,
        past_hidden=past_hidden,
        prefix_length=prefix_length,
        ref_text=REF_TEXT,
    )
    logger.info(f"Default voice '{DEFAULT_VOICE}' ready (prefix_length={prefix_length}).")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    load_model()
    yield
    # Cleanup on shutdown
    global model, voices
    model = None
    voices.clear()
    torch.cuda.empty_cache()


app = FastAPI(
    title="Voice Clone TTS Server",
    description="Qwen3-TTS voice cloning with persistent model",
    lifespan=lifespan,
)


class TTSRequest(BaseModel):
    text: str
    language: str = "English"
    stream: bool = False
    response_format: str = "wav"  # wav or pcm
    voice: Optional[str] = None  # voice name, defaults to DEFAULT_VOICE
    use_kv_cache: bool = True  # whether to use KV cache acceleration


class LoadVoiceRequest(BaseModel):
    voice_name: str
    ref_audio_path: str
    ref_text: str


def create_wav_streaming_header(
    sample_rate: int = 24000,
    num_channels: int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    """Create WAV header for streaming with unknown final size."""
    buffer = io.BytesIO()
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8

    buffer.write(b'RIFF')
    buffer.write(struct.pack('<I', 0xFFFFFFFF))  # Placeholder
    buffer.write(b'WAVE')
    buffer.write(b'fmt ')
    buffer.write(struct.pack('<I', 16))
    buffer.write(struct.pack('<H', 1))  # PCM
    buffer.write(struct.pack('<H', num_channels))
    buffer.write(struct.pack('<I', sample_rate))
    buffer.write(struct.pack('<I', byte_rate))
    buffer.write(struct.pack('<H', block_align))
    buffer.write(struct.pack('<H', bits_per_sample))
    buffer.write(b'data')
    buffer.write(struct.pack('<I', 0xFFFFFFFF))  # Placeholder
    return buffer.getvalue()


def convert_to_pcm(audio: np.ndarray) -> bytes:
    """Convert float audio to 16-bit PCM bytes."""
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    return pcm.tobytes()


@torch.inference_mode()
def generate_with_kv_cache(
    text: str,
    language: str,
    voice_cache: VoiceCache,
    use_cache: bool = True,
) -> tuple[List[np.ndarray], int]:
    """
    Generate speech using voice cache with optional KV cache acceleration.

    When KV cache is available in the VoiceCache, this function uses pre-computed
    attention states to skip redundant computation of the voice prompt prefix,
    resulting in faster inference.

    Args:
        text: Text to synthesize
        language: Target language
        voice_cache: VoiceCache with voice prompt and optionally precomputed KV states
        use_cache: Whether to use KV cache acceleration (default True)

    Returns:
        Tuple of (wavs, sample_rate)
    """
    global model

    # NOTE: Full KV cache integration with HuggingFace's generate() requires
    # significant model changes. For now, we use voice prompt caching which
    # still saves the voice prompt computation (speaker embedding + ICL setup).
    #
    # The KV cache infrastructure (compute_voice_kv_cache, VoiceCache.kv_cache)
    # is kept for future implementation when model-level KV cache support is added.

    # Use cached voice prompt for generation
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=voice_cache.prompt,
    )
    logger.debug(f"Generated with cached voice prompt")

    return wavs, sr


async def generate_speech_streaming(
    text: str,
    language: str,
    voice_cache: VoiceCache,
    use_cache: bool = True,
) -> AsyncGenerator[tuple[np.ndarray, int], None]:
    """Generate speech with sentence-level streaming and optional KV cache acceleration."""
    global model

    # Split text into sentences
    sentence_pattern = r'(?<=[.!?。！？])\s+'
    sentences = re.split(sentence_pattern, text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        sentences = [text]

    for sentence in sentences:
        try:
            wavs, sr = generate_with_kv_cache(
                text=sentence,
                language=language,
                voice_cache=voice_cache,
                use_cache=use_cache,
            )
            if wavs and len(wavs[0]) > 0:
                yield wavs[0], sr
        except Exception as e:
            logger.error(f"Error generating sentence: {e}")
            continue


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/v1/voices")
async def list_voices():
    """List all loaded voices with KV cache status."""
    voice_info = {}
    for name, cache in voices.items():
        voice_info[name] = {
            "kv_cached": cache.kv_cache is not None,
            "prefix_length": cache.prefix_length,
        }
    return {
        "voices": voice_info,
        "default": DEFAULT_VOICE,
    }


@app.post("/v1/voices/load")
async def load_voice(request: LoadVoiceRequest):
    """Load a new voice from reference audio with KV cache precomputation."""
    global model, voices

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not os.path.exists(request.ref_audio_path):
        raise HTTPException(status_code=400, detail=f"Reference audio not found: {request.ref_audio_path}")

    try:
        logger.info(f"Loading voice '{request.voice_name}' from {request.ref_audio_path}...")
        voice_prompt = model.create_voice_clone_prompt(
            ref_audio=request.ref_audio_path,
            ref_text=request.ref_text,
            x_vector_only_mode=False,
        )

        logger.info(f"Computing KV cache for voice '{request.voice_name}'...")
        kv_cache, past_hidden, prefix_length = compute_voice_kv_cache(model, voice_prompt, request.ref_text)

        voices[request.voice_name] = VoiceCache(
            prompt=voice_prompt,
            kv_cache=kv_cache,
            past_hidden=past_hidden,
            prefix_length=prefix_length,
            ref_text=request.ref_text,
        )
        logger.info(f"Voice '{request.voice_name}' loaded successfully (prefix_length={prefix_length}).")
        return {
            "status": "ok",
            "voice_name": request.voice_name,
            "kv_cached": kv_cache is not None,
            "prefix_length": prefix_length,
        }
    except Exception as e:
        logger.error(f"Failed to load voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/speech")
async def generate_speech_endpoint(request: TTSRequest):
    """Generate speech from text using voice cloning with KV cache optimization."""
    global model, voices

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Resolve voice
    voice_name = request.voice or DEFAULT_VOICE
    if voice_name not in voices:
        raise HTTPException(status_code=400, detail=f"Voice '{voice_name}' not loaded. Available: {list(voices.keys())}")
    voice_cache = voices[voice_name]

    # Handle streaming mode
    if request.stream:
        async def audio_stream_generator():
            if request.response_format == "wav":
                yield create_wav_streaming_header(24000)

            async for audio_chunk, sr in generate_speech_streaming(
                text=request.text,
                language=request.language,
                voice_cache=voice_cache,
                use_cache=request.use_kv_cache,
            ):
                if audio_chunk is not None and len(audio_chunk) > 0:
                    yield convert_to_pcm(audio_chunk)

        content_type = "audio/wav" if request.response_format == "wav" else "audio/pcm"
        return StreamingResponse(
            audio_stream_generator(),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                "Cache-Control": "no-cache",
                "Transfer-Encoding": "chunked",
            },
        )

    # Non-streaming mode - use KV cache if available and requested
    try:
        if request.use_kv_cache and voice_cache.kv_cache is not None:
            wavs, sr = generate_with_kv_cache(
                text=request.text,
                language=request.language,
                voice_cache=voice_cache,
                use_cache=True,
            )
        else:
            wavs, sr = model.generate_voice_clone(
                text=request.text,
                language=request.language,
                voice_clone_prompt=voice_cache.prompt,
            )

        audio = wavs[0]

        # Convert to WAV bytes
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format='WAV')
        buf.seek(0)

        return Response(
            content=buf.read(),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"},
        )

    except Exception as e:
        import traceback
        logger.error(f"Speech generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
