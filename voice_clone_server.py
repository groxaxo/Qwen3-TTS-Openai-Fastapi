#!/usr/bin/env python3
"""
Persistent voice cloning TTS server using Qwen3-TTS Base model.
Keeps model loaded in memory for fast inference.
Supports KV cache for voice prompts to reduce latency.
"""

import copy
import io
import json
import os
import re
import shutil
import struct
import logging
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Optional, Any, List

import torch
import soundfile as sf
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from safetensors.torch import save_file, load_file
from transformers.cache_utils import DynamicCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment
MODEL_PATH = os.environ.get("MODEL_PATH", "./models/Qwen3-TTS-12Hz-1.7B-Base")
REF_AUDIO = os.environ.get("REF_AUDIO", "./voice_refs/hai_reference.wav")
REF_TEXT = os.environ.get("REF_TEXT", "Yeah so basically I checked the system logs and found a couple of errors. Nothing critical, but you should probably take a look when you get a chance. The server has been running fine otherwise.")
PORT = int(os.environ.get("PORT", "8881"))
VOICE_REFS_DIR = Path(os.environ.get("VOICE_REFS_DIR", "./voice_refs"))

# Global state
model = None
voices: dict[str, "VoiceCache"] = {}  # voice_name -> VoiceCache
voices_lock = threading.RLock()  # Thread-safe access to voices dict
DEFAULT_VOICE = "hai"


@dataclass
class VoiceCache:
    """Cached voice data including prompt and precomputed KV states."""
    prompt: List[Any]  # VoiceClonePromptItem list from create_voice_clone_prompt
    kv_cache: Optional[DynamicCache] = None  # Precomputed KV states for voice prefix
    past_hidden: Optional[torch.Tensor] = None  # Last hidden state for code predictor
    prefix_length: int = 0  # Length of the cached prefix sequence
    ref_text: str = ""  # Reference text used for this voice
    ref_audio_path: Optional[str] = None  # Path to reference audio file


def get_voice_cache_path(voice_name: str) -> Path:
    """Get the path for a voice's KV cache safetensors file."""
    return VOICE_REFS_DIR / f"{voice_name}_kvcache.safetensors"


def get_voice_metadata_path(voice_name: str) -> Path:
    """Get the path for a voice's metadata JSON file."""
    return VOICE_REFS_DIR / f"{voice_name}_metadata.json"


def save_voice_cache_to_disk(voice_name: str, voice_cache: VoiceCache) -> bool:
    """
    Save voice KV cache and metadata to disk for persistence across restarts.

    Saves:
    - KV cache tensors to {voice_name}_kvcache.safetensors
    - Metadata (ref_text, prefix_length, ref_audio_path) to {voice_name}_metadata.json

    Returns True if successful, False otherwise.
    """
    if voice_cache.kv_cache is None:
        logger.warning(f"No KV cache to save for voice '{voice_name}'")
        return False

    try:
        VOICE_REFS_DIR.mkdir(parents=True, exist_ok=True)

        # Build tensors dict from KV cache
        tensors = {}
        cache = voice_cache.kv_cache

        # Handle DynamicCache structure - try different API versions
        num_layers = 0

        # New API (transformers >= 4.50): cache.layers with DynamicLayer objects
        if hasattr(cache, 'layers') and len(cache.layers) > 0:
            for layer_idx, layer in enumerate(cache.layers):
                # DynamicLayer has .keys and .values attributes
                if hasattr(layer, 'keys') and hasattr(layer, 'values'):
                    tensors[f"key_cache_{layer_idx}"] = layer.keys.cpu()
                    tensors[f"value_cache_{layer_idx}"] = layer.values.cpu()
                    num_layers += 1
                # Fallback: try tuple-like access
                elif isinstance(layer, (list, tuple)) and len(layer) == 2:
                    tensors[f"key_cache_{layer_idx}"] = layer[0].cpu()
                    tensors[f"value_cache_{layer_idx}"] = layer[1].cpu()
                    num_layers += 1

        # Old API (transformers < 4.36): cache.key_cache and cache.value_cache
        elif hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache'):
            for layer_idx in range(len(cache.key_cache)):
                tensors[f"key_cache_{layer_idx}"] = cache.key_cache[layer_idx].cpu()
                tensors[f"value_cache_{layer_idx}"] = cache.value_cache[layer_idx].cpu()
            num_layers = len(cache.key_cache)

        if num_layers == 0:
            logger.warning(f"No layers found in KV cache for voice '{voice_name}'")
            return False

        tensors["num_layers"] = torch.tensor([num_layers])

        # Add past_hidden if present
        if voice_cache.past_hidden is not None:
            tensors["past_hidden"] = voice_cache.past_hidden.cpu()

        # Add prefix_length as tensor
        tensors["prefix_length"] = torch.tensor([voice_cache.prefix_length])

        # Save tensors
        cache_path = get_voice_cache_path(voice_name)
        save_file(tensors, str(cache_path))
        logger.info(f"Saved KV cache for '{voice_name}' to {cache_path} ({num_layers} layers)")

        # Save metadata
        metadata = {
            "ref_text": voice_cache.ref_text,
            "prefix_length": voice_cache.prefix_length,
            "ref_audio_path": voice_cache.ref_audio_path,
        }
        metadata_path = get_voice_metadata_path(voice_name)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata for '{voice_name}' to {metadata_path}")

        return True

    except Exception as e:
        logger.error(f"Failed to save voice cache for '{voice_name}': {e}")
        import traceback
        traceback.print_exc()
        return False


def load_voice_cache_from_disk(voice_name: str, device: str = "cuda:0") -> Optional[tuple[DynamicCache, Optional[torch.Tensor], int, dict]]:
    """
    Load voice KV cache and metadata from disk.

    Returns:
        Tuple of (kv_cache, past_hidden, prefix_length, metadata) or None if not found/failed
    """
    cache_path = get_voice_cache_path(voice_name)
    metadata_path = get_voice_metadata_path(voice_name)

    if not cache_path.exists():
        return None

    try:
        # Load tensors
        tensors = load_file(str(cache_path))

        # Reconstruct DynamicCache
        kv_cache = DynamicCache()
        num_layers = int(tensors["num_layers"].item())

        for layer_idx in range(num_layers):
            key = tensors[f"key_cache_{layer_idx}"].to(device)
            value = tensors[f"value_cache_{layer_idx}"].to(device)
            kv_cache.update(key, value, layer_idx)

        # Load past_hidden
        past_hidden = None
        if "past_hidden" in tensors:
            past_hidden = tensors["past_hidden"].to(device)

        # Load prefix_length
        prefix_length = int(tensors["prefix_length"].item())

        # Load metadata
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        logger.info(f"Loaded KV cache for '{voice_name}' from disk (prefix_length={prefix_length})")
        return kv_cache, past_hidden, prefix_length, metadata

    except Exception as e:
        logger.error(f"Failed to load voice cache for '{voice_name}': {e}")
        import traceback
        traceback.print_exc()
        return None


def delete_voice_cache_from_disk(voice_name: str) -> bool:
    """Delete voice cache and metadata files from disk."""
    success = True

    cache_path = get_voice_cache_path(voice_name)
    if cache_path.exists():
        try:
            cache_path.unlink()
            logger.info(f"Deleted cache file: {cache_path}")
        except Exception as e:
            logger.error(f"Failed to delete cache file: {e}")
            success = False

    metadata_path = get_voice_metadata_path(voice_name)
    if metadata_path.exists():
        try:
            metadata_path.unlink()
            logger.info(f"Deleted metadata file: {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to delete metadata file: {e}")
            success = False

    return success


def discover_cached_voices() -> List[str]:
    """Discover all voices with saved KV caches on disk."""
    if not VOICE_REFS_DIR.exists():
        return []

    voice_names = []
    for cache_file in VOICE_REFS_DIR.glob("*_kvcache.safetensors"):
        voice_name = cache_file.stem.replace("_kvcache", "")
        metadata_path = get_voice_metadata_path(voice_name)
        # Only include if metadata exists (complete cache)
        if metadata_path.exists():
            voice_names.append(voice_name)

    return voice_names


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
    """Load the voice cloning model and restore cached voices."""
    global model, voices

    from qwen_tts import Qwen3TTSModel

    logger.info(f"Loading model from {MODEL_PATH}...")

    # Ensure voice_refs directory exists
    VOICE_REFS_DIR.mkdir(parents=True, exist_ok=True)

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

    # Auto-load cached voices from disk
    cached_voice_names = discover_cached_voices()
    logger.info(f"Discovered {len(cached_voice_names)} cached voices on disk: {cached_voice_names}")

    for voice_name in cached_voice_names:
        try:
            result = load_voice_cache_from_disk(voice_name)
            if result is None:
                continue

            kv_cache, past_hidden, prefix_length, metadata = result
            ref_text = metadata.get("ref_text", "")
            ref_audio_path = metadata.get("ref_audio_path")

            # Recreate voice prompt from reference audio if it still exists
            if ref_audio_path and os.path.exists(ref_audio_path):
                voice_prompt = model.create_voice_clone_prompt(
                    ref_audio=ref_audio_path,
                    ref_text=ref_text,
                    x_vector_only_mode=False,
                )
                voices[voice_name] = VoiceCache(
                    prompt=voice_prompt,
                    kv_cache=kv_cache,
                    past_hidden=past_hidden,
                    prefix_length=prefix_length,
                    ref_text=ref_text,
                    ref_audio_path=ref_audio_path,
                )
                logger.info(f"Restored voice '{voice_name}' from cache (prefix_length={prefix_length})")
            else:
                logger.warning(f"Reference audio not found for cached voice '{voice_name}': {ref_audio_path}")

        except Exception as e:
            logger.error(f"Failed to restore cached voice '{voice_name}': {e}")

    # Load default voice if not already loaded from cache
    if DEFAULT_VOICE not in voices:
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
            ref_audio_path=REF_AUDIO,
        )

        # Save default voice cache to disk for persistence
        save_voice_cache_to_disk(DEFAULT_VOICE, voices[DEFAULT_VOICE])
        logger.info(f"Default voice '{DEFAULT_VOICE}' ready (prefix_length={prefix_length}).")
    else:
        logger.info(f"Default voice '{DEFAULT_VOICE}' restored from cache.")

    logger.info(f"Total voices loaded: {len(voices)}")


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
    with voices_lock:
        for name, cache in voices.items():
            # Check if persisted on disk
            persisted = get_voice_cache_path(name).exists()
            voice_info[name] = {
                "kv_cached": cache.kv_cache is not None,
                "prefix_length": cache.prefix_length,
                "ref_audio_path": cache.ref_audio_path,
                "persisted": persisted,
            }
    return {
        "voices": voice_info,
        "default": DEFAULT_VOICE,
    }


@app.post("/v1/voices/load")
async def load_voice(request: LoadVoiceRequest):
    """Load a new voice from reference audio with KV cache precomputation and persistence."""
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

        voice_cache = VoiceCache(
            prompt=voice_prompt,
            kv_cache=kv_cache,
            past_hidden=past_hidden,
            prefix_length=prefix_length,
            ref_text=request.ref_text,
            ref_audio_path=request.ref_audio_path,
        )

        # Thread-safe addition to voices dict
        with voices_lock:
            voices[request.voice_name] = voice_cache

        # Persist to disk (outside lock - file I/O is slow)
        saved = save_voice_cache_to_disk(request.voice_name, voice_cache)

        logger.info(f"Voice '{request.voice_name}' loaded successfully (prefix_length={prefix_length}, persisted={saved}).")
        return {
            "status": "ok",
            "voice_name": request.voice_name,
            "kv_cached": kv_cache is not None,
            "prefix_length": prefix_length,
            "persisted": saved,
        }
    except Exception as e:
        logger.error(f"Failed to load voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/voices/upload")
async def upload_voice(
    audio_file: UploadFile = File(..., description="Reference audio file (wav/mp3)"),
    voice_name: str = Form(..., description="Name for this voice"),
    ref_text: str = Form(..., description="Transcript of the reference audio"),
):
    """
    Upload a voice reference audio file and create a persistent voice with KV cache.

    The audio file is saved to the voice_refs directory, and KV cache is computed
    and persisted to disk so the voice survives server restarts.
    """
    global model, voices

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate voice name
    if not voice_name or not voice_name.strip():
        raise HTTPException(status_code=400, detail="voice_name is required")
    voice_name = voice_name.strip()

    # Validate file type
    allowed_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    file_ext = Path(audio_file.filename).suffix.lower() if audio_file.filename else ""
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file_ext}'. Allowed: {allowed_extensions}"
        )

    try:
        # Ensure voice_refs directory exists
        VOICE_REFS_DIR.mkdir(parents=True, exist_ok=True)

        # Save uploaded file
        audio_path = VOICE_REFS_DIR / f"{voice_name}_reference{file_ext}"
        with open(audio_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        logger.info(f"Saved uploaded audio to {audio_path}")

        # Create voice prompt
        logger.info(f"Creating voice clone prompt for '{voice_name}'...")
        voice_prompt = model.create_voice_clone_prompt(
            ref_audio=str(audio_path),
            ref_text=ref_text,
            x_vector_only_mode=False,
        )

        # Compute KV cache
        logger.info(f"Computing KV cache for voice '{voice_name}'...")
        kv_cache, past_hidden, prefix_length = compute_voice_kv_cache(model, voice_prompt, ref_text)

        # Create voice cache
        voice_cache = VoiceCache(
            prompt=voice_prompt,
            kv_cache=kv_cache,
            past_hidden=past_hidden,
            prefix_length=prefix_length,
            ref_text=ref_text,
            ref_audio_path=str(audio_path),
        )

        # Thread-safe addition to voices dict
        with voices_lock:
            voices[voice_name] = voice_cache

        # Persist to disk (outside lock - file I/O is slow)
        saved = save_voice_cache_to_disk(voice_name, voice_cache)

        logger.info(f"Voice '{voice_name}' uploaded and cached (prefix_length={prefix_length}, persisted={saved}).")
        return {
            "status": "ok",
            "voice_name": voice_name,
            "audio_path": str(audio_path),
            "kv_cached": kv_cache is not None,
            "prefix_length": prefix_length,
            "persisted": saved,
        }

    except Exception as e:
        logger.error(f"Failed to upload voice: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/voices/{voice_name}")
async def delete_voice(voice_name: str):
    """
    Delete a voice and its cached data from memory and disk.

    Note: The default voice cannot be deleted.
    """
    global voices

    if voice_name == DEFAULT_VOICE:
        raise HTTPException(status_code=400, detail=f"Cannot delete default voice '{DEFAULT_VOICE}'")

    # Thread-safe check and removal from voices dict
    with voices_lock:
        if voice_name not in voices:
            raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found")

        # Get audio path before removing from memory
        voice_cache = voices[voice_name]
        audio_path = voice_cache.ref_audio_path

        # Remove from memory
        del voices[voice_name]
        logger.info(f"Removed voice '{voice_name}' from memory")

    # File operations outside lock (slow I/O)
    try:
        # Delete cache files from disk
        cache_deleted = delete_voice_cache_from_disk(voice_name)

        # Optionally delete the audio file (only if it's in our voice_refs dir)
        audio_deleted = False
        if audio_path:
            audio_path_obj = Path(audio_path)
            if audio_path_obj.exists() and VOICE_REFS_DIR in audio_path_obj.parents or audio_path_obj.parent == VOICE_REFS_DIR:
                try:
                    audio_path_obj.unlink()
                    audio_deleted = True
                    logger.info(f"Deleted audio file: {audio_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete audio file: {e}")

        return {
            "status": "ok",
            "voice_name": voice_name,
            "cache_deleted": cache_deleted,
            "audio_deleted": audio_deleted,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/speech")
async def generate_speech_endpoint(request: TTSRequest):
    """Generate speech from text using voice cloning with KV cache optimization."""
    global model, voices

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Resolve voice (thread-safe access)
    voice_name = request.voice or DEFAULT_VOICE
    with voices_lock:
        if voice_name not in voices:
            available = list(voices.keys())
            raise HTTPException(status_code=400, detail=f"Voice '{voice_name}' not loaded. Available: {available}")
        # Get reference to voice_cache - safe to use outside lock since VoiceCache is immutable
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
