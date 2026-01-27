#!/usr/bin/env python3
"""
Persistent voice cloning TTS server using Qwen3-TTS Base model.
Keeps model loaded in memory for fast inference.
Supports KV cache for voice prompts to reduce latency.
Supports multiple models (1.7B and 0.6B) with hot-swapping.
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
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Optional, Any, List, Literal

import torch
import soundfile as sf
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from safetensors.torch import save_file, load_file
from transformers.cache_utils import DynamicCache
from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment
MODEL_PATH_LARGE = os.environ.get("MODEL_PATH", os.environ.get("MODEL_PATH_LARGE", "./models/Qwen3-TTS-12Hz-1.7B-Base"))
MODEL_PATH_SMALL = os.environ.get("MODEL_PATH_SMALL", "./models/Qwen3-TTS-12Hz-0.6B-Base")
REF_AUDIO = os.environ.get("REF_AUDIO", "./voice_refs/hai_reference.wav")
REF_TEXT = os.environ.get("REF_TEXT", "Yeah so basically I checked the system logs and found a couple of errors. Nothing critical, but you should probably take a look when you get a chance. The server has been running fine otherwise.")
PORT = int(os.environ.get("PORT", "8881"))
VOICE_REFS_DIR = Path(os.environ.get("VOICE_REFS_DIR", "./voice_refs"))

# Model configuration
# TEMPORARY: Large model disabled to rule out model swapping issues
MODEL_CONFIGS = {
    # "large": {"path": MODEL_PATH_LARGE, "name": "Qwen3-TTS-12Hz-1.7B-Base"},  # DISABLED
    "small": {"path": MODEL_PATH_SMALL, "name": "Qwen3-TTS-12Hz-0.6B-Base"},
}
# Force small model as default since large is disabled
DEFAULT_MODEL = "small"  # os.environ.get("DEFAULT_MODEL", "large")
MAX_LOADED_MODELS = int(os.environ.get("MAX_LOADED_MODELS", "1"))

# Global state - lazy loading with LRU eviction
models: dict[str, Any] = {}  # model_name -> Qwen3TTSModel
model_last_used: dict[str, float] = {}  # model_name -> timestamp (for LRU)
voices: dict[str, dict[str, "VoiceCache"]] = {}  # model_name -> voice_name -> VoiceCache
models_lock = threading.RLock()  # Thread-safe access to models dict
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


def get_voice_cache_path(voice_name: str, model_name: str = "large") -> Path:
    """Get the path for a voice's KV cache safetensors file."""
    # Include model name in path to keep caches separate per model
    suffix = "" if model_name == "large" else f"_{model_name}"
    return VOICE_REFS_DIR / f"{voice_name}{suffix}_kvcache.safetensors"


def get_voice_metadata_path(voice_name: str, model_name: str = "large") -> Path:
    """Get the path for a voice's metadata JSON file."""
    suffix = "" if model_name == "large" else f"_{model_name}"
    return VOICE_REFS_DIR / f"{voice_name}{suffix}_metadata.json"


def save_voice_cache_to_disk(voice_name: str, voice_cache: VoiceCache, model_name: str = "large") -> bool:
    """
    Save voice KV cache and metadata to disk for persistence across restarts.

    Saves:
    - KV cache tensors to {voice_name}_kvcache.safetensors
    - Metadata (ref_text, prefix_length, ref_audio_path) to {voice_name}_metadata.json

    Returns True if successful, False otherwise.
    """
    if voice_cache.kv_cache is None:
        logger.warning(f"No KV cache to save for voice '{voice_name}' (model={model_name})")
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
            logger.warning(f"No layers found in KV cache for voice '{voice_name}' (model={model_name})")
            return False

        tensors["num_layers"] = torch.tensor([num_layers])

        # Add past_hidden if present
        if voice_cache.past_hidden is not None:
            tensors["past_hidden"] = voice_cache.past_hidden.cpu()

        # Add prefix_length as tensor
        tensors["prefix_length"] = torch.tensor([voice_cache.prefix_length])

        # Save tensors
        cache_path = get_voice_cache_path(voice_name, model_name)
        save_file(tensors, str(cache_path))
        logger.info(f"Saved KV cache for '{voice_name}' (model={model_name}) to {cache_path} ({num_layers} layers)")

        # Save metadata
        metadata = {
            "ref_text": voice_cache.ref_text,
            "prefix_length": voice_cache.prefix_length,
            "ref_audio_path": voice_cache.ref_audio_path,
        }
        metadata_path = get_voice_metadata_path(voice_name, model_name)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata for '{voice_name}' (model={model_name}) to {metadata_path}")

        return True

    except Exception as e:
        logger.error(f"Failed to save voice cache for '{voice_name}' (model={model_name}): {e}")
        import traceback
        traceback.print_exc()
        return False


def load_voice_cache_from_disk(voice_name: str, model_name: str = "large", device: str = "cuda:0") -> Optional[tuple[DynamicCache, Optional[torch.Tensor], int, dict]]:
    """
    Load voice KV cache and metadata from disk.

    Returns:
        Tuple of (kv_cache, past_hidden, prefix_length, metadata) or None if not found/failed
    """
    cache_path = get_voice_cache_path(voice_name, model_name)
    metadata_path = get_voice_metadata_path(voice_name, model_name)

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

        logger.info(f"Loaded KV cache for '{voice_name}' (model={model_name}) from disk (prefix_length={prefix_length})")
        return kv_cache, past_hidden, prefix_length, metadata

    except Exception as e:
        logger.error(f"Failed to load voice cache for '{voice_name}' (model={model_name}): {e}")
        import traceback
        traceback.print_exc()
        return None


def delete_voice_cache_from_disk(voice_name: str, model_name: str = "large") -> bool:
    """Delete voice cache and metadata files from disk."""
    success = True

    cache_path = get_voice_cache_path(voice_name, model_name)
    if cache_path.exists():
        try:
            cache_path.unlink()
            logger.info(f"Deleted cache file: {cache_path}")
        except Exception as e:
            logger.error(f"Failed to delete cache file: {e}")
            success = False

    metadata_path = get_voice_metadata_path(voice_name, model_name)
    if metadata_path.exists():
        try:
            metadata_path.unlink()
            logger.info(f"Deleted metadata file: {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to delete metadata file: {e}")
            success = False

    return success


def discover_cached_voices(model_name: str = "large") -> List[str]:
    """Discover all voices with saved KV caches on disk for a specific model."""
    if not VOICE_REFS_DIR.exists():
        return []

    voice_names = []
    suffix = "" if model_name == "large" else f"_{model_name}"
    pattern = f"*{suffix}_kvcache.safetensors" if suffix else "*_kvcache.safetensors"

    for cache_file in VOICE_REFS_DIR.glob(pattern):
        stem = cache_file.stem.replace("_kvcache", "")
        # Remove model suffix if present
        if suffix and stem.endswith(suffix.replace("_kvcache", "")):
            voice_name = stem[:-len(suffix.replace("_kvcache", ""))]
        else:
            voice_name = stem
            # Skip if this is a different model's cache
            if model_name != "large" and "_" not in stem:
                continue

        metadata_path = get_voice_metadata_path(voice_name, model_name)
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

    This pre-computes attention KV states for the voice-specific portion of the prompt,
    which includes: role tokens, codec prefix (language), and speaker embedding.

    The ICL portion (reference text/code alignment) is NOT cached because it needs
    to be computed together with the text-to-generate for proper alignment.

    Args:
        tts_model: The Qwen3TTSModel instance
        voice_prompt_items: VoiceClonePromptItem list from create_voice_clone_prompt
        ref_text: Reference text transcript (used for metadata, not cached)
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

        # Build codec prefix embeddings based on language
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

        # Full prefix: role + codec/tts aligned (WITHOUT ICL - that's computed per-request)
        prefix_embed = torch.cat([role_embed, prefix_tts_embed], dim=1)

        prefix_length = prefix_embed.shape[1]

        # Create attention mask
        attention_mask = torch.ones(1, prefix_length, device=talker.device, dtype=torch.long)

        # Run forward pass through the talker to compute KV cache
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

        logger.info(f"Computed KV cache for voice prefix (length={prefix_length})")
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


def clone_voice_prompt(prompt: List[Any]) -> List[Any]:
    """
    Deep clone voice prompt items to prevent tensor mutation during generation.

    This ensures that cached voice prompts are not corrupted by the model's
    internal operations, which could cause quality degradation (stuttering,
    hallucinations) on subsequent generations.
    """
    cloned = []
    for item in prompt:
        cloned.append(VoiceClonePromptItem(
            ref_code=item.ref_code.clone() if item.ref_code is not None else None,
            ref_spk_embedding=item.ref_spk_embedding.clone() if item.ref_spk_embedding is not None else None,
            x_vector_only_mode=item.x_vector_only_mode,
            icl_mode=item.icl_mode,
            ref_text=item.ref_text,
        ))
    return cloned


def load_single_model(model_name: str, model_path: str) -> Optional[Any]:
    """Load a single model with Flash Attention 2 if available."""
    from qwen_tts import Qwen3TTSModel

    if not os.path.exists(model_path):
        logger.warning(f"Model path not found for '{model_name}': {model_path}")
        return None

    logger.info(f"Loading model '{model_name}' from {model_path}...")

    try:
        model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        logger.info(f"Model '{model_name}' loaded with Flash Attention 2.")
        return model
    except Exception as e:
        logger.warning(f"Flash Attention 2 not available for '{model_name}': {e}, falling back to eager")
        try:
            model = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map="cuda:0",
                dtype=torch.bfloat16,
                attn_implementation="eager",
            )
            logger.info(f"Model '{model_name}' loaded with eager attention.")
            return model
        except Exception as e2:
            logger.error(f"Failed to load model '{model_name}': {e2}")
            return None


def load_voices_for_model(model_name: str, tts_model: Any) -> dict[str, VoiceCache]:
    """Load voices for a specific model."""
    model_voices: dict[str, VoiceCache] = {}

    # Auto-load cached voices from disk
    cached_voice_names = discover_cached_voices(model_name)
    logger.info(f"Discovered {len(cached_voice_names)} cached voices for model '{model_name}': {cached_voice_names}")

    for voice_name in cached_voice_names:
        try:
            result = load_voice_cache_from_disk(voice_name, model_name)
            if result is None:
                continue

            kv_cache, past_hidden, prefix_length, metadata = result
            ref_text = metadata.get("ref_text", "")
            ref_audio_path = metadata.get("ref_audio_path")

            # Recreate voice prompt from reference audio if it still exists
            if ref_audio_path and os.path.exists(ref_audio_path):
                voice_prompt = tts_model.create_voice_clone_prompt(
                    ref_audio=ref_audio_path,
                    ref_text=ref_text,
                    x_vector_only_mode=False,
                )
                model_voices[voice_name] = VoiceCache(
                    prompt=voice_prompt,
                    kv_cache=kv_cache,
                    past_hidden=past_hidden,
                    prefix_length=prefix_length,
                    ref_text=ref_text,
                    ref_audio_path=ref_audio_path,
                )
                logger.info(f"Restored voice '{voice_name}' for model '{model_name}' (prefix_length={prefix_length})")
            else:
                logger.warning(f"Reference audio not found for voice '{voice_name}': {ref_audio_path}")

        except Exception as e:
            logger.error(f"Failed to restore voice '{voice_name}' for model '{model_name}': {e}")

    # Load default voice if not already loaded from cache
    if DEFAULT_VOICE not in model_voices:
        if os.path.exists(REF_AUDIO):
            logger.info(f"Creating default voice clone prompt ({DEFAULT_VOICE}) for model '{model_name}'...")
            voice_prompt = tts_model.create_voice_clone_prompt(
                ref_audio=REF_AUDIO,
                ref_text=REF_TEXT,
                x_vector_only_mode=False,
            )

            logger.info(f"Computing KV cache for default voice (model={model_name})...")
            kv_cache, past_hidden, prefix_length = compute_voice_kv_cache(tts_model, voice_prompt, REF_TEXT)

            model_voices[DEFAULT_VOICE] = VoiceCache(
                prompt=voice_prompt,
                kv_cache=kv_cache,
                past_hidden=past_hidden,
                prefix_length=prefix_length,
                ref_text=REF_TEXT,
                ref_audio_path=REF_AUDIO,
            )

            # Save default voice cache to disk
            save_voice_cache_to_disk(DEFAULT_VOICE, model_voices[DEFAULT_VOICE], model_name)
            logger.info(f"Default voice '{DEFAULT_VOICE}' ready for model '{model_name}' (prefix_length={prefix_length}).")
        else:
            logger.warning(f"Default reference audio not found: {REF_AUDIO}")
    else:
        logger.info(f"Default voice '{DEFAULT_VOICE}' restored from cache for model '{model_name}'.")

    return model_voices


def unload_model(model_name: str) -> bool:
    """Unload a model from memory to free VRAM."""
    global models, voices, model_last_used

    if model_name not in models:
        return False

    logger.info(f"Unloading model '{model_name}' to free VRAM...")

    # Remove model and its voices from memory
    del models[model_name]
    if model_name in voices:
        del voices[model_name]
    if model_name in model_last_used:
        del model_last_used[model_name]

    # Force CUDA memory cleanup
    torch.cuda.empty_cache()

    logger.info(f"Model '{model_name}' unloaded.")
    return True


def ensure_model_loaded(model_name: str) -> Any:
    """
    Ensure a model is loaded, using lazy loading with LRU eviction.

    If the model is already loaded, returns it immediately.
    If at capacity (MAX_LOADED_MODELS), unloads the least recently used model first.

    Returns the loaded model, or raises an exception if loading fails.
    """
    global models, voices, model_last_used

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_CONFIGS.keys())}")

    with models_lock:
        # Already loaded - just update timestamp and return
        if model_name in models:
            model_last_used[model_name] = time.time()
            return models[model_name]

        # Need to load - check capacity first
        if len(models) >= MAX_LOADED_MODELS and MAX_LOADED_MODELS > 0:
            # Find LRU model to evict
            lru_model = min(model_last_used.keys(), key=lambda k: model_last_used[k])
            logger.info(f"At capacity ({MAX_LOADED_MODELS} models). Evicting LRU model '{lru_model}'...")
            unload_model(lru_model)

        # Load the requested model
        config = MODEL_CONFIGS[model_name]
        model_path = config["path"]

        tts_model = load_single_model(model_name, model_path)
        if tts_model is None:
            raise RuntimeError(f"Failed to load model '{model_name}' from {model_path}")

        models[model_name] = tts_model
        model_last_used[model_name] = time.time()

        # Load voices for this model
        voices[model_name] = load_voices_for_model(model_name, tts_model)

        logger.info(f"Model '{model_name}' loaded on demand with {len(voices[model_name])} voices")
        return tts_model


def load_models():
    """Load the default model on startup (lazy loading for others)."""
    global models, voices, model_last_used

    # Ensure voice_refs directory exists
    VOICE_REFS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Lazy loading enabled: MAX_LOADED_MODELS={MAX_LOADED_MODELS}, DEFAULT_MODEL={DEFAULT_MODEL}")

    # Only load the default model on startup - others load on demand
    try:
        ensure_model_loaded(DEFAULT_MODEL)
    except Exception as e:
        raise RuntimeError(f"Failed to load default model '{DEFAULT_MODEL}': {e}")

    logger.info(f"Startup complete. Default model '{DEFAULT_MODEL}' loaded. Other models load on demand.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    load_models()
    yield
    # Cleanup on shutdown
    global models, voices
    models.clear()
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
    use_kv_cache: bool = False  # KV cache disabled - needs architectural fixes for prefix caching
    model: Optional[str] = None  # model to use: "large" (1.7B) or "small" (0.6B), defaults to DEFAULT_MODEL


class LoadVoiceRequest(BaseModel):
    voice_name: str
    ref_audio_path: str
    ref_text: str
    model: Optional[str] = None  # model to load voice for, defaults to DEFAULT_MODEL


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
    tts_model: Any,
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
        tts_model: The Qwen3TTSModel to use for generation
        use_cache: Whether to use KV cache acceleration (default True)

    Returns:
        Tuple of (wavs, sample_rate)
    """
    # Clone voice prompt to prevent tensor mutation during generation.
    # This is critical to avoid quality degradation (stuttering, hallucinations)
    # on subsequent generations with the same voice.
    cloned_prompt = clone_voice_prompt(voice_cache.prompt)

    # Prepare KV cache parameters if available and requested
    kv_cache_kwargs = {}
    if use_cache and voice_cache.kv_cache is not None:
        # Clone the KV cache to avoid mutation
        kv_cache_kwargs["voice_kv_cache"] = clone_kv_cache(voice_cache.kv_cache)
        kv_cache_kwargs["voice_prefix_length"] = voice_cache.prefix_length
        if voice_cache.past_hidden is not None:
            kv_cache_kwargs["voice_past_hidden"] = voice_cache.past_hidden.clone()
        logger.debug(f"Using KV cache with prefix_length={voice_cache.prefix_length}")

    # Generate with voice prompt and optional KV cache acceleration
    wavs, sr = tts_model.generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=cloned_prompt,
        **kv_cache_kwargs,
    )

    return wavs, sr


async def generate_speech_streaming(
    text: str,
    language: str,
    voice_cache: VoiceCache,
    tts_model: Any,
    use_cache: bool = True,
) -> AsyncGenerator[tuple[np.ndarray, int], None]:
    """Generate speech with sentence-level streaming and optional KV cache acceleration."""
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
                tts_model=tts_model,
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
    loaded_models = list(models.keys())
    return {
        "status": "ok",
        "model_loaded": len(models) > 0,
        "models_loaded": loaded_models,
        "models_available": list(MODEL_CONFIGS.keys()),
        "default_model": DEFAULT_MODEL,
        "max_loaded_models": MAX_LOADED_MODELS,
    }


@app.get("/v1/models")
async def list_models():
    """List all available models (loaded and unloaded)."""
    model_info = {}
    for model_name, config in MODEL_CONFIGS.items():
        is_loaded = model_name in models
        model_info[model_name] = {
            "name": config.get("name", model_name),
            "path": config.get("path", "unknown"),
            "loaded": is_loaded,
            "voices_loaded": len(voices.get(model_name, {})) if is_loaded else 0,
            "last_used": model_last_used.get(model_name) if is_loaded else None,
        }
    return {
        "models": model_info,
        "default": DEFAULT_MODEL,
        "max_loaded": MAX_LOADED_MODELS,
        "currently_loaded": list(models.keys()),
    }


@app.get("/v1/voices")
async def list_voices(model: Optional[str] = None):
    """List all loaded voices with KV cache status. Optionally filter by model."""
    result = {}

    with voices_lock:
        models_to_list = [model] if model else list(voices.keys())

        for model_name in models_to_list:
            if model_name not in voices:
                continue

            voice_info = {}
            for name, cache in voices[model_name].items():
                # Check if persisted on disk
                persisted = get_voice_cache_path(name, model_name).exists()
                voice_info[name] = {
                    "kv_cached": cache.kv_cache is not None,
                    "prefix_length": cache.prefix_length,
                    "ref_audio_path": cache.ref_audio_path,
                    "persisted": persisted,
                }
            result[model_name] = voice_info

    return {
        "voices": result,
        "default_voice": DEFAULT_VOICE,
        "default_model": DEFAULT_MODEL,
    }


@app.post("/v1/voices/load")
async def load_voice(request: LoadVoiceRequest):
    """Load a new voice from reference audio with KV cache precomputation and persistence."""
    global models, voices

    # Lazy load model if needed
    model_name = request.model or DEFAULT_MODEL
    try:
        tts_model = ensure_model_loaded(model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    if not os.path.exists(request.ref_audio_path):
        raise HTTPException(status_code=400, detail=f"Reference audio not found: {request.ref_audio_path}")

    try:
        logger.info(f"Loading voice '{request.voice_name}' for model '{model_name}' from {request.ref_audio_path}...")
        voice_prompt = tts_model.create_voice_clone_prompt(
            ref_audio=request.ref_audio_path,
            ref_text=request.ref_text,
            x_vector_only_mode=False,
        )

        logger.info(f"Computing KV cache for voice '{request.voice_name}' (model={model_name})...")
        kv_cache, past_hidden, prefix_length = compute_voice_kv_cache(tts_model, voice_prompt, request.ref_text)

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
            if model_name not in voices:
                voices[model_name] = {}
            voices[model_name][request.voice_name] = voice_cache

        # Persist to disk (outside lock - file I/O is slow)
        saved = save_voice_cache_to_disk(request.voice_name, voice_cache, model_name)

        logger.info(f"Voice '{request.voice_name}' loaded for model '{model_name}' (prefix_length={prefix_length}, persisted={saved}).")
        return {
            "status": "ok",
            "voice_name": request.voice_name,
            "model": model_name,
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
    model: Optional[str] = Form(None, description="Model to load voice for (large or small)"),
):
    """
    Upload a voice reference audio file and create a persistent voice with KV cache.

    The audio file is saved to the voice_refs directory, and KV cache is computed
    and persisted to disk so the voice survives server restarts.
    """
    global models, voices

    # Lazy load model if needed
    model_name = model or DEFAULT_MODEL
    try:
        tts_model = ensure_model_loaded(model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

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
        logger.info(f"Creating voice clone prompt for '{voice_name}' (model={model_name})...")
        voice_prompt = tts_model.create_voice_clone_prompt(
            ref_audio=str(audio_path),
            ref_text=ref_text,
            x_vector_only_mode=False,
        )

        # Compute KV cache
        logger.info(f"Computing KV cache for voice '{voice_name}' (model={model_name})...")
        kv_cache, past_hidden, prefix_length = compute_voice_kv_cache(tts_model, voice_prompt, ref_text)

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
            if model_name not in voices:
                voices[model_name] = {}
            voices[model_name][voice_name] = voice_cache

        # Persist to disk (outside lock - file I/O is slow)
        saved = save_voice_cache_to_disk(voice_name, voice_cache, model_name)

        logger.info(f"Voice '{voice_name}' uploaded for model '{model_name}' (prefix_length={prefix_length}, persisted={saved}).")
        return {
            "status": "ok",
            "voice_name": voice_name,
            "model": model_name,
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
async def delete_voice(voice_name: str, model: Optional[str] = None):
    """
    Delete a voice and its cached data from memory and disk.

    Note: The default voice cannot be deleted.
    """
    global voices

    model_name = model or DEFAULT_MODEL

    if voice_name == DEFAULT_VOICE:
        raise HTTPException(status_code=400, detail=f"Cannot delete default voice '{DEFAULT_VOICE}'")

    # Thread-safe check and removal from voices dict
    with voices_lock:
        if model_name not in voices or voice_name not in voices[model_name]:
            raise HTTPException(status_code=404, detail=f"Voice '{voice_name}' not found for model '{model_name}'")

        # Get audio path before removing from memory
        voice_cache = voices[model_name][voice_name]
        audio_path = voice_cache.ref_audio_path

        # Remove from memory
        del voices[model_name][voice_name]
        logger.info(f"Removed voice '{voice_name}' from model '{model_name}'")

    # File operations outside lock (slow I/O)
    try:
        # Delete cache files from disk
        cache_deleted = delete_voice_cache_from_disk(voice_name, model_name)

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
            "model": model_name,
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
    global models, voices

    # Resolve model - lazy load if needed
    model_name = request.model or DEFAULT_MODEL
    try:
        tts_model = ensure_model_loaded(model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    # Resolve voice (thread-safe access)
    voice_name = request.voice or DEFAULT_VOICE
    with voices_lock:
        if model_name not in voices or voice_name not in voices[model_name]:
            available = list(voices.get(model_name, {}).keys())
            raise HTTPException(status_code=400, detail=f"Voice '{voice_name}' not loaded for model '{model_name}'. Available: {available}")
        # Get reference to voice_cache - safe to use outside lock since VoiceCache is immutable
        voice_cache = voices[model_name][voice_name]

    # Handle streaming mode
    if request.stream:
        async def audio_stream_generator():
            if request.response_format == "wav":
                yield create_wav_streaming_header(24000)

            async for audio_chunk, sr in generate_speech_streaming(
                text=request.text,
                language=request.language,
                voice_cache=voice_cache,
                tts_model=tts_model,
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
                tts_model=tts_model,
                use_cache=True,
            )
        else:
            wavs, sr = tts_model.generate_voice_clone(
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
