# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Official Qwen3-TTS backend implementation.

This backend uses the official Qwen3-TTS Python implementation
from the qwen_tts package.
"""

import logging
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from .base import TTSBackend

logger = logging.getLogger(__name__)

# Optional librosa import for speed adjustment
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class OfficialQwen3TTSBackend(TTSBackend):
    """Official Qwen3-TTS backend using the qwen_tts package."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"):
        """
        Initialize the official backend.
        
        Args:
            model_name: HuggingFace model identifier
        """
        super().__init__()
        self.model_name = model_name
        self._ready = False
    
    async def initialize(self) -> None:
        """Initialize the backend and load the model."""
        if self._ready:
            logger.info("Official backend already initialized")
            return
        
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
            
            # Determine device
            if torch.cuda.is_available():
                self.device = "cuda:0"
                self.dtype = torch.bfloat16
            else:
                self.device = "cpu"
                self.dtype = torch.float32
            
            logger.info(f"Loading Qwen3-TTS model '{self.model_name}' on {self.device}...")

            # Try loading with Flash Attention 2, fallback to SDPA or eager if not supported
            # (e.g., RTX 5090/Blackwell GPUs don't have pre-built flash-attn wheels yet)
            attn_implementations = ["flash_attention_2", "sdpa", "eager"]
            model_loaded = False

            last_error = None
            for attn_impl in attn_implementations:
                try:
                    logger.info(f"Attempting to load model with attention: {attn_impl}")
                    self.model = Qwen3TTSModel.from_pretrained(
                        self.model_name,
                        device_map=self.device,
                        dtype=self.dtype,
                        attn_implementation=attn_impl,
                    )
                    logger.info(f"Successfully loaded model with {attn_impl} attention")
                    model_loaded = True
                    break
                except Exception as attn_error:
                    last_error = attn_error
                    logger.warning(f"Could not load with {attn_impl}: {attn_error}")
                    if attn_impl != attn_implementations[-1]:
                        logger.info(f"Falling back to next attention implementation...")

            if not model_loaded:
                # If GPU loading failed completely, try CPU as last resort
                if self.device != "cpu":
                    logger.warning("All GPU attention implementations failed. Falling back to CPU...")
                    self.device = "cpu"
                    self.dtype = torch.float32
                    try:
                        self.model = Qwen3TTSModel.from_pretrained(
                            self.model_name,
                            device_map=self.device,
                            dtype=self.dtype,
                            attn_implementation="eager",
                        )
                        logger.info("Successfully loaded model on CPU (GPU not compatible)")
                        model_loaded = True
                    except Exception as cpu_error:
                        raise RuntimeError(f"Failed to load model on CPU: {cpu_error}")
                else:
                    raise RuntimeError(f"Failed to load model with any attention implementation. Last error: {last_error}")

            # Apply torch.compile() optimization for faster inference
            if torch.cuda.is_available() and hasattr(torch, 'compile'):
                logger.info("Applying torch.compile() optimization...")
                try:
                    # Compile the model with reduce-overhead mode for faster inference
                    self.model.model = torch.compile(
                        self.model.model,
                        mode="reduce-overhead",  # Optimize for inference speed
                        fullgraph=False,  # Allow graph breaks for compatibility
                    )
                    logger.info("torch.compile() optimization applied successfully")
                except Exception as e:
                    logger.warning(f"Could not apply torch.compile(): {e}")
            
            # Enable cuDNN benchmarking for optimal convolution algorithms
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                logger.info("Enabled cuDNN benchmark mode")
            
            # Enable TF32 for faster matmul on Ampere+ GPUs (RTX 30xx/40xx)
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Enabled TF32 precision for faster matmul")
            
            self._ready = True
            logger.info(f"Official Qwen3-TTS backend loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load official TTS backend: {e}")
            raise RuntimeError(f"Failed to initialize official TTS backend: {e}")
    
    async def generate_speech(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech from text using the official Qwen3-TTS model.
        
        Args:
            text: The text to synthesize
            voice: Voice name to use
            language: Language code
            instruct: Optional instruction for voice style
            speed: Speech speed multiplier
        
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if not self._ready:
            await self.initialize()
        
        try:
            # Generate speech
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                language=language,
                speaker=voice,
                instruct=instruct,
            )
            
            audio = wavs[0]
            
            # Apply speed adjustment if needed
            if speed != 1.0 and LIBROSA_AVAILABLE:
                audio = librosa.effects.time_stretch(audio.astype(np.float32), rate=speed)
            elif speed != 1.0:
                logger.warning("Speed adjustment requested but librosa not available")
            
            return audio, sr
            
        except Exception as e:
            logger.error(f"Speech generation failed: {e}")
            raise RuntimeError(f"Speech generation failed: {e}")
    
    def get_backend_name(self) -> str:
        """Return the name of this backend."""
        return "official"
    
    def get_model_id(self) -> str:
        """Return the model identifier."""
        return self.model_name
    
    def get_supported_voices(self) -> List[str]:
        """Return list of supported voice names."""
        if not self._ready or not self.model:
            # Return default voices when model is not loaded
            return ["Vivian", "Ryan", "Sophia", "Isabella", "Evan", "Lily"]
        
        try:
            if hasattr(self.model.model, 'get_supported_speakers'):
                speakers = self.model.model.get_supported_speakers()
                if speakers:
                    return list(speakers)
        except Exception as e:
            logger.warning(f"Could not get speakers from model: {e}")
        
        # Fallback to default voices
        return ["Vivian", "Ryan", "Sophia", "Isabella", "Evan", "Lily"]
    
    def get_supported_languages(self) -> List[str]:
        """Return list of supported language names."""
        if not self._ready or not self.model:
            # Return default languages when model is not loaded
            return ["English", "Chinese", "Japanese", "Korean", "German", "French", 
                    "Spanish", "Russian", "Portuguese", "Italian"]
        
        try:
            if hasattr(self.model.model, 'get_supported_languages'):
                languages = self.model.model.get_supported_languages()
                if languages:
                    return list(languages)
        except Exception as e:
            logger.warning(f"Could not get languages from model: {e}")
        
        # Fallback to default languages
        return ["English", "Chinese", "Japanese", "Korean", "German", "French", 
                "Spanish", "Russian", "Portuguese", "Italian"]
    
    def is_ready(self) -> bool:
        """Return whether the backend is initialized and ready."""
        return self._ready
    
    def get_device_info(self) -> Dict[str, Any]:
        """Return device information."""
        info = {
            "device": str(self.device) if self.device else "unknown",
            "gpu_available": False,
            "gpu_name": None,
            "vram_total": None,
            "vram_used": None,
        }

        try:
            import torch

            if torch.cuda.is_available():
                info["gpu_available"] = True
                if torch.cuda.current_device() >= 0:
                    device_idx = torch.cuda.current_device()
                    info["gpu_name"] = torch.cuda.get_device_name(device_idx)

                    # Get VRAM info
                    props = torch.cuda.get_device_properties(device_idx)
                    info["vram_total"] = f"{props.total_memory / 1024**3:.2f} GB"

                    if self._ready:
                        allocated = torch.cuda.memory_allocated(device_idx)
                        info["vram_used"] = f"{allocated / 1024**3:.2f} GB"
        except Exception as e:
            logger.warning(f"Could not get device info: {e}")

        return info

    def supports_voice_cloning(self) -> bool:
        """
        Check if this backend supports voice cloning.

        Voice cloning requires the Base model (Qwen3-TTS-12Hz-1.7B-Base).
        The CustomVoice model does not support voice cloning.
        """
        # Check if we're using the Base model (not CustomVoice)
        return "Base" in self.model_name and "CustomVoice" not in self.model_name

    def get_model_type(self) -> str:
        """Return the model type (base or customvoice)."""
        if "Base" in self.model_name:
            return "base"
        elif "CustomVoice" in self.model_name:
            return "customvoice"
        return "unknown"

    async def generate_voice_clone(
        self,
        text: str,
        ref_audio: np.ndarray,
        ref_audio_sr: int,
        ref_text: Optional[str] = None,
        language: str = "Auto",
        x_vector_only_mode: bool = False,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech by cloning a voice from reference audio.

        Args:
            text: The text to synthesize
            ref_audio: Reference audio as numpy array
            ref_audio_sr: Sample rate of reference audio
            ref_text: Transcript of reference audio (required for ICL mode)
            language: Language code (e.g., "English", "Chinese", "Auto")
            x_vector_only_mode: If True, use x-vector only (no ref_text needed)
            speed: Speech speed multiplier (0.25 to 4.0)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if not self._ready:
            await self.initialize()

        if not self.supports_voice_cloning():
            raise RuntimeError(
                "Voice cloning requires the Base model (Qwen3-TTS-12Hz-1.7B-Base). "
                "The current model does not support voice cloning."
            )

        try:
            # Call the model's voice cloning method
            # ref_audio expects a tuple of (waveform, sample_rate)
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                ref_audio=(ref_audio, ref_audio_sr),
                ref_text=ref_text,
                language=language,
                x_vector_only_mode=x_vector_only_mode,
            )

            audio = wavs[0]

            # Apply speed adjustment if needed
            if speed != 1.0 and LIBROSA_AVAILABLE:
                audio = librosa.effects.time_stretch(audio.astype(np.float32), rate=speed)
            elif speed != 1.0:
                logger.warning("Speed adjustment requested but librosa not available")

            return audio, sr

        except Exception as e:
            logger.error(f"Voice cloning failed: {e}")
            raise RuntimeError(f"Voice cloning failed: {e}")
