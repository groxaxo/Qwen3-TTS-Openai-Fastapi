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
            
            # Determine attention implementation based on device
            # Flash Attention 2 requires CUDA and is not available on CPU
            if torch.cuda.is_available():
                attn_impl = "flash_attention_2"
            else:
                # On CPU, force a standard/non-flash attention implementation
                attn_impl = "eager"
            
            # Load model with optimizations
            model_kwargs = {
                "device_map": self.device,
                "dtype": self.dtype,
                "attn_implementation": attn_impl,
            }
            if attn_impl == "flash_attention_2":
                logger.info("Using Flash Attention 2 for faster inference")
            else:
                logger.info("Running on CPU - using standard attention implementation")
            
            self.model = Qwen3TTSModel.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
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
