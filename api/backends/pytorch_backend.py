# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
CPU-optimized PyTorch backend for Qwen3-TTS.

This backend is specifically tuned for CPU inference with Intel processors,
particularly the i5-1240P and similar CPUs. It includes optimizations for:
- Thread management (PyTorch, OpenMP, MKL)
- Efficient attention mechanisms (SDPA/eager, no Flash Attention)
- Optimal dtype selection (float32 for CPU stability)
- Optional Intel Extension for PyTorch (IPEX)
"""

import logging
import os
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch

from .base import TTSBackend
from ..config import (
    CPU_THREADS, CPU_INTEROP, TTS_DEVICE, TTS_DTYPE, TTS_ATTN,
    USE_IPEX, IPEX_AVAILABLE
)

logger = logging.getLogger(__name__)

# Optional librosa import for speed adjustment
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class PyTorchCPUBackend(TTSBackend):
    """
    CPU-optimized PyTorch backend for Qwen3-TTS.
    
    This backend is designed for efficient CPU inference, with special
    optimizations for Intel processors like the i5-1240P.
    """
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device: str = "cpu",
        dtype: str = "float32",
        attn_implementation: str = "sdpa",
        cpu_threads: int = 12,
        cpu_interop_threads: int = 2,
        use_ipex: bool = False,
    ):
        """
        Initialize the CPU-optimized PyTorch backend.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to run on (should be 'cpu')
            dtype: Data type (recommended: 'float32' for CPU)
            attn_implementation: Attention implementation ('sdpa' or 'eager')
            cpu_threads: Number of threads for PyTorch operations
            cpu_interop_threads: Number of threads for inter-op parallelism
            use_ipex: Whether to use Intel Extension for PyTorch
        """
        super().__init__()
        self.model_name = model_id
        self.device_name = device
        self.dtype_name = dtype
        self.attn_implementation = attn_implementation
        self.cpu_threads = cpu_threads
        self.cpu_interop_threads = cpu_interop_threads
        self.use_ipex = use_ipex and IPEX_AVAILABLE
        self._ready = False
        
        # Configure CPU threading
        if device == "cpu":
            try:
                torch.set_num_threads(cpu_threads)
                torch.set_num_interop_threads(cpu_interop_threads)
                logger.info(f"Set PyTorch CPU threads: {cpu_threads}, interop: {cpu_interop_threads}")
            except RuntimeError as e:
                # Thread settings may fail if already set, this is non-critical
                logger.warning(f"Could not set thread counts (already set): {e}")
                logger.info(f"Using existing PyTorch thread settings")
    
    async def initialize(self) -> None:
        """Initialize the backend and load the model."""
        if self._ready:
            logger.info("CPU PyTorch backend already initialized")
            return
        
        try:
            from qwen_tts import Qwen3TTSModel
            
            # Map dtype string to torch dtype
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            self.dtype = dtype_map.get(self.dtype_name, torch.float32)
            
            # Warn if using non-float32 on CPU
            if self.device_name == "cpu" and self.dtype != torch.float32:
                logger.warning(
                    f"Using {self.dtype_name} on CPU may cause issues. "
                    f"float32 is recommended for CPU inference."
                )
            
            self.device = self.device_name
            
            logger.info(
                f"Loading Qwen3-TTS model '{self.model_name}' on {self.device} "
                f"with dtype={self.dtype_name}, attn={self.attn_implementation}"
            )
            
            # Load model
            # For CPU, avoid flash_attention_2 (not supported/efficient on CPU)
            attn_impl = self.attn_implementation
            if attn_impl == "flash_attention_2":
                logger.warning("Flash Attention 2 not recommended for CPU, using sdpa instead")
                attn_impl = "sdpa"
            
            try:
                self.model = Qwen3TTSModel.from_pretrained(
                    self.model_name,
                    device_map=self.device,
                    dtype=self.dtype,
                    attn_implementation=attn_impl,
                )
                logger.info(f"Successfully loaded model with {attn_impl} attention")
            except Exception as e:
                logger.warning(f"Failed to load with {attn_impl}, trying 'eager': {e}")
                self.model = Qwen3TTSModel.from_pretrained(
                    self.model_name,
                    device_map=self.device,
                    dtype=self.dtype,
                    attn_implementation="eager",
                )
                logger.info("Successfully loaded model with eager attention")
            
            # Apply IPEX optimizations if enabled
            if self.use_ipex:
                try:
                    import intel_extension_for_pytorch as ipex
                    logger.info("Applying Intel Extension for PyTorch optimizations...")
                    self.model.model = ipex.optimize(self.model.model, dtype=self.dtype)
                    logger.info("IPEX optimization applied successfully")
                except Exception as e:
                    logger.warning(f"Could not apply IPEX optimization: {e}")
            
            self._ready = True
            logger.info(f"CPU PyTorch backend loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load CPU PyTorch backend: {e}")
            raise RuntimeError(f"Failed to initialize CPU PyTorch backend: {e}")
    
    async def generate_speech(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech from text.
        
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
        return "pytorch_cpu"
    
    def get_model_id(self) -> str:
        """Return the model identifier."""
        return self.model_name
    
    def get_supported_voices(self) -> List[str]:
        """Return list of supported voice names."""
        if not self._ready or not self.model:
            return ["Vivian", "Ryan", "Sophia", "Isabella", "Evan", "Lily"]
        
        try:
            if hasattr(self.model.model, 'get_supported_speakers'):
                speakers = self.model.model.get_supported_speakers()
                if speakers:
                    return list(speakers)
        except Exception as e:
            logger.warning(f"Could not get speakers from model: {e}")
        
        return ["Vivian", "Ryan", "Sophia", "Isabella", "Evan", "Lily"]
    
    def get_supported_languages(self) -> List[str]:
        """Return list of supported language names."""
        if not self._ready or not self.model:
            return ["English", "Chinese", "Japanese", "Korean", "German", "French", 
                    "Spanish", "Russian", "Portuguese", "Italian"]
        
        try:
            if hasattr(self.model.model, 'get_supported_languages'):
                languages = self.model.model.get_supported_languages()
                if languages:
                    return list(languages)
        except Exception as e:
            logger.warning(f"Could not get languages from model: {e}")
        
        return ["English", "Chinese", "Japanese", "Korean", "German", "French", 
                "Spanish", "Russian", "Portuguese", "Italian"]
    
    def is_ready(self) -> bool:
        """Return whether the backend is initialized and ready."""
        return self._ready
    
    def get_device_info(self) -> Dict[str, Any]:
        """Return device information."""
        info = {
            "device": str(self.device) if self.device else "cpu",
            "gpu_available": False,
            "gpu_name": None,
            "vram_total": None,
            "vram_used": None,
            "cpu_threads": self.cpu_threads,
            "cpu_interop_threads": self.cpu_interop_threads,
            "ipex_enabled": self.use_ipex,
        }
        
        return info
    
    def supports_voice_cloning(self) -> bool:
        """
        Check if this backend supports voice cloning.
        
        Voice cloning requires the Base model.
        """
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
            language: Language code
            x_vector_only_mode: If True, use x-vector only
            speed: Speech speed multiplier
        
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if not self._ready:
            await self.initialize()
        
        if not self.supports_voice_cloning():
            raise RuntimeError(
                "Voice cloning requires the Base model (Qwen3-TTS-12Hz-*-Base). "
                "The current model does not support voice cloning."
            )
        
        try:
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
