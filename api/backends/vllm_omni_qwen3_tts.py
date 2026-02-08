# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
vLLM-Omni Qwen3-TTS backend implementation.

This backend uses vLLM-Omni for faster inference with Qwen3-TTS models.
Uses the official vLLM-Omni API with correct imports from `vllm_omni`.

See: https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/offline_inference/qwen3_tts/
"""

import os
import io
import logging
import asyncio
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from .base import TTSBackend

logger = logging.getLogger(__name__)

# Set multiprocessing method for vLLM-Omni (must be done before import)
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# Optional librosa import for speed adjustment
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class VLLMOmniQwen3TTSBackend(TTSBackend):
    """
    vLLM-Omni backend for Qwen3-TTS.
    
    Uses the same input structure as the official vLLM-Omni example:
    - inputs["prompt"]
    - inputs["additional_information"] with list-wrapped fields
    - output.multimodal_output["audio"] and ["sr"]
    """
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        stage_configs_path: Optional[str] = None,
        enable_stats: bool = False,
        stage_init_timeout_s: int = 300,
        seed: int = 42,
        max_tokens: int = 2048,
    ):
        """
        Initialize the vLLM-Omni backend.
        
        Args:
            model_name: HuggingFace model identifier
                Options:
                - Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice (recommended)
                - Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
                - Qwen/Qwen3-TTS-12Hz-1.7B-Base
                - Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
            stage_configs_path: Optional path to stage configs
            enable_stats: Whether to log statistics
            stage_init_timeout_s: Timeout for stage initialization
            seed: Random seed for reproducibility
            max_tokens: Maximum tokens for generation
        """
        super().__init__()
        self.model_name = model_name
        self.stage_configs_path = stage_configs_path
        self.enable_stats = enable_stats
        self.stage_init_timeout_s = stage_init_timeout_s
        self.seed = seed
        self.max_tokens = max_tokens
        
        self._ready = False
        self._lock = asyncio.Lock()
        self.omni = None
        self.sampling_params_list = None
    
    async def initialize(self) -> None:
        """Initialize the backend and load the model."""
        if self._ready:
            logger.info("vLLM-Omni backend already initialized")
            return
        
        async with self._lock:
            if self._ready:
                return
            
            try:
                # Import vLLM-Omni (note: from vllm_omni, not vllm)
                from vllm import SamplingParams
                from vllm_omni import Omni
                
                logger.info(f"Loading vLLM-Omni model '{self.model_name}'...")
                
                # Initialize vLLM-Omni
                self.omni = Omni(
                    model=self.model_name,
                    stage_configs_path=self.stage_configs_path,
                    log_stats=self.enable_stats,
                    stage_init_timeout=self.stage_init_timeout_s,
                )
                
                # Pre-create sampling params for reuse
                self.sampling_params_list = [
                    SamplingParams(
                        temperature=0.9,
                        top_p=1.0,
                        top_k=50,
                        max_tokens=self.max_tokens,
                        seed=self.seed,
                        detokenize=False,
                        repetition_penalty=1.05,
                    )
                ]
                
                self._ready = True
                logger.info(f"vLLM-Omni backend loaded successfully!")
                
            except ImportError as e:
                logger.error(f"vLLM-Omni not installed: {e}")
                raise RuntimeError(
                    "vLLM-Omni is not installed. Please use Dockerfile.vllm or install: "
                    "pip install vllm-omni (requires Python 3.12 and CUDA)"
                )
            except Exception as e:
                logger.error(f"Failed to load vLLM-Omni backend: {e}")
                raise RuntimeError(f"Failed to initialize vLLM-Omni backend: {e}")
    
    async def generate_speech(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech from text using vLLM-Omni.
        
        Args:
            text: The text to synthesize
            voice: Voice/speaker name (e.g., "Vivian", "Ryan")
            language: Language code (e.g., "English", "Chinese", "Auto")
            instruct: Optional instruction for voice style
            speed: Speech speed multiplier
        
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if not self._ready:
            await self.initialize()
        
        try:
            # Build prompt and inputs following official vLLM-Omni example
            prompt = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
            
            # Determine task type based on model
            if "CustomVoice" in self.model_name:
                task_type = "CustomVoice"
            elif "VoiceDesign" in self.model_name:
                task_type = "VoiceDesign"
            else:
                task_type = "Base"
            
            inputs = {
                "prompt": prompt,
                "additional_information": {
                    "task_type": [task_type],
                    "text": [text],
                    "instruct": [instruct or ""],
                    "language": [language],
                    "speaker": [voice],
                    "max_new_tokens": [self.max_tokens],
                },
            }
            
            # Generate using vLLM-Omni
            omni_generator = self.omni.generate(inputs, self.sampling_params_list)
            
            # Process outputs
            for stage_outputs in omni_generator:
                for output in stage_outputs.request_output:
                    # Extract audio from multimodal output
                    audio_tensor = output.multimodal_output["audio"]
                    sr = int(output.multimodal_output["sr"].item())
                    
                    # Convert to numpy
                    audio_np = audio_tensor.float().detach().cpu().numpy()
                    if audio_np.ndim > 1:
                        audio_np = audio_np.flatten()
                    
                    # Apply speed adjustment if needed
                    if speed != 1.0 and LIBROSA_AVAILABLE:
                        audio_np = librosa.effects.time_stretch(
                            audio_np.astype(np.float32), 
                            rate=speed
                        )
                    elif speed != 1.0:
                        logger.warning("Speed adjustment requested but librosa not available")
                    
                    return audio_np, sr
            
            raise RuntimeError("No audio returned from vLLM-Omni (no stage outputs)")
            
        except Exception as e:
            logger.error(f"vLLM-Omni speech generation failed: {e}")
            raise RuntimeError(f"vLLM-Omni speech generation failed: {e}")
    
    def synthesize_wav_bytes(
        self,
        text: str,
        speaker: str = "Vivian",
        language: str = "Auto",
        instruct: str = "",
        task_type: str = "CustomVoice",
    ) -> Tuple[bytes, int]:
        """
        Synchronous method to synthesize and return WAV bytes.
        
        This provides a direct interface matching the official vLLM-Omni example.
        
        Args:
            text: Text to synthesize
            speaker: Speaker/voice name
            language: Language code
            instruct: Optional instruction
            task_type: Task type (CustomVoice, VoiceDesign, Base)
        
        Returns:
            Tuple of (wav_bytes, sample_rate)
        """
        import soundfile as sf
        
        # Run async method synchronously
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, create a new loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.generate_speech(text, speaker, language, instruct)
                )
                audio_np, sr = future.result()
        else:
            audio_np, sr = loop.run_until_complete(
                self.generate_speech(text, speaker, language, instruct)
            )
        
        # Convert to WAV bytes
        buf = io.BytesIO()
        sf.write(buf, audio_np, samplerate=sr, format="WAV")
        return buf.getvalue(), sr
    
    def close(self):
        """Close the backend and release resources."""
        if self.omni is not None:
            try:
                self.omni.close()
            except Exception as e:
                logger.warning(f"Error closing vLLM-Omni: {e}")
            finally:
                self.omni = None
                self._ready = False
    
    def get_backend_name(self) -> str:
        """Return the name of this backend."""
        return "vllm_omni"
    
    def get_model_id(self) -> str:
        """Return the model identifier."""
        return self.model_name
    
    def get_supported_voices(self) -> List[str]:
        """Return list of supported voice names."""
        # Same voices as official Qwen3-TTS
        return [
            "Vivian", "Ryan", "Sophia", "Isabella", "Evan", "Lily",
            "Serena", "Dylan", "Eric", "Aiden"
        ]
    
    def get_supported_languages(self) -> List[str]:
        """Return list of supported language names."""
        return [
            "Auto", "English", "Chinese", "Japanese", "Korean", 
            "German", "French", "Spanish", "Russian", "Portuguese", "Italian"
        ]
    
    def is_ready(self) -> bool:
        """Return whether the backend is initialized and ready."""
        return self._ready
    
    def get_device_info(self) -> Dict[str, Any]:
        """Return device information."""
        info = {
            "device": "cuda",
            "gpu_available": False,
            "gpu_name": None,
            "vram_total": None,
            "vram_used": None,
        }
        
        try:
            import torch
            
            if torch.cuda.is_available():
                info["gpu_available"] = True
                device_idx = torch.cuda.current_device()
                info["gpu_name"] = torch.cuda.get_device_name(device_idx)
                
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

    async def load_custom_voices(self, custom_voices_dir: str) -> None:
        """Log a warning if custom voice directories exist, as vLLM doesn't support them."""
        from pathlib import Path

        voices_path = Path(custom_voices_dir)
        if not voices_path.exists():
            return

        # Check if there are any voice subdirectories
        voice_dirs = [d for d in voices_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if voice_dirs:
            logger.warning(
                f"Found {len(voice_dirs)} custom voice folder(s) in {custom_voices_dir}, "
                "but the vLLM backend does not support voice cloning. "
                "Custom voices will be ignored. Use the official backend with the Base model instead."
            )

    def get_model_type(self) -> str:
        """Return the model type (base, customvoice, or voicedesign)."""
        if "Base" in self.model_name:
            return "base"
        elif "CustomVoice" in self.model_name:
            return "customvoice"
        elif "VoiceDesign" in self.model_name:
            return "voicedesign"
        return "unknown"
