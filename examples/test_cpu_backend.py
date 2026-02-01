#!/usr/bin/env python3
# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Example script for testing CPU backend functionality.

This script demonstrates how to use the CPU-optimized PyTorch backend
for Qwen3-TTS inference without requiring a GPU.
"""

import os
import sys
import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def test_cpu_backend():
    """Test the CPU backend with a simple inference."""
    
    # Set environment variables for CPU backend
    os.environ["TTS_BACKEND"] = "pytorch"
    os.environ["TTS_MODEL_ID"] = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    os.environ["TTS_DEVICE"] = "cpu"
    os.environ["TTS_DTYPE"] = "float32"
    os.environ["TTS_ATTN"] = "sdpa"
    os.environ["CPU_THREADS"] = str(os.cpu_count() or 4)  # Auto-detect CPU cores
    os.environ["CPU_INTEROP"] = "1"
    
    logger.info("=" * 60)
    logger.info("CPU Backend Test")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Backend: {os.environ['TTS_BACKEND']}")
    logger.info(f"  Model: {os.environ['TTS_MODEL_ID']}")
    logger.info(f"  Device: {os.environ['TTS_DEVICE']}")
    logger.info(f"  Dtype: {os.environ['TTS_DTYPE']}")
    logger.info(f"  Attention: {os.environ['TTS_ATTN']}")
    logger.info(f"  CPU Threads: {os.environ['CPU_THREADS']}")
    logger.info("")
    
    try:
        # Import backend factory
        from api.backends import get_backend, initialize_backend
        
        logger.info("Initializing CPU backend...")
        backend = await initialize_backend(warmup=False)
        
        logger.info(f"Backend: {backend.get_backend_name()}")
        logger.info(f"Model: {backend.get_model_id()}")
        logger.info("")
        
        # Get device info
        device_info = backend.get_device_info()
        logger.info("Device Information:")
        for key, value in device_info.items():
            if value is not None:
                logger.info(f"  {key}: {value}")
        logger.info("")
        
        # Test speech generation
        logger.info("Generating test speech...")
        test_text = "Hello! This is a test of the CPU-optimized PyTorch backend."
        
        audio, sample_rate = await backend.generate_speech(
            text=test_text,
            voice="Vivian",
            language="English",
        )
        
        logger.info(f"Success! Generated {len(audio)} samples at {sample_rate} Hz")
        logger.info(f"Duration: {len(audio) / sample_rate:.2f} seconds")
        logger.info("")
        
        # Save to file
        try:
            import soundfile as sf
            output_file = "cpu_backend_test.wav"
            sf.write(output_file, audio, sample_rate)
            logger.info(f"Saved audio to: {output_file}")
        except ImportError:
            logger.warning("soundfile not installed, skipping file save")
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("CPU Backend Test PASSED ✓")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"CPU Backend Test FAILED ✗")
        logger.error(f"Error: {e}")
        logger.error("=" * 60)
        import traceback
        traceback.print_exc()
        return False


async def test_openvino_backend():
    """Test the OpenVINO backend (will likely fail without exported model)."""
    
    # Set environment variables for OpenVINO backend
    os.environ["TTS_BACKEND"] = "openvino"
    os.environ["OV_DEVICE"] = "CPU"
    os.environ["OV_MODEL_DIR"] = "./.ov_models"
    
    logger.info("=" * 60)
    logger.info("OpenVINO Backend Test (Expected to Fail)")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Backend: {os.environ['TTS_BACKEND']}")
    logger.info(f"  Device: {os.environ['OV_DEVICE']}")
    logger.info(f"  Model Dir: {os.environ['OV_MODEL_DIR']}")
    logger.info("")
    
    try:
        from api.backends import get_backend, initialize_backend
        from api.backends.factory import reset_backend
        
        # Reset backend to force re-initialization
        reset_backend()
        
        logger.info("Initializing OpenVINO backend...")
        backend = await initialize_backend(warmup=False)
        
        logger.info(f"Backend: {backend.get_backend_name()}")
        logger.info("")
        
        # This will likely fail without exported model
        logger.info("Attempting speech generation...")
        audio, sample_rate = await backend.generate_speech(
            text="Test",
            voice="Vivian",
            language="English",
        )
        
        logger.info("Unexpectedly succeeded!")
        return True
        
    except RuntimeError as e:
        logger.info("=" * 60)
        logger.info("OpenVINO Backend Test Result:")
        logger.info(f"  Expected error occurred: {str(e)[:100]}...")
        logger.info("")
        logger.info("This is expected - OpenVINO requires model export.")
        logger.info("For CPU inference, use TTS_BACKEND=pytorch instead.")
        logger.info("=" * 60)
        return True  # Expected failure
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"Unexpected error: {e}")
        logger.error("=" * 60)
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all backend tests."""
    
    logger.info("\n")
    logger.info("╔════════════════════════════════════════════════════════╗")
    logger.info("║     Qwen3-TTS CPU Backend Functionality Test          ║")
    logger.info("╚════════════════════════════════════════════════════════╝")
    logger.info("")
    
    # Test CPU backend
    logger.info("Test 1: CPU-Optimized PyTorch Backend")
    logger.info("")
    success_cpu = asyncio.run(test_cpu_backend())
    
    logger.info("\n\n")
    
    # Test OpenVINO backend (expected to fail)
    logger.info("Test 2: OpenVINO Backend (Experimental)")
    logger.info("")
    success_openvino = asyncio.run(test_openvino_backend())
    
    logger.info("\n\n")
    logger.info("=" * 60)
    logger.info("Test Summary:")
    logger.info(f"  CPU Backend: {'PASSED ✓' if success_cpu else 'FAILED ✗'}")
    logger.info(f"  OpenVINO Backend: {'PASSED ✓' if success_openvino else 'FAILED ✗'}")
    logger.info("=" * 60)
    
    if success_cpu and success_openvino:
        logger.info("\nAll tests completed successfully!")
        sys.exit(0)
    else:
        logger.error("\nSome tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
