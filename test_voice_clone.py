#!/usr/bin/env python3
"""Test voice cloning with Qwen3-TTS Base model."""

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Reference audio and transcript
REF_AUDIO = "voice_refs/hai_reference.wav"
REF_TEXT = "Yeah so basically I checked the system logs and found a couple of errors. Nothing critical, but you should probably take a look when you get a chance. The server has been running fine otherwise."

# Test text
TEST_TEXT = "Alright, testing out the voice cloning. Let me know if this sounds anything like the original."

print("Loading Base model for voice cloning...")
model = Qwen3TTSModel.from_pretrained(
    "./models/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="eager",  # fallback since flash_attention_2 not installed
)
print("Model loaded.")

print("Creating voice clone prompt...")
prompt_items = model.create_voice_clone_prompt(
    ref_audio=REF_AUDIO,
    ref_text=REF_TEXT,
    x_vector_only_mode=False,
)
print("Voice clone prompt created.")

print("Generating cloned speech...")
wavs, sr = model.generate_voice_clone(
    text=TEST_TEXT,
    language="English",
    voice_clone_prompt=prompt_items,
)

output_file = "voice_refs/cloned_output.wav"
sf.write(output_file, wavs[0], sr)
print(f"Saved to {output_file}")
