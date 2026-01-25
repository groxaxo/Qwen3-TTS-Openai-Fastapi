#!/usr/bin/env python3
"""
Voice cloning TTS using Qwen3-TTS Base model.
Outputs WAV audio to stdout or a file.

Usage:
    python voice_clone_tts.py "text to speak" [-o output.wav]
    python voice_clone_tts.py "text" > output.wav
"""

import sys
import argparse
import io

# Global model instance (lazy loaded)
_model = None
_voice_prompt = None

# Paths
MODEL_PATH = "./models/Qwen3-TTS-12Hz-1.7B-Base"
REF_AUDIO = "./voice_refs/hai_reference.wav"
REF_TEXT = "Yeah so basically I checked the system logs and found a couple of errors. Nothing critical, but you should probably take a look when you get a chance. The server has been running fine otherwise."


def get_model():
    """Lazy load the model."""
    global _model
    if _model is None:
        import torch
        from qwen_tts import Qwen3TTSModel

        print("Loading model...", file=sys.stderr)
        _model = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        print("Model loaded.", file=sys.stderr)
    return _model


def get_voice_prompt():
    """Lazy load the voice clone prompt."""
    global _voice_prompt
    if _voice_prompt is None:
        model = get_model()
        print("Creating voice clone prompt...", file=sys.stderr)
        _voice_prompt = model.create_voice_clone_prompt(
            ref_audio=REF_AUDIO,
            ref_text=REF_TEXT,
            x_vector_only_mode=False,
        )
        print("Voice prompt ready.", file=sys.stderr)
    return _voice_prompt


def generate_speech(text: str, language: str = "English") -> tuple:
    """Generate speech from text using voice cloning."""
    model = get_model()
    voice_prompt = get_voice_prompt()

    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=voice_prompt,
    )

    return wavs[0], sr


def main():
    parser = argparse.ArgumentParser(description="Voice cloning TTS")
    parser.add_argument("text", help="Text to speak")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument("-l", "--language", default="English", help="Language (default: English)")
    args = parser.parse_args()

    import soundfile as sf

    audio, sr = generate_speech(args.text, args.language)

    if args.output:
        sf.write(args.output, audio, sr)
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        # Write WAV to stdout
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format='WAV')
        buf.seek(0)
        sys.stdout.buffer.write(buf.read())


if __name__ == "__main__":
    main()
