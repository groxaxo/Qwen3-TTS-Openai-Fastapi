"""Measure warm WAV RTF or PCM first-audio latency against any running API."""
import argparse
import io
import json
from pathlib import Path
import statistics
import time
import urllib.request
import wave


def measure(base_url, voice, text, streaming, model="tts-1"):
    payload = dict(model=model, voice=voice, input=text,
                   response_format="pcm" if streaming else "wav", stream=streaming)
    request = urllib.request.Request(base_url.rstrip('/') + '/audio/speech',
        data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        first = response.read1(4096)
        first_audio = time.perf_counter() - start
        data = first + response.read()
    elapsed = time.perf_counter() - start
    if streaming:
        duration = len(data) / (24000 * 2)
    else:
        with wave.open(io.BytesIO(data)) as audio:
            duration = audio.getnframes() / audio.getframerate()
    if duration <= 0: raise ValueError('API returned empty audio')
    return {'wall_seconds':elapsed, 'audio_seconds':duration, 'rtf':elapsed/duration,
            **({'first_audio_seconds':first_audio} if streaming else {})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8880/v1')
    parser.add_argument('--voice', default='Ryan')
    parser.add_argument('--model', default='tts-1', help='Proxy model ID when using a shared gateway')
    parser.add_argument('--requests', type=int, default=4)
    parser.add_argument('--mode', choices=['wav','pcm'], default='wav')
    parser.add_argument('--text', default='The speech service is ready. This short sample measures synthesis speed and response latency.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.requests < 1: parser.error('--requests must be positive')
    # Warmup is excluded. Generated output duration varies between requests.
    measure(args.base_url, args.voice, args.text, args.mode == 'pcm', args.model)
    results = [measure(args.base_url, args.voice, args.text, args.mode == 'pcm', args.model) for _ in range(args.requests)]
    report = {'mode':args.mode, 'model':args.model, 'voice':args.voice, 'text':args.text, 'samples':results,
              'median_rtf':statistics.median(r['rtf'] for r in results)}
    if args.mode == 'pcm':
        report['median_first_audio_seconds'] = statistics.median(r['first_audio_seconds'] for r in results)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output: args.output.write_text(rendered + '\n')


if __name__ == '__main__': main()
