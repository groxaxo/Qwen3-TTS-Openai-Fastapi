"""A disconnected stream must release the single MLX inference worker."""
from pathlib import Path
import subprocess
import sys


def test_close_stream_with_backpressure_allows_next_request():
    # A separate process bounds the old deadlock, whose blocked non-daemon
    # executor thread otherwise prevents the test runner itself from exiting.
    script = '''
import asyncio
from types import SimpleNamespace
import numpy as np
from api.backends.mlx_qwen3_tts import MLXQwen3TTSBackend

class Model:
    sample_rate = 24000
    def generate(self, **kwargs):
        try:
            for _ in range(100):
                yield SimpleNamespace(audio=np.ones(240, dtype=np.float32))
        finally:
            self.closed = True
    def generate_custom_voice(self, **kwargs):
        yield SimpleNamespace(audio=np.ones(240, dtype=np.float32))

async def check():
    backend = MLXQwen3TTSBackend()
    backend.model = Model()
    backend._ready = True
    stream = backend.generate_speech_streaming('A long stream', 'Ryan')
    await anext(stream)
    await asyncio.sleep(.05)  # Fill the bounded queue while the client is idle.
    await asyncio.wait_for(stream.aclose(), timeout=1)
    assert backend.model.closed
    audio, rate = await asyncio.wait_for(backend.generate_speech('Next request', 'Ryan'), timeout=1)
    assert len(audio) == 240 and rate == 24000
    backend._executor.shutdown(wait=True)

asyncio.run(check())
print('Recovered after disconnect')
'''
    result = subprocess.run([sys.executable, '-c', script],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=4)
    assert result.returncode == 0, result.stderr
    assert 'Recovered after disconnect' in result.stdout
