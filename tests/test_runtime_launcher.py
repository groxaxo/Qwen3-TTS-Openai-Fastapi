"""Exercise the real supervisor with lightweight HTTP/process stand-ins."""
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER = '''
import json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
state = Path(os.environ['STUB_STATE'])
(state/'worker.json').write_text(json.dumps({'pid':os.getpid(), 'cwd':os.getcwd(),
    'gpu':os.getenv('CUDA_VISIBLE_DEVICES'), 'pythonpath':os.getenv('PYTHONPATH')}))
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(json.dumps({'ready':not os.getenv('STUB_NOT_READY')}).encode())
    def log_message(self, *args): pass
server = HTTPServer(('127.0.0.1', int(os.environ['TTS_GRAPH_PORT'])), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
while True:
    if os.getenv('STUB_WORKER_FAILURE') and (state/'frontend.json').exists(): sys.exit(7)
    time.sleep(.02)
'''
FRONTEND = '''
import json, os, sys, time
from pathlib import Path
Path(os.environ['STUB_STATE'],'frontend.json').write_text(json.dumps({
    'pid':os.getpid(), 'gpu':os.getenv('CUDA_VISIBLE_DEVICES'),
    'backend':os.getenv('TTS_BACKEND'), 'model':os.getenv('TTS_MODEL_ID'),
    'host':os.getenv('HOST'), 'worker_url':os.getenv('TTS_GRAPH_WORKER_URL')}))
if os.getenv('STUB_FRONTEND_FAILURE'): sys.exit(5)
while True: time.sleep(.1)
'''


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@pytest.fixture
def runtime(tmp_path):
    (tmp_path/'scripts').mkdir()
    (tmp_path/'api').mkdir()
    shutil.copy(ROOT/'scripts/serve.py', tmp_path/'scripts/serve.py')
    (tmp_path/'scripts/qwen_graph_worker.py').write_text(WORKER)
    (tmp_path/'api/__init__.py').touch()
    (tmp_path/'api/main.py').write_text(FRONTEND)
    env = {k:v for k,v in os.environ.items() if not k.startswith(('TTS_', 'STUB_'))}
    env.update(STUB_STATE=str(tmp_path), TTS_GRAPH_PORT=str(free_port()),
               CUDA_VISIBLE_DEVICES='2', PYTHONPATH='/deliberately/wrong', WORKERS='1')
    processes = []

    def launch(**overrides):
        process = subprocess.Popen([
            sys.executable, str(tmp_path/'scripts/serve.py'), '--worker-python', sys.executable,
            '--startup-timeout', '3',
        ], env=dict(env, **overrides), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        processes.append(process)
        return process

    yield tmp_path, launch
    for process in processes:
        if process.poll() is None: process.terminate()
        process.communicate(timeout=10)


def wait_file(path, process):
    deadline = time.monotonic() + 5
    while not path.exists():
        assert process.poll() is None, process.communicate()[0]
        assert time.monotonic() < deadline
        time.sleep(.02)
    return json.loads(path.read_text())


def assert_stopped(state):
    for name in ('worker.json', 'frontend.json'):
        if (state/name).exists():
            pid = json.loads((state/name).read_text())['pid']
            with pytest.raises(ProcessLookupError): os.kill(pid, 0)


def test_default_isolation_and_sigterm_cleanup(runtime):
    state, launch = runtime
    process = launch()
    frontend = wait_file(state/'frontend.json', process)
    worker = json.loads((state/'worker.json').read_text())
    assert frontend['backend'] == 'cuda_graphs'
    assert frontend['gpu'] == ''
    assert frontend['host'] == '0.0.0.0'
    assert frontend['model'] == 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'
    assert worker['gpu'] == '2'
    assert worker['pythonpath'] is None
    assert Path(worker['cwd']) != state
    process.send_signal(signal.SIGTERM)
    process.communicate(timeout=10)
    assert process.returncode == 0
    assert_stopped(state)


@pytest.mark.parametrize('failure,code', [('STUB_WORKER_FAILURE',7), ('STUB_FRONTEND_FAILURE',5)])
def test_either_child_failure_stops_the_other(runtime, failure, code):
    state, launch = runtime
    process = launch(**{failure:'1'})
    process.communicate(timeout=10)
    assert process.returncode == code
    assert_stopped(state)


def test_startup_timeout_releases_worker(runtime):
    state, launch = runtime
    process = launch(STUB_NOT_READY='1')
    output = process.communicate(timeout=10)[0]
    assert process.returncode == 1
    assert 'startup timed out' in output
    assert not (state/'frontend.json').exists()
    assert_stopped(state)


def test_occupied_worker_port_is_rejected_without_spawning(runtime):
    state, launch = runtime
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        process = launch(TTS_GRAPH_PORT=str(sock.getsockname()[1]))
        output = process.communicate(timeout=5)[0]
    assert process.returncode == 1
    assert 'Address already in use' in output
    assert not (state/'worker.json').exists()


def test_multiple_workers_are_rejected_before_loading_model(runtime):
    state, launch = runtime
    process = launch(WORKERS='2')
    assert 'WORKERS=1' in process.communicate(timeout=5)[0]
    assert process.returncode == 1
    assert not (state/'worker.json').exists()


def test_installer_rejects_shared_environment(tmp_path):
    env = dict(os.environ, TTS_FRONTEND_ENV=str(tmp_path), TTS_GRAPH_ENV=str(tmp_path))
    process = subprocess.run(['bash', str(ROOT/'scripts/install.sh')], env=env,
                             capture_output=True, text=True, timeout=10)
    assert process.returncode == 1
    assert 'different environments' in process.stderr
