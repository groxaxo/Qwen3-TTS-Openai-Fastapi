"""Supervise the public API and private CUDA-graph worker without loading GPU code."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def stop_process(process: subprocess.Popen, timeout: float = 20) -> None:
    """Stop the owned process group, including any inference subprocesses."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run(worker_python: str, startup_timeout: float) -> int:
    env = dict(os.environ)
    backend = env.setdefault("TTS_BACKEND", "cuda_graphs").strip().lower()
    if backend != "cuda_graphs":
        os.chdir(ROOT)
        os.execvpe(sys.executable, [sys.executable, "-m", "api.main"], env)

    if not Path(worker_python).is_file():
        raise ValueError("Worker Python not found. Run ./scripts/install.sh or set TTS_GRAPH_ENV.")
    if env.get("WORKERS", "1") != "1" or env.get("TTS_MAX_CONCURRENT", "1") != "1":
        raise ValueError("cuda_graphs requires WORKERS=1 and TTS_MAX_CONCURRENT=1")
    port = int(env.get("TTS_GRAPH_PORT", "18881"))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    model = env.get("TTS_MODEL_NAME") or env.get("TTS_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    if Path(model).is_dir():
        model = str(Path(model).resolve())
    env.update(TTS_MODEL_ID=model, TTS_MODEL_NAME=model, TTS_GRAPH_PORT=str(port),
               TTS_GRAPH_WORKER_URL=f"http://127.0.0.1:{port}")
    for key, value in {
        "HOST": "0.0.0.0", "PORT": "8880", "WORKERS": "1", "TTS_MAX_CONCURRENT": "1",
        "TTS_LAZY_LOAD": "false", "TTS_WARMUP_ON_START": "false",
        "TTS_MIN_CHUNK_CHARS": "200", "TTS_MAX_CHUNK_CHARS": "600",
        "OMP_NUM_THREADS": "8", "TOKENIZERS_PARALLELISM": "false", "PYTHONNOUSERSITE": "1",
    }.items():
        env.setdefault(key, value)
    worker_env = dict(env)
    worker_env.pop("PYTHONPATH", None)
    frontend_env = dict(env, CUDA_VISIBLE_DEVICES="")
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    children = []
    try:
        worker = subprocess.Popen(
            [worker_python, str(ROOT / "scripts/qwen_graph_worker.py")],
            cwd=tempfile.gettempdir(), env=worker_env, start_new_session=True,
        )
        children.append(worker)
        deadline = time.monotonic() + startup_timeout
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        print(f"Waiting for CUDA-graph warmup on 127.0.0.1:{port}...", flush=True)
        while not stopped.is_set():
            if worker.poll() is not None:
                raise RuntimeError(f"Graph worker exited during startup ({worker.returncode})")
            try:
                with opener.open(env["TTS_GRAPH_WORKER_URL"] + "/health", timeout=1) as response:
                    if json.load(response).get("ready"):
                        break
            except (OSError, ValueError, urllib.error.URLError):
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError("Graph worker startup timed out; inspect the worker log above")
            stopped.wait(.2)
        if stopped.is_set():
            return 0
        frontend = subprocess.Popen([sys.executable, "-m", "api.main"], cwd=ROOT,
                                    env=frontend_env, start_new_session=True)
        children.append(frontend)
        while not stopped.wait(.2):
            for child in children:
                if child.poll() is not None:
                    return child.returncode or (1 if child is worker else 0)
        return 0
    finally:
        for child in reversed(children):
            stop_process(child)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-python", default=str(Path(os.getenv("TTS_GRAPH_ENV", "/missing")) / "bin/python"))
    parser.add_argument("--startup-timeout", type=float, default=float(os.getenv("TTS_STARTUP_TIMEOUT", "900")))
    args = parser.parse_args()
    try:
        return run(args.worker_python, args.startup_timeout)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"TTS startup failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
