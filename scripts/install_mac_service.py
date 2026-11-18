"""Install the local Apple Silicon Qwen TTS API as a launchd login service."""
import argparse
import os
from pathlib import Path
import platform
import plistlib
import subprocess
import sys
from urllib.parse import urlsplit

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
LABEL = "com.local.qwen3-tts-mlx"
CUSTOM = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
CLONE = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
REVISIONS = {
    CUSTOM: "41d3337e8b7f2843a75841595fc14e4b9a7a4b96",
    CLONE: "e7dd0585652209fa0d7783659aad4e8a324de11c",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18882)
    parser.add_argument("--library-dir", type=Path,
                        default=Path.home() / "Library/Application Support/Qwen3-TTS/voices")
    parser.add_argument("--tailscale-origin", help="HTTPS origin reported by Tailscale Serve")
    parser.add_argument("--tailscale-login", help="Tailscale login allowed to use the studio")
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("This launchd installer requires Apple Silicon macOS.")
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535.")
    if bool(args.tailscale_origin) != bool(args.tailscale_login):
        parser.error("--tailscale-origin and --tailscale-login must be supplied together.")
    if args.tailscale_origin:
        try:
            origin = urlsplit(args.tailscale_origin.rstrip("/"))
            valid_origin = (
                origin.scheme == "https" and (origin.hostname or "").endswith(".ts.net")
                and origin.username is None and not origin.path and not origin.query
                and not origin.fragment and (origin.port is None or origin.port > 0)
                and args.tailscale_origin.rstrip("/") == f"https://{origin.netloc}"
            )
        except ValueError:
            valid_origin = False
        if not valid_origin or not args.tailscale_login.strip():
            parser.error("Use a Tailscale HTTPS origin with no path and a nonempty login.")
    models = {name: snapshot_download(name, revision=revision)
              for name, revision in REVISIONS.items()}
    library = args.library_dir.expanduser().resolve()
    library.mkdir(parents=True, exist_ok=True, mode=0o700)
    logs = Path.home() / "Library/Logs/Qwen3-TTS"
    logs.mkdir(parents=True, exist_ok=True)
    agents = Path.home() / "Library/LaunchAgents"
    agents.mkdir(parents=True, exist_ok=True)
    path = agents / f"{LABEL}.plist"
    domain = f"gui/{os.getuid()}"
    service = f"{domain}/{LABEL}"
    env = {
        "HOST": "127.0.0.1", "PORT": str(args.port), "WORKERS": "1",
        "TTS_BACKEND": "mlx", "MLX_MODEL_ID": models[CUSTOM], "MLX_CLONE_MODEL_ID": models[CLONE],
        "HF_HUB_OFFLINE": "1", "TTS_LAZY_LOAD": "false", "TTS_WARMUP_ON_START": "true",
        "TTS_WARMUP_MAX_SECONDS": "60", "TTS_MAX_CONCURRENT": "1",
        "TTS_MIN_CHUNK_CHARS": "200", "TTS_MAX_CHUNK_CHARS": "600",
        "VOICE_LIBRARY_DIR": str(library), "ENABLE_LOCAL_VOICE_STUDIO": "true",
        "PYTHONUNBUFFERED": "1", "CORS_ORIGINS": f"http://127.0.0.1:{args.port},http://localhost:{args.port}",
        "PATH": "/opt/homebrew/bin:/usr/bin:/bin",
    }
    if args.tailscale_origin:
        env["VOICE_STUDIO_TAILSCALE_ORIGIN"] = f"https://{origin.netloc.lower()}"
        env["VOICE_STUDIO_TAILSCALE_LOGIN"] = args.tailscale_login.strip()
    payload = {
        "Label": LABEL, "ProgramArguments": [sys.executable, "-m", "api.main"],
        "WorkingDirectory": str(ROOT), "EnvironmentVariables": env,
        "RunAtLoad": True, "KeepAlive": {"SuccessfulExit": False}, "ThrottleInterval": 10,
        "StandardOutPath": str(logs / "api.log"), "StandardErrorPath": str(logs / "api.log"),
    }
    if subprocess.run(["launchctl", "print", service], capture_output=True).returncode == 0:
        subprocess.run(["launchctl", "bootout", service], check=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(plistlib.dumps(payload))
    temporary.replace(path)
    subprocess.run(["launchctl", "enable", service], check=True)
    subprocess.run(["launchctl", "bootstrap", domain, str(path)], check=True)
    print(f"Starting Qwen Voice Studio at http://127.0.0.1:{args.port}/studio")


if __name__ == "__main__":
    main()
