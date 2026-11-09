"""Local lifecycle manager. Run with the project's .venv Python."""
import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "platform"
RUN = ROOT / "runtime"

def environment():
    RUN.mkdir(exist_ok=True)
    config_file = RUN / "local-config.json"
    if not config_file.exists():
        config_file.write_text(json.dumps({"JWT_SECRET_KEY": secrets.token_urlsafe(48),
            "DEMO_PASSWORD": secrets.token_urlsafe(16)}, indent=2), encoding="utf-8")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    backend = RUN / "results"
    backend.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(config)
    env.setdefault("SQLALCHEMY_DATABASE_URL", "sqlite:///" + (RUN / "meereis.db").as_posix())
    env.setdefault("CELERY_BROKER_URL", "filesystem://")
    env.setdefault("CELERY_RESULT_BACKEND", backend.as_uri() + "/")
    env.setdefault("MEEREIS_RUNTIME_DIR", str(RUN))
    env.setdefault("API_BASE_URL", "http://127.0.0.1:8001")
    env["PYTHONUNBUFFERED"] = "1"
    return env, config

def start(check_only=False):
    env, config = environment()
    for port in (8001, 8501):
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise SystemExit(f"Port {port} already in use. Stop the existing service first.")
    commands = {
        "api": ["-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8001"],
        "worker": ["-m", "celery", "-A", "celery_worker.celery_app", "worker", "--pool=solo", "--loglevel=info"],
        "web": ["-m", "streamlit", "run", "web_gui.py", "--server.address", "127.0.0.1", "--server.port", "8501", "--server.headless", "true"],
    }
    processes = []
    try:
        for name, args in commands.items():
            with (RUN / f"{name}.log").open("a", encoding="utf-8") as log:
                processes.append(subprocess.Popen([sys.executable, *args], cwd=BASE, env=env,
                    stdout=log, stderr=log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)))
        for _ in range(90):
            if any(proc.poll() is not None for proc in processes):
                raise RuntimeError("A service exited; inspect runtime/*.log")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8501/_stcore/health", timeout=1):
                    with urllib.request.urlopen("http://127.0.0.1:8001/", timeout=1):
                        break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError("Startup timed out; inspect runtime/*.log")
        print("Ready: http://127.0.0.1:8501")
        print("Login: researcher; password is in runtime/local-config.json")
        print("Keep this window open. Ctrl+C stops all three services.")
        if check_only:
            return
        while True:
            if any(proc.poll() is not None for proc in processes):
                raise RuntimeError("A service stopped; inspect runtime/*.log")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping services...")
    finally:
        for proc in reversed(processes):
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["start"])
    parser.add_argument("--check", action="store_true", help="Start, check readiness, then stop")
    args = parser.parse_args()
    start(args.check)
