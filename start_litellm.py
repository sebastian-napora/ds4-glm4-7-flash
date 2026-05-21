#!/usr/bin/env python3
"""Start the GLM LiteLLM proxy in a detached subprocess with local logging."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "litellm_proxy.log"

LOG_DIR.mkdir(exist_ok=True)

python = ROOT / "venv" / "bin" / "python"
if not python.exists():
    python = Path(sys.executable)

print(f"Starting LiteLLM proxy, logging to {LOG_FILE}")

with LOG_FILE.open("w") as f:
    f.write(f"=== Starting LiteLLM at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

log = LOG_FILE.open("a")
proc = subprocess.Popen(
    [str(python), str(ROOT / "server_compress.py")],
    cwd=str(ROOT),
    stdin=open(os.devnull, "r"),
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
    env=os.environ.copy(),
)

print(f"Started with PID: {proc.pid}")

for i in range(10):
    time.sleep(5)
    if proc.poll() is not None:
        print(f"Process died with exit code: {proc.returncode}")
        break
    lines = LOG_FILE.read_text(errors="replace").splitlines()
    print(f"[{i + 1}/10] Still running. Last 5 lines:")
    for line in lines[-5:]:
        print(f"  {line}")
