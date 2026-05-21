#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

if [ -x "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/venv/bin/python"
else
    PYTHON="${PYTHON_BIN:-python3}"
fi

exec "$PYTHON" "$SCRIPT_DIR/glm_server.py" 2>&1 | tee -a "$SCRIPT_DIR/logs/vllm_backend.log"
