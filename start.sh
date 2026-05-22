#!/bin/bash
#
# Start the custom native GLM backend, with optional LiteLLM proxy for
# VS Code / GitHub Copilot plugin compatibility.
#
# This script does not use llama.cpp, vLLM, SGLang, Ollama, or any other model
# inference engine. LiteLLM is only a protocol proxy.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${GLM_MODEL:-models/GLM-4.7-Flash-Q8_0.gguf}"
HOST="${GLM_HOST:-0.0.0.0}"
PORT="${GLM_PORT:-11112}"
LITELLM_HOST="${LITE_LLM_PROXY_HOST:-0.0.0.0}"
LITELLM_PORT="${LITE_LLM_PROXY_PORT:-11111}"

usage() {
    cat <<EOF
Usage: $0 [both|server|proxy|inspect|summary]

Modes:
  both      Start native backend and LiteLLM proxy.
  server    Start native backend only.
  proxy     Start LiteLLM proxy only; backend must already be running.
  inspect   Load model and print native summary.

Environment:
  GLM_MODEL             Local GGUF path. Default: models/GLM-4.7-Flash-Q8_0.gguf
  GLM_HOST              Native backend bind host. Default: 0.0.0.0
  GLM_PORT              Native backend port. Default: 11112
  LITE_LLM_PROXY_HOST   LiteLLM bind host. Default: 0.0.0.0
  LITE_LLM_PROXY_PORT   LiteLLM port. Default: 11111

Notes:
  The backend is a custom native engine skeleton. It loads the GLM GGUF directly.
  Generation is not implemented yet.
EOF
}

if [ "${1:-both}" = "--help" ] || [ "${1:-both}" = "-h" ]; then
    usage
    exit 0
fi

if [ "${1:-both}" != "proxy" ] && [ ! -f "$MODEL" ]; then
    echo "Model not found: $MODEL" >&2
    echo "Download the native target first:" >&2
    echo "  ./download_model.sh Q8_0" >&2
    exit 1
fi

if [ -d "$SCRIPT_DIR/venv" ]; then
    VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
else
    VENV_PYTHON="${PYTHON_BIN:-python3}"
fi

mkdir -p "$SCRIPT_DIR/logs"

start_backend() {
    make -C native
    echo "Starting glm-native-server on ${HOST}:${PORT}"
    "$SCRIPT_DIR/native/bin/glm-native-server" -m "$MODEL" --host "$HOST" --port "$PORT" \
        2>&1 | tee "$SCRIPT_DIR/logs/glm-native-server.log" &
    echo "$!" > "$SCRIPT_DIR/logs/glm-native-server.pid"
    echo "Native backend PID: $!"
}

start_proxy() {
    if ! "$VENV_PYTHON" -c 'import litellm' >/dev/null 2>&1; then
        echo "LiteLLM is not installed. Run ./setup-venv.sh first." >&2
        exit 1
    fi
    export LITE_LLM_PROXY_HOST="$LITELLM_HOST"
    export LITE_LLM_PROXY_PORT="$LITELLM_PORT"
    echo "Starting LiteLLM proxy on ${LITELLM_HOST}:${LITELLM_PORT}"
    "$VENV_PYTHON" "$SCRIPT_DIR/server_compress.py" \
        2>&1 | tee "$SCRIPT_DIR/logs/litellm_proxy.log" &
    echo "$!" > "$SCRIPT_DIR/logs/litellm_proxy.pid"
    echo "LiteLLM PID: $!"
}

case "${1:-both}" in
    both)
        start_backend
        sleep 2
        start_proxy
        echo
        echo "Services:"
        echo "  Native backend: http://${HOST}:${PORT}"
        echo "  LiteLLM proxy:  http://${LITELLM_HOST}:${LITELLM_PORT}/v1"
        ;;
    server)
        make -C native
        exec "$SCRIPT_DIR/native/bin/glm-native-server" -m "$MODEL" --host "$HOST" --port "$PORT"
        ;;
    proxy)
        start_proxy
        ;;
    inspect|summary)
        make -C native
        exec "$SCRIPT_DIR/native/bin/glm-native" -m "$MODEL" --summary
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
