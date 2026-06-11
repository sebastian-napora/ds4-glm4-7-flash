#!/bin/bash
#
# Start llama.cpp + LiteLLM together in one foreground terminal session.
# Both processes log directly to this terminal. Ctrl+C stops both.
#
# Default topology:
#   LiteLLM:  http://0.0.0.0:12111/v1
#   llama.cpp: http://0.0.0.0:12112/v1
#
# Tool calling is enabled through the llama.cpp OpenAI-compatible endpoint and
# advertised in lite_llm_config_llamacpp.yaml. Default context is 65,536 tokens.
# Optional speculative decoding can be enabled with LLAMA_SPEC_TYPE=ngram-cache.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LLAMA_HOST="${GLM_HOST:-0.0.0.0}"
LLAMA_PORT="${GLM_PORT:-12112}"
LITELLM_HOST="${LITE_LLM_PROXY_HOST:-0.0.0.0}"
LITELLM_PORT="${LITE_LLM_PROXY_PORT:-12111}"
LLAMA_CTX="${LLAMA_CTX:-130000}"
LLAMA_SPEC_TYPE="${LLAMA_SPEC_TYPE:-none}"
usage() {
    cat <<EOF
Usage: $0

Starts both services in the current terminal:
  1. llama.cpp server on \${GLM_HOST:-0.0.0.0}:\${GLM_PORT:-12112}
  2. LiteLLM proxy on \${LITE_LLM_PROXY_HOST:-0.0.0.0}:\${LITE_LLM_PROXY_PORT:-12111}

Environment:
  GLM_HOST              llama.cpp bind host. Default: 0.0.0.0
  GLM_PORT              llama.cpp port. Default: 12112
  LITE_LLM_PROXY_HOST   LiteLLM bind host. Default: 0.0.0.0
  LITE_LLM_PROXY_PORT   LiteLLM port. Default: 12111
  LLAMA_CTX             Context size. Default: 130000
  LLAMA_NGL             GPU layers to offload. Default: 999
  LLAMA_THREADS         CPU threads for llama.cpp. Default: nproc
  LLAMA_PARALLEL        Parallel request slots. Default: 1
  LLAMA_FLASH_ATTN      on|off|auto. Default: auto
  LLAMA_SPEC_TYPE       Speculative decoding mode. Default: none
  LLAMA_SPEC_DRAFT_MODEL Draft model path for draft-based speculation
  LLAMA_SPEC_DRAFT_N_MAX Max drafted tokens. Default: 3
  LLAMA_SERVER_EXTRA_ARGS Extra raw args appended to llama-server
  GLM_MODEL             GGUF path. Default: models/GLM-4.7-Flash-Q6_K.gguf
  PYTHON_BIN            Override python if venv/bin/python3 is unavailable
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

if [ -d "$SCRIPT_DIR/venv" ]; then
    VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
else
    VENV_PYTHON="${PYTHON_BIN:-python3}"
fi

if [ ! -x "$SCRIPT_DIR/third_party/llama.cpp/build/bin/llama-server" ]; then
    echo "llama-server not built: $SCRIPT_DIR/third_party/llama.cpp/build/bin/llama-server" >&2
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/lite_llm_config_llamacpp.yaml" ]; then
    echo "LiteLLM config not found: $SCRIPT_DIR/lite_llm_config_llamacpp.yaml" >&2
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/server_compress_llamacpp.py" ]; then
    echo "LiteLLM llama.cpp wrapper not found: $SCRIPT_DIR/server_compress_llamacpp.py" >&2
    exit 1
fi

if ! "$VENV_PYTHON" -c 'import litellm' >/dev/null 2>&1; then
    echo "LiteLLM is not installed for $VENV_PYTHON" >&2
    echo "Run ./setup-venv.sh first." >&2
    exit 1
fi

export PYTHONUNBUFFERED=1

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    echo
    echo "Stopping stack..."
    if [ -n "${LITELLM_PID:-}" ] && kill -0 "$LITELLM_PID" >/dev/null 2>&1; then
        kill "$LITELLM_PID" >/dev/null 2>&1 || true
    fi
    if [ -n "${LLAMA_PID:-}" ] && kill -0 "$LLAMA_PID" >/dev/null 2>&1; then
        kill "$LLAMA_PID" >/dev/null 2>&1 || true
    fi
    wait "${LITELLM_PID:-}" "${LLAMA_PID:-}" 2>/dev/null || true
    exit "$rc"
}

trap cleanup EXIT INT TERM

echo
echo "┌─────────────────────────────────────────────────────┐"
echo "│  llama.cpp + LiteLLM  GLM-4.7-Flash (Q6_K)         │"
echo "├─────────────────────────────────────────────────────┤"
printf "│  llama.cpp:  http://%s:%s                  │\n" "$LLAMA_HOST" "$LLAMA_PORT"
printf "│  LiteLLM:    http://%s:%s/v1               │\n" "$LITELLM_HOST" "$LITELLM_PORT"
printf "│  Context:    %-37s │\n" "$LLAMA_CTX"
printf "│  Spec:       %-37s │\n" "$LLAMA_SPEC_TYPE"
printf "│  Config:     %-37s │\n" "lite_llm_config_llamacpp.yaml"
echo "└─────────────────────────────────────────────────────┘"
echo

echo "Starting llama.cpp..."
GLM_HOST="$LLAMA_HOST" GLM_PORT="$LLAMA_PORT" LLAMA_CTX="$LLAMA_CTX" \
LLAMA_SPEC_TYPE="$LLAMA_SPEC_TYPE" \
    "$SCRIPT_DIR/llamacpp_server.sh" &
LLAMA_PID=$!

for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${LLAMA_PORT}/v1/models" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$LLAMA_PID" >/dev/null 2>&1; then
        wait "$LLAMA_PID"
        exit 1
    fi
    sleep 1
done

if ! curl -fsS "http://127.0.0.1:${LLAMA_PORT}/v1/models" >/dev/null 2>&1; then
    echo "llama.cpp did not become ready on port ${LLAMA_PORT}" >&2
    exit 1
fi

echo "Starting LiteLLM..."
LLAMA_BACKEND_PORT="$LLAMA_PORT" \
LITELLM_BACKEND_API_BASE="http://127.0.0.1:${LLAMA_PORT}/v1" \
LITE_LLM_PROXY_HOST="$LITELLM_HOST" \
LITE_LLM_PROXY_PORT="$LITELLM_PORT" \
    "$VENV_PYTHON" "$SCRIPT_DIR/server_compress_llamacpp.py" &
LITELLM_PID=$!

for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:${LITELLM_PORT}/v1/models" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$LITELLM_PID" >/dev/null 2>&1; then
        wait "$LITELLM_PID"
        exit 1
    fi
    sleep 1
done

if ! curl -fsS "http://127.0.0.1:${LITELLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "LiteLLM did not become ready on port ${LITELLM_PORT}" >&2
    exit 1
fi

echo
echo "Ready:"
echo "  llama.cpp: http://0.0.0.0:${LLAMA_PORT}/v1/models"
echo "  LiteLLM:   http://0.0.0.0:${LITELLM_PORT}/v1/models"
echo "Press Ctrl+C to stop both."
echo

wait -n "$LLAMA_PID" "$LITELLM_PID"
