#!/bin/bash
#
# Start llama.cpp + LiteLLM together in one foreground terminal session.
# Both processes log directly to this terminal. Ctrl+C stops both.
#
# Architecture:
#   VS Code / GitHub Copilot plugin -> LiteLLM (12111) -> llama-server (12112)
#
# No XML proxy — llama.cpp b9282 natively handles tools/tool_calls.
# Default context is 65,536 tokens.
# Optional speculative decoding with: LLAMA_SPEC_TYPE=ngram-cache

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LLAMA_HOST="${GLM_HOST:-0.0.0.0}"
LLAMA_PORT="${GLM_PORT:-12112}"
LITELLM_HOST="${LITE_LLM_PROXY_HOST:-0.0.0.0}"
LITELLM_PORT="${LITE_LLM_PROXY_PORT:-12111}"
LLAMA_CTX="${LLAMA_CTX:-200072}"
LLAMA_NGL="${LLAMA_NGL:-999}"
LLAMA_PARALLEL="${LLAMA_PARALLEL:-1}"
LLAMA_REASONING_BUDGET="${LLAMA_REASONING_BUDGET:-0}"
LLAMA_SPEC_TYPE="${LLAMA_SPEC_TYPE:-none}"

usage() {
    cat <<EOF
Usage: $0

Starts both services in the current terminal:
  1. llama-server on \${GLM_HOST:-0.0.0.0}:\${GLM_PORT:-12112}
  2. LiteLLM proxy on \${LITE_LLM_PROXY_HOST:-0.0.0.0}:\${LITE_LLM_PROXY_PORT:-12111}

Environment:
  GLM_HOST              llama-server bind host. Default: 0.0.0.0
  GLM_PORT              llama-server port. Default: 12112
  LITE_LLM_PROXY_HOST   LiteLLM bind host. Default: 0.0.0.0
  LITE_LLM_PROXY_PORT   LiteLLM port. Default: 12111
  LLAMA_CTX             Context size. Default: 131072
  LLAMA_NGL             GPU layers to offload. Default: 999
  LLAMA_THREADS         CPU threads for llama.cpp. Default: nproc
  LLAMA_PARALLEL        Parallel request slots. Default: 1
  LLAMA_REASONING_BUDGET Extended-thinking token budget (0 = disabled). Default: 0
  LLAMA_FLASH_ATTN      on|off|auto. Default: auto
  LLAMA_SPEC_TYPE       Speculative decoding mode. Default: none
  LLAMA_SPEC_DRAFT_MODEL Draft model path for draft-based speculation
  LLAMA_SPEC_DRAFT_N_MAX Max drafted tokens. Default: 3
  LLAMA_SERVER_EXTRA_ARGS Extra raw args appended to llama-server
  GLM_MODEL             GGUF path. Default: models/GLM-4.7-Flash-Q6_K.gguf
  GLM_GGUF_QUANT        Quantization variant (appended to model name)
  PYTHON_BIN            Override python if venv/bin/python3 is unavailable
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

# ── Detect Python ─────────────────────────────────────────────────────────────
if [ -d "$SCRIPT_DIR/venv" ]; then
    VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
elif [ -d "$HOME/ai-projects/lunch-model/venv" ]; then
    VENV_PYTHON="$HOME/ai-projects/lunch-model/venv/bin/python3"
else
    VENV_PYTHON="${PYTHON_BIN:-python3}"
fi

# ── Validate dependencies ──────────────────────────────────────────────────────
if [ ! -x "$SCRIPT_DIR/third_party/llama.cpp/build/bin/llama-server" ]; then
    echo "llama-server not found: $SCRIPT_DIR/third_party/llama.cpp/build/bin/llama-server" >&2
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/lite_llm_config_llamacpp_direct.yaml" ]; then
    echo "LiteLLM config not found: $SCRIPT_DIR/lite_llm_config_llamacpp_direct.yaml" >&2
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/server_compress_llamacpp_direct.py" ]; then
    echo "LiteLLM wrapper not found: $SCRIPT_DIR/server_compress_llamacpp_direct.py" >&2
    exit 1
fi

if ! "$VENV_PYTHON" -c 'import litellm' >/dev/null 2>&1; then
    echo "LiteLLM is not installed for $VENV_PYTHON" >&2
    echo "Run ./setup-venv.sh first." >&2
    exit 1
fi

# ── Resolve model path ─────────────────────────────────────────────────────────
QUANT="${GLM_GGUF_QUANT:-Q6_K}"
GGUF_PATH="${GLM_MODEL:-}"
if [ -z "$GGUF_PATH" ]; then
    GGUF_PATH="$SCRIPT_DIR/models/GLM-4.7-Flash-$QUANT.gguf"
fi
if [ ! -f "$GGUF_PATH" ]; then
    echo "Model not found: $GGUF_PATH" >&2
    echo "Available:" >&2
    ls -1 "$SCRIPT_DIR/models"/*.gguf 2>/dev/null || echo "  (none)" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1

# ── Cleanup handler ────────────────────────────────────────────────────────────
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

# ── Print banner ────────────────────────────────────────────────────────────────
echo
echo "┌─────────────────────────────────────────────────────┐"
echo "│  llama.cpp + LiteLLM  GLM-4.7-Flash ($QUANT)         │"
echo "├─────────────────────────────────────────────────────┤"
printf "│  Model:      %-37s │\n" "$(basename "$GGUF_PATH")"
printf "│  llama.cpp:  http://%s:%s                  │\n" "$LLAMA_HOST" "$LLAMA_PORT"
printf "│  LiteLLM:    http://%s:%s/v1               │\n" "$LITELLM_HOST" "$LITELLM_PORT"
printf "│  Context:    %-37s │\n" "$LLAMA_CTX (extended thinking: ${LLAMA_REASONING_BUDGET})"
printf "│  GPU layers: %-37s │\n" "$LLAMA_NGL"
printf "│  Spec:       %-37s │\n" "${LLAMA_SPEC_TYPE:-none}"
printf "│  Config:     %-37s │\n" "lite_llm_config_llamacpp_direct.yaml"
printf "│  Transform:  %-37s │\n" "XML → OpenAI tool_calls (port 12180)"
echo "└─────────────────────────────────────────────────────┘"
echo

# ── Start llama-server ──────────────────────────────────────────────────────────
echo "Starting llama-server (port $LLAMA_PORT)..."
"$SCRIPT_DIR/third_party/llama.cpp/build/bin/llama-server" \
    -m "$GGUF_PATH" \
    --host "$LLAMA_HOST" \
    --port "$LLAMA_PORT" \
    -ngl "$LLAMA_NGL" \
    -c "$LLAMA_CTX" \
    -t "${LLAMA_THREADS:-$(nproc)}" \
    --parallel "$LLAMA_PARALLEL" \
    --reasoning-budget "$LLAMA_REASONING_BUDGET" \
    --jinja \
    --alias glm-4.7-flash-llamacpp \
    ${LLAMA_SPEC_TYPE:+--spec-type "$LLAMA_SPEC_TYPE"} \
    ${LLAMA_SPEC_DRAFT_MODEL:+--spec-draft-model "$LLAMA_SPEC_DRAFT_MODEL" --spec-draft-n-max "${LLAMA_SPEC_DRAFT_N_MAX:-3}"} \
    ${LLAMA_SERVER_EXTRA_ARGS:+ $LLAMA_SERVER_EXTRA_ARGS} \
    &
LLAMA_PID=$!

# Wait for llama-server to become ready
for _ in $(seq 1 60); do
    if curl -fsS "http://0.0.0.0:${LLAMA_PORT}/v1/models" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$LLAMA_PID" >/dev/null 2>&1; then
        wait "$LLAMA_PID"
        exit 1
    fi
    sleep 1
done

if ! curl -fsS "http://0.0.0.0:${LLAMA_PORT}/v1/models" >/dev/null 2>&1; then
    echo "llama-server did not become ready on port ${LLAMA_PORT}" >&2
    exit 1
fi

echo "✅ llama-server ready on port $LLAMA_PORT"

# ── Start LiteLLM (which internally starts XML proxy on 12180) ─────────────────
echo "Starting LiteLLM + XML transform proxy..."
echo "  Architecture:"
echo "    Copilot → LiteLLM (12111) → XML proxy (12180) → llama-server ($LLAMA_PORT)"
echo ""
LITE_LLM_PROXY_HOST="$LITELLM_HOST" \
LITE_LLM_PROXY_PORT="$LITELLM_PORT" \
LLAMA_BACKEND_PORT="$LLAMA_PORT" \
    "$VENV_PYTHON" "$SCRIPT_DIR/server_compress_llamacpp_direct.py" \
    &
LITELLM_PID=$!

# Wait for LiteLLM to become ready (XML proxy also starts inside that process)
for _ in $(seq 1 60); do
    if curl -fsS "http://0.0.0.0:${LITELLM_PORT}/v1/models" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$LITELLM_PID" >/dev/null 2>&1; then
        wait "$LITELLM_PID"
        exit 1
    fi
    sleep 1
done

if ! curl -fsS "http://0.0.0.0:${LITELLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "LiteLLM did not become ready on port ${LITELLM_PORT}" >&2
    exit 1
fi

echo "✅ LiteLLM proxy ready on port $LITELLM_PORT"
echo "✅ XML transform proxy ready on port 12180"
echo
echo "┌─────────────────────────────────────────────────────┐"
echo "│  Ready — configure Copilot LLM Gateway:            │"
echo "├─────────────────────────────────────────────────────┤"
printf "│  Endpoint: http://localhost:%s/v1/chat/completions │\n" "$LITELLM_PORT"
printf "│  Model:    %-37s │\n" "glm-4.7-flash-llamacpp"
echo "└─────────────────────────────────────────────────────┘"
echo
echo "Press Ctrl+C to stop both."
echo

# ── Wait for either process (logs stream to terminal) ───────────────────────────
wait -n "$LLAMA_PID" "$LITELLM_PID"
