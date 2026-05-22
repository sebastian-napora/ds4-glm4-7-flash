#!/bin/bash
#
# Launch GLM-4.7-Flash GGUF via the custom native GLM engine server.
#
# Architecture:
#   Copilot -> LiteLLM (11111) -> glm-native-server (11112)
#
# Usage:
#   ./glm_llama_server.sh              # defaults: Q8_0, models/GLM-4.7-Flash-Q8_0.gguf
#   GLM_GGUF_QUANT=Q4_K_XL ./glm_llama_server.sh
#   GLM_MODEL=/path/to/file.gguf ./glm_llama_server.sh
#
# Environment:
#   GLM_GGUF_QUANT         Quantization variant (default: Q8_0)
#   GLM_MODEL              Path to GGUF file (default: models/GLM-4.7-Flash-<QUANT>.gguf)
#   GLM_HOST               Bind host (default: 0.0.0.0)
#   GLM_PORT               Server port (default: 11112)
#   LITE_LLM_PROXY_HOST    LiteLLM bind host (default: 0.0.0.0)
#   LITE_LLM_PROXY_PORT    LiteLLM port (default: 11111)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ────────────────────────────────────────────────────────────────────
QUANT="${GLM_GGUF_QUANT:-Q6_K}"
MODEL_DIR="${GLM_MODEL_DIR:-models}"

# Resolve model path
MODEL="${GLM_MODEL:-}"
if [ -z "$MODEL" ]; then
    case "$QUANT" in
        GLM-4.7-Flash-*.gguf) FILE="$QUANT" ;;
        *) FILE="GLM-4.7-Flash-$QUANT.gguf" ;;
    esac
    MODEL="$MODEL_DIR/$FILE"
fi

# Server host/port
GLM_HOST="${GLM_HOST:-0.0.0.0}"
GLM_PORT="${GLM_PORT:-11112}"

# LiteLLM proxy defaults
LITE_LLM_PROXY_HOST="${LITE_LLM_PROXY_HOST:-0.0.0.0}"
LITE_LLM_PROXY_PORT="${LITE_LLM_PROXY_PORT:-11111}"

# ── Validate model ──────────────────────────────────────────────────────────────
if [ ! -s "$MODEL" ]; then
    echo "Model not found: $MODEL"
    echo
    echo "Download it first:"
    echo "  ./ensure-model.sh $QUANT"
    exit 1
fi

# ── Locate native server binary ─────────────────────────────────────────────────
NATIVE_DIR="$SCRIPT_DIR/native"
SERVER_BIN="$NATIVE_DIR/bin/glm-native-server"

if [ ! -x "$SERVER_BIN" ]; then
    echo "glm-native-server not found: $SERVER_BIN"
    echo
    echo "Building native engine..."
    make -C "$NATIVE_DIR"
    if [ ! -x "$SERVER_BIN" ]; then
        echo "Build failed. Check $NATIVE_DIR/Makefile."
        exit 1
    fi
fi

echo "Using native server: $SERVER_BIN"

mkdir -p "$SCRIPT_DIR/logs"

# ── Print config ────────────────────────────────────────────────────────────────
echo
echo "┌─────────────────────────────────────────────────────┐"
echo "│  GLM Native Engine  GLM-4.7-Flash ($QUANT)         │"
echo "├─────────────────────────────────────────────────────┤"
printf "│  Model:      %-37s │\n" "$MODEL"
printf "│  Endpoint:   http://%s:%s                  │\n" "$GLM_HOST" "$GLM_PORT"
printf "│  LiteLLM:    http://%s:%s/v1               │\n" "$LITE_LLM_PROXY_HOST" "$LITE_LLM_PROXY_PORT"
echo "└─────────────────────────────────────────────────────┘"
echo
echo "⏳ Starting glm-native-server…"
echo

# ── Start server ────────────────────────────────────────────────────────────────
exec "$SERVER_BIN" -m "$MODEL" --host "$GLM_HOST" --port "$GLM_PORT"