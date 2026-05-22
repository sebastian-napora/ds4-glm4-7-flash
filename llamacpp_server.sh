#!/bin/bash
#
# Launch GLM-4.7-Flash GGUF via llama.cpp's llama-server, with CUDA on the
# DGX Spark GB10 (Blackwell, sm_121). This runs as a *separate* process tree
# from glm_native_server.sh — different binary, different ports.
#
# Architecture:
#   Copilot -> LiteLLM (12111) -> llama-server (12112)
#
# Native engine and llama.cpp can run side-by-side:
#   native:    ports 11111 (LiteLLM) / 11112 (server)
#   llama.cpp: ports 12111 (LiteLLM) / 12112 (server)
#
# Usage:
#   ./llamacpp_server.sh                                # defaults: Q6_K, GPU offload all
#   GLM_GGUF_QUANT=Q6_K ./llamacpp_server.sh
#   GLM_MODEL=/path/to/file.gguf ./llamacpp_server.sh
#   LLAMA_NGL=20 ./llamacpp_server.sh                   # offload only 20 layers to GPU
#   LLAMA_CTX=65536 ./llamacpp_server.sh
#
# Environment:
#   GLM_GGUF_QUANT        Quantization variant (default: Q6_K)
#   GLM_MODEL             Path to GGUF file (default: models/GLM-4.7-Flash-<QUANT>.gguf)
#   GLM_HOST              Bind host (default: 0.0.0.0)
#   GLM_PORT              llama-server port (default: 12112)
#   LLAMA_NGL             # of layers to offload to GPU (default: 999 = all)
#   LLAMA_CTX             Context size (default: 65536)
#   LLAMA_THREADS         CPU threads (default: nproc)
#   LLAMA_PARALLEL        Parallel slots (default: 1)
#   LLAMA_FLASH_ATTN      Enable flash attention (default: 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ────────────────────────────────────────────────────────────────────
QUANT="${GLM_GGUF_QUANT:-Q6_K}"
MODEL_DIR="${GLM_MODEL_DIR:-models}"

MODEL="${GLM_MODEL:-}"
if [ -z "$MODEL" ]; then
    case "$QUANT" in
        GLM-4.7-Flash-*.gguf) FILE="$QUANT" ;;
        *) FILE="GLM-4.7-Flash-$QUANT.gguf" ;;
    esac
    MODEL="$MODEL_DIR/$FILE"
fi

GLM_HOST="${GLM_HOST:-0.0.0.0}"
GLM_PORT="${GLM_PORT:-12112}"

NGL="${LLAMA_NGL:-999}"
CTX="${LLAMA_CTX:-65536}"
THREADS="${LLAMA_THREADS:-$(nproc)}"
PARALLEL="${LLAMA_PARALLEL:-1}"
FLASH_ATTN="${LLAMA_FLASH_ATTN:-1}"

# ── Validate model ──────────────────────────────────────────────────────────────
if [ ! -e "$MODEL" ]; then
    echo "Model not found: $MODEL"
    echo
    echo "Available models:"
    ls -1 "$MODEL_DIR"/*.gguf 2>/dev/null || echo "  (none)"
    exit 1
fi

# ── Locate llama-server binary ──────────────────────────────────────────────────
LLAMA_BIN="$SCRIPT_DIR/third_party/llama.cpp/build/bin/llama-server"

if [ ! -x "$LLAMA_BIN" ]; then
    echo "llama-server not built: $LLAMA_BIN"
    echo
    echo "Build it with CUDA for the GB10:"
    echo "  cd third_party/llama.cpp"
    echo "  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DLLAMA_CURL=OFF"
    echo "  cmake --build build --target llama-server -j\$(nproc)"
    exit 1
fi

mkdir -p "$SCRIPT_DIR/logs"

FA_ARG=""
if [ "$FLASH_ATTN" = "1" ]; then FA_ARG="--flash-attn auto"; fi

# ── Print config ────────────────────────────────────────────────────────────────
echo
echo "┌─────────────────────────────────────────────────────┐"
echo "│  llama.cpp (CUDA)  GLM-4.7-Flash ($QUANT)          │"
echo "├─────────────────────────────────────────────────────┤"
printf "│  Model:      %-37s │\n" "$MODEL"
printf "│  Endpoint:   http://%s:%s                  │\n" "$GLM_HOST" "$GLM_PORT"
printf "│  GPU layers: %-37s │\n" "$NGL"
printf "│  Context:    %-37s │\n" "$CTX"
printf "│  Threads:    %-37s │\n" "$THREADS"
echo "└─────────────────────────────────────────────────────┘"
echo

exec "$LLAMA_BIN" \
    -m "$MODEL" \
    --host "$GLM_HOST" \
    --port "$GLM_PORT" \
    -ngl "$NGL" \
    -c "$CTX" \
    -t "$THREADS" \
    -np "$PARALLEL" \
    --jinja \
    --alias glm-4.7-flash-llamacpp \
    $FA_ARG
