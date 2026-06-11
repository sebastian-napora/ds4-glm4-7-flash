#!/bin/bash
#
# Start llama.cpp + LiteLLM with MTP speculative decoding enabled.
#
# This uses the GGUF model's embedded Multi-Token Prediction heads via
# llama.cpp's `--spec-type draft-mtp`, so no separate draft model is required.
#
# Best for:
#   - models whose GGUF preserves MTP / NextN heads
#   - low-latency chat and code generation
#   - comparing built-in draft heads against n-gram speculation
#
# Usage:
#   ./start_mtp_llamacpp.sh
#   LLAMA_CTX=32768 ./start_mtp_llamacpp.sh
#   LLAMA_SPEC_DRAFT_N_MAX=4 ./start_mtp_llamacpp.sh
#   LLAMA_SERVER_EXTRA_ARGS="--spec-draft-p-min 0.05" ./start_mtp_llamacpp.sh
#
# Notes:
#   - The default llama.cpp + LiteLLM ports remain 12111 / 12112.
#   - The GGUF must contain MTP tensors; otherwise llama-server may fail to load.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
    cat <<EOF
Usage: $0

Starts the llama.cpp + LiteLLM stack with MTP speculative decoding.

Environment:
  GLM_HOST                llama.cpp bind host. Default: 0.0.0.0
  GLM_PORT                llama.cpp port. Default: 12112
  LITE_LLM_PROXY_HOST     LiteLLM bind host. Default: 0.0.0.0
  LITE_LLM_PROXY_PORT     LiteLLM port. Default: 12111
  GLM_MODEL               GGUF path. Default: models/GLM-4.7-Flash-Q6_K.gguf
  LLAMA_CTX               Context size. Default: 65536
  LLAMA_NGL               GPU layers. Default: 999
  LLAMA_THREADS           CPU threads. Default: nproc
  LLAMA_PARALLEL          Parallel slots. Default: 1
  LLAMA_FLASH_ATTN        on|off|auto. Default: on
  LLAMA_SPEC_TYPE         Default: draft-mtp
  LLAMA_SPEC_DRAFT_N_MAX  Drafted tokens per step. Default: 3
  LLAMA_SERVER_EXTRA_ARGS Extra raw llama-server args.

Examples:
  ./start_mtp_llamacpp.sh
  LLAMA_SPEC_DRAFT_N_MAX=4 ./start_mtp_llamacpp.sh
  LLAMA_SERVER_EXTRA_ARGS="--spec-draft-p-min 0.05" ./start_mtp_llamacpp.sh
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

export LLAMA_FLASH_ATTN="${LLAMA_FLASH_ATTN:-on}"
export LLAMA_SPEC_TYPE="${LLAMA_SPEC_TYPE:-draft-mtp}"
export LLAMA_CTX="${LLAMA_CTX:-65536}"
export LLAMA_NGL="${LLAMA_NGL:-999}"
export LLAMA_PARALLEL="${LLAMA_PARALLEL:-1}"
export LLAMA_SPEC_DRAFT_N_MAX="${LLAMA_SPEC_DRAFT_N_MAX:-3}"

echo
echo "╔═════════════════════════════════════════════════════╗"
echo "║         llama.cpp + LiteLLM with MTP decoding      ║"
echo "╚═════════════════════════════════════════════════════╝"
echo
echo "  Spec type:   ${LLAMA_SPEC_TYPE}"
echo "  Flash attn:  ${LLAMA_FLASH_ATTN}"
echo "  Context:     ${LLAMA_CTX}"
echo "  Draft n max: ${LLAMA_SPEC_DRAFT_N_MAX}"
if [ -n "${LLAMA_SERVER_EXTRA_ARGS:-}" ]; then
    echo "  Extra args:  ${LLAMA_SERVER_EXTRA_ARGS}"
fi
echo

exec "$SCRIPT_DIR/start_llamacpp_litellm.sh"
