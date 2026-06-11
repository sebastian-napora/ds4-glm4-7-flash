#!/bin/bash
#
# Start llama.cpp + LiteLLM with n-gram speculative decoding enabled.
#
# Best for:
#   - RAG / document Q&A
#   - summarization
#   - code completion / editing
#   - prompts with repeated phrases or copied context
#
# Less effective for:
#   - short prompts
#   - open-ended creative chat
#
# Default topology:
#   LiteLLM:   http://0.0.0.0:12111/v1
#   llama.cpp: http://0.0.0.0:12112/v1
#
# Default fast path:
#   --flash-attn on
#   --spec-type ngram-cache
#   --ctx-size 65536
#
# Usage:
#   ./start_ngram_llamacpp.sh
#   LLAMA_CTX=32768 ./start_ngram_llamacpp.sh
#   LLAMA_SPEC_TYPE=ngram-simple \
#   LLAMA_SERVER_EXTRA_ARGS="--spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 8 --spec-ngram-simple-min-hits 2" \
#   ./start_ngram_llamacpp.sh
#
# Important:
#   MTP / EAGLE are NOT enabled here. They require draft-capable model support
#   that the current GLM Q6_K llama.cpp setup does not provide.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
    cat <<EOF
Usage: $0

Starts the llama.cpp + LiteLLM stack with n-gram speculative decoding.

Environment:
  GLM_HOST              llama.cpp bind host. Default: 0.0.0.0
  GLM_PORT              llama.cpp port. Default: 12112
  LITE_LLM_PROXY_HOST   LiteLLM bind host. Default: 0.0.0.0
  LITE_LLM_PROXY_PORT   LiteLLM port. Default: 12111
  GLM_MODEL             GGUF path. Default: models/GLM-4.7-Flash-Q6_K.gguf
  LLAMA_CTX             Context size. Default: 65536
  LLAMA_NGL             GPU layers. Default: 999
  LLAMA_THREADS         CPU threads. Default: nproc
  LLAMA_PARALLEL        Parallel slots. Default: 1
  LLAMA_FLASH_ATTN      on|off|auto. Default: on
  LLAMA_SPEC_TYPE       Default: ngram-cache
  LLAMA_SERVER_EXTRA_ARGS
                        Extra raw llama-server args.

Examples:
  ./start_ngram_llamacpp.sh
  LLAMA_CTX=32768 ./start_ngram_llamacpp.sh
  LLAMA_SPEC_TYPE=ngram-simple \\
  LLAMA_SERVER_EXTRA_ARGS="--spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 8 --spec-ngram-simple-min-hits 2" \\
  ./start_ngram_llamacpp.sh
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

export LLAMA_FLASH_ATTN="${LLAMA_FLASH_ATTN:-on}"
export LLAMA_SPEC_TYPE="${LLAMA_SPEC_TYPE:-ngram-cache}"
export LLAMA_CTX="${LLAMA_CTX:-65536}"
export LLAMA_NGL="${LLAMA_NGL:-999}"
export LLAMA_PARALLEL="${LLAMA_PARALLEL:-1}"

echo
echo "╔═════════════════════════════════════════════════════╗"
echo "║      llama.cpp + LiteLLM with n-gram decoding      ║"
echo "╚═════════════════════════════════════════════════════╝"
echo
echo "  Spec type:   ${LLAMA_SPEC_TYPE}"
echo "  Flash attn:  ${LLAMA_FLASH_ATTN}"
echo "  Context:     ${LLAMA_CTX}"
if [ -n "${LLAMA_SERVER_EXTRA_ARGS:-}" ]; then
    echo "  Extra args:  ${LLAMA_SERVER_EXTRA_ARGS}"
fi
echo

exec "$SCRIPT_DIR/start_llamacpp_litellm.sh"
