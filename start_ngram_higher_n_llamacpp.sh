#!/bin/bash
#
# Start llama.cpp + LiteLLM with a higher-n n-gram-simple preset.
#
# Preset:
#   --spec-type ngram-simple
#   --spec-ngram-simple-size-n 8
#   --spec-ngram-simple-size-m 16
#   --spec-ngram-simple-min-hits 2
#
# This is a stricter lookup preset meant to test whether larger n-grams improve
# acceptance and throughput for your workload.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
    cat <<EOF
Usage: $0

Starts the llama.cpp + LiteLLM stack with a higher-n ngram-simple preset.

Preset:
  LLAMA_SPEC_TYPE=ngram-simple
  --spec-ngram-simple-size-n 8
  --spec-ngram-simple-size-m 16
  --spec-ngram-simple-min-hits 2

Environment:
  LLAMA_SERVER_EXTRA_ARGS Extra raw llama-server args appended after the preset.

Examples:
  ./start_ngram_higher_n_llamacpp.sh
  LLAMA_CTX=32768 ./start_ngram_higher_n_llamacpp.sh
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

export LLAMA_FLASH_ATTN="${LLAMA_FLASH_ATTN:-on}"
export LLAMA_SPEC_TYPE="ngram-simple"
export LLAMA_CTX="${LLAMA_CTX:-65536}"
export LLAMA_NGL="${LLAMA_NGL:-999}"
export LLAMA_PARALLEL="${LLAMA_PARALLEL:-1}"
PRESET_ARGS="--spec-ngram-simple-size-n 8 --spec-ngram-simple-size-m 16 --spec-ngram-simple-min-hits 2"
export LLAMA_SERVER_EXTRA_ARGS="${PRESET_ARGS}${LLAMA_SERVER_EXTRA_ARGS:+ ${LLAMA_SERVER_EXTRA_ARGS}}"

echo
echo "╔═════════════════════════════════════════════════════╗"
echo "║   llama.cpp + LiteLLM with higher-n n-gram setup   ║"
echo "╚═════════════════════════════════════════════════════╝"
echo
echo "  Spec type:   ${LLAMA_SPEC_TYPE}"
echo "  Context:     ${LLAMA_CTX}"
echo "  Extra args:  ${LLAMA_SERVER_EXTRA_ARGS}"
echo

exec "$SCRIPT_DIR/start_llamacpp_litellm.sh"
