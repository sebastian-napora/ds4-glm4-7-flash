#!/bin/bash
#
# Start llama.cpp + LiteLLM with a longer-draft n-gram-simple preset.
#
# Preset:
#   --spec-type ngram-simple
#   --spec-ngram-simple-size-n 4
#   --spec-ngram-simple-size-m 32
#   --spec-ngram-simple-min-hits 2
#
# This is a more aggressive draft-length preset meant to test whether longer
# speculative bursts improve throughput when prompt/output overlap is high.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
    cat <<EOF
Usage: $0

Starts the llama.cpp + LiteLLM stack with a long-draft ngram-simple preset.

Preset:
  LLAMA_SPEC_TYPE=ngram-simple
  --spec-ngram-simple-size-n 4
  --spec-ngram-simple-size-m 32
  --spec-ngram-simple-min-hits 2

Environment:
  LLAMA_SERVER_EXTRA_ARGS Extra raw llama-server args appended after the preset.

Examples:
  ./start_ngram_long_draft_llamacpp.sh
  LLAMA_CTX=32768 ./start_ngram_long_draft_llamacpp.sh
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
PRESET_ARGS="--spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 32 --spec-ngram-simple-min-hits 2"
export LLAMA_SERVER_EXTRA_ARGS="${PRESET_ARGS}${LLAMA_SERVER_EXTRA_ARGS:+ ${LLAMA_SERVER_EXTRA_ARGS}}"

echo
echo "╔═════════════════════════════════════════════════════╗"
echo "║  llama.cpp + LiteLLM with long-draft n-gram setup  ║"
echo "╚═════════════════════════════════════════════════════╝"
echo
echo "  Spec type:   ${LLAMA_SPEC_TYPE}"
echo "  Context:     ${LLAMA_CTX}"
echo "  Extra args:  ${LLAMA_SERVER_EXTRA_ARGS}"
echo

exec "$SCRIPT_DIR/start_llamacpp_litellm.sh"
