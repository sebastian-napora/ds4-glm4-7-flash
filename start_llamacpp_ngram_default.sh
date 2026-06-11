#!/bin/bash
#
# Start the maintained llama.cpp + LiteLLM direct stack with the faster
# benchmark-style defaults.
#
# Defaults can still be overridden by exporting the env vars before launch:
#   LLAMA_CTX=32768 ./start_llamacpp_ngram_default.sh
#   LLAMA_SPEC_TYPE=none ./start_llamacpp_ngram_default.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LLAMA_CTX="${LLAMA_CTX:-65536}"
export LLAMA_SPEC_TYPE="${LLAMA_SPEC_TYPE:-ngram-cache}"

exec "$SCRIPT_DIR/start_llamacpp_direct.sh" "$@"
