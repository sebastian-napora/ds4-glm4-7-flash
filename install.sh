#!/bin/bash
#
# End-to-end setup for DGX Spark / GB10 native GLM development.
#
# This prepares helper Python deps, ensures the Q8_0 GGUF exists locally or in
# the Hugging Face cache, and builds the custom native binaries.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

QUANT="${1:-${GLM_GGUF_QUANT:-Q8_0}}"

case "$QUANT" in
    -h|--help|help)
        cat <<EOF
Usage: $0 [QUANT]

Default quant: Q8_0

Runs:
  ./setup-dev-venv.sh
  ./ensure-model.sh <QUANT>
  make -C native
EOF
        exit 0
        ;;
esac

echo "==> Setting up developer venv"
"$SCRIPT_DIR/setup-dev-venv.sh"

echo
echo "==> Ensuring GGUF model"
"$SCRIPT_DIR/ensure-model.sh" "$QUANT"

echo
echo "==> Building native engine binaries"
make -C native

echo
echo "Install complete."
echo "Run native backend + LiteLLM with:"
echo "  ./run-server-litellm.sh"
