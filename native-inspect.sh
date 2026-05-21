#!/bin/bash
#
# Build and run the native GLM GGUF inspector.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${GLM_MODEL:-models/GLM-4.7-Flash-Q8_0.gguf}"
ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --metadata|--tensors|--help|-h)
            ARGS+=("$1")
            ;;
        *)
            MODEL="$1"
            ;;
    esac
    shift
done

make -C native
exec "$SCRIPT_DIR/native/bin/glm-inspect" "${ARGS[@]}" "$MODEL"
