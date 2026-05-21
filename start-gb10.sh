#!/bin/bash
#
# DGX Spark / GB10 launcher for the custom native GLM engine.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export GLM_MODEL="${GLM_MODEL:-models/GLM-4.7-Flash-Q8_0.gguf}"
export GLM_HOST="${GLM_HOST:-0.0.0.0}"
export GLM_PORT="${GLM_PORT:-11112}"

echo "=============================================="
echo "  GLM-4.7-Flash native custom engine"
echo "=============================================="
echo "  machine:     DGX Spark / GB10 target"
echo "  model:       $GLM_MODEL"
echo "  host:        $GLM_HOST"
echo "  port:        $GLM_PORT"
echo "  proxy:       LiteLLM for VS Code / Copilot compatibility"
echo "  inference:   native forward pass pending"
echo "=============================================="
echo

exec "$SCRIPT_DIR/run-server-litellm.sh" "${1:-both}"
