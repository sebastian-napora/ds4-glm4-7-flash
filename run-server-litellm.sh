#!/bin/bash
#
# Run custom glm-native-server plus LiteLLM proxy.
#
# This is the script to use for VS Code / GitHub Copilot plugin integration.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-both}"
MODEL="${GLM_MODEL:-models/GLM-4.7-Flash-Q8_0.gguf}"

case "$MODE" in
    both|server|proxy|inspect|summary) ;;
    -h|--help|help)
        cat <<EOF
Usage: $0 [both|server|proxy|inspect]

Default: both

Environment:
  GLM_MODEL             Default: models/GLM-4.7-Flash-Q8_0.gguf
  GLM_HOST              Native backend host. Default: 0.0.0.0
  GLM_PORT              Native backend port. Default: 11112
  LITE_LLM_PROXY_HOST   LiteLLM host. Default: 0.0.0.0
  LITE_LLM_PROXY_PORT   LiteLLM port. Default: 11111
EOF
        exit 0
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        exit 2
        ;;
esac

if [ "$MODE" != "proxy" ] && [ ! -s "$MODEL" ]; then
    echo "Model missing: $MODEL" >&2
    echo "Ensuring model first..."
    "$SCRIPT_DIR/ensure-model.sh" "${GLM_GGUF_QUANT:-Q8_0}"
fi

exec "$SCRIPT_DIR/start.sh" "$MODE" 2>&1 | tee "$SCRIPT_DIR/logs/startup.log"
