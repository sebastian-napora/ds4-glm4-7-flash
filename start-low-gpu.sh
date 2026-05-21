#!/bin/bash
#
# Conservative wrapper for the custom native GLM server.
# At this stage the native server only loads the model, so GPU throttling is
# mostly future-facing for when CUDA kernels land.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GPU_POWER_LIMIT_PERCENT="${GPU_POWER_LIMIT_PERCENT:-35}"

if command -v nvidia-smi >/dev/null 2>&1; then
    MAX_POWER=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -n 1 | awk '{print int($1)}')
    if [ -n "$MAX_POWER" ] && [ "$MAX_POWER" -gt 0 ]; then
        NEW_POWER=$(( MAX_POWER * GPU_POWER_LIMIT_PERCENT / 100 ))
        echo "Setting GPU power limit to ${NEW_POWER}W (${GPU_POWER_LIMIT_PERCENT}% of ${MAX_POWER}W)"
        sudo -n nvidia-smi -pl "$NEW_POWER" 2>/dev/null || nvidia-smi -pl "$NEW_POWER" 2>/dev/null || true
    fi
fi

exec "$SCRIPT_DIR/start-gb10.sh" "${1:-both}"
