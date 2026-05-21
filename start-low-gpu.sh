#!/bin/bash
#
# Conservative launcher for shared GB10 machines.
# Applies an NVIDIA power limit when nvidia-smi allows it, then starts the
# normal GB10 vLLM/LiteLLM stack.

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
else
    echo "nvidia-smi not found; skipping GPU power limit"
fi

export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.45}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
export VLLM_OPT_LEVEL="${VLLM_OPT_LEVEL:-0}"

exec "$SCRIPT_DIR/start-gb10.sh" "${1:-both}"
