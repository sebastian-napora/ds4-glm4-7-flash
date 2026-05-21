#!/bin/bash
#
# DGX Spark / GB10 defaults for GLM-4.7-Flash-NVFP4.
#
# This mirrors the practical role of ds4-changed/start-lazy-experts-128gb.sh,
# but the implementation is GLM/vLLM specific: no ds4 GGUF loader and no
# DS4_CUDA_LAZY_ROUTED_EXPERTS toggles.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export VLLM_USE_FLASHINFER_MOE_FP4="${VLLM_USE_FLASHINFER_MOE_FP4:-0}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.50}"
export VLLM_OPT_LEVEL="${VLLM_OPT_LEVEL:-1}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
export GLM_MAX_RAM_GB="${GLM_MAX_RAM_GB:-118}"
export GLM_MAX_RAM_PERCENT="${GLM_MAX_RAM_PERCENT:-94}"

if [ -d "$SCRIPT_DIR/model" ]; then
    export GLM_MODEL_PATH="${GLM_MODEL_PATH:-$SCRIPT_DIR/model}"
fi

echo "=============================================="
echo "  GLM-4.7-Flash-NVFP4 on DGX Spark GB10"
echo "=============================================="
echo "  model:            ${GLM_MODEL_PATH:-GadflyII/GLM-4.7-Flash-NVFP4}"
echo "  max_model_len:    $VLLM_MAX_MODEL_LEN"
echo "  gpu_mem_util:     $VLLM_GPU_MEM_UTIL"
echo "  optimization:     $VLLM_OPT_LEVEL"
echo "  batched_tokens:   $VLLM_MAX_NUM_BATCHED_TOKENS"
echo "  flashinfer_moe:   $VLLM_USE_FLASHINFER_MOE_FP4"
echo "  ram guard:        ${GLM_MAX_RAM_GB}GB / ${GLM_MAX_RAM_PERCENT}%"
echo "=============================================="
echo

exec "$SCRIPT_DIR/start.sh" "${1:-both}"
