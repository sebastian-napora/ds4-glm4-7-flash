#!/bin/sh
#
# Download GadflyII/GLM-4.7-Flash-NVFP4 into a local snapshot directory.
#
# Unlike ds4-changed, this is not a GGUF download. GLM is served by vLLM from
# Safetensors + compressed-tensors NVFP4 metadata.

set -e

REPO="${GLM_REPO:-GadflyII/GLM-4.7-Flash-NVFP4}"
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MODEL_DIR="${GLM_MODEL_DIR:-$ROOT/model}"
TOKEN="${HF_TOKEN:-}"

usage() {
    cat <<EOF
GLM-4.7-Flash-NVFP4 snapshot downloader

Usage:
  ./download_model.sh [--token TOKEN] [--model-dir DIR]

Environment:
  GLM_REPO        Hugging Face repo. Default: GadflyII/GLM-4.7-Flash-NVFP4
  GLM_MODEL_DIR   Local snapshot directory. Default: ./model
  HF_TOKEN        Optional Hugging Face token.

After download, start with:
  GLM_MODEL_PATH="$MODEL_DIR" ./start-gb10.sh both

If ./model exists, start.sh and start-gb10.sh use it automatically.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --token)
            shift
            [ $# -gt 0 ] || { echo "Missing value after --token" >&2; exit 1; }
            TOKEN="$1"
            ;;
        --model-dir)
            shift
            [ $# -gt 0 ] || { echo "Missing value after --model-dir" >&2; exit 1; }
            MODEL_DIR="$1"
            ;;
        -h|--help|help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

case "$MODEL_DIR" in
    /*) ;;
    *) MODEL_DIR="$ROOT/$MODEL_DIR" ;;
esac

if [ -z "$TOKEN" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi

PYTHON="$ROOT/venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    PYTHON=python3
fi

mkdir -p "$MODEL_DIR"

echo "Downloading $REPO"
echo "Target: $MODEL_DIR"
echo "This uses huggingface_hub snapshot_download so large Xet-backed files resume cleanly."

if [ -n "$TOKEN" ]; then
    HF_TOKEN="$TOKEN" "$PYTHON" - "$REPO" "$MODEL_DIR" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

repo, out = sys.argv[1], sys.argv[2]
snapshot_download(
    repo_id=repo,
    local_dir=out,
    token=os.environ.get("HF_TOKEN") or None,
    resume_download=True,
    allow_patterns=[
        "*.json",
        "*.jinja",
        "*.safetensors",
        "*.txt",
        "README.md",
    ],
)
PY
else
    "$PYTHON" - "$REPO" "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo, out = sys.argv[1], sys.argv[2]
snapshot_download(
    repo_id=repo,
    local_dir=out,
    resume_download=True,
    allow_patterns=[
        "*.json",
        "*.jinja",
        "*.safetensors",
        "*.txt",
        "README.md",
    ],
)
PY
fi

echo
echo "Done."
echo "Local model path:"
echo "  $MODEL_DIR"
