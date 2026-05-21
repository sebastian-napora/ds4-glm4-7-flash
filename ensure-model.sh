#!/bin/bash
#
# Ensure the native GLM GGUF target is available locally.
#
# The script first checks ./models, then the Hugging Face cache. If the GGUF is
# already cached on the DGX Spark, it creates ./models/<file>.gguf as a symlink
# to the cached file. If not cached, it downloads via huggingface_hub and then
# creates the same symlink.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO="${GLM_GGUF_REPO:-unsloth/GLM-4.7-Flash-GGUF}"
QUANT="${GLM_GGUF_QUANT:-Q8_0}"
MODEL_DIR="${GLM_GGUF_DIR:-$SCRIPT_DIR/models}"
TOKEN="${HF_TOKEN:-}"

usage() {
    cat <<EOF
Usage: $0 [QUANT] [--repo REPO] [--model-dir DIR] [--token TOKEN]

Default:
  repo:  $REPO
  quant: $QUANT
  dir:   $MODEL_DIR

Examples:
  ./ensure-model.sh
  ./ensure-model.sh Q8_0
  ./ensure-model.sh UD-Q4_K_XL

Environment:
  GLM_GGUF_REPO     Default: unsloth/GLM-4.7-Flash-GGUF
  GLM_GGUF_QUANT    Default: Q8_0
  GLM_GGUF_DIR      Default: ./models
  HF_TOKEN          Optional Hugging Face token
EOF
}

if [ $# -gt 0 ]; then
    case "$1" in
        -h|--help|help)
            usage
            exit 0
            ;;
        --*) ;;
        *)
            QUANT="$1"
            shift
            ;;
    esac
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --repo)
            shift
            [ $# -gt 0 ] || { echo "Missing value after --repo" >&2; exit 1; }
            REPO="$1"
            ;;
        --model-dir)
            shift
            [ $# -gt 0 ] || { echo "Missing value after --model-dir" >&2; exit 1; }
            MODEL_DIR="$1"
            ;;
        --token)
            shift
            [ $# -gt 0 ] || { echo "Missing value after --token" >&2; exit 1; }
            TOKEN="$1"
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
    *) MODEL_DIR="$SCRIPT_DIR/$MODEL_DIR" ;;
esac

case "$QUANT" in
    GLM-4.7-Flash-*.gguf) FILE="$QUANT" ;;
    *) FILE="GLM-4.7-Flash-$QUANT.gguf" ;;
esac

TARGET="$MODEL_DIR/$FILE"

PYTHON="$SCRIPT_DIR/venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    PYTHON="${PYTHON_BIN:-python3}"
fi

if ! "$PYTHON" -c 'import huggingface_hub' >/dev/null 2>&1; then
    echo "huggingface_hub is not installed. Run ./setup-dev-venv.sh first." >&2
    exit 1
fi

mkdir -p "$MODEL_DIR"

if [ -s "$TARGET" ]; then
    echo "Model already available:"
    echo "  $TARGET"
    exit 0
fi

if [ -z "$TOKEN" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi

echo "Ensuring model:"
echo "  repo:   $REPO"
echo "  file:   $FILE"
echo "  target: $TARGET"

if [ -n "$TOKEN" ]; then
    HF_TOKEN="$TOKEN" "$PYTHON" - "$REPO" "$FILE" "$TARGET" <<'PY'
import os
import pathlib
import sys
from huggingface_hub import hf_hub_download

repo, filename, target = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
token = os.environ.get("HF_TOKEN") or None

def resolve(local_only: bool):
    return pathlib.Path(hf_hub_download(
        repo_id=repo,
        filename=filename,
        token=token,
        local_files_only=local_only,
        resume_download=not local_only,
    ))

try:
    path = resolve(True)
    print(f"Found in Hugging Face cache: {path}")
except Exception:
    print("Not present in Hugging Face cache; downloading now.")
    path = resolve(False)
    print(f"Downloaded: {path}")

target.parent.mkdir(parents=True, exist_ok=True)
if target.exists() or target.is_symlink():
    target.unlink()
target.symlink_to(path)
print(f"Linked: {target} -> {path}")
PY
else
    "$PYTHON" - "$REPO" "$FILE" "$TARGET" <<'PY'
import pathlib
import sys
from huggingface_hub import hf_hub_download

repo, filename, target = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])

def resolve(local_only: bool):
    return pathlib.Path(hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_files_only=local_only,
        resume_download=not local_only,
    ))

try:
    path = resolve(True)
    print(f"Found in Hugging Face cache: {path}")
except Exception:
    print("Not present in Hugging Face cache; downloading now.")
    path = resolve(False)
    print(f"Downloaded: {path}")

target.parent.mkdir(parents=True, exist_ok=True)
if target.exists() or target.is_symlink():
    target.unlink()
target.symlink_to(path)
print(f"Linked: {target} -> {path}")
PY
fi

echo
echo "Ready:"
echo "  GLM_MODEL=\"$TARGET\" ./run-server-litellm.sh"
