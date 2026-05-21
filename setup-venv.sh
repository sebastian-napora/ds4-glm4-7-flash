#!/bin/bash
#
# Create a local Python environment for the GLM-4.7-Flash-NVFP4 stack.
#
# The model relies on fast-moving vLLM/Transformers support for
# compressed-tensors NVFP4 and glm4_moe_lite, so this script installs from the
# project requirements and leaves CUDA wheel selection to pip/index config.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Creating venv: $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"

echo "Upgrading pip tooling"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

echo "Installing GLM serving requirements"
"$VENV_DIR/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"

echo
echo "Setup complete."
echo "Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo
echo "Start the GB10 stack with:"
echo "  ./start-gb10.sh both"
