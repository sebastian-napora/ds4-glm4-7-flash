#!/bin/bash
#
# Idempotent developer venv setup for the native GLM project.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating developer venv: $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "Using existing developer venv: $VENV_DIR"
fi

echo "Upgrading pip tooling"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

echo "Installing project helper/proxy requirements"
"$VENV_DIR/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"

echo
echo "Developer venv ready:"
echo "  source $VENV_DIR/bin/activate"
