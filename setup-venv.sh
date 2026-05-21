#!/bin/bash
#
# Backwards-compatible alias for the developer venv setup.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/setup-dev-venv.sh" "$@"
