#!/bin/bash
#
# Backwards-compatible model downloader.
#
# Prefer ./ensure-model.sh for the cache-aware flow. This script delegates to
# it so an already-cached DGX Spark model is reused instead of downloaded again.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/ensure-model.sh" "$@"
