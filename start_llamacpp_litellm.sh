#!/bin/bash
#
# Compatibility launcher for the maintained llama.cpp + LiteLLM path.
#
# The older implementation forwarded large OpenAI tool schemas to llama.cpp's
# native tool parser, which can produce empty responses with Copilot-style tool
# sets. Use the direct launcher instead: it starts LiteLLM through
# server_compress_llamacpp_direct.py, prompts GLM tools as XML, removes native
# tool schemas before llama.cpp sees them, and converts GLM XML tool calls back
# to OpenAI tool_calls.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$SCRIPT_DIR/start_llamacpp_direct.sh" "$@"
