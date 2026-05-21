#!/bin/bash
#
# Start GLM-4.7-Flash-NVFP4 serving stack.
#
# Usage:
#   ./start.sh          # vLLM backend + token stats + LiteLLM proxy
#   ./start.sh backend  # vLLM backend only (11112)
#   ./start.sh proxy    # LiteLLM proxy only (11111)
#   ./start.sh stats    # token stats only (11113)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d "$SCRIPT_DIR/venv" ]; then
    VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
else
    VENV_PYTHON="${PYTHON_BIN:-python3}"
fi

mkdir -p "$SCRIPT_DIR/logs"

if [ -d "$SCRIPT_DIR/model" ]; then
    export GLM_MODEL_PATH="${GLM_MODEL_PATH:-$SCRIPT_DIR/model}"
fi

export GLM_HOST="${GLM_HOST:-0.0.0.0}"
export GLM_PORT="${GLM_PORT:-11112}"
export LITE_LLM_PROXY_HOST="${LITE_LLM_PROXY_HOST:-0.0.0.0}"
export LITE_LLM_PROXY_PORT="${LITE_LLM_PROXY_PORT:-11111}"
export GLM_TOKEN_STATS_HOST="${GLM_TOKEN_STATS_HOST:-0.0.0.0}"
export GLM_TOKEN_STATS_PORT="${GLM_TOKEN_STATS_PORT:-11113}"
export GLM_STATS_HOST="${GLM_STATS_HOST:-$GLM_TOKEN_STATS_HOST}"
export GLM_STATS_PORT="${GLM_STATS_PORT:-$GLM_TOKEN_STATS_PORT}"

start_backend() {
    echo "Starting vLLM backend on ${GLM_HOST}:${GLM_PORT}"
    "$VENV_PYTHON" "$SCRIPT_DIR/glm_server.py" \
        >> "$SCRIPT_DIR/logs/vllm_server.log" 2>&1 &
    echo "$!" > "$SCRIPT_DIR/logs/vllm_server.pid"
    echo "Backend PID: $!"
}

new_token_session() {
    NEW_SID=$("$VENV_PYTHON" -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
import glm_token_tracker
print(glm_token_tracker.new_session(), end='')
" 2>/dev/null || true)
    echo "Token session: ${NEW_SID:-unknown}"
}

start_stats() {
    echo "Starting token stats server on ${GLM_TOKEN_STATS_HOST}:${GLM_TOKEN_STATS_PORT}"
    "$VENV_PYTHON" "$SCRIPT_DIR/token_stats_server.py" \
        >> "$SCRIPT_DIR/logs/token_stats_server.log" 2>&1 &
    echo "$!" > "$SCRIPT_DIR/logs/token_stats_server.pid"
    echo "Stats PID: $!"
}

start_proxy() {
    echo "Starting LiteLLM proxy on ${LITE_LLM_PROXY_HOST}:${LITE_LLM_PROXY_PORT}"
    "$VENV_PYTHON" "$SCRIPT_DIR/server_compress.py" \
        >> "$SCRIPT_DIR/logs/litellm_proxy.log" 2>&1 &
    echo "$!" > "$SCRIPT_DIR/logs/litellm_proxy.pid"
    echo "Proxy PID: $!"
}

stop_from_pidfile() {
    local pidfile="$1"
    if [ -s "$pidfile" ]; then
        kill "$(cat "$pidfile")" 2>/dev/null || true
    fi
}

start_memory_guard() {
    if ! command -v free >/dev/null 2>&1; then
        echo "Memory guard skipped: free(1) not found"
        return
    fi

    local max_gb="${GLM_MAX_RAM_GB:-0}"
    local max_pct="${GLM_MAX_RAM_PERCENT:-0}"
    if [ "$max_gb" = "0" ] && [ "$max_pct" = "0" ]; then
        return
    fi

    (
        while true; do
            sleep 15
            used_gb=$(free -b | awk 'NR==2 {printf "%.2f", $3/1024/1024/1024}')
            used_pct=$(free | awk 'NR==2 {printf "%d", ($3/$2)*100}')
            exceeded=0
            if [ "$max_gb" != "0" ] && awk -v u="$used_gb" -v m="$max_gb" 'BEGIN { exit !(u > m) }'; then
                exceeded=1
            fi
            if [ "$max_pct" != "0" ] && [ "$used_pct" -ge "$max_pct" ]; then
                exceeded=1
            fi
            if [ "$exceeded" -eq 1 ]; then
                echo "Memory guard exceeded: ${used_gb}GB / ${used_pct}%"
                stop_from_pidfile "$SCRIPT_DIR/logs/litellm_proxy.pid"
                stop_from_pidfile "$SCRIPT_DIR/logs/token_stats_server.pid"
                stop_from_pidfile "$SCRIPT_DIR/logs/vllm_server.pid"
                exit 1
            fi
        done
    ) >> "$SCRIPT_DIR/logs/memory_guard.log" 2>&1 &
    echo "$!" > "$SCRIPT_DIR/logs/memory_guard.pid"
    echo "Memory guard PID: $! (${GLM_MAX_RAM_GB:-0}GB / ${GLM_MAX_RAM_PERCENT:-0}%)"
}

usage() {
    cat <<EOF
Usage: $0 [both|backend|proxy|stats]

Environment:
  GLM_MODEL_PATH                  Local model path or HF repo
  VLLM_MAX_MODEL_LEN              Default: 65536 via start-gb10.sh
  VLLM_GPU_MEM_UTIL               Default: 0.50 via start-gb10.sh
  VLLM_OPT_LEVEL                  Default: 1 via start-gb10.sh
  GLM_MAX_RAM_GB                  Optional memory guard, set by start-gb10.sh
  GLM_MAX_RAM_PERCENT             Optional memory guard, set by start-gb10.sh
  GLM_PORT                        Default: 11112
  LITE_LLM_PROXY_PORT             Default: 11111
  GLM_TOKEN_STATS_PORT            Default: 11113

Logs:
  logs/vllm_server.log
  logs/litellm_proxy.log
  logs/token_stats_server.log
EOF
}

case "${1:-both}" in
    both)
        start_backend
        new_token_session
        echo "Waiting 5s before starting proxy services..."
        sleep 5
        start_stats
        sleep 1
        start_proxy
        start_memory_guard
        echo
        echo "Services started:"
        echo "  vLLM backend:   http://${GLM_HOST}:${GLM_PORT}"
        echo "  LiteLLM proxy:  http://${LITE_LLM_PROXY_HOST}:${LITE_LLM_PROXY_PORT}"
        echo "  Token stats:    http://${GLM_TOKEN_STATS_HOST}:${GLM_TOKEN_STATS_PORT}"
        echo
        echo "Health checks:"
        echo "  curl http://localhost:${GLM_PORT}/health"
        echo "  curl http://localhost:${LITE_LLM_PROXY_PORT}/health"
        ;;
    backend)
        start_backend
        ;;
    proxy)
        new_token_session
        start_proxy
        ;;
    stats)
        start_stats
        ;;
    --help|-h|help)
        usage
        ;;
    *)
        usage >&2
        exit 1
        ;;
esac
