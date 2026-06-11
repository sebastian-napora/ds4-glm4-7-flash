#!/usr/bin/env bash
# Kill all repo-managed serving stacks and free all known ports at once.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORTS=(11111 11112 11113 12111 12112 12180 8080)
PORT_PATTERN='11111|11112|11113|12111|12112|12180|8080'
PROCESS_PATTERNS=(
    "$SCRIPT_DIR/native/bin/glm-native-server"
    "$SCRIPT_DIR/glm_server.py"
    "$SCRIPT_DIR/glm_server_draft.py"
    "$SCRIPT_DIR/glm_sglang_server.py"
    "$SCRIPT_DIR/server_compress.py"
    "$SCRIPT_DIR/server_compress_llamacpp.py"
    "$SCRIPT_DIR/server_compress_llamacpp_direct.py"
    "$SCRIPT_DIR/token_stats_server.py"
    "$SCRIPT_DIR/run-server-litellm.sh"
    "$SCRIPT_DIR/start.sh"
    "$SCRIPT_DIR/start_ngram.sh"
    "$SCRIPT_DIR/start_draft.sh"
    "$SCRIPT_DIR/start_sglang.sh"
    "$SCRIPT_DIR/start_llamacpp_litellm.sh"
    "$SCRIPT_DIR/start_ngram_llamacpp.sh"
    "$SCRIPT_DIR/start_ngram_higher_n_llamacpp.sh"
    "$SCRIPT_DIR/start_ngram_long_draft_llamacpp.sh"
    "$SCRIPT_DIR/start_mtp_llamacpp.sh"
    "$SCRIPT_DIR/third_party/llama.cpp/build/bin/llama-server"
    "VLLM::EngineCore"
)

port_pids() {
    local port="$1"

    if command -v ss >/dev/null 2>&1; then
        ss -ltnp "( sport = :$port )" 2>/dev/null \
            | grep -o 'pid=[0-9]\+' \
            | cut -d= -f2 \
            | sort -u
        return 0
    fi

    if command -v lsof >/dev/null 2>&1; then
        lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null | sort -u
    fi
}

pattern_pids() {
    local pattern="$1"

    if command -v pgrep >/dev/null 2>&1; then
        pgrep -f -- "$pattern" | sort -u || true
        return 0
    fi

    ps -eo pid=,args= 2>/dev/null \
        | awk -v pat="$pattern" '$0 ~ pat {print $1}' \
        | sort -u
}

stop_pid_tree() {
    local pid="$1"
    local children=()

    if [[ -z "$pid" || "$pid" == "$$" ]] || ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi

    mapfile -t children < <(ps -o pid= --ppid "$pid" 2>/dev/null | awk '{print $1}')
    for child in "${children[@]}"; do
        stop_pid_tree "$child"
    done

    kill "$pid" 2>/dev/null || true
    for _ in {1..10}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
}

declare -A SEEN_PIDS=()
stopped=0

stop_once() {
    local pid="$1"

    [[ -n "$pid" ]] || return 0
    [[ "$pid" == "$$" ]] && return 0
    [[ -n "${SEEN_PIDS[$pid]:-}" ]] && return 0
    SEEN_PIDS["$pid"]=1

    if kill -0 "$pid" 2>/dev/null; then
        stop_pid_tree "$pid"
        stopped=$((stopped + 1))
    fi
}

for pattern in "${PROCESS_PATTERNS[@]}"; do
    while read -r pid; do
        stop_once "$pid"
    done < <(pattern_pids "$pattern")
done

for port in "${PORTS[@]}"; do
    while read -r pid; do
        stop_once "$pid"
    done < <(port_pids "$port" || true)
done

echo "✅ GLM serving processes stopped (${stopped} process trees)"
ss -tlnp 2>/dev/null | grep -E "$PORT_PATTERN" || echo "  (ports are free)"
ps -eo pid=,args= 2>/dev/null \
    | grep -E 'glm-native-server|glm_server.py|glm_server_draft.py|glm_sglang_server.py|server_compress|token_stats_server|llama-server|VLLM::EngineCore' \
    | grep -v 'grep -E' \
    || echo "  (no remaining processes)"
