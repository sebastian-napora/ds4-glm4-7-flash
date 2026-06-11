#!/usr/bin/env bash
#
# benchmark_llamacpp_speculative.sh — Compare llama.cpp plain, n-gram presets,
# and MTP through the LiteLLM proxy stack on ports 12111 / 12112.
#
# Usage:
#   ./benchmark_llamacpp_speculative.sh
#   ./benchmark_llamacpp_speculative.sh --runs 10
#   ./benchmark_llamacpp_speculative.sh --test code
#   ./benchmark_llamacpp_speculative.sh --test all
#   ./benchmark_llamacpp_speculative.sh --skip-mtp
#   ./benchmark_llamacpp_speculative.sh --only-mtp
#
# Options:
#   --runs N                    Measured runs per mode          (default 5)
#   --warmup N                  Warmup runs per mode            (default 2)
#   --max-tokens N              Output tokens per request       (default 512)
#   --test TYPE                 Prompt type: short|rag|code|all (default rag)
#   --startup-timeout S         Seconds to wait for stack       (default 600)
#   --skip-normal               Skip plain llama.cpp mode
#   --skip-ngram                Skip default ngram-cache mode
#   --skip-ngram-higher-n       Skip higher-n ngram-simple mode
#   --skip-ngram-long-draft     Skip long-draft ngram-simple mode
#   --skip-mtp                  Skip MTP mode
#   --only-mtp                  Run only the MTP phase
#   -h, --help                  Show this help
#
# Results saved to logs/benchmark_llamacpp_speculative_<timestamp>.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUNS=5
WARMUP=2
MAX_TOKENS=512
TEST_TYPE="rag"
STARTUP_TIMEOUT=600
POLL_INTERVAL=2
SKIP_NORMAL=0
SKIP_NGRAM=0
SKIP_NGRAM_HIGHER_N=0
SKIP_NGRAM_LONG_DRAFT=0
SKIP_MTP=0

usage() { sed -n '3,26p' "$0" | sed 's/^# \?//'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs)                    RUNS="$2"; shift 2 ;;
        --warmup)                  WARMUP="$2"; shift 2 ;;
        --max-tokens)              MAX_TOKENS="$2"; shift 2 ;;
        --test)                    TEST_TYPE="$2"; shift 2 ;;
        --startup-timeout)         STARTUP_TIMEOUT="$2"; shift 2 ;;
        --skip-normal)             SKIP_NORMAL=1; shift ;;
        --skip-ngram)              SKIP_NGRAM=1; shift ;;
        --skip-ngram-higher-n)     SKIP_NGRAM_HIGHER_N=1; shift ;;
        --skip-ngram-long-draft)   SKIP_NGRAM_LONG_DRAFT=1; shift ;;
        --skip-mtp)                SKIP_MTP=1; shift ;;
        --only-mtp)
            SKIP_NORMAL=1
            SKIP_NGRAM=1
            SKIP_NGRAM_HIGHER_N=1
            SKIP_NGRAM_LONG_DRAFT=1
            SKIP_MTP=0
            shift
            ;;
        -h|--help)                 usage ;;
        *) echo "❌ Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -x "$SCRIPT_DIR/venv/bin/python3" ]]; then
    PY="$SCRIPT_DIR/venv/bin/python3"
else
    PY="python3"
fi

LITELLM_PORT="${LITE_LLM_PROXY_PORT:-12111}"
LLAMA_PORT="${GLM_PORT:-12112}"
LITELLM_URL="http://127.0.0.1:${LITELLM_PORT}/v1"
LITELLM_MODELS_URL="${LITELLM_URL}/models"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$SCRIPT_DIR/logs"
RESULTS_JSON="$RESULTS_DIR/benchmark_llamacpp_speculative_${TIMESTAMP}.json"
mkdir -p "$RESULTS_DIR"

SHORT_PROMPT="Explain what prefix caching is and why it matters for LLM inference. Be concise."

RAG_PROMPT='You are given the following document extract. Answer the question based only on this text.

DOCUMENT:
Speculative decoding is an inference optimization technique that accelerates large language model
generation without changing output quality. The technique works by using a smaller, faster draft
model to propose multiple candidate tokens in advance. The target (large) model then verifies
all proposed tokens in a single forward pass, accepting correct tokens and regenerating from the
first mismatch. Because the target model processes multiple tokens at once during verification,
the effective throughput is higher than standard autoregressive decoding. The speedup depends on
the acceptance rate — how often the draft tokens match what the target model would have generated.
N-gram speculative decoding is a special case where, instead of a separate draft model, candidate
tokens are looked up directly in the input prompt using n-gram matching. If the model is likely to
repeat or paraphrase text from the prompt (as in RAG, summarisation, or code editing), n-gram
speculative decoding can achieve significant speedups with zero additional memory overhead and no
compatibility issues. Multi-Token Prediction (MTP) uses extra heads trained into the main model to
predict several future tokens directly, avoiding a second draft model while still providing a true
model-based proposal path.

QUESTION: Compare plain autoregressive decoding, default ngram-cache, stricter higher-n ngram-simple,
long-draft ngram-simple, and MTP speculative decoding. Focus on when each one is likely to help.'

CODE_PROMPT='Complete the following Python function. Return only the completed function with no explanation.

def compute_token_stats(completions: list[dict]) -> dict:
    """
    Given a list of OpenAI-style completion response dicts, compute:
    - total_requests: number of completions
    - total_prompt_tokens: sum of usage.prompt_tokens
    - total_completion_tokens: sum of usage.completion_tokens
    - avg_completion_tokens: average completion tokens per request
    - avg_tokens_per_second: average completion_tokens / latency_seconds
      (each dict may have a latency_seconds key)
    Returns a dict with the above keys.
    """'

declare -A PROMPTS
PROMPTS["short"]="$SHORT_PROMPT"
PROMPTS["rag"]="$RAG_PROMPT"
PROMPTS["code"]="$CODE_PROMPT"

if [[ "$TEST_TYPE" == "all" ]]; then
    TEST_TYPES=("short" "rag" "code")
else
    TEST_TYPES=("$TEST_TYPE")
fi

declare -a MODE_IDS=()
declare -a MODE_LABELS=()
declare -a MODE_SCRIPTS=()
declare -a MODE_LOGS=()

add_mode() {
    MODE_IDS+=("$1")
    MODE_LABELS+=("$2")
    MODE_SCRIPTS+=("$3")
    MODE_LOGS+=("$4")
}

if [[ "$SKIP_NORMAL" == "0" ]]; then
    add_mode "plain" "plain" "start_llamacpp_litellm.sh" "bench_llamacpp_plain.log"
fi
if [[ "$SKIP_NGRAM" == "0" ]]; then
    add_mode "ngram-cache" "ngram-cache" "start_ngram_llamacpp.sh" "bench_llamacpp_ngram_cache.log"
fi
if [[ "$SKIP_NGRAM_HIGHER_N" == "0" ]]; then
    add_mode "ngram-higher-n" "ngram-higher-n" "start_ngram_higher_n_llamacpp.sh" "bench_llamacpp_ngram_higher_n.log"
fi
if [[ "$SKIP_NGRAM_LONG_DRAFT" == "0" ]]; then
    add_mode "ngram-long-draft" "ngram-long-draft" "start_ngram_long_draft_llamacpp.sh" "bench_llamacpp_ngram_long_draft.log"
fi
if [[ "$SKIP_MTP" == "0" ]]; then
    add_mode "mtp" "mtp" "start_mtp_llamacpp.sh" "bench_llamacpp_mtp.log"
fi

if (( ${#MODE_IDS[@]} == 0 )); then
    echo "❌ Nothing to run: all benchmark modes are skipped."
    exit 1
fi

MODE_IDS_CSV="$(IFS=,; printf '%s' "${MODE_IDS[*]}")"
MODE_LABELS_CSV="$(IFS=,; printf '%s' "${MODE_LABELS[*]}")"

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
    for _ in {1..20}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
}

stop_stack() {
    local port=""
    local pid=""

    for port in "$LITELLM_PORT" "$LLAMA_PORT"; do
        while read -r pid; do
            [[ -n "$pid" ]] || continue
            stop_pid_tree "$pid"
        done < <(port_pids "$port" || true)
    done
}

cleanup() {
    stop_stack
}

trap cleanup EXIT

LAST_STACK_FAILURE_REASON=""

describe_stack_failure() {
    local log_file="$1"
    local details=""

    LAST_STACK_FAILURE_REASON="startup failed"

    if [[ ! -f "$log_file" ]]; then
        return 0
    fi

    if grep -q "model doesn't contain MTP layers" "$log_file"; then
        LAST_STACK_FAILURE_REASON="current GGUF does not contain MTP layers for llama.cpp draft-mtp"
        echo "   ⚠️  ${LAST_STACK_FAILURE_REASON}."
        echo "   💡 MTP needs a GGUF with NextN / MTP layers preserved during conversion."
        return 0
    fi

    details="$(grep -E 'failed to|error|exiting due to|process exited|timed out' "$log_file" | tail -n 5 || true)"
    if [[ -n "$details" ]]; then
        echo "   ⚠️  Startup failure details:"
        while IFS= read -r line; do
            [[ -n "$line" ]] || continue
            echo "      $line"
        done <<< "$details"
    else
        echo "   ⚠️  Last startup log lines:"
        tail -n 20 "$log_file" | sed 's/^/      /' || true
    fi
}

wait_for_stack() {
    local label="$1"
    local monitor_pid="$2"
    local log_file="$3"
    local deadline=$((SECONDS + STARTUP_TIMEOUT))

    echo -n "   ⏳ Waiting for $label"
    while ! curl -fsS --max-time 5 "$LITELLM_MODELS_URL" >/dev/null 2>&1; do
        if ! kill -0 "$monitor_pid" 2>/dev/null; then
            echo " ❌ process exited"
            describe_stack_failure "$log_file"
            return 1
        fi
        if (( SECONDS >= deadline )); then
            echo " ❌ timed out"
            LAST_STACK_FAILURE_REASON="startup timed out"
            describe_stack_failure "$log_file"
            return 1
        fi
        echo -n "."
        sleep "$POLL_INTERVAL"
    done
    echo " ready ✅"
}

run_bench() {
    local prompt="$1"
    local out_file="$2"

    "$PY" "$SCRIPT_DIR/glm_benchmark.py" \
        --target litellm \
        --litellm-url "$LITELLM_URL" \
        --runs "$RUNS" \
        --warmup-runs "$WARMUP" \
        --max-tokens "$MAX_TOKENS" \
        --temperature 0.0 \
        --prompt "$prompt" \
        --format json \
        > "$out_file" 2>/dev/null
}

start_phase() {
    local label="$1"
    local script_name="$2"
    local log_file="$3"

    stop_stack
    echo "   🚀 Starting ${label}..."
    GLM_PORT="$LLAMA_PORT" \
    LITE_LLM_PROXY_PORT="$LITELLM_PORT" \
        "$SCRIPT_DIR/$script_name" \
        >>"$log_file" 2>&1 &
    local stack_pid="$!"

    if wait_for_stack "$label" "$stack_pid" "$log_file"; then
        return 0
    fi

    stop_pid_tree "$stack_pid"
    return 1
}

render_mode_table() {
    local test_name="$1"
    "$PY" - "$RESULTS_DIR" "$TIMESTAMP" "$test_name" "$MODE_IDS_CSV" "$MODE_LABELS_CSV" <<'PYEOF'
import json
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
timestamp = sys.argv[2]
test_name = sys.argv[3]
mode_ids = [m for m in sys.argv[4].split(",") if m]
mode_labels = [m for m in sys.argv[5].split(",") if m]

def load(path: Path):
    try:
        data = json.loads(path.read_text())
        return data[0] if isinstance(data, list) else data
    except Exception:
        return None

def fmt(value, unit=""):
    return f"{value:.2f}{unit}" if value is not None else "n/a"

records = []
plain_tps = None
for mode_id, label in zip(mode_ids, mode_labels):
    path = results_dir / f"bench_{timestamp}_{test_name}_{mode_id}.json"
    data = load(path)
    tps = data.get("avg_completion_tokens_per_second") if data else None
    lat = data.get("avg_latency_seconds") if data else None
    if mode_id == "plain":
        plain_tps = tps
    records.append((label, tps, lat))

print(f"  {'Mode':<20}  {'tok/s':>12}  {'latency (s)':>12}  {'speedup vs plain':>17}")
print(f"  {'-' * 20:<20}  {'-' * 12:>12}  {'-' * 12:>12}  {'-' * 17:>17}")
for label, tps, lat in records:
    if plain_tps and tps and plain_tps > 0:
        speedup = f'{tps / plain_tps:.2f}x'
    elif label == "plain" and plain_tps:
        speedup = "1.00x"
    else:
        speedup = "n/a"
    print(f"  {label:<20}  {fmt(tps):>12}  {fmt(lat, 's'):>12}  {speedup:>17}")
PYEOF
}

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   llama.cpp + LiteLLM: Plain, N-gram presets, and MTP Bench ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Config:  runs=${RUNS}  warmup=${WARMUP}  max_tokens=${MAX_TOKENS}  test=${TEST_TYPE}"
echo "  Ports:   LiteLLM=${LITELLM_PORT}  llama.cpp=${LLAMA_PORT}"
echo "  Modes:   ${MODE_IDS_CSV}"
echo "  Results: ${RESULTS_JSON}"
echo ""

for TEST in "${TEST_TYPES[@]}"; do
    PROMPT="${PROMPTS[$TEST]}"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Prompt type: ${TEST^^}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for idx in "${!MODE_IDS[@]}"; do
        mode_id="${MODE_IDS[$idx]}"
        mode_label="${MODE_LABELS[$idx]}"
        mode_script="${MODE_SCRIPTS[$idx]}"
        mode_log="${RESULTS_DIR}/${MODE_LOGS[$idx]}"
        out_file="${RESULTS_DIR}/bench_${TIMESTAMP}_${TEST}_${mode_id}.json"

        echo ""
        echo "  [$((idx + 1))/${#MODE_IDS[@]}] llama.cpp — ${mode_label}"
        if start_phase "${mode_label} stack" "$mode_script" "$mode_log"; then
            echo "   📊 Running ${RUNS} measured requests (${WARMUP} warmup)..."
            run_bench "$PROMPT" "$out_file"
            echo "   ✅ ${mode_label} done."
        else
            echo "   ⚠️  ${mode_label} skipped — ${LAST_STACK_FAILURE_REASON}."
        fi
    done

    stop_stack

    echo ""
    echo "  ┌─ ${TEST^^} results ───────────────────────────────────────────────────────┐"
    render_mode_table "$TEST"
    echo "  └───────────────────────────────────────────────────────────────────────────┘"
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    COMPARISON RESULTS                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"

"$PY" - "$RESULTS_DIR" "$TIMESTAMP" "$RESULTS_JSON" "$MODE_IDS_CSV" "$MODE_LABELS_CSV" "${TEST_TYPES[@]}" <<'PYEOF'
import json
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
timestamp = sys.argv[2]
combined_path = Path(sys.argv[3])
mode_ids = [m for m in sys.argv[4].split(",") if m]
mode_labels = [m for m in sys.argv[5].split(",") if m]
tests = sys.argv[6:]

def load(path: Path):
    try:
        data = json.loads(path.read_text())
        return data[0] if isinstance(data, list) else data
    except Exception:
        return None

def fmt(value, unit=""):
    return f"{value:.2f}{unit}" if value is not None else "n/a"

combined = {}

for test_name in tests:
    print()
    print(f"  {test_name.upper()}:")
    print(f"  {'Mode':<20}  {'tok/s':>12}  {'latency (s)':>12}  {'speedup vs plain':>17}")
    print(f"  {'-' * 20:<20}  {'-' * 12:>12}  {'-' * 12:>12}  {'-' * 17:>17}")

    plain_tps = None
    combined[test_name] = {}

    for mode_id, label in zip(mode_ids, mode_labels):
        path = results_dir / f"bench_{timestamp}_{test_name}_{mode_id}.json"
        data = load(path)
        tps = data.get("avg_completion_tokens_per_second") if data else None
        lat = data.get("avg_latency_seconds") if data else None
        if mode_id == "plain":
            plain_tps = tps
        if plain_tps and tps and plain_tps > 0:
            speedup = f"{tps / plain_tps:.2f}x"
        elif mode_id == "plain" and plain_tps:
            speedup = "1.00x"
        else:
            speedup = "n/a"

        print(f"  {label:<20}  {fmt(tps):>12}  {fmt(lat, 's'):>12}  {speedup:>17}")
        combined[test_name][mode_id] = {
            "summary": data,
            "speedup_vs_plain": None if speedup == "n/a" else speedup,
        }

combined_path.write_text(json.dumps(combined, indent=2))
print()
print(f"  💾 Full results saved to: {combined_path}")
print()
PYEOF

echo "  Logs:"
for idx in "${!MODE_IDS[@]}"; do
    printf '    %-18s logs/%s\n' "${MODE_LABELS[$idx]}:" "${MODE_LOGS[$idx]}"
done
echo ""
