#!/bin/bash
# ============================================================================
# RetroAgent main evaluation entry point
#
# Runs the iterative-planning evaluation used in the paper: for each target,
# rollouts are issued until the search budget (unique single-step model calls)
# is exhausted or a route is found, repeated num_exp times for mean/std, with
# pass@1 available via max_rollouts=1.
#
# This is a THIN WRAPPER around `python -m slime_retro.eval`. It only:
#   1. checks the host tool server and the SGLang server(s) are reachable,
#   2. pins the reported protocol (greedy temperature 0, budget 500, num_exp 10),
#   3. execs the eval module, forwarding any extra key=value overrides.
#
# ----------------------------------------------------------------------------
# PREREQUISITES
#   1. Serve the checkpoint on SGLang (Docker); point --model-path at your
#      exported HF checkpoint directory:
#        docker run --rm --gpus '"device=0,1"' --network=host --ipc=host \
#          --shm-size=16g -v "$PWD:$PWD" -w "$PWD" slimerl/slime:v0.3.0 \
#          python3 -m sglang.launch_server --model-path /path/to/retroagent_hf --tp 2 \
#          --mem-fraction-static 0.85 --context-length 131072 \
#          --port 30000 --host 0.0.0.0 --trust-remote-code
#      (untrained baseline: use --model-path Qwen/Qwen3-4B-Instruct-2507)
#   2. Start the tool server on the host:
#        python -m tools.tool_server base_port=8100 num_instances=16
#
# ----------------------------------------------------------------------------
# USAGE
#   bash scripts/eval.sh                                   # USPTO-190, budget 500, greedy (T=0)
#   TARGETS=chembl-1000 bash scripts/eval.sh               # out-of-distribution set
#   TEMPERATURE=0.6 bash scripts/eval.sh                   # untrained zero-shot baseline
#   SGLANG_URL=http://localhost:30000,http://localhost:30001 bash scripts/eval.sh   # multi-replica
#   bash scripts/eval.sh max_rollouts=1                    # pass@1
#   bash scripts/eval.sh max_budget=200 num_exp=4          # forward any eval override
#
# ----------------------------------------------------------------------------
# KEY ENV VARS (all optional)
#   SGLANG_URL=http://localhost:30000     served checkpoint URL(s), comma-separated
#   TARGETS=uspto-190                     uspto-190 | chembl-1000
#   TEMPERATURE=0.0                       greedy for the trained model; 0.6 for zero-shot
#   RETRO_TOOL_SERVER_PORT=8100           host tool-server base port
#   RETRO_TOOL_SERVER_INSTANCES=16        host tool-server instance count
#   LOG_DIR=logs/slime_retro_eval         output directory
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

SGLANG_URL="${SGLANG_URL:-http://localhost:30000}"
TARGETS="${TARGETS:-uspto-190}"
RETRO_TOOL_SERVER_PORT="${RETRO_TOOL_SERVER_PORT:-8100}"
RETRO_TOOL_SERVER_INSTANCES="${RETRO_TOOL_SERVER_INSTANCES:-16}"
LOG_DIR="${LOG_DIR:-logs/slime_retro_eval}"

fail() { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Preflight checks (skip with SKIP_PREFLIGHT=1)
# ---------------------------------------------------------------------------
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
    # Tool server: base port and the top port of the requested instance range.
    base="${RETRO_TOOL_SERVER_PORT}"
    top=$(( base + RETRO_TOOL_SERVER_INSTANCES - 1 ))
    hint="Start it on the host:  python -m tools.tool_server base_port=${base} num_instances=${RETRO_TOOL_SERVER_INSTANCES}"
    curl -sf "http://127.0.0.1:${base}/metrics" >/dev/null 2>&1 \
        || fail "tool server not reachable at 127.0.0.1:${base}. ${hint}"
    curl -sf "http://127.0.0.1:${top}/metrics" >/dev/null 2>&1 \
        || fail "tool server instance on port ${top} not reachable; fewer than ${RETRO_TOOL_SERVER_INSTANCES} instances are running. ${hint}"
    echo ">>> Tool server OK on ports ${base}-${top}"

    # SGLang server(s): each comma-separated URL must answer.
    IFS=',' read -ra urls <<< "${SGLANG_URL}"
    for u in "${urls[@]}"; do
        curl -sf "${u}/get_model_info" >/dev/null 2>&1 \
            || fail "SGLang server not reachable at ${u}. Serve a checkpoint first (see the header of this script)."
    done
    echo ">>> SGLang OK: ${SGLANG_URL}"
fi

# ---------------------------------------------------------------------------
# Run the iterative-planning evaluation (released-weights defaults).
# Any extra key=value args are forwarded to the eval module.
# ---------------------------------------------------------------------------
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
echo ">>> Evaluating: targets=${TARGETS} sglang_url=${SGLANG_URL}"
# Trained-model evaluation uses greedy decoding (temperature 0), matching the
# paper. For the untrained zero-shot baseline, pass temperature=0.6 (greedy
# decoding produces degenerate outputs on the untrained model).
exec python -m slime_retro.eval \
    sglang_url="${SGLANG_URL}" \
    targets="${TARGETS}" \
    temperature="${TEMPERATURE:-0.0}" \
    max_budget=500 \
    num_exp=10 \
    tool_server_port="${RETRO_TOOL_SERVER_PORT}" \
    tool_server_instances="${RETRO_TOOL_SERVER_INSTANCES}" \
    log_dir="${LOG_DIR}" \
    "$@"
