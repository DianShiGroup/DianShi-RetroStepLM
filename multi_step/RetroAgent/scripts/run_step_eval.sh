#!/usr/bin/env bash
# Single entry point for every single-step evaluation.
#
# The shipped configuration is configs/dianshi_direct.yaml, which selects the
# `dianshi_direct` backend from the config's `backend` field rather than from a
# flag.
#
#   configs/dianshi_direct.yaml
#
# To evaluate a variant instead, copy that config, change what you need, and run
# it with --config <your copy>; the README lists what each key does.
#
# What it does, in order:
#   1. loads .env (real asset paths, endpoints, credentials)
#   2. checks the planner and the tool server are reachable
#   3. starts a tool server bound to the given step config
#   4. runs the evaluation against it, then shuts the tool server down
#
# Usage:
#   bash scripts/run_step_eval.sh --config configs/dianshi_direct.yaml \
#       --targets DianShi-RetroPlanSystem-Assets/test_dataset/uspto_190_targets_smiles.txt --out results/direct
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

# shellcheck disable=SC1091
[[ -f "${PROJECT_DIR}/.env" ]] && set -a && . "${PROJECT_DIR}/.env" && set +a

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
CONFIG=""
TARGETS="${TARGETS:-}"
OUT_DIR=""
LOG_DIR=""
PLANNER_URL="${PLANNER_BASE_URL:-http://127.0.0.1:8000/v1}"
PLANNER_MODEL="${PLANNER_MODEL_NAME:-planner}"
TOOL_PORT="${TOOL_SERVER_BASE_PORT:-8100}"
TOOL_INSTANCES="${TOOL_SERVER_INSTANCES:-1}"
CONCURRENCY="${CONCURRENCY:-8}"
KEEP_TOOL_SERVER="${KEEP_TOOL_SERVER:-0}"

die() { echo "ERROR: $*" >&2; exit 1; }
usage() { sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)      CONFIG="$2"; shift 2 ;;
    --targets)     TARGETS="$2"; shift 2 ;;
    --out)         OUT_DIR="$2"; shift 2 ;;
    --log-dir)     LOG_DIR="$2"; shift 2 ;;
    --planner-url) PLANNER_URL="$2"; shift 2 ;;
    --planner-model) PLANNER_MODEL="$2"; shift 2 ;;
    --instances)   TOOL_INSTANCES="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    -h|--help)     usage ;;
    *)             die "unknown argument: $1 (try --help)" ;;
  esac
done

[[ -n "${CONFIG}" ]] || die "--config is required (try --help)"
[[ -f "${CONFIG}" ]] || die "config not found: ${CONFIG}"
[[ -n "${TARGETS}" ]] || die "--targets is required (or set TARGETS in .env)"
[[ -f "${TARGETS}" ]] || die "targets file not found: ${TARGETS}"
[[ -x "${PYTHON_BIN}" ]] || die "python not found at ${PYTHON_BIN} (set PYTHON_BIN)"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export RETRO_SINGLE_STEP_CONFIG="${PROJECT_DIR}/${CONFIG}"

# Name results after the config and target set so datasets and A/B variants
# cannot overwrite each other.
config_stem="$(basename "${CONFIG}" .yaml)"
targets_stem="$(basename "${TARGETS}" .txt)"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/results/${config_stem}/${targets_stem}}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs/${config_stem}/${targets_stem}}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

# Which step backend did we pick? Read it back through the same loader the
# server uses, so ${VAR} expansion and any future validation stay identical.
backend="$("${PYTHON_BIN}" - "${CONFIG}" <<'PY'
import sys
from inference.support.config_loader import load_yaml
cfg = load_yaml(sys.argv[1]) or {}
print(cfg.get("backend") or ((cfg.get("retro") or {}).get("backend")) or "dianshi_direct")
PY
)"
gate="$("${PYTHON_BIN}" - "${CONFIG}" <<'PY'
import sys
from inference.support.config_loader import load_yaml
cfg = load_yaml(sys.argv[1]) or {}
print("on" if (cfg.get("template_filter") or {}).get("enabled") else "off")
PY
)"

echo ">>> config     : ${CONFIG}"
echo ">>> backend    : ${backend} (template_gate=${gate})"
echo ">>> targets    : ${TARGETS}"
echo ">>> out        : ${OUT_DIR}"

# --- preflight -------------------------------------------------------------
curl --noproxy '*' -sf "${PLANNER_URL%/}/models" >/dev/null \
  || die "planner not reachable at ${PLANNER_URL}; start scripts/serve_planner.sh"
echo ">>> planner OK : ${PLANNER_URL} (${PLANNER_MODEL})"

top_port=$(( TOOL_PORT + TOOL_INSTANCES - 1 ))
for port in $(seq "${TOOL_PORT}" "${top_port}"); do
  if curl --noproxy '*' -sf "http://127.0.0.1:${port}/metrics" >/dev/null 2>&1; then
    [[ "${KEEP_TOOL_SERVER}" == "1" ]] && die "port ${port} is already serving; stop it or pass --instances"
  fi
done

# --- tool server -----------------------------------------------------------
echo ">>> starting tool server on ${TOOL_PORT}-${top_port} with ${CONFIG}"
"${PYTHON_BIN}" -m tools.tool_server \
    base_port="${TOOL_PORT}" \
    num_instances="${TOOL_INSTANCES}" \
    > "${LOG_DIR}/tool_server.log" 2>&1 &
tool_pid=$!

cleanup() {
  if [[ "${KEEP_TOOL_SERVER}" != "1" ]] && kill -0 "${tool_pid}" 2>/dev/null; then
    echo ">>> stopping tool server (pid ${tool_pid})"
    kill "${tool_pid}" 2>/dev/null || true
    wait "${tool_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for port in $(seq "${TOOL_PORT}" "${top_port}"); do
  ready=0
  for _ in $(seq 1 120); do
    if curl --noproxy '*' -sf "http://127.0.0.1:${port}/metrics" >/dev/null 2>&1; then
      ready=1; break
    fi
    kill -0 "${tool_pid}" 2>/dev/null || { tail -40 "${LOG_DIR}/tool_server.log" >&2; die "tool server exited"; }
    sleep 1
  done
  [[ "${ready}" == "1" ]] || { tail -40 "${LOG_DIR}/tool_server.log" >&2; die "tool server on ${port} never became ready"; }
done
echo ">>> tool server OK"

# --- evaluation ------------------------------------------------------------
"${PYTHON_BIN}" -m slime_retro.eval \
  planner_backend=vllm \
  planner_url="${PLANNER_URL}" \
  planner_model="${PLANNER_MODEL}" \
  planner_api_key="${PLANNER_API_KEY:-}" \
  model_name="${TOKENIZER_MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}" \
  targets="${TARGETS}" \
  max_budget="${MAX_BUDGET:-500}" \
  num_exp="${NUM_EXP:-1}" \
  milestones="${MILESTONES:-500}" \
  temperature="${TEMPERATURE:-0.0}" \
  predict_templates_top_k="${PREDICT_TEMPLATES_TOP_K:-10}" \
  concurrency="${CONCURRENCY}" \
  tool_server_port="${TOOL_PORT}" \
  tool_server_instances="${TOOL_INSTANCES}" \
  log_dir="${LOG_DIR}" \
  "$@"

echo ">>> done: ${LOG_DIR}"