#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"

INPUT_FILE="${INPUT_FILE:?Set INPUT_FILE to a one-SMILES-per-line text file}"
PLANNER_URL="${PLANNER_URL:?Set PLANNER_URL to the planner OpenAI base URL, e.g. http://host:port/v1}"
PLANNER_MODEL="${PLANNER_MODEL:-RetroAgent}"
DIANSHI_DIRECT_API_KEY="${DIANSHI_DIRECT_API_KEY:-}"
TOOL_SERVER_URL="${TOOL_SERVER_URL:-http://127.0.0.1}"
TOOL_SERVER_PORT="${RETRO_TOOL_SERVER_PORT:-8100}"
TOOL_SERVER_INSTANCES="${RETRO_TOOL_SERVER_INSTANCES:-4}"
LOG_DIR="${LOG_DIR:?Set LOG_DIR to an adapter-specific output directory}"
CONCURRENCY="${CONCURRENCY:-8}"

test -x "${PYTHON_BIN}"
test -f "${INPUT_FILE}"
"${PYTHON_BIN}" -c 'import sys; print("python:", sys.executable)'

base_port="${TOOL_SERVER_PORT}"
top_port=$((base_port + TOOL_SERVER_INSTANCES - 1))
PLANNER_HEALTH_HEADERS=()
if [[ -n "${PLANNER_API_KEY}" ]]; then
  PLANNER_HEALTH_HEADERS=(-H "Authorization: Bearer ${PLANNER_API_KEY}")
fi
curl --noproxy '*' -sf "${TOOL_SERVER_URL}:${base_port}/metrics" >/dev/null
curl --noproxy '*' -sf "${TOOL_SERVER_URL}:${top_port}/metrics" >/dev/null
curl --noproxy '*' -sf "${PLANNER_HEALTH_HEADERS[@]}" "${PLANNER_URL%/}/models" >/dev/null

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

exec "${PYTHON_BIN}" -m slime_retro.eval \
  planner_backend=vllm \
  planner_url="${PLANNER_URL}" \
  planner_model="${PLANNER_MODEL}" \
  planner_api_key="${PLANNER_API_KEY}" \
  model_name="${TOKENIZER_MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}" \
  targets="${INPUT_FILE}" \
  max_budget=500 \
  num_exp=1 \
  milestones=500 \
  temperature=0.0 \
  predict_templates_top_k=10 \
  concurrency="${CONCURRENCY}" \
  tool_server_url="${TOOL_SERVER_URL}" \
  tool_server_port="${TOOL_SERVER_PORT}" \
  tool_server_instances="${TOOL_SERVER_INSTANCES}" \
  log_dir="${LOG_DIR}" \
  "$@"
