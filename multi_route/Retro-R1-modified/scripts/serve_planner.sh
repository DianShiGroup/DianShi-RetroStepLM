#!/usr/bin/env bash
# Serve the planner LLM behind an OpenAI-compatible endpoint.
#
# Plain vLLM, no cluster scheduler: run it on the machine that has the GPUs,
# then point the evaluation at it. Every value comes from .env (see
# .env.example), so nothing here needs editing.
#
#   cp .env.example .env && $EDITOR .env
#   bash scripts/serve_planner.sh              # foreground
#   bash scripts/serve_planner.sh --port 8080  # override one value
#
# The single-step model is served separately (scripts/serve_single_step.sh) or
# by any other OpenAI-compatible server; it does not have to be this one.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

# shellcheck disable=SC1091
[[ -f "${PROJECT_DIR}/.env" ]] && set -a && . "${PROJECT_DIR}/.env" && set +a

MODEL_PATH="${PLANNER_MODEL_PATH:?Set PLANNER_MODEL_PATH in .env (or pass --model-path)}"
SERVED_NAME="${PLANNER_MODEL_NAME:-planner}"
PORT="${PLANNER_PORT:-8000}"
TP="${PLANNER_TP:-8}"
MAX_MODEL_LEN="${PLANNER_MAX_MODEL_LEN:-262144}"
GPU_MEM_UTIL="${PLANNER_GPU_MEM_UTIL:-0.90}"
LOG_DIR="${PLANNER_LOG_DIR:-${PROJECT_DIR}/logs/planner}"

die() { echo "ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)    MODEL_PATH="$2"; shift 2 ;;
    --model-name)    SERVED_NAME="$2"; shift 2 ;;
    --port)          PORT="$2"; shift 2 ;;
    --tp)            TP="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    -h|--help)       sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)               die "unknown argument: $1" ;;
  esac
done

# Any OpenAI-compatible server works; this is the one we used.
command -v vllm >/dev/null 2>&1 || die "vllm not on PATH (pip install 'vllm>=0.18'; validated with 0.26)"
[[ -e "${MODEL_PATH}" ]] || die "model path does not exist: ${MODEL_PATH}"
mkdir -p "${LOG_DIR}"

echo ">>> Serving ${MODEL_PATH}"
echo ">>> registered as: ${SERVED_NAME}"
# Point configs/planner.yaml at this; it reads $PLANNER_BASE_URL / $PLANNER_MODEL_NAME.
echo ">>> endpoint: http://127.0.0.1:${PORT}/v1  (model ${SERVED_NAME})"
echo ">>>   name=${SERVED_NAME} port=${PORT} tp=${TP} max_model_len=${MAX_MODEL_LEN}"
echo ">>> Log: ${LOG_DIR}/planner-${PORT}.log"

# The served name must match PLANNER_MODEL_NAME in the eval config, otherwise
# the client asks for a model the server does not know.
exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --trust-remote-code \
  2>&1 | tee "${LOG_DIR}/planner-${PORT}.log"