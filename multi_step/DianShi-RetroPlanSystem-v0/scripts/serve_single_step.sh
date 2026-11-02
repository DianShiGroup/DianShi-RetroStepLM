#!/usr/bin/env bash
# Serve the single-step retrosynthesis model behind an OpenAI-compatible
# endpoint. Same shape as scripts/serve_planner.sh; kept separate because the
# two models are usually different sizes on different GPUs.
#
#   bash scripts/serve_single_step.sh --model-path /path/to/checkpoint
#
# The name it is served under must match SINGLE_STEP_MODEL in the config.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

# shellcheck disable=SC1091
[[ -f "${PROJECT_DIR}/.env" ]] && set -a && . "${PROJECT_DIR}/.env" && set +a

MODEL_PATH="${SINGLE_STEP_MODEL_PATH:-}"
SERVED_NAME="${SINGLE_STEP_MODEL:-qwen3-4b}"
PORT="${SINGLE_STEP_PORT:-10120}"
TP="${SINGLE_STEP_TP:-1}"
MAX_MODEL_LEN="${SINGLE_STEP_MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${SINGLE_STEP_GPU_MEM_UTIL:-0.90}"
LOG_DIR="${SINGLE_STEP_LOG_DIR:-${PROJECT_DIR}/logs/single_step}"

die() { echo "ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)    MODEL_PATH="$2"; shift 2 ;;
    --model-name)    SERVED_NAME="$2"; shift 2 ;;
    --port)          PORT="$2"; shift 2 ;;
    --tp)            TP="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    -h|--help)       sed -n '2,8p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)               die "unknown argument: $1" ;;
  esac
done

[[ -n "${MODEL_PATH}" ]] || die "set SINGLE_STEP_MODEL_PATH in .env or pass --model-path"
# Any OpenAI-compatible server works; this is the one we used.
command -v vllm >/dev/null 2>&1 || die "vllm not on PATH (pip install 'vllm>=0.18'; validated with 0.26)"
[[ -e "${MODEL_PATH}" ]] || die "model path does not exist: ${MODEL_PATH}"
mkdir -p "${LOG_DIR}"

echo ">>> Serving ${MODEL_PATH}"
echo ">>> registered as: ${SERVED_NAME}"
# The evaluation YAMLs in configs/bench/ read $SINGLE_STEP_API_URL and
# $SINGLE_STEP_MODEL from .env.
echo ">>> endpoint: http://127.0.0.1:${PORT}/v1  (model ${SERVED_NAME})"
echo ">>>   name=${SERVED_NAME} port=${PORT} tp=${TP} max_model_len=${MAX_MODEL_LEN}"

exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --trust-remote-code \
  2>&1 | tee "${LOG_DIR}/single_step-${PORT}.log"