#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=/mnt/shared-storage-user/mineru4s/anxiao/ms-swift/output/0314_cot_v1_4b_10e_atomwise/v0-20260811-101041/checkpoint-47682
VLLM_API_KEY="${VLLM_API_KEY:-}"
PORT_START="${PORT_START:-10120}"
PORT_END="${PORT_END:-10150}"
GPU_NUM="${GPU_NUM:-8}"
LOG_DIR="${LOG_DIR:-logs/vllm_instances_direct}"
SERVICE_INFO_FILE="${SERVICE_INFO_FILE:-}"

SCRIPT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
RJOB_NAME="${RJOB_NAME:-rjob-lxb-vllm-retro-0314_cot_v1_4b_10e_atomwise}"
VLLM_ENV=/mnt/shared-storage-user/mineru4s/lixibao/envs/vllm026
CUDA_HOME=/mnt/shared-storage-user/mineru4s/share/library/cuda-13.2
VLLM_BIN="${VLLM_ENV}/bin/vllm"
HOST_NETWORK_ARG=()
if [[ "${HOST_NETWORK:-0}" == "1" ]]; then
  HOST_NETWORK_ARG+=(--host-network=true)
fi

if [[ "${INSIDE_RJOB:-0}" != "1" ]]; then
  rjob submit \
    --name="${RJOB_NAME}" \
    --gpu="${GPU_NUM}" \
    --memory=512000 \
    --cpu=64 \
    --enable-sshd \
    "${HOST_NETWORK_ARG[@]}" \
    --namespace=ailab-chemdata \
    --charged-group=chemdata_gpu_pool \
    --mount=gpfs://gpfs1/mineru4s:/mnt/shared-storage-user/mineru4s \
    --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
    --image=registry.h.pjlab.org.cn/library/ml-base:22.04-pjlab \
    -P 1 \
    -e DISTRIBUTED_JOB=false \
    -e INSIDE_RJOB=1 \
    -e PRIVATE_LOG_DIR="${LOG_DIR}" \
    -e PRIVATE_GPU_NUM="${GPU_NUM}" \
    -e PRIVATE_PORT_START="${PORT_START}" \
    -e PRIVATE_PORT_END="${PORT_END}" \
    -e PRIVATE_SERVICE_INFO_FILE="${SERVICE_INFO_FILE}" \
    -- bash -exc "bash ${SCRIPT_PATH}"
  exit 0
fi

export HF_HOME="/mnt/shared-storage-user/mineru4s/share/huggingface"
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${VLLM_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

PROJECT_ROOT="/mnt/shared-storage-user/mineru4s/lixibao/retro_app/retro_app_backend"
cd "${PROJECT_ROOT}"

LOG_DIR="${PRIVATE_LOG_DIR:-${LOG_DIR}}"
GPU_NUM="${PRIVATE_GPU_NUM:-${GPU_NUM}}"
PORT_START="${PRIVATE_PORT_START:-${PORT_START}}"
PORT_END="${PRIVATE_PORT_END:-${PORT_END}}"
SERVICE_INFO_FILE="${PRIVATE_SERVICE_INFO_FILE:-${SERVICE_INFO_FILE}}"

is_port_in_use() {
  local port="$1"
  timeout 0.25 bash -c ">/dev/tcp/127.0.0.1/${port}" >/dev/null 2>&1
}

find_next_available_port() {
  local start_port="$1"
  for p in $(seq "${start_port}" "${PORT_END}"); do
    if ! is_port_in_use "$p"; then
      echo "$p"
      return 0
    fi
  done
  return 1
}

mkdir -p "${LOG_DIR}"

CURRENT_PORT="$(find_next_available_port "${PORT_START}")" || {
  echo "No available port in range ${PORT_START}-${PORT_END}, exiting." >&2
  exit 1
}

if is_port_in_use "${CURRENT_PORT}"; then
  echo "Port ${CURRENT_PORT} is already in use, exiting." >&2
  exit 1
fi

if [[ -n "${SERVICE_INFO_FILE}" ]]; then
  SERVICE_IP="$(hostname -I | awk '{print $1}')"
  test -n "${SERVICE_IP}"
  mkdir -p "$(dirname "${SERVICE_INFO_FILE}")"
  {
    printf 'ip=%s\n' "${SERVICE_IP}"
    printf 'model=qwen3-4b\n'
    printf 'endpoint=http://%s:%s/v1/chat/completions\n' "${SERVICE_IP}" "${CURRENT_PORT}"
  } > "${SERVICE_INFO_FILE}"
  echo "Service manifest: ${SERVICE_INFO_FILE}"
  sed 's/^/  /' "${SERVICE_INFO_FILE}"
fi

echo "Starting single vLLM server: dp=${GPU_NUM}, tp=1, port=${CURRENT_PORT}"

API_KEY_ARGS=()
[[ -n "${VLLM_API_KEY}" ]] && API_KEY_ARGS+=(--api-key "${VLLM_API_KEY}")

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --dtype auto \
  "${API_KEY_ARGS[@]}" \
  --served-model-name qwen3-4b \
  --tensor-parallel-size 1 \
  --data-parallel-size "${GPU_NUM}" \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.85 \
  --port "${CURRENT_PORT}"
