#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
RJOB_NAME="${RJOB_NAME:-rjob-lxb-vllm-serve-qwen35-397b}"
VLLM_ENV=/mnt/shared-storage-user/mineru4s/lixibao/envs/vllm026
CUDA_HOME=/mnt/shared-storage-user/mineru4s/share/library/cuda-13.2
VLLM_BIN="${VLLM_ENV}/bin/vllm"
MODEL_PATH=/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/9f1f3de9a3a48cfd340fd73ca98c02625b7afb3b
SERVED_MODEL=qwen3.5-397b
PORT="${PORT:-8099}"
GPU_COUNT=8
VLLM_API_KEY="${VLLM_API_KEY:-}"

if [[ "${INSIDE_RJOB:-0}" == "1" ]]; then
  PORT="${PRIVATE_PORT:-${PORT}}"
  export HF_HOME="/mnt/shared-storage-user/mineru4s/share/huggingface"
  export CUDA_HOME
  export PATH="${CUDA_HOME}/bin:${VLLM_ENV}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

  API_KEY_ARGS=()
  [[ -n "${VLLM_API_KEY}" ]] && API_KEY_ARGS+=(--api-key "${VLLM_API_KEY}")

  exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
    "${API_KEY_ARGS[@]}" \
    --max-model-len 262144 \
    --port "${PORT}" \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --served-model-name "${SERVED_MODEL}" \
    --tensor-parallel-size "${GPU_COUNT}" \
    --language-model-only \
    --enable-prefix-caching \
    --reasoning-parser qwen3
fi

rjob submit \
  --name="${RJOB_NAME}" \
  --gpu="${GPU_COUNT}" \
  --memory=1048576 \
  --cpu=128 \
  --enable-sshd \
  --host-network=true \
  --namespace=ailab-chemdata \
  --charged-group=chemdata_gpu_pool \
  --mount=gpfs://gpfs1/mineru4s:/mnt/shared-storage-user/mineru4s \
  --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
  --image=registry.h.pjlab.org.cn/library/ml-base:22.04-pjlab \
  -P 1 \
  --custom-resources rdma/mlnx_shared="${GPU_COUNT}" \
  --custom-resources mellanox.com/mlnx_rdma=1 \
  -e DISTRIBUTED_JOB=false \
  -e INSIDE_RJOB=1 \
  -e PRIVATE_PORT="${PORT}" \
  -- bash -exc "bash ${SCRIPT_PATH}"
