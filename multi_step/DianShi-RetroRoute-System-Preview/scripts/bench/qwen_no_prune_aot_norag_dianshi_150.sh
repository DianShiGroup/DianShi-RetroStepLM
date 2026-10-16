#!/usr/bin/env bash
# CPU rjob: Qwen 规划、不开主动剪枝、AOT* apply no-RAG / dianshi_150
# Config: configs/bench/qwen_no_prune_aot_norag_dianshi_150.yaml
#
#   bash scripts/bench/qwen_no_prune_aot_norag_dianshi_150.sh
set -euo pipefail

SCRIPT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/bench/qwen_no_prune_aot_norag_dianshi_150.yaml"
PYTHON_BIN="/mnt/shared-storage-user/mineru4s/lixibao/retro_app/retro_app_backend/.venv/bin/python"
RJOB_NAME="${RJOB_NAME:-rjob-lxb-bench-qwen_no_prune_aot_norag_dianshi_150}"
CHARGED_GROUP="${CHARGED_GROUP:-chemdata_gpu_pool}"
CPU_COUNT="${CPU_COUNT:-8}"
MEMORY_MIB="${MEMORY_MIB:-32768}"

if [[ "${INSIDE_RJOB:-0}" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES=""
  cd "${PROJECT_ROOT}"
  exec "${PYTHON_BIN}" cli.py --config "${CONFIG}"
fi

test -f "${CONFIG}"
test -x "${PYTHON_BIN}"

rjob submit \
  --name="${RJOB_NAME}" \
  --gpu=0 \
  --memory="${MEMORY_MIB}" \
  --cpu="${CPU_COUNT}" \
  --enable-sshd \
  --namespace=ailab-chemdata \
  --charged-group="${CHARGED_GROUP}" \
  --mount=gpfs://gpfs1/mineru4s:/mnt/shared-storage-user/mineru4s \
  --mount=gpfs://gpfs2/gpfs2-shared-public:/mnt/shared-storage-gpfs2/gpfs2-shared-public \
  --image=registry.h.pjlab.org.cn/library/ml-base:22.04-pjlab \
  -P 1 \
  -e DISTRIBUTED_JOB=false \
  -e INSIDE_RJOB=1 \
  -- bash -exc "bash '${SCRIPT_PATH}'"
