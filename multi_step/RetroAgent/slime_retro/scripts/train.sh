#!/bin/bash
# Start slime training inside Docker container.
#
# Prerequisites:
#   1. Prepare data (see scripts/README.md)
#   2. Start tool server on host:
#      python -m tools.tool_server base_port=8100 num_instances=16
#
# Usage:
#   bash slime_retro/scripts/train.sh                                    # Qwen3-4B (default)
#
# Override defaults:
#   OFF_POLICY=0 DYNAMIC_SAMPLING=0 bash slime_retro/scripts/train.sh

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
DOCKER_IMAGE="${DOCKER_IMAGE:-slimerl/slime:v0.3.0}"
MODEL="${MODEL:-qwen3_4b_instr_2507}"

# Enable TIS and dynamic sampling by default
OFF_POLICY="${OFF_POLICY:-1}"
DYNAMIC_SAMPLING="${DYNAMIC_SAMPLING:-1}"

docker run --rm --gpus all --network=host --ipc=host --shm-size=16g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "${PROJECT_DIR}:${PROJECT_DIR}" \
    -v "$HOME/.netrc:/root/.netrc:ro" \
    -e OFF_POLICY="${OFF_POLICY}" \
    -e DYNAMIC_SAMPLING="${DYNAMIC_SAMPLING}" \
    -e RETRO_TOOL_SERVER_INSTANCES="${RETRO_TOOL_SERVER_INSTANCES:-16}" \
    -e COLOCATE="${COLOCATE:-1}" \
    -e ACTOR_NUM_GPUS="${ACTOR_NUM_GPUS:-8}" \
    -e ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-0}" \
    -e ADV_ESTIMATOR="${ADV_ESTIMATOR:-gspo}" \
    -e KL_COEF="${KL_COEF:-0.001}" \
    -e ENTROPY_COEF="${ENTROPY_COEF:-0.00}" \
    ${EVAL_TEMPERATURE:+-e EVAL_TEMPERATURE="${EVAL_TEMPERATURE}"} \
    ${EVAL_TOP_K:+-e EVAL_TOP_K="${EVAL_TOP_K}"} \
    ${EVAL_TOP_P:+-e EVAL_TOP_P="${EVAL_TOP_P}"} \
    ${SAVE_DIR:+-e SAVE_DIR="${SAVE_DIR}"} \
    ${RETRO_PREDICT_TEMPLATES_TOP_K:+-e RETRO_PREDICT_TEMPLATES_TOP_K="${RETRO_PREDICT_TEMPLATES_TOP_K}"} \
    ${RETRO_EXPAND_MODE:+-e RETRO_EXPAND_MODE="${RETRO_EXPAND_MODE}"} \
    ${RETRO_REWARD_DEPTH_THRESHOLD:+-e RETRO_REWARD_DEPTH_THRESHOLD="${RETRO_REWARD_DEPTH_THRESHOLD}"} \
    ${RETRO_REWARD_DEPTH_RATE:+-e RETRO_REWARD_DEPTH_RATE="${RETRO_REWARD_DEPTH_RATE}"} \
    ${RETRO_REWARD_DEPTH_CAP:+-e RETRO_REWARD_DEPTH_CAP="${RETRO_REWARD_DEPTH_CAP}"} \
    ${RETRO_MAX_TOTAL_TOKENS:+-e RETRO_MAX_TOTAL_TOKENS="${RETRO_MAX_TOTAL_TOKENS}"} \
    "${DOCKER_IMAGE}" \
    bash "${PROJECT_DIR}/slime_retro/scripts/models/${MODEL}.sh"
