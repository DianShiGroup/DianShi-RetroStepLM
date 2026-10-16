#!/bin/bash
# Export SLIME Megatron torch_dist checkpoint to HuggingFace format.
#
# Usage:
#   # Direct checkpoint path (preferred)
#   bash slime_retro/scripts/export_hf.sh --checkpoint-dir checkpoints_OG/slime_retro/iter_0000179
#
#   # Auto-detect from SAVE_DIR + STEP
#   STEP=89 bash slime_retro/scripts/export_hf.sh
#
#   # Custom output dir
#   bash slime_retro/scripts/export_hf.sh --checkpoint-dir /path/to/iter_0000179 --output-dir models/my_export

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
DOCKER_IMAGE="${DOCKER_IMAGE:-slimerl/slime:v0.3.0}"
MODEL="${MODEL:-qwen3_4b_instr_2507}"

# Parse named arguments
CHECKPOINT_DIR=""
OUTPUT_DIR=""
for arg in "$@"; do
    case "$arg" in
        --checkpoint-dir=*) CHECKPOINT_DIR="${arg#*=}" ;;
        --output-dir=*)     OUTPUT_DIR="${arg#*=}" ;;
    esac
done

# Resolve model name
case "${MODEL}" in
    qwen3_4b_instr_2507)    MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507" ;;
    *)                      echo "Unknown MODEL=${MODEL}"; exit 1 ;;
esac
MODEL_SHORT=$(echo "${MODEL_NAME}" | tr '/' '-' | tr '[:upper:]' '[:lower:]')
HF_DIR="${PROJECT_DIR}/models/${MODEL_NAME##*/}"

# Resolve input directory
if [[ -n "${CHECKPOINT_DIR}" ]]; then
    # Direct path: make absolute if relative
    if [[ "${CHECKPOINT_DIR}" != /* ]]; then
        INPUT_DIR="${PROJECT_DIR}/${CHECKPOINT_DIR}"
    else
        INPUT_DIR="${CHECKPOINT_DIR}"
    fi
else
    # Legacy: SAVE_DIR + STEP
    SAVE_DIR="${SAVE_DIR:-${PROJECT_DIR}/checkpoints/slime_retro_${MODEL_SHORT}}"
    if [[ -z "${STEP}" ]]; then
        STEP=$(cat "${SAVE_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null || echo "")
        if [[ -z "${STEP}" ]]; then
            echo "ERROR: No --checkpoint-dir or STEP specified"
            exit 1
        fi
        echo ">>> Auto-detected latest step: ${STEP}"
    fi
    INPUT_DIR="${SAVE_DIR}/iter_$(printf '%07d' ${STEP})"
fi

# Resolve output directory
if [[ -z "${OUTPUT_DIR}" ]]; then
    ITER_NAME=$(basename "${INPUT_DIR}")
    STEP_NUM=$(echo "${ITER_NAME}" | grep -oP '\d+' | sed 's/^0*//' || echo "unknown")
    CKPT_TAG=$(basename "$(dirname "$(dirname "${INPUT_DIR}")")" | tr '+' '_' | tr -cd 'a-zA-Z0-9_-')
    if [[ "${CKPT_TAG}" == "retrosynthesis" || -z "${CKPT_TAG}" ]]; then
        CKPT_TAG="default"
    fi
    OUTPUT_DIR="${PROJECT_DIR}/models/${MODEL_NAME##*/}_${CKPT_TAG}_step${STEP_NUM}"
fi
if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR}"
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "ERROR: Checkpoint not found: ${INPUT_DIR}"
    exit 1
fi

echo ">>> Exporting checkpoint:"
echo "  Input:  ${INPUT_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Origin HF: ${HF_DIR}"

docker run --rm --gpus all \
    -v "${PROJECT_DIR}:${PROJECT_DIR}" \
    "${DOCKER_IMAGE}" \
    python /root/slime/tools/convert_torch_dist_to_hf.py \
        --input-dir "${INPUT_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --origin-hf-dir "${HF_DIR}" \
        --force

echo ">>> Export complete: ${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}"/*.safetensors 2>/dev/null
