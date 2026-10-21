#!/bin/bash
# Standalone checkpoint conversion: HF → Megatron torch_dist
#
# Usage:
#   bash slime_retro/scripts/convert_chkpt.sh                                    # Qwen3-4B-Instruct-2507
#
# Prerequisites: nvidia-container-toolkit, Docker

set -ex

MODEL_NAME="${MODEL:-Qwen3-4B-Instruct-2507}"
MODELS_DIR="${MODELS_DIR:-$(pwd)/models}"
SLIME_IMAGE="slimerl/slime:v0.3.0"

# Map model name to SLIME's arch script and extra args
case "${MODEL_NAME}" in
    Qwen3-4B-Instruct-2507)
        SLIME_MODEL_SCRIPT="qwen3-4B-Instruct-2507.sh"
        MAX_POS_EMBED=262144
        EXTRA_ARGS=""
        ;;
    *)
        echo "Unknown model: ${MODEL_NAME}"; exit 1
        ;;
esac

HF_REPO="Qwen/${MODEL_NAME}"

# Download HF model if not present
if [ ! -d "${MODELS_DIR}/${MODEL_NAME}" ]; then
    pip install -q huggingface_hub
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${HF_REPO}', local_dir='${MODELS_DIR}/${MODEL_NAME}')
"
fi

echo ">>> HF checkpoint: ${MODELS_DIR}/${MODEL_NAME}"
ls "${MODELS_DIR}/${MODEL_NAME}"

# Convert HF → Megatron torch_dist inside container
docker run --rm --gpus all \
    -v "${MODELS_DIR}:/models" \
    "${SLIME_IMAGE}" \
    bash -c "
source /root/slime/scripts/models/${SLIME_MODEL_SCRIPT}
PYTHONPATH=/root/Megatron-LM python /root/slime/tools/convert_hf_to_torch_dist.py \
    \${MODEL_ARGS[@]} \
    --max-position-embeddings ${MAX_POS_EMBED} \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${HF_REPO} \
    --bf16 \
    ${EXTRA_ARGS} \
    --hf-checkpoint /models/${MODEL_NAME} \
    --save /models/${MODEL_NAME}_torch_dist
"

echo ">>> Conversion complete"
ls -la "${MODELS_DIR}/${MODEL_NAME}_torch_dist/"
