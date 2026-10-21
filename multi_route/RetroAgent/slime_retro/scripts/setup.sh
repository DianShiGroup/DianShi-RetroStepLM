#!/bin/bash
# Setup script for a RetroAgent training/evaluation server.
# Installs host dependencies, downloads data + weights from HuggingFace, and
# pulls the SLIME Docker image.
#
# Architecture:
#   - Tool server runs on the HOST (Python 3.11 + project deps in requirements.txt)
#   - SLIME training runs inside DOCKER (ray, megatron, sglang pre-installed)
#   - Docker uses --network=host so training can reach the tool server at localhost:8100
#
# Usage:
#   git clone https://github.com/SXKDZ/RetroAgent.git && cd RetroAgent
#   bash slime_retro/scripts/setup.sh

set -ex

HF_REPO="${HF_REPO:-SXKDZ/RetroAgent}"          # HuggingFace repo hosting data + weights
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"

# ==========================================================================
# Host dependencies (for the tool server)
# ==========================================================================

# chz requires Python >= 3.11
if ! command -v python3.11 &>/dev/null; then
    echo ">>> Please install Python 3.11+ before running this script." >&2
    exit 1
fi

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install "huggingface_hub[cli]"

# ==========================================================================
# Data + trained weights (hosted on HuggingFace)
# ==========================================================================
# The trained RetroAgent checkpoint lives on HuggingFace.
huggingface-cli download "${HF_REPO}" --repo-type model --local-dir "${PROJECT_DIR}/hf_release"

# The datasets/ files are NOT on HuggingFace: they come from upstream releases
# (DESP, LLM-Syn-Planner, PDVN, SCScore). See README "Data Preparation".
python3 -m scripts.prepare_datasets || {
    echo ">>> scripts.prepare_datasets failed. datasets/ is incomplete; see README 'Data Preparation'." >&2
    exit 1
}
for f in processed/building_blocks.parquet idx2template_retro.json model_retro.pt \
         scscore/model.ckpt-10654.as_numpy.json.gz; do
    [ -s "${PROJECT_DIR}/datasets/${f}" ] || {
        echo ">>> Missing required data file: datasets/${f}" >&2
        exit 1
    }
done

# ==========================================================================
# SLIME (model config scripts) + Docker image (for training)
# ==========================================================================
if [ ! -d "${PROJECT_DIR}/external/slime" ]; then
    git clone https://github.com/THUDM/slime.git "${PROJECT_DIR}/external/slime"
fi
docker pull slimerl/slime:v0.3.0

# Optional: log in to Weights & Biases to record training metrics
# wandb login

echo ">>> Setup complete. Project at: ${PROJECT_DIR}"
echo ">>> Next steps:"
echo "  1. Start the tool server (host):"
echo "     PYTHONPATH=${PROJECT_DIR} python3 -m tools.tool_server base_port=8100 num_instances=16"
echo "  2. Start training (docker):  bash slime_retro/scripts/train.sh"
