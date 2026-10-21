#!/bin/bash
# ============================================================================
# RetroAgent main training entry point
#
# Trains the retrosynthesis planning agent with RL (GSPO by default) on top of
# Qwen3-4B-Instruct-2507, using the slime framework (Megatron + SGLang) inside
# the slimerl/slime:v0.3.0 Docker image on a single 8xH200 node.
#
# This is a THIN WRAPPER around slime_retro/scripts/train.sh. It does NOT
# reimplement the Docker/Ray/Megatron plumbing; it only:
#   1. runs preflight checks that a fresh clone otherwise fails silently on,
#   2. pins the released/paper defaults (model, GSPO, KL, reward penalties),
#   3. execs the real launcher (slime_retro/scripts/train.sh).
#
# ----------------------------------------------------------------------------
# ARCHITECTURE (host / Docker split, the most non-obvious operational detail)
#   - The HTTP tool server runs on the HOST (not in Docker).
#   - Training runs INSIDE Docker; it reaches the host tool server at
#     127.0.0.1:8100 because the container is launched with --network=host.
#
# ----------------------------------------------------------------------------
# PREREQUISITES (run in this order; see slime_retro/scripts/README.md)
#   1. Docker image:      docker pull slimerl/slime:v0.3.0
#   2. Host Python deps:  pip install -r requirements.txt
#   3. Data files in datasets/ (provided separately; see README Data Preparation):
#        building blocks, templates, single-step model (model_retro.pt),
#        target sets, and the reference routes used for data preparation.
#   4. Start the tool server ON THE HOST (leave it running):
#        python -m tools.tool_server base_port=8100 num_instances=16
#   5. Prepare training + eval parquets (tool server must be up):
#        python -m slime_retro.prepare_data depth_min=5 depth_max=30
#        python -m slime_retro.prepare_data output=datasets/slime_retro_eval_uspto190.parquet targets=uspto-190
#   6. Convert HF weights -> Megatron torch_dist (once):
#        bash slime_retro/scripts/convert_chkpt.sh
#   7. Train:
#        bash scripts/train.sh
#
# ----------------------------------------------------------------------------
# USAGE
#   bash scripts/train.sh                                  # Qwen3-4B, GSPO (defaults)
#   ADV_ESTIMATOR=grpo bash scripts/train.sh               # switch algorithm to GRPO
#   SKIP_PREFLIGHT=1 bash scripts/train.sh                 # bypass the preflight checks
#   EVAL_TEMPERATURE=0.6 EVAL_TOP_K=20 bash scripts/train.sh   # sampling-based in-run eval
#
# ----------------------------------------------------------------------------
# RELEASED CONFIG (single 8xH200 node, colocated, TP=2)
#   Model      : Qwen/Qwen3-4B-Instruct-2507  (context 65536, TP=2, PP=1)
#   Algorithm  : GSPO (sequence-level clip 3.5e-4); GRPO available via override
#   KL         : low_var_kl, coef 0.001;  entropy coef 0.0
#   Optimizer  : Adam, lr 1e-6 constant, wd 0.01, betas (0.9, 0.98)
#   Rollout    : batch 32 x group 8, response<=8192 tok, temp 1.0
#   Off-policy : TIS on (num-steps-per-rollout 2 -> global batch 128)
#   Sampling   : DAPO dynamic sampling on (over-sampling 48, nonzero-std filter)
#
#   Reward (per episode; see slime_retro/config.py + slime_retro/utils.py):
#     success base           = +1.0
#     delta_N  (budget)      = -0.02 per predict_templates call above 20, cap 0.3
#     delta_D  (route depth) = -0.02 per depth level above 20,            cap 0.3
#     per-turn tool failure / no-parseable-call = -0.05 each
#     route not completed    = -1.0
#     zero valid tool calls  = -1.5  (format failure / overthinking)
#   N-penalty knobs are fixed in config.py; delta_D knobs are env-overridable
#   below (RETRO_REWARD_DEPTH_*).
#
# ----------------------------------------------------------------------------
# KEY ENV VARS (all optional; defaults shown are the released config)
#   MODEL=qwen3_4b_instr_2507      per-model script under slime_retro/scripts/models/
#   ADV_ESTIMATOR=gspo             gspo | grpo
#   KL_COEF=0.001                  KL loss coefficient
#   ENTROPY_COEF=0.00              entropy coefficient
#   ACTOR_NUM_GPUS=8               GPUs for training (released config: 8)
#   COLOCATE=1                     share GPUs between SGLang inference + training
#   OFF_POLICY=1                   TIS off-policy correction
#   DYNAMIC_SAMPLING=1             DAPO dynamic sampling
#   RETRO_TOOL_SERVER_PORT=8100    host tool-server base port
#   RETRO_TOOL_SERVER_INSTANCES=16 host tool-server instance count
#   RETRO_MAX_TOTAL_TOKENS=56000   per-episode token budget (< sglang ctx - 8192)
#   RETRO_REWARD_DEPTH_THRESHOLD=20 / _RATE=0.02 / _CAP=0.3   depth penalty knobs
#   DOCKER_IMAGE=slimerl/slime:v0.3.0
# ============================================================================

set -euo pipefail

# Derive PROJECT_DIR from this script's location, no hardcoded absolute paths.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

# ---------------------------------------------------------------------------
# Released / paper defaults (override any of these from the environment)
# ---------------------------------------------------------------------------
export MODEL="${MODEL:-qwen3_4b_instr_2507}"     # Qwen3-4B-Instruct-2507
export ADV_ESTIMATOR="${ADV_ESTIMATOR:-gspo}"    # GSPO is the project/paper default
export KL_COEF="${KL_COEF:-0.001}"
export ENTROPY_COEF="${ENTROPY_COEF:-0.00}"

# GPU / rollout layout (single 8xH200 node, colocated)
export ACTOR_NUM_GPUS="${ACTOR_NUM_GPUS:-8}"
export ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-0}"
export COLOCATE="${COLOCATE:-1}"
export OFF_POLICY="${OFF_POLICY:-1}"
export DYNAMIC_SAMPLING="${DYNAMIC_SAMPLING:-1}"

# Episode / tool-server config
export RETRO_TOOL_SERVER_PORT="${RETRO_TOOL_SERVER_PORT:-8100}"
export RETRO_TOOL_SERVER_INSTANCES="${RETRO_TOOL_SERVER_INSTANCES:-16}"
export RETRO_MAX_TOTAL_TOKENS="${RETRO_MAX_TOTAL_TOKENS:-56000}"

# Reward: route-depth penalty (delta_D). N-penalty is fixed in config.py.
export RETRO_REWARD_DEPTH_THRESHOLD="${RETRO_REWARD_DEPTH_THRESHOLD:-20}"
export RETRO_REWARD_DEPTH_RATE="${RETRO_REWARD_DEPTH_RATE:-0.02}"
export RETRO_REWARD_DEPTH_CAP="${RETRO_REWARD_DEPTH_CAP:-0.3}"

export DOCKER_IMAGE="${DOCKER_IMAGE:-slimerl/slime:v0.3.0}"

# ---------------------------------------------------------------------------
# Preflight checks (skip with SKIP_PREFLIGHT=1)
# ---------------------------------------------------------------------------
fail() { echo "ERROR: $*" >&2; exit 1; }

# Map MODEL script -> HF short name used for the torch_dist checkpoint dir.
case "${MODEL}" in
    qwen3_4b_instr_2507) MODEL_HF_SHORT="Qwen3-4B-Instruct-2507" ;;
    *)                   MODEL_HF_SHORT="" ;;  # unknown -> soft glob check below
esac

preflight() {
    echo ">>> Preflight checks (PROJECT_DIR=${PROJECT_DIR})"

    # 1. Model script must exist.
    local model_script="${PROJECT_DIR}/slime_retro/scripts/models/${MODEL}.sh"
    [[ -f "${model_script}" ]] || fail "Missing model script: ${model_script}"

    # 2. Docker must be available (the launcher runs one container).
    command -v docker >/dev/null 2>&1 || fail "docker not found on PATH"
    docker info >/dev/null 2>&1 || fail "cannot talk to the Docker daemon (is it running / do you have permission?)"

    # 3. GPUs (warn only; the container requests --gpus all).
    if command -v nvidia-smi >/dev/null 2>&1; then
        local n_gpus
        n_gpus="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
        echo "    GPUs visible on host: ${n_gpus} (released config expects ${ACTOR_NUM_GPUS})"
    else
        echo "    WARN: nvidia-smi not found; assuming GPUs are exposed to Docker."
    fi

    # 4. Host tool server must be up. Check the base port and the top port of
    #    the requested instance range (ports base .. base+instances-1).
    local base="${RETRO_TOOL_SERVER_PORT}"
    local top=$(( base + RETRO_TOOL_SERVER_INSTANCES - 1 ))
    local hint="Start it on the host:  python -m tools.tool_server base_port=${base} num_instances=${RETRO_TOOL_SERVER_INSTANCES}"
    curl -sf "http://127.0.0.1:${base}/metrics" >/dev/null 2>&1 \
        || fail "tool server not reachable at 127.0.0.1:${base}. ${hint}"
    curl -sf "http://127.0.0.1:${top}/metrics" >/dev/null 2>&1 \
        || fail "tool server instance on port ${top} not reachable; fewer than ${RETRO_TOOL_SERVER_INSTANCES} instances are running. ${hint}"
    echo "    Tool server OK on ports ${base}-${top}"

    # 5. Training + eval parquets must exist.
    local train_path="${TRAIN_DATA:-${PROJECT_DIR}/datasets/slime_retro_train.parquet}"
    local eval_path="${EVAL_DATA_USPTO:-${PROJECT_DIR}/datasets/slime_retro_eval_uspto190.parquet}"
    [[ -f "${train_path}" ]] || fail "missing training data: ${train_path}
    Prepare it:  python -m slime_retro.prepare_data depth_min=5 depth_max=30   (or see slime_retro/scripts/README.md)"
    [[ -f "${eval_path}" ]] || fail "missing eval data: ${eval_path}
    Prepare it:  python -m slime_retro.prepare_data output=datasets/slime_retro_eval_uspto190.parquet targets=uspto-190"
    echo "    Train data OK: ${train_path}"
    echo "    Eval data OK:  ${eval_path}"

    # 6. Megatron torch_dist checkpoint (used as --ref-load and initial --load).
    local ckpt="${TORCH_DIST_CHECKPOINT:-}"
    if [[ -z "${ckpt}" && -n "${MODEL_HF_SHORT}" ]]; then
        ckpt="${PROJECT_DIR}/models/${MODEL_HF_SHORT}_torch_dist"
    fi
    if [[ -n "${ckpt}" ]]; then
        [[ -d "${ckpt}" ]] || fail "missing Megatron checkpoint: ${ckpt}
    Convert it once:  MODEL=${MODEL_HF_SHORT:-<HF-model>} bash slime_retro/scripts/convert_chkpt.sh"
        echo "    Checkpoint OK: ${ckpt}"
    elif ! compgen -G "${PROJECT_DIR}/models/*_torch_dist" >/dev/null; then
        fail "no models/*_torch_dist checkpoint found. Run: bash slime_retro/scripts/convert_chkpt.sh"
    fi

    echo ">>> Preflight OK"
}

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
    preflight
else
    echo ">>> SKIP_PREFLIGHT=1, skipping preflight checks"
fi

# ---------------------------------------------------------------------------
# Hand off to the real launcher (Docker + Ray + Megatron + SGLang).
# All settings above are exported, so they flow through train.sh -> the
# per-model script -> common.sh, and into the container via -e / runtime_env.
# ---------------------------------------------------------------------------
echo ">>> Launching: MODEL=${MODEL} ADV_ESTIMATOR=${ADV_ESTIMATOR} KL_COEF=${KL_COEF}"
exec bash "${PROJECT_DIR}/slime_retro/scripts/train.sh"
