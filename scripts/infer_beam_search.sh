#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-models/DianShi-RetroStep-Direct}"
CACHED_DATA_PATH="${CACHED_DATA_PATH:-model_tests/test_out/sl_cot/testset_sl_cot_new_cached}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
BEAM_WIDTH="${BEAM_WIDTH:-10}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
EVAL_KS="${EVAL_KS:-1 3 5 10}"
NO_THINKING=1
OVERWRITE=0

usage() {
    echo "Usage: scripts/infer_beam_search.sh [options]"
    echo
    echo "Options:"
    echo "  --model-path PATH         Local model path or Hugging Face model ID"
    echo "  --data-path PATH          Hugging Face cached dataset directory"
    echo "  --output-dir PATH         Result directory when --output-path is omitted"
    echo "  --output-path PATH        Exact beam-search JSONL output path"
    echo "  --num-gpus N              Number of one-GPU Ray workers"
    echo "  --batch-size N            Total batch size across all workers"
    echo "  --max-tokens N            Maximum generated tokens per beam"
    echo "  --beam-width N            Number of beams"
    echo "  --gpu-memory-utilization F"
    echo "  --enable-thinking         Do not disable thinking in the chat template"
    echo "  --overwrite               Replace an existing result file"
    echo "  -h, --help                Show this help"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --data-path) CACHED_DATA_PATH="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --output-path) OUTPUT_PATH="$2"; shift 2 ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --beam-width) BEAM_WIDTH="$2"; shift 2 ;;
        --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --enable-thinking) NO_THINKING=0; shift ;;
        --overwrite) OVERWRITE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ ! -d "$CACHED_DATA_PATH" ]; then
    echo "Cached dataset does not exist: $CACHED_DATA_PATH" >&2
    echo "Build it first with scripts/build_model_tests.sh." >&2
    exit 2
fi

if [ -z "$OUTPUT_PATH" ]; then
    MODEL_NAME="$(basename -- "$MODEL_PATH")"
    DATASET_NAME="$(basename -- "$CACHED_DATA_PATH")"
    OUTPUT_PATH="$OUTPUT_DIR/${MODEL_NAME}_${DATASET_NAME}_beam${BEAM_WIDTH}.jsonl"
fi
METRICS_PATH="${OUTPUT_PATH%.jsonl}_metrics.json"

INFER_ARGS=(
    --model-path "$MODEL_PATH"
    --cached-data-path "$CACHED_DATA_PATH"
    --output-path "$OUTPUT_PATH"
    --num-gpus "$NUM_GPUS"
    --batch-size "$BATCH_SIZE"
    --max-tokens "$MAX_TOKENS"
    --beam-width "$BEAM_WIDTH"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
)
if [ "$NO_THINKING" -eq 1 ]; then
    INFER_ARGS+=(--no-thinking)
fi
if [ "$OVERWRITE" -eq 1 ]; then
    INFER_ARGS+=(--overwrite)
fi

echo "Model: $MODEL_PATH"
echo "Dataset: $CACHED_DATA_PATH"
echo "Results: $OUTPUT_PATH"

"$PYTHON_BIN" -m dianshi_retrostep.inference.vllm_beamsearch_infer "${INFER_ARGS[@]}"

echo
echo "Beam search completed. Evaluating exact-match accuracy..."
# EVAL_KS is intentionally split into separate integer arguments.
# shellcheck disable=SC2086
"$PYTHON_BIN" -m dianshi_retrostep.evaluation.eval_retro_beam_topk "$OUTPUT_PATH" --ks $EVAL_KS --output-json "$METRICS_PATH"
