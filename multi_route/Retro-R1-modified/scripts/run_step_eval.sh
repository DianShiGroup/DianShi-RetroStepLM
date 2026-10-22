#!/usr/bin/env bash
# Single entry point for every Retro-R1 evaluation.
#
# The shipped configuration is configs/dianshi_direct.yaml, which selects the
# `dianshi_direct` backend from the config's `backend` field rather than from a
# flag.
#
#   configs/dianshi_direct.yaml
#   configs/planner.yaml                      the LLM that issues CALL / SELECT
#
# To evaluate a variant instead, copy the step config, change what you need, and
# run it with --config <your copy>; the README lists what each key does.
#
# What it does, in order:
#   1. loads .env (real asset paths, endpoints, credentials)
#   2. checks the planner and the single-step endpoint are reachable
#   3. runs cli.py against the chosen step config
#   4. names the output directory after the config, so an A/B pair cannot
#      overwrite itself; completed targets are resumed on a re-run
#
# Usage:
#   bash scripts/run_step_eval.sh \
#       --config configs/dianshi_direct.yaml \
#       --targets DianShi-RetroRoute-Dataset/test_dataset/uspto_190_targets_smiles.txt
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

# shellcheck disable=SC1091
[[ -f "${PROJECT_DIR}/.env" ]] && set -a && . "${PROJECT_DIR}/.env" && set +a

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
CONFIG=""
TARGETS="${TARGETS_FILE:-}"
PLANNER_CONFIG="${PLANNER_CONFIG:-configs/planner.yaml}"
MAX_TURNS="${MAX_TURNS:-100}"
MAXSTEP="${MAXSTEP:-30}"
TOPK="${TOPK:-10}"
WORKERS="${WORKERS:-8}"
OUTPUT_DIR=""
PASSTHROUGH=()

die() { echo "ERROR: $*" >&2; exit 1; }
usage() { sed -n '2,28p' "${BASH_SOURCE[0]}"; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)      CONFIG="$2"; shift 2 ;;
    --targets)     TARGETS="$2"; shift 2 ;;
    --planner-config) PLANNER_CONFIG="$2"; shift 2 ;;
    --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
    --workers)     WORKERS="$2"; shift 2 ;;
    --max-turns)   MAX_TURNS="$2"; shift 2 ;;
    --maxstep)     MAXSTEP="$2"; shift 2 ;;
    --topk)        TOPK="$2"; shift 2 ;;
    -h|--help)     usage ;;
    *)             PASSTHROUGH+=("$1"); shift ;;   # forwarded to cli.py
  esac
done

[[ -n "${CONFIG}" ]]    || die "--config is required (try --help)"
[[ -f "${CONFIG}" ]]    || die "step config not found: ${CONFIG}"
[[ -n "${TARGETS}" ]]   || die "--targets is required (or set TARGETS_FILE in .env)"
[[ -f "${TARGETS}" ]]   || die "targets file not found: ${TARGETS}"
[[ -f "${PLANNER_CONFIG}" ]] || die "planner config not found: ${PLANNER_CONFIG}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# Name results after the config and target set so datasets and variants cannot
# overwrite each other.
config_stem="$(basename "${CONFIG}" .yaml)"
targets_stem="$(basename "${TARGETS}" .txt)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results/${config_stem}/${targets_stem}}"
mkdir -p "${OUTPUT_DIR}"

# Read both files back through the same loader the runner uses, so ${VAR}
# expansion and validation stay identical.
read -r backend gate step_url < <("${PYTHON_BIN}" - "${CONFIG}" <<'PY'
import sys
from inference.runner import load_single_step_config
cfg = load_single_step_config(sys.argv[1]) or {}
backend = cfg.get("backend") or "dianshi_direct"
urls = cfg.get("api_urls") or cfg.get("api_url") or []
if isinstance(urls, str):
    urls = [urls]
urls = [u for u in urls if u]
gate = "on" if (cfg.get("template_filter") or {}).get("enabled") else "off"
print(backend, gate, urls[0] if urls else "-")
PY
)

echo ">>> step config : ${CONFIG}"
echo ">>> backend     : ${backend} (template_gate=${gate})"
echo ">>> step endpoint: ${step_url}"
echo ">>> planner     : ${PLANNER_CONFIG}"
echo ">>> targets     : ${TARGETS}"
echo ">>> output      : ${OUTPUT_DIR}"

# --- preflight -------------------------------------------------------------
if [[ "${step_url}" != "-" ]]; then
  probe="${step_url%/chat/completions}"; probe="${probe%/v1}"
  curl --noproxy '*' -sf "${probe}/models" >/dev/null \
    || echo "WARNING: single-step endpoint ${step_url} did not answer; continuing anyway"
fi

"${PYTHON_BIN}" - "${PLANNER_CONFIG}" <<'PY'
import sys
from inference.runner import load_llm_config
cfg = load_llm_config(sys.argv[1])
print(">>> planner url : %s (%s)" % (cfg["base_url"], cfg["model_name"]))
PY

probe="$(mktemp -d)"
"${PYTHON_BIN}" - "${PLANNER_CONFIG}" > "${probe}/url" <<'PY'
import sys
from inference.runner import load_llm_config
print(load_llm_config(sys.argv[1])["base_url"])
PY
planner_url="$(cat "${probe}/url")"
curl --noproxy '*' -sf "${planner_url%/}/models" >/dev/null \
  || die "planner not reachable at ${planner_url}; start scripts/serve_planner.sh"

# --- evaluation ------------------------------------------------------------
# --resume is what makes a re-run cheap: targets already in the output jsonl
# are skipped, so an interrupted lane can simply be restarted.
exec "${PYTHON_BIN}" cli.py \
  --data_file "${TARGETS}" \
  --llm "${PLANNER_CONFIG}" \
  --single_step_config "${CONFIG}" \
  --max_turns "${MAX_TURNS}" \
  --maxstep "${MAXSTEP}" \
  --topk "${TOPK}" \
  --workers "${WORKERS}" \
  --output_dir "${OUTPUT_DIR}" \
  --resume \
  "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"