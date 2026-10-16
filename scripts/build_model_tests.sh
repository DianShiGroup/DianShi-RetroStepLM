#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m dianshi_retrostep.data.build_model_tests --config configs/model_test_config.json "$@"
