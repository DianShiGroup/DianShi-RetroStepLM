#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" build_model_tests.py --config model_test_config.json "$@"
