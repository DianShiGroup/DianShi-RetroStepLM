#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Environment: RetroAgent/.venv, Python 3.11, PyTorch cu129.
# Override CUDA_HOME when your CUDA toolkit lives elsewhere; the exports below
# are skipped entirely when it is unset.
CUDA_HOME="${CUDA_HOME:-}"
PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"

source "${PROJECT_DIR}/.venv/bin/activate"

if [[ -n "${CUDA_HOME}" ]]; then
  export CUDA_HOME
  export CUDA_PATH="${CUDA_HOME}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" -c 'import sys; print("python:", sys.executable)'

# TODO: add the actual inference command here later.
:
