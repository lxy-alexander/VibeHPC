#!/usr/bin/env bash
set -euo pipefail

if [ -z "${INFERENCE_DIR:-}" ]; then
    echo "ERROR: INFERENCE_DIR environment variable is not set or is empty"
    exit 1
fi

# Use loadgen from the local 3rdparty/mlc-inference submodule
LOADGEN_DIR="${INFERENCE_DIR}/loadgen"

if [ ! -d "${LOADGEN_DIR}" ]; then
    echo "ERROR: Loadgen directory not found at ${LOADGEN_DIR}"
    echo "Please ensure the mlc-inference submodule is initialized:"
    echo "  git submodule update --init --recursive 3rdparty/mlc-inference"
    exit 1
fi

echo "Installing loadgen from ${LOADGEN_DIR}..."
cd "${LOADGEN_DIR}"
pip install .

