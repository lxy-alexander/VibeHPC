#!/usr/bin/env bash
set -euo pipefail

if [ -z "${MITTEN_DIR:-}" ]; then
    echo "ERROR: MITTEN_DIR environment variable is not set or is empty"
    exit 1
fi

if [ ! -d "${MITTEN_DIR}" ]; then
    echo "ERROR: Mitten directory not found at ${MITTEN_DIR}"
    echo "Please ensure the mitten submodule is initialized:"
    echo "  git submodule update --init --recursive 3rdparty/mitten"
    exit 1
fi

echo "Installing mitten from ${MITTEN_DIR}..."
cd "${MITTEN_DIR}"
sed -i 's/numpy >=1.22.0, <1.24.0/numpy >=1.26.4/' ./setup.cfg
pip install .