#!/bin/bash
set -Eeo pipefail
shopt -s nullglob
trap 'echo "[install.sh] Error on line $LINENO" >&2' ERR

# Resolve script directory for robust relative pathing
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Default values
base_requirements=0
mitten=0
loadgen=0
llm_requirements=0
whisper_requirements=0
trt=0
sdxl_requirements=0
wan22_a14b_requirements=0
q3vl_requirements=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --base_requirements)
            base_requirements=1
            shift 1
            ;;
        --mitten)
            mitten=1
            shift 1
            ;;
        --loadgen)
            loadgen=1
            shift 1
            ;;
        --llm_requirements)
            llm_requirements=1
            shift 1
            ;;
        --whisper_requirements)
            whisper_requirements=1
            shift 1
            ;;
        --trt)
            trt=1
            shift 1
            ;;
        --sdxl_requirements)
            sdxl_requirements=1
            shift 1
            ;;
        --wan22_a14b_requirements)
            wan22_a14b_requirements=1
            shift 1
            ;;
        --q3vl_requirements)
            q3vl_requirements=1
            shift 1
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ $base_requirements -eq 1 ]; then
    echo "Installing base dependencies..."
    REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.${BUILD_CONTEXT}.txt"
    if [ ! -f "${REQUIREMENTS_FILE}" ]; then
        echo "[ERROR] Could not find requirements.${BUILD_CONTEXT}.txt at ${REQUIREMENTS_FILE}"
        exit 1
    fi

    python3 -m pip install --upgrade pip
    python3 -m pip install -r "${REQUIREMENTS_FILE}" || { echo "[ERROR] pip install -r ${REQUIREMENTS_FILE} failed with exit code $?"; exit 1; }
    # there is a bug in pynvml in current version of container, so we uninstall it
    python3 -m pip uninstall -y pynvml
fi

if [ $mitten -eq 1 ]; then
    echo "Installing MLPerf mitten..."
    if [ -f "${SCRIPT_DIR}/install_mitten.sh" ]; then
        bash "${SCRIPT_DIR}/install_mitten.sh" || { echo "[ERROR] install_mitten.sh failed with exit code $?"; exit 1; }
    else
        echo "[ERROR] ${SCRIPT_DIR}/install_mitten.sh not found"
        exit 1
    fi
fi

if [ $loadgen -eq 1 ]; then
    echo "Installing loadgen..."
    if [ -f "${SCRIPT_DIR}/install_loadgen.sh" ]; then
        bash "${SCRIPT_DIR}/install_loadgen.sh" || { echo "[ERROR] install_loadgen.sh failed with exit code $?"; exit 1; }
    else
        echo "[ERROR] ${SCRIPT_DIR}/install_loadgen.sh not found"
        exit 1
    fi
fi

if [ $llm_requirements -eq 1 ]; then
    echo "Installing LLM dependencies..."
    REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements/requirements.llm.txt"
    if [ -f "${REQUIREMENTS_FILE}" ]; then
        pip install -r "${REQUIREMENTS_FILE}" || { echo "[ERROR] pip install -r ${REQUIREMENTS_FILE} failed with exit code $?"; exit 1; }
    else
        echo "[ERROR] requirements.llm.txt not found at ${REQUIREMENTS_FILE}"
        exit 1
    fi
fi

if [ $whisper_requirements -eq 1 ]; then
    echo "Installing Whisper dependencies..."
    REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements/requirements.whisper.txt"
    if [ -f "${REQUIREMENTS_FILE}" ]; then
        pip install -r "${REQUIREMENTS_FILE}" || { echo "[ERROR] pip install -r ${REQUIREMENTS_FILE} failed with exit code $?"; exit 1; }
    else
        echo "[ERROR] requirements.whisper.txt not found at ${REQUIREMENTS_FILE}"
        exit 1
    fi
fi

if [ $trt -eq 1 ]; then
    echo "Installing TensorRT..."
    if [ -f "${SCRIPT_DIR}/install_tensorrt.sh" ]; then
        bash "${SCRIPT_DIR}/install_tensorrt.sh" || { echo "[ERROR] install_tensorrt.sh failed with exit code $?"; exit 1; }
    else
        echo "[ERROR] ${SCRIPT_DIR}/install_tensorrt.sh not found"
        exit 1
    fi
fi

if [ $sdxl_requirements -eq 1 ]; then
    echo "Installing SDXL dependencies..."
    REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements/requirements.stable-diffusion-xl.txt"
    if [ -f "${REQUIREMENTS_FILE}" ]; then
        pip install -r "${REQUIREMENTS_FILE}" || { echo "[ERROR] pip install -r ${REQUIREMENTS_FILE} failed with exit code $?"; exit 1; }
    else
        echo "[ERROR] requirements.stable-diffusion-xl.txt not found at ${REQUIREMENTS_FILE}"
        exit 1
    fi
fi

if [ $wan22_a14b_requirements -eq 1 ]; then
    echo "Installing Wan-2.2 A14B / VisionFly dependencies..."

    # Install system dependencies
    apt-get update && apt-get install -y libxcb1

    REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements/requirements.wan22-a14b.txt"
    if [ -f "${REQUIREMENTS_FILE}" ]; then
        # Install base requirements (packages not in trtllm base image)
        pip install -r "${REQUIREMENTS_FILE}" || { echo "[ERROR] pip install -r ${REQUIREMENTS_FILE} failed with exit code $?"; exit 1; }

        # Install flashinfer-vx
        pip install git+https://gitlab-master.nvidia.com/ruqingx/flashinfer-vx --no-build-isolation || { echo "[WARNING] flashinfer-vx installation failed"; }

        # Install visual_gen
        VISUAL_GEN_DIR="/work/3rdparty/trtllm/tensorrt_llm/visual_gen"
        if [ -d "${VISUAL_GEN_DIR}" ]; then
            echo "Installing visual_gen from ${VISUAL_GEN_DIR}..."
            pip install -e "${VISUAL_GEN_DIR}" --no-build-isolation --no-deps || { echo "[ERROR] visual_gen installation failed"; exit 1; }
        else
            echo "[ERROR] visual_gen source not found at ${VISUAL_GEN_DIR}"
            exit 1
        fi
    else
        echo "[ERROR] requirements.wan22-a14b.txt not found at ${REQUIREMENTS_FILE}"
        exit 1
    fi

fi

if [ $q3vl_requirements -eq 1 ]; then
    echo "Installing Q3VL dependencies..."
    REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements/requirements.q3vl.txt"
    if [ -f "${REQUIREMENTS_FILE}" ]; then
        pip install -r "${REQUIREMENTS_FILE}" || { echo "[ERROR] pip install -r ${REQUIREMENTS_FILE} failed with exit code $?"; exit 1; }
    else
        echo "[ERROR] requirements.q3vl.txt not found at ${REQUIREMENTS_FILE}"
        exit 1
    fi
fi
