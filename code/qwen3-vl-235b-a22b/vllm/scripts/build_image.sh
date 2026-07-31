#!/bin/bash

set -eux
set -o pipefail

PROJECT_ROOT=$(dirname "${BASH_SOURCE[0]}")/../
PROJECT_ROOT=$(realpath "${PROJECT_ROOT}")

DEFAULT_VLLM_REPO=https://github.com/CentML/vllm.git
vllm_repo=${DEFAULT_VLLM_REPO}

DEFAULT_VLLM_REVISION=mlperf-inf-mm-q3vl-v6.0
vllm_revision=${DEFAULT_VLLM_REVISION}

DEFAULT_VLLM_BUILD_MAX_JOBS=256
vllm_build_max_jobs=${DEFAULT_VLLM_BUILD_MAX_JOBS}

DEFAULT_VLLM_BUILD_NVCC_THREADS=2
vllm_build_nvcc_threads=${DEFAULT_VLLM_BUILD_NVCC_THREADS}

DEFAULT_VLLM_BUILD_CUDA_VERSION=13.0.1
vllm_build_cuda_version=${DEFAULT_VLLM_BUILD_CUDA_VERSION}

# Automatically detect architecture: x86_64 -> amd64, aarch64/arm64 -> arm64
DETECTED_ARCH=$(uname -m)
if [[ "${DETECTED_ARCH}" == "x86_64" ]]; then
    DEFAULT_VLLM_BUILD_ARCH=amd64
elif [[ "${DETECTED_ARCH}" == "aarch64" || "${DETECTED_ARCH}" == "arm64" ]]; then
    DEFAULT_VLLM_BUILD_ARCH=arm64
else
    echo "Warning: Unknown architecture ${DETECTED_ARCH}, defaulting to amd64"
    DEFAULT_VLLM_BUILD_ARCH=amd64
fi
vllm_build_arch=${DEFAULT_VLLM_BUILD_ARCH}

DEFAULT_VLLM_BUILD_IMAGE_REPO=gitlab-master.nvidia.com:5005/mlpinf/mlperf-inference/vllm-openai
vllm_build_image_repo=${DEFAULT_VLLM_BUILD_IMAGE_REPO}

vllm_build_push_flag=""
vllm_build_no_cache_flag=""
result_image_tag=""

DEFAULT_RESULT_IMAGE_REPO=gitlab-master.nvidia.com:5005/mlpinf/mlperf-inference/mlperf-inf-mm-q3vl-nv
result_image_repo=${DEFAULT_RESULT_IMAGE_REPO}

DEFAULT_DYNAMO_REPO=https://github.com/CentML/dynamo.git
dynamo_repo=${DEFAULT_DYNAMO_REPO}

DEFAULT_DYNAMO_REVISION=mlperf-inf-mm-q3vl-v6.0
dynamo_revision=${DEFAULT_DYNAMO_REVISION}

DEFAULT_MLPERF_INF_MM_Q3VL_INSTALL_URL=git+https://github.com/mlcommons/inference.git#subdirectory=multimodal/qwen3-vl/
mlperf_inf_mm_q3vl_install_url=${DEFAULT_MLPERF_INF_MM_Q3VL_INSTALL_URL}

DEFAULT_MLPERF_INF_MM_Q3VL_NV_INSTALL_URL=$(realpath --relative-to=${PROJECT_ROOT} $(pwd))
mlperf_inf_mm_q3vl_nv_install_url=${DEFAULT_MLPERF_INF_MM_Q3VL_NV_INSTALL_URL}

result_push_flag=""
result_no_cache_flag=""

function _exit_with_help_msg() {
  cat <<EOF
Build the docker image for NVIDIA's submission to the MLPerf Inference VLM benchmark.

Usage: ${BASH_SOURCE[0]}
  [-h | --help]     Print this help message.
  [--vllm-repo <vllm_repo>]           The repository to use for vLLM (default: ${DEFAULT_VLLM_REPO}).
  [--vllm-revision <vllm_revision>]   The revision to use for vLLM (default: ${DEFAULT_VLLM_REVISION}).
  [--vllm-build-max-jobs <vllm_build_max_jobs>]   The maximum number of processes to use for vLLM build (default: ${DEFAULT_VLLM_BUILD_MAX_JOBS}).
  [--vllm-build-nvcc-threads <vllm_build_nvcc_threads>]   The number of nvcc threads to use for vLLM build (default: ${DEFAULT_VLLM_BUILD_NVCC_THREADS}).
  [--vllm-build-cuda-version <vllm_build_cuda_version>]   The CUDA version to use for vLLM build (default: ${DEFAULT_VLLM_BUILD_CUDA_VERSION}).
  [--vllm-build-arch <vllm_build_arch>]   The CPU architecture to use for vLLM build (default: ${DEFAULT_VLLM_BUILD_ARCH}).
  [--vllm-build-image-repo <vllm_build_image_repo>]   The docker image repository to use for the resulting vLLM image (default: ${DEFAULT_VLLM_BUILD_IMAGE_REPO}).
  [--result-image-tag <result_image_tag>]   Full result image tag (overrides repo+tag).
  [--vllm-build-push]   Push the resulting vLLM image to the docker image repository.
  [--vllm-force-rebuild]   Force the rebuild of the vLLM image.
  [--result-image-repo <result_image_repo>]   The docker image repository to use for the resulting MLPerf Inference VLM benchmark image (default: ${DEFAULT_RESULT_IMAGE_REPO}).
  [--dynamo-repo <dynamo_repo>]   The repository to use for Dynamo (default: ${DEFAULT_DYNAMO_REPO}).
  [--dynamo-revision <dynamo_revision>]   The revision to use for Dynamo (default: ${DEFAULT_DYNAMO_REVISION}).
  [--mlperf-inf-mm-q3vl-install-url <mlperf_inf_mm_q3vl_install_url>]   The URL to use for mlperf-inf-mm-q3vl (default: ${DEFAULT_MLPERF_INF_MM_Q3VL_INSTALL_URL}).
  [--mlperf-inf-mm-q3vl-nv-install-url <mlperf_inf_mm_q3vl_nv_install_url>]   The URL to use for mlperf-inf-mm-q3vl-nv (default: ${DEFAULT_MLPERF_INF_MM_Q3VL_NV_INSTALL_URL}).
  [--result-push]   Push the resulting MLPerf Inference VLM benchmark image to the docker image repository.
  [--result-force-rebuild]   Force the rebuild of the MLPerf Inference VLM benchmark image.
EOF
  if [ -n "$1" ]; then
    echo "$(tput bold setab 1)$1$(tput sgr0)"
  fi
  exit "$2"
}

while [[ $# -gt 0 ]]; do
    case $1 in
    -h | --help)
        _exit_with_help_msg "" 0
        ;;
    --vllm-repo)
        vllm_repo=$2
        shift
        shift
        ;;
    --vllm-repo=*)
        vllm_repo=${1#*=}
        shift
        ;;
    --vllm-revision)
        vllm_revision=$2
        shift
        shift
        ;;
    --vllm-revision=*)
        vllm_revision=${1#*=}
        shift
        ;;
    --vllm-build-max-jobs)
        vllm_build_max_jobs=$2
        shift
        shift
        ;;
    --vllm-build-max-jobs=*)
        vllm_build_max_jobs=${1#*=}
        shift
        ;;
    --vllm-build-nvcc-threads)
        vllm_build_nvcc_threads=$2
        shift
        shift
        ;;
    --vllm-build-nvcc-threads=*)
        vllm_build_nvcc_threads=${1#*=}
        shift
        ;;
    --vllm-build-cuda-version)
        vllm_build_cuda_version=$2
        shift
        shift
        ;;
    --vllm-build-cuda-version=*)
        vllm_build_cuda_version=${1#*=}
        shift
        ;;
    --vllm-build-arch)
        vllm_build_arch=$2
        shift
        shift
        ;;
    --vllm-build-arch=*)
        vllm_build_arch=${1#*=}
        shift
        ;;
    --vllm-build-image-repo)
        vllm_build_image_repo=$2
        shift
        shift
        ;;
    --vllm-build-image-repo=*)
        vllm_build_image_repo=${1#*=}
        shift
        ;;
    --result-image-tag)
        result_image_tag=$2
        shift
        shift
        ;;
    --result-image-tag=*)
        result_image_tag=${1#*=}
        shift
        ;;
    --vllm-build-push)
        vllm_build_push_flag="--push"
        shift
        ;;
    --vllm-force-rebuild)
        vllm_build_no_cache_flag="--no-cache"
        shift
        ;;
    --result-image-repo)
        result_image_repo=$2
        shift
        shift
        ;;
    --result-image-repo=*)
        result_image_repo=${1#*=}
        shift
        ;;
    --dynamo-repo)
        dynamo_repo=$2
        shift
        shift
        ;;
    --dynamo-repo=*)
        dynamo_repo=${1#*=}
        shift
        ;;
    --dynamo-revision)
        dynamo_revision=$2
        shift
        shift
        ;;
    --dynamo-revision=*)
        dynamo_revision=${1#*=}
        shift
        ;;
    --mlperf-inf-mm-q3vl-install-url)
        mlperf_inf_mm_q3vl_install_url=$2
        shift
        shift
        ;;
    --mlperf-inf-mm-q3vl-install-url=*)
        mlperf_inf_mm_q3vl_install_url=${1#*=}
        shift
        ;;
    --mlperf-inf-mm-q3vl-nv-install-url)
        mlperf_inf_mm_q3vl_nv_install_url=$2
        shift
        shift
        ;;
    --mlperf-inf-mm-q3vl-nv-install-url=*)
        mlperf_inf_mm_q3vl_nv_install_url=${1#*=}
        shift
        ;;
    --result-push)
        result_push_flag="--push"
        shift
        ;;
    --result-force-rebuild)
        result_no_cache_flag="--no-cache"
        shift
        ;;
    *)
        _exit_with_help_msg "[ERROR] Unknown option: $1" 1
        ;;
  esac
done

vllm_build_image_tag=${vllm_build_arch}_cuda${vllm_build_cuda_version}_$(echo "${vllm_repo}" | sed -e 's|https://github.com/||' -e 's|\.git$||' -e 's|/|_|g')-${vllm_revision}
if [ ${#vllm_build_image_tag} -gt 128 ]; then
    vllm_build_image_tag="${vllm_build_image_tag:0:128}"
fi
vllm_build_image_tag=${vllm_build_image_repo}:${vllm_build_image_tag}

if { ! docker image inspect "${vllm_build_image_tag}" > /dev/null 2>&1 && ! docker manifest inspect "${vllm_build_image_tag}" > /dev/null 2>&1; } || [ "${vllm_build_no_cache_flag}" = "--no-cache" ]; then
    VLLM_CLONE_DIR=${VLLM_CLONE_DIR:-${PWD}/vllm}
    rm -rf "${VLLM_CLONE_DIR}"
    git clone "${vllm_repo}" "${VLLM_CLONE_DIR}"
    cd "${VLLM_CLONE_DIR}"
    git checkout "${vllm_revision}"

    DOCKER_BUILDKIT=1 docker build \
        --pull \
        ${vllm_build_push_flag} \
        ${vllm_build_no_cache_flag} \
        --build-arg max_jobs="${vllm_build_max_jobs}" \
        --build-arg nvcc_threads="${vllm_build_nvcc_threads}" \
        --build-arg RUN_WHEEL_CHECK=false \
        --build-arg CUDA_VERSION="${vllm_build_cuda_version}" \
        --build-arg BUILD_BASE_IMAGE="nvidia/cuda:${vllm_build_cuda_version}-devel-ubuntu22.04" \
        --build-arg torch_cuda_arch_list='9.0 10.0+PTX 10.3' \
        --build-arg INSTALL_KV_CONNECTORS=true \
        --platform "linux/${vllm_build_arch}" \
        --tag "${vllm_build_image_tag}" \
        --target vllm-openai \
        --progress plain \
        -f docker/Dockerfile \
        .

    cd -
    rm -rf "${VLLM_CLONE_DIR}"
fi

if ! docker image inspect "${vllm_build_image_tag}" > /dev/null 2>&1; then
    docker pull "${vllm_build_image_tag}"
fi

if [ -z "${result_image_tag}" ]; then
    result_image_tag=${vllm_build_arch}_cuda${vllm_build_cuda_version}_$(echo "${dynamo_repo}" | sed -e 's|https://github.com/||' -e 's|\.git$||' -e 's|/|_|g')-${dynamo_revision}_$(echo "${vllm_repo}" | sed -e 's|https://github.com/||' -e 's|\.git$||' -e 's|/|_|g')-${vllm_revision}
    if [ ${#result_image_tag} -gt 128 ]; then
        result_image_tag="${result_image_tag:0:128}"
    fi
    result_image_tag=${result_image_repo}:${result_image_tag}
fi


DOCKER_BUILDKIT=1 docker build \
    --ulimit nofile=65536:65536 \
    ${result_push_flag} \
    ${result_no_cache_flag} \
    --build-arg VLLM_BASE_IMAGE="${vllm_build_image_tag}" \
    --build-arg ARCH="${vllm_build_arch}" \
    --build-arg CUDA_MAJOR_VERSION="$(echo "${vllm_build_cuda_version}" | cut -d. -f1)" \
    --build-arg DYNAMO_REPO="${dynamo_repo}" \
    --build-arg DYNAMO_REVISION="${dynamo_revision}" \
    --build-arg MLPERF_INF_MM_Q3VL_INSTALL_URL="${mlperf_inf_mm_q3vl_install_url}" \
    --build-arg MLPERF_INF_MM_Q3VL_NV_INSTALL_URL="${mlperf_inf_mm_q3vl_nv_install_url}" \
    --platform "linux/${vllm_build_arch}" \
    --tag "${result_image_tag}" \
    --progress plain \
    -f "${PROJECT_ROOT}/docker/mpi-dynamo-vllm.Dockerfile" \
    .
