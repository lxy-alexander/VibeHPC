ARG VLLM_BASE_IMAGE
FROM ${VLLM_BASE_IMAGE} 

ARG ARCH=arm64
ARG CUDA_MAJOR_VERSION=13

WORKDIR /vllm-workspace

########################
# Install dependencies #
# ######################
# System dependencies.
RUN apt-get update && \
    apt-get -y install \
		build-essential \
		ca-certificates \
		cmake \
		curl \
		libclang-dev \
		libhwloc-dev \
		libopenmpi-dev \
		libudev-dev \
		numactl \
		openmpi-bin \
		pkg-config \
		protobuf-compiler \
		python3-dev \
		tmux \
		vim \
	    git \
	&& \
	rm -rf /var/lib/apt/lists/*
# Install Rust.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y 
# Currently vLLM's upstream Dockerfile installs the wrong versions of nixl and triton 
# in the case of cuda 13. Once vLLM upstream fixes these problems, we can get rid of the
# nixl and triton dependencies in this Dockerfile.
RUN uv pip install --system --no-cache --verbose \
		triton>=3.5.1 \
    	maturin \
    	nixl \
    	nixl-cu${CUDA_MAJOR_VERSION} \
    	pip
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV TRITON_PTXAS_BLACKWELL_PATH=${TRITON_PTXAS_PATH}

#####################
# nsight-system-cli #
#####################
# Install Nsight Systems CLI for profiling (supports x86_64 and ARM64/GB200)
# Version format: YEAR.MINOR.PATCH (e.g., 2025.4.1)
ARG NSIGHT_SYSTEMS_VERSION=2025.4.1
ARG NSIGHT_SYSTEMS_BUILD=172
ARG NSIGHT_SYSTEMS_X86_BUILD=3634357
ARG NSIGHT_SYSTEMS_URL
RUN YEAR_MINOR=$(echo ${NSIGHT_SYSTEMS_VERSION} | cut -d. -f1-2 | tr '.' '_') && \
    if [ "${ARCH}" = "arm64" ]; then \
        NSIGHT_SYSTEMS_URL="https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/${YEAR_MINOR}/nsight-systems-cli-${NSIGHT_SYSTEMS_VERSION}_${NSIGHT_SYSTEMS_VERSION}.${NSIGHT_SYSTEMS_BUILD}-1_${ARCH}.deb"; \
    elif [ "${ARCH}" = "amd64" ]; then \
        NSIGHT_SYSTEMS_URL="https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/${YEAR_MINOR}/NsightSystems-linux-cli-public-${NSIGHT_SYSTEMS_VERSION}.${NSIGHT_SYSTEMS_BUILD}-${NSIGHT_SYSTEMS_X86_BUILD}.deb"; \
    else \
        echo "Unsupported architecture: ${ARCH}"; \
        exit 1; \
    fi && \
    curl -fsSL -o /tmp/nsight-systems.deb "${NSIGHT_SYSTEMS_URL}" && \
    apt-get install -y --no-install-recommends /tmp/nsight-systems.deb && \
    which nsys && \
    nsys --version && \
    rm /tmp/nsight-systems.deb && \
    rm -rf /var/lib/apt/lists/*

##################
# Install Dynamo #
##################
# Clone the repository.
ARG DYNAMO_REPO=https://github.com/ai-dynamo/dynamo.git
RUN git clone ${DYNAMO_REPO}
# Checkout the desired revision.
WORKDIR /vllm-workspace/dynamo
ARG DYNAMO_REVISION=main
RUN git checkout ${DYNAMO_REVISION}
# Build and install Rust-Python bindings.
WORKDIR /vllm-workspace/dynamo/lib/bindings/python
RUN . $HOME/.cargo/env && maturin build --release -i python3
RUN uv pip install target/wheels/*.whl --system
# Install the Dynamo Python package.
WORKDIR /vllm-workspace/dynamo
RUN uv pip install . --system

#########################
# Install NATS and ETCD #
#########################
# Install ETCD.
ARG ETCD_VER=v3.6.6
ARG ETCD_DOWNLOAD_URL=https://storage.googleapis.com/etcd
ARG ETCD_TARGET_DIR=/opt/etcd
RUN mkdir -p ${ETCD_TARGET_DIR} && \
	curl -L ${ETCD_DOWNLOAD_URL}/${ETCD_VER}/etcd-${ETCD_VER}-linux-${ARCH}.tar.gz -o /tmp/etcd-${ETCD_VER}-linux-${ARCH}.tar.gz && \
	tar xzvf /tmp/etcd-${ETCD_VER}-linux-${ARCH}.tar.gz -C ${ETCD_TARGET_DIR} --strip-components=1 --no-same-owner
# Install NATS.
ARG NATS_VER=v2@v2.11.6
WORKDIR /opt
RUN curl -fsSL https://binaries.nats.dev/nats-io/nats-server/${NATS_VER} | sh

ENV PATH="/opt/:/opt/etcd/:$PATH"

##############################
# Install mlperf-inf-mm-q3vl #
##############################
WORKDIR /vllm-workspace
ARG LOADGEN_INSTALL_URL=""
ARG MLPERF_INF_MM_Q3VL_INSTALL_URL=git+https://github.com/mlcommons/inference.git#subdirectory=multimodal/qwen3-vl/
# Install the mlcommons-loadgen package if LOADGEN_INSTALL_URL is not empty.
RUN if [ -n "${LOADGEN_INSTALL_URL}" ]; then \
        uv pip install --system --no-cache --verbose "${LOADGEN_INSTALL_URL}"; \
    fi;
# Install mlperf-inf-mm-q3vl.
RUN uv pip install --system --no-cache --verbose "${MLPERF_INF_MM_Q3VL_INSTALL_URL}"

#################################
# Install mlperf-inf-mm-q3vl-nv #
#################################
ARG BUILD_CONTEXT_DIR=/tmp/mm_q3vl_nv_build_context
ARG MLPERF_INF_MM_Q3VL_NV_INSTALL_URL=git+https://github.com/mlcommons/inference_results_v6.0.git#subdirectory=closed/NVIDIA/code/qwen3-vl-235b-a22b/vllm

COPY . ${BUILD_CONTEXT_DIR}/

RUN if echo "${MLPERF_INF_MM_Q3VL_NV_INSTALL_URL}" | grep -q "^git+"; then \
        echo "Installing from git URL: ${MLPERF_INF_MM_Q3VL_NV_INSTALL_URL}"; \
        uv pip install --system --no-cache --verbose "${MLPERF_INF_MM_Q3VL_NV_INSTALL_URL}"; \
    else \
        echo "Installing from local path: ${MLPERF_INF_MM_Q3VL_NV_INSTALL_URL}"; \
        uv pip install --system --no-cache --verbose "${BUILD_CONTEXT_DIR}/${MLPERF_INF_MM_Q3VL_NV_INSTALL_URL}"; \
    fi;

####################################
# Upgrade NVIDIA cuDNN to 9.18.1.3 #
####################################
RUN uv pip install --upgrade --system --no-cache --verbose nvidia-cudnn-cu13==9.18.1.3

ENTRYPOINT ["/bin/bash"]
