# syntax=docker/dockerfile:1.4

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG ENV
ARG BUILD_CONTEXT
ARG MITTEN_DIR
ARG INFERENCE_DIR
ARG INSTALL_ARGS=""

ENV ENV=${ENV}
ENV BUILD_CONTEXT=${BUILD_CONTEXT}
ENV MITTEN_DIR=${MITTEN_DIR}
ENV INFERENCE_DIR=${INFERENCE_DIR}

SHELL ["/bin/bash", "-c"]

# Copy the entire common directory structure to preserve paths
COPY --from=common / /work/docker/common

# Copy 3rdparty submodules for installation
COPY --from=mitten / ${MITTEN_DIR}
COPY --from=mlc-inference / ${INFERENCE_DIR}
COPY --from=visual_gen / /work/3rdparty/trtllm/tensorrt_llm/visual_gen

WORKDIR /work

RUN if [[ "${INSTALL_ARGS}" == *"--base_requirements"* ]]; then \
        BUILD_CONTEXT=${BUILD_CONTEXT} bash docker/common/install.sh --base_requirements; \
    fi

RUN --mount=type=ssh if [[ "${INSTALL_ARGS}" == *"--mitten"* ]]; then \
        bash docker/common/install.sh --mitten; \
    fi

RUN --mount=type=ssh if [[ "${INSTALL_ARGS}" == *"--loadgen"* ]]; then \
        bash docker/common/install.sh --loadgen; \
    fi

RUN if [[ "${INSTALL_ARGS}" == *"--llm_requirements"* ]]; then \
        bash docker/common/install.sh --llm_requirements; \
    fi

RUN if [[ "${INSTALL_ARGS}" == *"--whisper_requirements"* ]]; then \
        bash docker/common/install.sh --whisper_requirements; \
    fi

RUN if [[ "${INSTALL_ARGS}" == *"--trt"* ]]; then \
        bash docker/common/install.sh --trt; \
    fi

RUN if [[ "${INSTALL_ARGS}" == *"--sdxl_requirements"* ]]; then \
        bash docker/common/install.sh --sdxl_requirements; \
    fi

# Install Wan-2.2 A14B / visual_gen dependencies (visual_gen source installed at runtime)
RUN if [[ "${INSTALL_ARGS}" == *"--wan22_a14b_requirements"* ]]; then \
        bash docker/common/install.sh --wan22_a14b_requirements; \
    fi

# Install Q3VL dependencies
RUN if [[ "${INSTALL_ARGS}" == *"--q3vl_requirements"* ]]; then \
        bash docker/common/install.sh --q3vl_requirements; \
    fi
