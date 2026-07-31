#!/bin/bash
set -e

export NVSHMEM_VERSION=2.10.1
export NVSHMEM_RC=3
export NVSHMEM_FILE_BASE=nvshmem_src_${NVSHMEM_VERSION}-${NVSHMEM_RC}
export MPI_HOME=/usr/local/mpi/
export UCX_HOME=/usr/local/ucx/
export CUDA_HOME=/usr/local/cuda
export NVSHMEM_MPI_SUPPORT=1
export NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70 \
                     -gencode=arch=compute_80,code=sm_80 \
                     -gencode=arch=compute_90,code=sm_90"
export CUDA_ARCH_BIN="60 70 80 90"
export CUDA_ARCH_PTX="90"
export NVSI_ARCH=$(nvidia-smi -q | grep -m 1 Architecture | cut -d: -f2 | sed 's/^ *//g')
if [ "${NVSI_ARCH}" = "Blackwell" ]; then
    export NVCC_GENCODE="${NVCC_GENCODE} -gencode=arch=compute_100,code=sm_100"
    export CUDA_ARCH_BIN="60 70 80 90 100"
    export CUDA_ARCH_PTX="100"
fi

export NVSHMEM_UCX_SUPPORT=1
export NVSHMEM_USE_GDRCOPY=0
export NVSHMEM_IBDEVX_SUPPORT=1
export NVSHMEM_IBRC_SUPPORT=1
export NVSHMEM_IBGDA_SUPPORT=1
export NVSHMEM_ENABLE_ALL_DEVICE_INLINING=0

export NVSHMEM_BOOTSTRAP_PLUGIN=/usr/local/nvshmem/lib/nvshmem_bootstrap_mpi.so
export NVSHMEM_BOOTSTRAP=plugin
export NVSHMEM_HEAP_KIND=SYSMEM
export NVSHMEM_REMOTE_TRANSPORT=ibrc
# IBGDA Tuning
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IBGDA_NUM_RC_PER_PE=216
export NVSHMEM_IBGDA_NUM_DCI=1

install_nvshmem() {
    wget https://developer.download.nvidia.com/compute/redist/nvshmem/${NVSHMEM_VERSION}/source/${NVSHMEM_FILE_BASE}.txz
    tar xf ${NVSHMEM_FILE_BASE}.txz
    rm -f ${NVSHMEM_FILE_BASE}.txz
    mv ${NVSHMEM_FILE_BASE} nvshmem
    ln -s libmlx5.so.1 /usr/lib/$(uname -m)-linux-gnu/libmlx5.so
    cd ./nvshmem && patch -p1 < /tmp/patches/nvshmem.ibgda.patch && make -j install && rm -rf build
}


install_wholegraph() {
    git clone https://github.com/fmtlib/fmt.git /opt/fmt
    cd /opt/fmt && git checkout 9.1.0 && mkdir build && cd build && \
        cmake -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE .. && \
        make && make install
    git clone https://github.com/gabime/spdlog.git /opt/spdlog
    cd /opt/spdlog && mkdir build && cd build && \
        cmake .. && make -j && cp libspdlog.a /usr/lib/libspdlog.a
    export PYTHON=/usr/bin/python
    pip install scikit-build-core
    pip install rapids_build_backend
    cd /opt && tar -xzf wholegraph.tar.gz
    # cd /opt/wholegraph && bash build.sh libwholegraph pylibwholegraph -v --enable-nvshmem
    cd /opt/wholegraph && bash build.sh libwholegraph pylibwholegraph -v --allgpuarch
}


install_dgl() {
    python3 -m pip install pynvml
    git clone https://github.com/dmlc/dgl /opt/dgl
    cd /opt/dgl && \
        git checkout 51907e048558d49b21044ea5955d3d46466b1a72 && \
        git submodule init && \
        git submodule update
    cd /opt/dgl/third_party/GKlib && patch -p1 < /tmp/patches/dgl.GKlib.no-march-native.patch
    cd /opt/dgl/third_party/METIS && patch -p1 < /tmp/patches/dgl.METIS.no-march-native.patch
    mkdir /opt/dgl/build && cd /opt/dgl/build
    export NCCL_ROOT=/usr
    cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release \
        -DUSE_CUDA=ON -DCUDA_ARCH_BIN="${CUDA_ARCH_BIN}" -DCUDA_ARCH_PTX="${CUDA_ARCH_PTX}" \
        -DCUDA_ARCH_NAME="Manual" \
        -DBUILD_TORCH=ON \
        -DBUILD_SPARSE=ON \
        -DBUILD_GRAPHBOLT=ON
    cmake --build .
    cd /opt/dgl/python && python setup.py bdist_wheel \
        && pip install ./dist/dgl*.whl \
        && rm -rf ./dist \
        && rm -rf ../build
    pip install --no-cache-dir ogb torchmetrics
}


case $BUILD_CONTEXT in
  x86_64)
    install_nvshmem
    install_wholegraph
    install_dgl
    ;;
  aarch64-Grace)
    ;;
  aarch64-Orin)
    ;;
  *)
    echo "Supported BUILD_CONTEXT are only x86_64."
    exit 1
    ;;
esac
