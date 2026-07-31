#!/bin/bash

set -e

install_gflags(){
    local VERSION=$1

    cd /tmp
    # -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DBUILD_gflags_LIB=ON .. \
    git clone -b v${VERSION} https://github.com/gflags/gflags.git
    cd gflags
    mkdir build && cd build
    cmake -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DBUILD_gflags_LIB=ON ..
    make -j
    make install
    cd /tmp && rm -rf gflags
}

install_glog(){
    local VERSION=$1

    cd /tmp
    git clone -b v${VERSION} https://github.com/google/glog.git
    cd glog
    cmake -H. -Bbuild -G "Unix Makefiles" -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON
    cmake --build build
    cmake --build build --target install
    cd /tmp && rm -rf glog

}

install_CUB(){
    local ARCH=$1

    # Install CUB, needed by NMS OPT plugin
    cd /tmp
    wget https://github.com/NVlabs/cub/archive/1.8.0.zip -O cub-1.8.0.zip
    unzip cub-1.8.0.zip
    mv cub-1.8.0/cub /usr/include/${ARCH}-linux-gnu/
    rm -rf cub-1.8.0.zip cub-1.8.0
}


install_boost_1_78(){
    apt purge libboost-all-dev \
      && apt autoremove -y libboost1.74-dev \
      && sudo rm -rf /usr/lib/libboost_* /usr/include/boost \
      && wget -O /tmp/boost.tar.gz \
          https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz \
      && (cd /tmp && tar xzf boost.tar.gz) \
      && mv /tmp/boost_1_80_0/boost /usr/include/boost
}


install_triton_link() {
    # create a symlink to triton installation directory from /opt/tritonserver
    ln -sf /work/build/triton-inference-server/out/opt/tritonserver/ /opt/tritonserver
}

install_pytest_deps(){
    python3 -m pip install pytest==8.4.0 pytest-metadata==3.1.0 pytest-html==4.1.1
}

case ${BUILD_CONTEXT} in
  x86_64)
    install_gflags 2.2.1
    install_glog 0.6.0
    install_CUB x86_64
    install_boost_1_78
    install_triton_link
    install_pytest_deps
    ;;
  aarch64-Grace)
    install_gflags 2.2.2
    install_glog 0.6.0
    install_CUB aarch64
    install_boost_1_78
    install_triton_link
    install_pytest_deps
    ;;
  *)
    echo "Supported BUILD_CONTEXT are only x86_64, aarch64-Grace."
    exit 1
    ;;
esac
