#!/bin/bash

set -e

if [[ $ARCH == "aarch64" ]]; then
    # Add mirror site as aarch64 pkgs sometimes get lost
	sed -i -e 's/http:\/\/archive/mirror:\/\/mirrors/' -e 's/\/ubuntu\//\/mirrors.txt/' /etc/apt/sources.list
else
    # MLPINF-1247 - Some partners in China are reporting DNS issues with Apt, specifically with cuda-repo. Remove the .list.
    rm -f /etc/apt/sources.list.d/cuda.list
fi


install_core_packages(){
    apt update
    apt install -y --no-install-recommends build-essential autoconf libtool git git-lfs \
        ccache curl wget pkg-config sudo ca-certificates automake libssl-dev tree \
        bc python3-dev python3-pip google-perftools gdb libglib2.0-dev clang sshfs libre2-dev \
        libboost-dev libnuma-dev numactl sysstat sshpass ntpdate less vim iputils-ping pybind11-dev
    apt install --only-upgrade libksba8
    apt remove -y cmake
    apt remove -y libgflags-dev
    apt remove -y libprotobuf-dev
    apt -y autoremove
    apt install -y --no-install-recommends pkg-config zip g++ zlib1g-dev unzip
    apt install -y --no-install-recommends libarchive-dev
    apt install -y --no-install-recommends rsync

    # Needed by Triton
    apt install -y rapidjson-dev
    apt install -y libb64-dev
    apt install -y libgtest-dev

    # Needed by mixtral submission_checker
    # Ref: https://github.com/mlcommons/inference/blob/master/language/mixtral-8x7b/Dockerfile.eval
    apt install -y --no-install-recommends apt-transport-https bison libffi-dev libgdbm-dev \
        libncurses5-dev libreadline-dev libyaml-dev lsb-release software-properties-common \
        tzdata zlib1g-dev
}

install_platform_specific_x86_64(){
    # Install libjemalloc2
    echo 'deb http://archive.ubuntu.com/ubuntu focal main restricted universe multiverse' | tee -a /etc/apt/sources.list.d/focal.list
    echo 'Package: *\nPin: release a=focal\nPin-Priority: -10\n' | tee -a /etc/apt/preferences.d/focal.pref
    apt update
    apt install --no-install-recommends -t focal -y libjemalloc2 libtcmalloc-minimal4

    # For cv2
    apt install -y libgl1-mesa-glx

    # For SDXL accuracy venv
    PY_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[0:2])))')
    apt install -y python${PY_VERSION}-venv
}

install_platform_specific_grace(){
   # Install libjemalloc2
    apt update
    apt install --no-install-recommends -y libjemalloc2 libtcmalloc-minimal4

    # for SDXL accuracy venv
    PY_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[0:2])))')
    apt install -y python${PY_VERSION}-venv

    # Some convenient tools
    apt install -y ripgrep

    # install RUST
    apt install -y cargo
}

case ${BUILD_CONTEXT} in
  x86_64)
    install_core_packages
    install_platform_specific_x86_64
    ;;
  aarch64-Grace)
    install_core_packages
    install_platform_specific_grace
    ;;
  *)
    echo "Supported BUILD_CONTEXT are only x86_64, aarch64-Grace."
    exit 1
    ;;
esac
