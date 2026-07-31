#!/bin/bash

SRC_ROOT=$(dirname $(dirname $(readlink -f "${BASH_SOURCE[0]}")))

nvidia-smi 2>&1 > /dev/null
if [ $? -ne 0 ]; then
    GPU_FLAG=""
else
    GPU_FLAG="--gpus=all"
fi

docker run $GPU_FLAG \
    -v $SRC_ROOT:/work \
    -w /work \
    gitlab-master.nvidia.com/mlpinf/mlperf-inference/liftoff-launcher/mitten-slim-x86_64-default:dev-250609 \
    get_sys_id 2> /dev/null

