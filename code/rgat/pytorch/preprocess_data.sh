#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


DATA_ROOT="${RGAT_SCRATCH_PATH:-/home/mlperf_inf_rgat}"
DL_ROOT=$DATA_ROOT/data
PREPROCESSED_ROOT=$DATA_ROOT/optimized/converted
CURR_DIR=$PWD
INFERENCE_DIR=${INFERENCE_DIR:-/work/3rdparty/mlc-inference}

mkdir -p ${PREPROCESSED_ROOT}
cd /work && make clone_loadgen
echo "Generating train and validation splits..."
mkdir -p /tmp/rgat/full/processed && \
    python3 ${INFERENCE_DIR}/graph/R-GAT/tools/split_seeds.py --dataset_size full --path /tmp/rgat
mv /tmp/rgat/full/processed/*.pt ${DL_ROOT}/igbh_full/
echo "Done generating train and validation splits"
md5sum ${DL_ROOT}/igbh_full/*.pt | tee ${DL_ROOT}/igbh_full/indices.checksum
rm -rf /tmp/rgat

echo "Converting dataset to fp16 and fp8 for wholegraph"
cd $CURR_DIR && python3 preprocess_data.py \
    --data_dir ${DL_ROOT} \
    --preprocessed_data_dir ${PREPROCESSED_ROOT} \
    --precision float16 \
    --size full \
    --shuffle paper author conference journal fos institute
