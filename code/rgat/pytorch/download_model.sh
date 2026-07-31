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

MLPERF_SCRATCH_PATH=${MLPERF_SCRATCH_PATH:-/home/mlperf_inference_storage}
MODEL_PATH=${MLPERF_SCRATCH_PATH}/models/rgat


# Install rclone
if ! command -v rclone 2>&1 >/dev/null
then
    sudo -v ; curl https://rclone.org/install.sh | sudo bash
fi

rclone config create mlc-inference s3 \
    provider=Cloudflare \
    access_key_id=f65ba5eef400db161ea49967de89f47b \
    secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b \
    endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com

mkdir -p ${MODEL_PATH}
rclone copy mlc-inference:mlcommons-inference-wg-public/R-GAT/RGAT.pt $MODEL_PATH -P
