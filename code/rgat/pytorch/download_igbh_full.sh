#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
echo "IGBH600M (Heteregeneous) download starting"

DATA_ROOT="${RGAT_SCRATCH_PATH:-/home/mlperf_inf_rgat}"
DL_ROOT=$DATA_ROOT/data/fp32/igbh_full

mkdir -p $DL_ROOT

# paper
mkdir $DL_ROOT/paper
cd $DL_ROOT/paper
wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/paper/node_feat.npy
wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/paper/node_label_19.npy
wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/paper/node_label_2K.npy
wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/paper/paper_id_index_mapping.npy

download_edge () {
    mkdir $DL_ROOT/$1
    cd $DL_ROOT/$1

    wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/$1/edge_index.npy
}

download_node () {
    mkdir $DL_ROOT/$1
    cd $DL_ROOT/$1

    wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/$1/$1_id_index_mapping.npy
    wget -c https://igb-public.s3.us-east-2.amazonaws.com/IGBH/processed/$1/node_feat.npy
}

# paper__cites__paper
download_edge paper__cites__paper

# author
download_node author

# conference
download_node conference

# institute
download_node institute

# journal
download_node journal

# fos
download_node fos

# author__affiliated_to__institute
download_edge author__affiliated_to__institute

# paper__published__journal
download_edge paper__published__journal

# paper__topic__fos
download_edge paper__topic__fos

# paper__venue__conference
download_edge paper__venue__conference

# paper__written_by__author
download_edge paper__written_by__author

echo "IGBH-IGBH (Heteregeneous) download complete"
