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


from dataclasses import dataclass
from typing import Optional
import os

from .dataset import IGBSize


@dataclass
class RGATConfig:
    dataset_size: IGBSize = IGBSize.Full
    embedding_path: os.PathLike = "/home/mlperf_inf_rgat/optimized/converted/embeddings"
    graph_path: os.PathLike = "/home/mlperf_inf_rgat/optimized/converted/graph"
    weights_path: os.PathLike = "build/models/rgat/RGAT.pt"
    num_classes: int = 2983
    fan_outs: str = "5,10,15"
    hidden_channels: int = 512
    num_heads = 4

    backend: str = "DGL"
    use_cuda_graph: bool = False
    infer_overlap: bool = True
    batch_size: int = 6144

    wg_sharding_location: str = "cuda"
    wg_sharding_partition: str = "node"
    wg_sharding_type: str = "continuous"
    sampling_device: str = "cuda"
    graph_device: str = "cuda"
    graph_sharding_partition: str = "node"
    concat_embedding_mode: Optional[str] = None
    wg_gather_sm: int = 16 
    fp8_embedding: bool = True
    gatconv_backend: str = "cugraph"
    cugraph_switches: str = "1100"
    pad_node_count_to: int = 3072
    gc_threshold_multiplier: int = 1
    high_priority_embed_stream: bool = True
    num_sampling_threads: int = 1
    num_workers: int = 0
