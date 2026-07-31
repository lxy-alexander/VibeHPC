# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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


# This file is based off NVIDIA GNN training code. Public link:
# https://github.com/mlcommons/training_results_v4.1/blob/main/NVIDIA/benchmarks/gnn/implementations/h200_ngc24.04_dgl/model/__init__.py


import torch
import math

try:
    import dgl
    DGL_AVAILABLE = True
except ModuleNotFoundError:
    DGL_AVAILABLE = False
    dgl = None


def check_dgl_available():
    assert DGL_AVAILABLE, "DGL Not available in the container"


from code.rgat.pytorch.model.dgl_model import RGAT_DGL, FeatureExtractor_DGL


def get_model(backend, gatconv_backend, switches, pad_node_count_to, etypes, **model_kwargs):
    check_dgl_available()

    return RGAT_DGL(
        etypes=etypes,
        **model_kwargs,
        gatconv_backend=gatconv_backend,
        switches=switches,
        pad_node_count_to=pad_node_count_to)


def get_feature_extractor(backend, formats=None):
    check_dgl_available()
    return FeatureExtractor_DGL(formats=formats)


def gen_synthetic_block(list_dict_node_count, list_dict_edge_count, batch_size, device):
    list_dict_node_count.append({ntype: batch_size if ntype == "paper" else 0 for ntype in list_dict_node_count[0]})
    blocks_synth = []
    for layer, dict_edge_count in enumerate(list_dict_edge_count):
        dict_graph = {}
        for etype, edge_count in dict_edge_count.items():
            rows = torch.tensor([], dtype=torch.int32, device=device)
            cols = torch.tensor([], dtype=torch.int32, device=device)
            if edge_count != 0:
                num_src_nodes = list_dict_node_count[layer][etype[0]]
                num_dst_nodes = list_dict_node_count[layer+1][etype[2]]

                rows = torch.arange(num_src_nodes-1, -1, step=-1, device=device, dtype=torch.int32).repeat(math.ceil(edge_count / num_src_nodes))[:edge_count]
                cols = torch.arange(num_dst_nodes-1, -1, step=-1, device=device, dtype=torch.int32).repeat(math.ceil(edge_count / num_dst_nodes))[:edge_count]

            dict_graph[etype] = (rows, cols)

        block = dgl.create_block(dict_graph, num_src_nodes=list_dict_node_count[layer], num_dst_nodes=list_dict_node_count[layer+1]).formats('csc')
        blocks_synth.append(block)
    return blocks_synth
