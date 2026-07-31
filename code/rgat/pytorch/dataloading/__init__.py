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
#https://github.com/mlcommons/training_results_v4.1/blob/main/NVIDIA/benchmarks/gnn/implementations/h200_ngc24.04_dgl/dataloading/__init__.py


import torch
from code.rgat.pytorch.dataloading.sampler import PyGSampler
from .overlap_dataloader import PrefetchInterleaver

try:
    import dgl
    DGL_AVAILABLE = True
except ModuleNotFoundError:
    DGL_AVAILABLE = False
    dgl = None


def check_dgl_available():
    assert DGL_AVAILABLE, "DGL Not available in the container"


def build_graph(graph_structure, features):
    check_dgl_available()

    graph = dgl.heterograph(graph_structure.edge_dict, graph_structure.node_counts)
    graph.predict = "paper"

    assert features is not None, "Features must not be none!"

    for node, d in features.config['nodes'].items():
        if graph.num_nodes(ntype=node) < d['node_count']:
            graph.add_nodes(d['node_count'] - graph.num_nodes(ntype=node), ntype=node)
        else:
            assert graph.num_nodes(ntype=node) == d['node_count'], f"\
            Graph has more {node} nodes ({graph.num_nodes(ntype=node)}) \
                than feature shape ({d['node_count']})"

    # graph.label = graph_structure.label.to(graph.device)
    graph.label = graph_structure.label
    return graph
