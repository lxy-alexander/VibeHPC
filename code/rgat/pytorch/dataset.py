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

from enum import Enum, unique
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional, List, Tuple

import dataclasses as dcls
import logging
import numpy as np
import os
import pylibwholegraph.torch as wgth
import torch
import yaml

from .common.helper import BlockWiseRoundRobinSharder, FP8Helper


@dcls.dataclass
class GraphMetadata:
    subdir: str
    n_paper_nodes: int
    n_author_nodes: int


@unique
class IGBSize(Enum):
    Tiny = GraphMetadata("tiny", 100000, 357041)
    Small = GraphMetadata("small", 1000000, 1926066)
    Medium = GraphMetadata("medium", 10000000, 15544654)
    Large = GraphMetadata("large", 100000000, 116959896)
    Full = GraphMetadata("full", 269346174, 277220883)


@dcls.dataclass(eq=True, frozen=True)
class Edge:
    src: str
    action: str
    dst: str

    def name(self):
        return f"{self.src}__{self.action}__{self.dst}"

    @classmethod
    def from_str(cls, s):
        return Edge(*s.split("__"))

    def as_tuple(self):
        return (self.src, self.action, self.dst)


# Taken from
# https://github.com/pytorch/pytorch/blob/e180ca652f8a38c479a3eff1080efe69cbc11621/torch/testing/_internal/common_utils.py#L349
numpy_to_torch_dtype_dict = {np.bool_: torch.bool,
                             np.uint8: torch.uint8,
                             np.int8: torch.int8,
                             np.int16: torch.int16,
                             np.int32: torch.int32,
                             np.int64: torch.int64,
                             np.float16: torch.float16,
                             np.float32: torch.float32,
                             np.float64: torch.float64,
                             np.complex64: torch.complex64,
                             np.complex128: torch.complex128}


def np_dtype_to_torch_dtype(dtype):
    if dtype in numpy_to_torch_dtype_dict:
        torch_dtype = numpy_to_torch_dtype_dict[dtype]
    else:
        # Try to infer from torch.from_numpy
        torch_dtype = torch.from_numpy(np.array([], dtype=dtype)).dtype
    return torch_dtype


def load_wholememory_tensor(path: os.PathLike,
                            shape: Tuple[int, ...],
                            dtype: np.dtype,
                            comm,
                            wmtype,
                            location,
                            sampling_device: str):
    assert len(shape) <= 2

    logging.debug(f"Creating WM tensor from {path} with type={dtype}, shape={shape} on {location}")

    wg_tensor = wgth.create_wholememory_tensor_from_filelist(
        comm,
        wmtype,
        location,
        path,
        np_dtype_to_torch_dtype(dtype),
        shape[-1] if len(shape) == 2 else 0)
    assert wg_tensor.shape == shape
    return wg_tensor.get_global_tensor(host_view=(sampling_device=='cpu'))

def get_edge_types(size_variant: IGBSize):
    edges = [Edge("paper", "cites", "paper"),
             Edge("paper", "written_by", "author"),
             Edge("author", "affiliated_to", "institute"),
             Edge("paper", "topic", "fos")]

    if size_variant in (IGBSize.Large, IGBSize.Full):
        edges.append(Edge("paper", "published", "journal"))
        edges.append(Edge("paper", "venue", "conference"))
    return edges

def get_node_types(size_variant: IGBSize):
    nodes = ["paper", "author", "institute", "fos"]

    if size_variant in (IGBSize.Large, IGBSize.Full):
        nodes.append("conference")
        nodes.append("journal")
    return nodes


class IGBHeteroGraphStructure:
    def __init__(self,
                 config: Dict,
                 data_root: os.PathLike = "/home/mlperf_inf_rgat/optimized/converted/graph",
                 size_variant: IGBSize = IGBSize.Full,
                 num_classes: int = 2983,
                 wholegraph_comms: Dict = None,
                 graph_device: str = "cpu",
                 sampling_device: str = "cuda",
                 graph_sharding_partition: str = "node"):
        assert wholegraph_comms is not None, "Cannot create WholeGraph graph structure without comms settings"

        self.data_root = Path(data_root)
        self.size = size_variant

        self._dir = self.data_root / self.size.value.subdir

        if num_classes == 2983:
            self.label_file = "node_label_2K.npy"
            self.full_num_trainable_nodes = 157675969
        else:
            self.label_file = "node_label_19.npy"
            self.full_num_trainable_nodes = 227130858

        self.config = config
        self.graph_device = graph_device
        self.sampling_device = sampling_device
        self.graph_sharding_partition = graph_sharding_partition

        self.node_comm = wholegraph_comms["node"]
        self.global_comm = wholegraph_comms["global"]

        if self.graph_sharding_partition == "node":
            self.edge_comm = self.node_comm
        else:
            self.edge_comm = self.global_comm

        self.edge_dict, self.node_counts = self.load_graph_data()
        self.label = self.load_labels()
        # self.train_indices = self.load_indices(subset="train")
        self.val_indices = self.load_indices(subset="val")

    def load_graph_data(self):
        node_counts = {name: _conf["node_count"]
                       for name, _conf in self.config["nodes"].items()}

        graph_data = dict()
        for name, _conf in self.config["edges"].items():
            e = Edge.from_str(name)
            graph_format = _conf["format"]
            graph_filenames = _conf["filenames"]
            graph_array_len = _conf["array_len"]
            graph_array_dtype = _conf["array_dtype"]

            assert len(graph_filenames) == 3
            # First tensor is column index pointer
            # Second is row indices
            # Third is edge IDs, which is not needed. Use an empty tensor
            _t = [load_wholememory_tensor(str(self._dir / graph_filenames[i]),
                                          (graph_array_len[i],),
                                          np.dtype(graph_array_dtype[i]),
                                          self.edge_comm,
                                          "continuous",
                                          self.graph_device,
                                          self.sampling_device)
                  for i in range(2)]
            _t.append(torch.tensor([], dtype=torch.int64))
            graph_data[e.as_tuple()] = (graph_format, tuple(_t))
        return graph_data, node_counts

    def load_labels(self):
        return torch.from_numpy(np.load(self._dir / self.label_file)).to(torch.long)

    def load_indices(self, subset: str = "train"):
        p = self._dir / f"{subset}_idx.pt"
        assert p.exists(), f"{subset} indices not found. Has preprocessing been run?"
        return torch.load(str(p))


class IGBHeteroLazyFeatures:
    """Lazily initializes features / embeddings for IGBH. Will only be initialized when .build_features is called.
    """
    def __init__(self,
                 tensor_dict: Dict,
                 data_root: os.PathLike = "/home/mlperf_inf_rgat/optimized/converted/embeddings",
                 size_variant: IGBSize = IGBSize.Full,
                 wholegraph_comms: Dict = None,
                 concat_embedding_mode: str = None,
                 wg_gather_sm: int = -1,
                 fp8_embedding: bool = False):
        self.data_root = Path(data_root)
        self.size = size_variant
        self.is_fp8 = fp8_embedding
        if fp8_embedding:
            self._dir = self.data_root / self.size.value.subdir / "float8"
            logging.info(f"Loading FP8 embeddings from {self._dir}")
        else:
            self._dir = self.data_root / self.size.value.subdir / "float16"
            self.fp8_helper = None

        with (self._dir / "config.yml").open(mode='r') as f:
            self.config = yaml.safe_load(f)

        if self.is_fp8:
            self.fp8_helper = FP8Helper(device="cuda",
                                        scale=self.config["fp8"]["scale"],
                                        fp8_format=self.config["fp8"]["format"])

        self.tensor_dict = tensor_dict
        self.wholegraph_comms = wholegraph_comms
        self.concat_embedding_mode = concat_embedding_mode

        self.edge_types = [Edge.from_str(canonical_etype).as_tuple() for canonical_etype in self.config["edges"]]

        node_config = self.config["nodes"]
        if self.concat_embedding_mode:
            assert self.concat_embedding_mode == 'offline', "Only 'offline' embedding mode is supported."
            # Check if concat_embedding_mode can be used
            _L = list(self.tensor_dict.values())
            wg_option = _L[0]
            for opt in _L[1:]:
                if opt != wg_option:
                    raise ValueError("concat embedding requires all embedding tables to have the same " \
                                     "WG sharding options")
            _nodes = list(node_config.values())
            feat_dtype = _nodes[0]["feat_dtype"]
            feat_dim = _nodes[0]["feat_dim"]
            for opst in _nodes[1:]:
                if opts["feat_dtype"] != feat_dtype:
                    raise ValueError("concat embedding requires all embedding tables to have the same dtype")
                if opts["feat_dim"] != feat_dim:
                    raise ValueError("concat embedding requires all embedding tables to have the same dim")

            feat_comm = self.wholegraph_comms[wg_option["partition"]]
            torch_dtype = np_dtype_to_torch_dtype(feat_dtype)
            self.node_names = list()
            counts = list()
            self.node_files = list()
            for name in self.config["concatenated_features"]["node_orders"]:
                self.node_names.append(name)
                counts.append(node_config[name]["node_count"])
                self.node_files.append(node_config[name]["feat_filename"])

            node_counts = torch.tensor(counts).to(torch.int64).to("cuda")
            self.node_offsets = torch.zeros(node_counts.numel() + 1, dtype=torch.int64, device="cuda")
            self.node_offsets[1:] = torch.cumsum(node_counts, dim=0)

            concatenated_config = self.config['concatenated_features']
            self.concat_embedding_file_path = self._dir / concatenated_config['path']
            self.sharder = BlockWiseRoundRobinSharder(concatenated_config["block_size"],
                                                      concatenated_config['num_buckets'],
                                                      self.node_offsets[-1])
            node_storage = wgth.create_embedding(comm=feat_comm,
                                                 memory_type=wg_option["type"],
                                                 memory_location=wg_option["location"],
                                                 dtype=torch_dtype,
                                                 sizes=(concatenated_config["total_number_of_nodes"], feat_dim),
                                                 gather_sms=wg_gather_sm)
            self.feature = node_storage
        else:
            self.feature = dict()
            for name, _conf in node_config.items():
                node_option = self.tensor_dict[name]
                node_count = _conf["node_count"]
                feat_dtype = np.dtype(_conf["feat_dtype"])
                feat_dim = _conf["feat_dim"]
                logging.info(f"Creating wholegraph {feat_dtype} embedding with dims: {node_count}x{feat_dim}")
                feat_comm = self.wholegraph_comms[node_option["partition"]]
                torch_dtype = np_dtype_to_torch_dtype(feat_dtype)
                node_storage = wgth.create_embedding(comm=feat_comm,
                                                     memory_type=node_option["type"],
                                                     memory_location=node_option["location"],
                                                     dtype=torch_dtype,
                                                     sizes=(node_count, feat_dim),
                                                     gather_sms=wg_gather_sm)
                self.feature[name] = node_storage

    def warmup(self):
        if self.concat_embedding_mode:
            indices = torch.zeros((1,), dtype=torch.int64, device='cuda')
            self.feature.gather(indices)
        else:
            for name in self.config['nodes']:
                indices = torch.zeros((1,), dtype=torch.int64, device='cuda')
                self.feature[name].gather(indices)

    def build_features(self):
        if self.concat_embedding_mode:
            self.feature.get_embedding_tensor().from_filelist(str(self.concat_embedding_file_path))
        else:
            for name, _conf in self.config["nodes"].items():
                feat_filename = _conf["feat_filename"]
                _p = self._dir / feat_filename
                self.feature[name].get_embedding_tensor().from_filelist(str(_p))

    def get_input_features(self, input_dict, device):
        if self.concat_embedding_mode:
            list_idx = [input_dict[ntype] + self.node_offsets[i]
                        for i, ntype in enumerate(self.node_names)]
            concat_idx = torch.concat(list_idx)
            # if self.concat_embedding_mode == 'offline':
            concat_idx = self.sharder.map(concat_idx)

            concat_out = self.feature.gather(concat_idx).to(device)
            num_requested_nodes = [idx.size(0) for idx in list_idx]
            tensor_num_requested_nodes = torch.tensor(num_requested_nodes).to(torch.int64)
            idx_offsets = torch.zeros(len(list_idx) + 1, dtype=torch.int64)
            idx_offsets[1:] = torch.cumsum(tensor_num_requested_nodes, dim=0)

            results = {
                key: concat_out[idx_offsets[i]:idx_offsets[i+1]].detach()
                for i, key in enumerate(self.node_names)
            }

            if self.fp8_helper is not None:
                return {
                    node: self.fp8_helper.fp8_to_fp16(embedding)
                    for node, embedding in results.items()
                }
            else:
                return results
        else:
            results = {
                key: self.feature[key].gather(value).to(device)
                for key, value in input_dict.items()
            }

            if self.fp8_helper is not None:
                return {
                    node: self.fp8_helper.fp8_to_fp16(embedding)
                    for node, embedding in results.items()
                }
            else:
                return results
