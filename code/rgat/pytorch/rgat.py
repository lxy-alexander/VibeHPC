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

# Standard library imports
import logging
import os
import re
import sys
import threading
import time
from typing import List

# Environment setup
os.environ["DGL_PREFETCHER_TIMEOUT"] = str(300)
os.environ["DGLBACKEND"] = "pytorch"

# Third-party imports
import dgl
import mlperf_loadgen as lg
import numpy as np
import pylibwholegraph.torch as wgth
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from tqdm import tqdm

# Local imports
# pylint: disable=wrong-import-position
sys.path.insert(0, "build/harness/lib")
from FFIUtils import QuerySamplesCompletePool

from .config import RGATConfig
from .dataloading import build_graph
from .dataloading.sampler import PyGSampler
from .dataloading.overlap_dataloader import PrefetchInterleaver
from .dataset import IGBHeteroGraphStructure, IGBHeteroLazyFeatures, get_node_types
from .model import get_model, get_feature_extractor, gen_synthetic_block
from .utils import SharedBuffer, SharedBufferCollection


# Force 'spawn'
mp.set_start_method("spawn", force=True)


def preprocess_model_state(weights):
    """Preprocesses model weights for compatibility with the current model.

    This function performs several preprocessing steps on the model weights:
    1. Concatenates attn_l and attn_r
    2. Removes prefix from parameter names
    3. Renames edges according to the translation dictionary

    Args:
        weights (dict): Dictionary containing the model state dict

    Returns:
        dict: Processed state dictionary ready for model loading
    """
    prefix_str = "model."
    edge_transl_dict = {
        "rev_written_by" : "('author', 'reverse_written_by', 'paper')",
        "rev_venue" : "('conference', 'reverse_venue', 'paper')",
        "rev_topic" : "('fos', 'reverse_topic', 'paper')",
        "rev_affiliated_to" : "('institute', 'reverse_affiliated_to', 'author')",
        "rev_published" : "('journal', 'reverse_published', 'paper')",
        "affiliated_to" : "('author', 'affiliated_to', 'institute')",
        "cites" : "('paper', 'cites', 'paper')",
        "published" : "('paper', 'published', 'journal')",
        "topic" : "('paper', 'topic', 'fos')",
        "venue" : "('paper', 'venue', 'conference')",
        "written_by" : "('paper', 'written_by', 'author')"
    }
    weight_transl_dict = {
        "fc.weight" : "lin.weight"
    }

    state_dict = {}

    # step 1: parameter renaming
    for (key, val) in weights['model_state_dict'].items():
        # remove prefix
        new_key = key.replace(prefix_str,"")
        # renamed edges
        for patt, new_patt in edge_transl_dict.items():
            if patt in new_key :
                new_key = re.sub(patt, new_patt, new_key)
                break
        # renamed weights
        for patt, new_patt in weight_transl_dict.items():
            new_key = re.sub(patt, new_patt, new_key)
        # assign dict
        state_dict[new_key] = val

    # step 2: attn_l and attn_r concatenation (and subsequent removal)
    for layer in [ "layers.0", "layers.1", "layers.2" ]:
        for edge in edge_transl_dict.values():
            dest_param = f"{layer}.mods.{edge}.attn_weights"
            src_param_1 = f"{layer}.mods.{edge}.attn_l"
            src_param_2 = f"{layer}.mods.{edge}.attn_r"
            state_dict[dest_param] = torch.cat([state_dict[src_param_1], state_dict[src_param_2]], dim=0)
    for layer in [ "layers.0", "layers.1", "layers.2" ]:
        for edge in edge_transl_dict.values():
            src_param_1 = f"{layer}.mods.{edge}.attn_l"
            src_param_2 = f"{layer}.mods.{edge}.attn_r"
            del state_dict[src_param_1]
            del state_dict[src_param_2]

    # step 3: reshape dimensions
    #   size mismatch for attn_weights: copying a param with shape torch.Size([2, 4, 128]) from checkpoint, the shape in current model is torch.Size([1024]).
    #   size mismatch for bias: copying a param with shape torch.Size([512]) from checkpoint, the shape in current model is torch.Size([4, 128]).
    for key, val in state_dict.items():
        if key == "linear.bias":
            continue
        if 'attn_weights' in key:
            state_dict[key] = torch.flatten(val)
        elif 'bias' in key:
            state_dict[key] = torch.reshape(val, (4, 128))
    return state_dict


def load_model(etypes, conf: RGATConfig, device):
    """Loads and initializes the RGAT model with the given configuration.

    Args:
        etypes (list): List of edge types in the graph
        conf (RGATConfig): Configuration object containing model parameters
        device: Device to load the model onto

    Returns:
        tuple: (model, feature_extractor) - The loaded model and its feature extractor
    """
    model = get_model(backend=conf.backend,
                      gatconv_backend=conf.gatconv_backend,
                      switches=conf.cugraph_switches,
                      pad_node_count_to=conf.pad_node_count_to,
                      etypes=etypes,
                      in_feats=1024,
                      h_feats=conf.hidden_channels,
                      num_classes=conf.num_classes,
                      num_layers=len(conf.fan_outs.split(',')),
                      n_heads=conf.num_heads,
                      dropout=0.2,
                      with_trim=False).to(device)

    weights = preprocess_model_state(torch.load(conf.weights_path))
    model.load_state_dict(weights, strict=False)

    model = model.half()  # Set to fp16
    model.eval()

    if conf.gatconv_backend == "cugraph" and conf.backend == "DGL":
        formats = ["csc"]
    else:
        formats = None
    feature_extractor = get_feature_extractor(backend=conf.backend, formats=formats)
    return model, feature_extractor


def init_wholegraph(local_rank,
                    conf: RGATConfig,
                    seed: int = 0,
                    world_size: int = 8):
    """Initializes WholeGraph environment for distributed graph processing.

    Args:
        local_rank (int): Local rank of the process
        conf (RGATConfig): Configuration object
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
        world_size (int, optional): Total number of processes. Defaults to 8.

    Returns:
        tuple: (embedding_comms, feature_store) - Communication handles and feature store
    """
    logging.info("Starting wholegraph on rank %d", local_rank)
    torch.manual_seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)

    # Only single-node is supported right now
    global_rank = local_rank
    node_size = world_size

    os.environ["RANK"] = f"{local_rank}"
    os.environ["WORLD_SIZE"] = f"{world_size}"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    wg_log_level = "warn"
    wgth.init_torch_env(global_rank,
                        world_size,
                        local_rank,
                        node_size,
                        wg_log_level)
    torch.cuda.set_device(local_rank)

    # Prevent following communicators to lock the tree
    os.environ["NCCL_SHARP_DISABLE"] = "1"
    os.environ["NCCL_COLLNET_ENABLE"] = "0"

    # WholeGraph pre dataset loading setups: set default arguments
    tensor_dict = {
        node_name: {"partition": conf.wg_sharding_partition,
                    "location": conf.wg_sharding_location,
                    "type": conf.wg_sharding_type}
        for node_name in get_node_types(conf.dataset_size)
    }

    # WholeGraph inter-node/GPU communications
    embedding_comms = {
        "node": wgth.get_local_node_communicator(),
        "global": wgth.get_global_communicator(),
        "local": wgth.get_local_device_communicator()
    }

    # Init WG storage
    feature_store = IGBHeteroLazyFeatures(tensor_dict,
                                          data_root=conf.embedding_path,
                                          size_variant=conf.dataset_size,
                                          wholegraph_comms=embedding_comms,
                                          concat_embedding_mode=conf.concat_embedding_mode,
                                          wg_gather_sm=conf.wg_gather_sm,
                                          fp8_embedding=conf.fp8_embedding)
    return embedding_comms, feature_store


def load_dataset(conf: RGATConfig,
                 torch_device,
                 embedding_comms,
                 feature_store: IGBHeteroLazyFeatures,
                 skip_embedding_init: bool = False):
    """Loads and initializes the dataset for RGAT processing.

    Args:
        conf (RGATConfig): Configuration object
        torch_device: Device to load the dataset onto
        embedding_comms: Communication handles for distributed processing
        feature_store (IGBHeteroLazyFeatures): Feature store for the dataset
        skip_embedding_init (bool, optional): Whether to skip embedding initialization. Defaults to False.

    Returns:
        tuple: (dataset, graph) - The loaded dataset and its graph representation
    """
    dataset = IGBHeteroGraphStructure(feature_store.config,
                                      data_root=conf.graph_path,
                                      size_variant=conf.dataset_size,
                                      num_classes=conf.num_classes,
                                      wholegraph_comms=embedding_comms,
                                      graph_device=conf.graph_device,
                                      sampling_device=conf.sampling_device,
                                      graph_sharding_partition=conf.graph_sharding_partition)

    if not skip_embedding_init:
        logging.info("Loading features to %s", torch_device)
        feature_store.build_features()

    logging.info("Building graph on %s", torch_device)
    graph = build_graph(graph_structure=dataset,
                        features=feature_store)

    if conf.sampling_device == "cuda":
        logging.info("Moving graph to %s memory", torch_device)
        graph = graph.to(torch_device)

        # MLPerf Inference only uses validation set
        dataset.val_indices = dataset.val_indices.to(torch_device)
    return dataset, graph


class RGATCore:
    """Core class for RGAT (Relational Graph Attention Network) processing.

    This class handles the main RGAT operations including model initialization,
    inference, and query processing.

    Attributes:
        device_id (int): ID of the device this core is running on
        torch_device: PyTorch device handle
        conf (RGATConfig): Configuration object
        response_buffer (SharedBuffer): Buffer for storing responses
        batch_buffer (SharedBuffer): Buffer for storing batches
        embedding_comms: Communication handles for distributed processing
        feature_store (IGBHeteroLazyFeatures): Feature store for the dataset
        model: The RGAT model
        feature_extractor: Feature extractor for the model
        ds: Dataset object
        graph: Graph representation
        sampler (PyGSampler): Sampler for graph operations
    """

    def __init__(self,
                 conf: RGATConfig,
                 device_id: int,
                 response_buffer: SharedBuffer,
                 batch_buffer: SharedBuffer):
        """Initializes the RGATCore instance.

        Args:
            conf (RGATConfig): Configuration object
            device_id (int): ID of the device to run on
            response_buffer (SharedBuffer): Buffer for storing responses
            batch_buffer (SharedBuffer): Buffer for storing batches
        """
        torch.cuda.init()
        self.device_id = device_id
        self.torch_device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(self.torch_device)
        torch.autograd.set_grad_enabled(False)

        self.conf = conf

        self.response_buffer = response_buffer
        self.batch_buffer = batch_buffer

        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logging.info("Instantiating RGAT engine on device %d", device_id)
        self.embedding_comms, self.feature_store = init_wholegraph(self.device_id, conf)
        self.print_gpu_stats()
        logging.info("Loading model on device %d", device_id)
        self.model, self.feature_extractor = load_model(self.feature_store.edge_types, conf, self.torch_device)
        self.print_gpu_stats()

        self.ds, self.graph = load_dataset(conf, self.torch_device, self.embedding_comms, self.feature_store)
        self.print_gpu_stats()

        fanouts = [int(fanout) for fanout in self.conf.fan_outs.split(",")]
        self.sampler = PyGSampler(fanouts=fanouts, num_threads=self.conf.num_sampling_threads)

    def trace(self):
        logging.info("[{}] MLPINF TRACING time = {}".format(os.environ['RANK'],time.time()))

    def print_gpu_stats(self):
        """Prints current GPU memory usage statistics."""
        free, total = torch.cuda.mem_get_info(self.torch_device)
        free /= 1e9
        total /= 1e9
        used = total - free
        perc = used / total
        logging.info("GPU%d Mem: %.1f GB / %.1f GB Used (%.2f%%)", self.device_id, used, total, perc * 100)

    def __del__(self):
        wgth.finalize()

    def warmup(self):
        """Performs warmup operations to initialize the model and feature store."""
        self.feature_store.warmup()

        bs = self.conf.batch_size
        edge_types = list(self.model.layers[0].mod_dict.keys())
        node_types = list(set([x[0] for x in edge_types]))
        num_layers = len(self.model.layers)
        node_counts, edge_counts = [], []
        for layer in range(num_layers):
            node_counts.append({ntype: bs for ntype in node_types})
            if layer < num_layers - 1:
                edge_counts.append({etype: bs for etype in edge_types})
            else:
                edge_counts.append({etype: bs if etype[2] == 'paper' else 0 for etype in edge_types})
        node_counts.append({ntype: bs if ntype == 'paper' else 0 for ntype in node_types})
        blocks = gen_synthetic_block(node_counts, edge_counts, batch_size=bs, device=self.torch_device)
        x = {
            node: torch.randn((blocks[0].num_src_nodes(node), 1024),
                              device=self.torch_device,
                              dtype=torch.half)
            for node in blocks[0].ntypes}
        y = self.model(blocks, x)
        del y

    def infer(self, indices, ids):
        """Performs inference on the given indices.

        Args:
            indices: Indices of samples to process
            ids: IDs of the samples
        """
        # Remap: sample_indices is index within val_indices
        sample_indices = self.ds.val_indices[indices]
        sample_loader = dgl.dataloading.DataLoader(
            self.graph,
            {"paper": sample_indices},
            self.sampler,
            batch_size=self.conf.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.conf.num_workers)
        prefetcher = PrefetchInterleaver(self.feature_extractor,
                                         self.feature_store,
                                         self.torch_device,
                                         use_high_priority_stream=self.conf.high_priority_embed_stream,
                                         dataloader=sample_loader)

        # TODO: CUDA graphs is currently not supported. Add this feature later
        preds_list = list()
        labels_list = list()
        with torch.no_grad():
            offset = 0
            for batch in prefetcher:
                subgraph, paper_nodes, labels = prefetcher.get_inputs_and_outputs()
                assert not labels.is_cuda
                labels_list.append(labels.numpy().astype(np.uint64))

                y = self.model(subgraph, paper_nodes)
                del subgraph
                del paper_nodes

                _arr = y.argmax(1).detach().cpu().numpy().astype(np.uint64)
                del y

                offset += len(_arr)
                preds_list.append(_arr)
        del prefetcher
        del sample_loader

        predictions = np.concatenate(preds_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        N = len(ids)
        assert len(predictions) == N
        assert len(labels) == N

        with self.response_buffer.write_context(N * 3) as buff:
            buff[:N] = ids[:]
            buff[N:N*2] = predictions
            buff[N*2:N*3] = labels

    def await_queries(self):
        """Continuously processes queries from the batch buffer until stopped."""
        while True:
            with self.batch_buffer.read_context() as dat:
                if self.batch_buffer.stopped:
                    logging.info("RGATCore %d received STOP signal", self.device_id)
                    break

                assert len(dat) % 2 == 0
                n_entries = len(dat) // 2
                indices = dat[:n_entries]
                ids = dat[n_entries:n_entries*2]

                self.infer(indices, ids)
        logging.info("RGATCore %d completed", self.device_id)

    @staticmethod
    def create_core_inst(barrier, *args):
        """Creates and initializes a new RGATCore instance.

        Args:
            barrier: Synchronization barrier
            *args: Arguments to pass to RGATCore constructor
        """
        logging.info("Starting RGATCore on device %d", args[1])
        core = RGATCore(*args)
        logging.info("RGATCore %d warming up...", args[1])
        core.warmup()

        logging.info("RGATCore %d active. Awaiting queries...", args[1])
        barrier.wait()
        core.await_queries()


class RGATOfflineServer:
    """Server class for offline RGAT processing.

    This class manages the offline processing of RGAT queries, including
    delegation of samples to cores and handling of responses.

    Attributes:
        conf (RGATConfig): Configuration object
        devices (List[int]): List of device IDs to use
        ds_size (int): Dataset size
        delegator_max_size (int): Maximum size for delegator
        verbose (bool): Whether to print verbose output
        sample_buff (SharedBufferCollection): Buffer collection for samples
        response_buff (SharedBufferCollection): Buffer collection for responses
        procs (dict): Dictionary of running processes
        _cache (dict): Cache for query responses
        comp_pool (QuerySamplesCompletePool): Pool for completing queries
    """

    def __init__(self,
                 devices: List[int],
                 conf: RGATConfig,
                 ds_size: int = 788379,
                 delegator_max_size: int = 788379,
                 n_complete_threads: int = 64,
                 n_io_buffers: int = 16,
                 verbose: bool = True):
        """Initializes the RGATOfflineServer instance.

        Args:
            devices (List[int]): List of device IDs to use
            conf (RGATConfig): Configuration object
            ds_size (int, optional): Dataset size. Defaults to 788379.
            delegator_max_size (int, optional): Maximum size for delegator. Defaults to 788379.
            n_complete_threads (int, optional): Number of completion threads. Defaults to 64.
            n_io_buffers (int, optional): Number of I/O buffers. Defaults to 16.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
        """
        self.conf = conf
        self.devices = devices
        self.ds_size = ds_size
        self.delegator_max_size = delegator_max_size
        self.verbose = verbose

        # In Offline Scenario, we can also choose when to call QuerySamplesComplete since Offline only cares about total
        # throughput and not latency. Because of this, we can make our response buffer the same size as the input sample
        # buffer.

        # Allocate buffers for each device
        self.sample_buff = SharedBufferCollection(n_io_buffers,
                                                  self.delegator_max_size * 2,
                                                  typecode='Q')
        self.response_buff = SharedBufferCollection(n_io_buffers,
                                                    self.delegator_max_size * 3,
                                                    typecode='Q')

        self.procs = self.start()

        logging.info("Creating QuerySampleComplete pool with %d threads", n_complete_threads)
        self._cache = dict()
        self.comp_pool = QuerySamplesCompletePool(n_complete_threads)

    def start(self):
        """Starts the server and initializes all cores.

        Returns:
            dict: Dictionary of running processes
        """
        procs = dict()
        barrier = mp.Barrier(len(self.devices) + 1)
        for dev_id in self.devices:
            _args = (barrier,
                     self.conf,
                     dev_id,
                     self.response_buff,
                     self.sample_buff)

            p = mp.Process(target=RGATCore.create_core_inst, args=_args)
            p.start()
            procs[dev_id] = p

        logging.info("Waiting for cores to start up...")
        barrier.wait()
        logging.info("All cores started")
        return procs

    def signal_stop(self):
        """Signals all processes to stop."""
        self.sample_buff.stop()

    def shutdown(self):
        """Shuts down the server and cleans up resources."""
        self.signal_stop()
        for dev_id in self.devices:
            if dev_id in self.procs:
                logging.info("Joining process %d", dev_id)
                self.procs[dev_id].join()
                self.procs.pop(dev_id)

        self.response_buff.stop()

    def handle_responses(self, n_expected: int):
        """Handles responses from the cores.

        Args:
            n_expected (int): Number of expected responses
        """
        logging.info("Waiting for %d responses", n_expected)
        n_completed = 0
        correct = 0

        if self.verbose:
            pbar = tqdm(total=n_expected, mininterval=1)

        while n_completed < n_expected:
            with self.response_buff.read_context() as dat:
                assert len(dat) % 3 == 0
                n_resp = len(dat) // 3

                ids = np.array(dat[:n_resp], dtype=np.uint64)
                preds = np.ascontiguousarray(dat[n_resp:n_resp*2], dtype=np.uint64)
                labels = np.array(dat[n_resp*2:n_resp*3], dtype=np.uint64)

                # Generating extremely large lists of QueryResponses via loops or list comp is very slow.
                # This is kind of an insane way to do this, but under the hood, a QueryResponse is just 4 int64s, where
                # the first 3 are unsigned.
                # QueryResponse is a struct of (id, data_ptr, byte_size, n_toks)
                qr = np.zeros(n_resp * 4, dtype=np.uint64)
                # Assign every 4th idx with offset 0 to id
                qr[::4] = ids
                # Assign every 4th idx with offset 1 to data_ptrs. This is why we force preds being contiguous.
                assert preds.itemsize == 8
                start_ptr = preds.ctypes.data
                end_ptr = preds.ctypes.data + (n_resp * preds.itemsize)
                qr[1::4] = np.arange(start_ptr, end_ptr, preds.itemsize, dtype=np.uint64)
                # Assign every 4th idx with offset 2 to itemsize
                qr[2::4] = preds.itemsize

                # Keep QueryResponses and predictions alive to prevent seg fault.
                # We can use result pointer as key for dict as it is unique
                self._cache[qr.ctypes.data] = (qr, preds)

            n_completed += n_resp
            correct += (preds == labels).sum()

            if self.verbose:
                if n_completed:
                    acc = correct / n_completed * 100
                    pbar.set_description(f"Acc: {acc: 3.2f}%")
                else:
                    pbar.set_description("Waiting...")
                pbar.update(n_resp)

            self.comp_pool.enqueue(qr.ctypes.data, n_resp)
        pbar.close()
        logging.info("%d responses handled with accuracy %.2f%%", n_completed, correct / n_completed * 100)

    def delegate_samples(self, query_samples):
        """Delegates samples to the cores for processing.

        Args:
            query_samples: List of query samples to process
        """
        n_devs = len(self.devices)
        S = min(len(query_samples) // n_devs, self.delegator_max_size)

        for i in range(0, len(query_samples), S):
            j = min(i + S, len(query_samples))  # Calculate end position so we can tell the core how many elems to read
            size = j - i

            with self.sample_buff.write_context(size * 2) as buf:
                buf[:size] = [q.index for q in query_samples[i:j]]
                buf[size:size*2] = [q.id for q in query_samples[i:j]]

    def issue_queries(self, query_samples):
        """Issues queries to the cores for processing.

        Args:
            query_samples: List of query samples to process
        """
        logging.info("issue_queries called with %d query samples", len(query_samples))
        logging.info("Sending queries to cores")
        t = threading.Thread(target=self.delegate_samples,
                             args=(query_samples,),
                             daemon=True)
        t.start()
        self.handle_responses(len(query_samples))
        t.join()

    def __del__(self):
        self.shutdown()

    def flush_queries(self):
        """Flushes all pending queries and stops processing."""
        logging.info("flush_queries called")
        self.signal_stop()

        self._cache.clear()
