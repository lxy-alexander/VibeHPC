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
from nvmitten.configurator import Field

import pathlib


__doc__ = """Triton configuration

Used to configure Triton Inference Server settings for harness runs.
"""


use_triton = Field(
    "use_triton",
    description="Use Triton harness",
    from_string=bool)

triton_verbose_frontend = Field(
    "triton_verbose_frontend",
    description="Enable verbose logging from triton frontend",
    from_string=bool)

triton_num_clients_per_frontend = Field(
    "triton_num_clients_per_frontend",
    description="Number of gRPC clients per frontend",
    from_string=int)

triton_num_frontends_per_model = Field(
    "triton_num_frontends_per_model",
    description="Number of frontend processes per GPU model",
    from_string=int)

triton_num_servers = Field(
    "triton_num_servers",
    description="Number of tritonserver instances",
    from_string=int)

triton_skip_server_spawn = Field(
    "triton_skip_server_spawn",
    description="Skip spawning the tritonserver instance (and use default ports)",
    from_string=bool)

triton_grpc_ports: Field = Field(
    "triton_grpc_ports",
    description="The hosts and grpc ports to run tritonserver procs on. Eg: node_0:port_0,port_1|node_1:port_0,port_1")

preferred_batch_size = Field(
    "preferred_batch_size",
    description="Preferred batch sizes")

max_queue_delay_usec = Field(
    "max_queue_delay_usec",
    description="Set max queuing delay for Triton in usec.",
    from_string=int)

instance_group_count = Field(
    "instance_group_count",
    description="Set number of instance groups on each GPU.",
    from_string=int)

request_timeout_usec = Field(
    "request_timeout_usec",
    description="Set the timeout for every request in usec.",
    from_string=int)

buffer_manager_thread_count = Field(
    "buffer_manager_thread_count",
    description="The number of threads used to accelerate copies and other operations required to manage input and output tensor contents.",
    from_string=int)

gather_kernel_buffer_threshold = Field(
    "gather_kernel_buffer_threshold",
    description="Set the threshold number of buffers for triton to use gather kernel to gather input data. 0 disables the gather kernel",
    from_string=int)

batch_triton_requests = Field(
    "batch_triton_requests",
    description="Send a batch of query samples to triton instead of single query at a time",
    from_string=bool)

output_pinned_memory = Field(
    "output_pinned_memory",
    description="Use pinned memory when data transfer for output is between device mem and non-pinned sys mem",
    from_string=bool)

use_concurrent_harness = Field(
    "use_concurrent_harness",
    description="Use multiple threads for batching and triton issue while using the triton harness",
    from_string=bool)

num_concurrent_batchers = Field(
    "num_concurrent_batchers",
    description="Number of threads that will batch samples to form triton requests. Only used when the concurrent triton harness is used",
    from_string=int)

num_concurrent_issuers = Field(
    "num_concurrent_issuers",
    description="Number of threads that will issue requests to triton. Only used when the concurrent triton harness is used.",
    from_string=int)

dla_num_batchers = Field(
    "dla_num_batchers",
    description="Number of threads that will batch samples to form triton requests. Only used when the concurrent DLA triton harness is used",
    from_string=int)

dla_num_issuers = Field(
    "dla_num_issuers",
    description="Number of threads that will issue requests to triton. Only used when the concurrent DLA triton harness is used.",
    from_string=int)
