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

"""WAN22 Fields"""

from nvmitten.configurator import Field

prompt_file = Field(
    "wan22_prompt_file",
    description="Path to prompts file for WAN22 T2V",
    from_string=str
)

negative_prompt = Field(
    "wan22_negative_prompt",
    description="Negative prompt for WAN22 T2V generation",
    from_string=str
)

model_path = Field(
    "wan22_model_path",
    description="Path to WAN22 model",
    from_string=str
)

num_frames = Field(
    "wan22_num_frames",
    description="Number of frames to generate",
    from_string=int
)

height = Field(
    "wan22_height",
    description="Video height",
    from_string=int
)

width = Field(
    "wan22_width",
    description="Video width",
    from_string=int
)

num_inference_steps = Field(
    "wan22_num_inference_steps",
    description="Number of inference steps",
    from_string=int
)

guidance_scale = Field(
    "wan22_guidance_scale",
    description="Guidance scale for generation",
    from_string=float
)

guidance_scale_2 = Field(
    "wan22_guidance_scale_2",
    description="Guidance scale for second transformer (Wan2.2 only)",
    from_string=float
)

attn_type = Field(
    "wan22_attn_type",
    description="Attention type (default, sage-attn, etc.)",
    from_string=str
)

linear_type = Field(
    "wan22_linear_type",
    description="Linear type for quantization",
    from_string=str
)

# Multi-device parallelism fields
ulysses_size = Field(
    "wan22_ulysses_size",
    description="Ulysses (sequence) parallelism size for multi-GPU",
    from_string=int
)

cfg_size = Field(
    "wan22_cfg_size",
    description="CFG (classifier-free guidance) parallelism size",
    from_string=int
)

# Context Parallelism fields
cp_size = Field(
    "wan22_cp_size",
    description="Context parallelism size - splits sequence across GPUs",
    from_string=int
)

ring_size = Field(
    "wan22_ring_size",
    description="Ring attention parallelism size for memory-efficient long sequences",
    from_string=int
)

# Tensor Parallelism fields
tp_size = Field(
    "wan22_tp_size",
    description="Tensor parallelism size - splits model weights across GPUs",
    from_string=int
)

# Data Parallelism fields
dp_size = Field(
    "wan22_dp_size",
    description="Data parallelism size for batch distribution",
    from_string=int
)

# FSDP fields
fsdp_size = Field(
    "wan22_fsdp_size",
    description="Fully Sharded Data Parallel size for DiT",
    from_string=int
)

t5_fsdp_size = Field(
    "wan22_t5_fsdp_size",
    description="Fully Sharded Data Parallel size for T5 text encoder",
    from_string=int
)

# VAE parallelism
disable_parallel_vae = Field(
    "wan22_disable_parallel_vae",
    description="Disable parallel VAE encoding/decoding",
    from_string=lambda x: x.lower() in ('true', '1', 'yes')
)

parallel_vae_split_dim = Field(
    "wan22_parallel_vae_split_dim",
    description="Dimension to split for parallel VAE ('height' or 'width')",
    from_string=str
)

enable_visual_gen_cpu_offload = Field(
    "wan22_enable_visual_gen_cpu_offload",
    description="Enable visual_gen CPU offload for large models",
    from_string=lambda x: x.lower() in ('true', '1', 'yes')
)

cpu_offload_stride = Field(
    "wan22_cpu_offload_stride",
    description="Stride for block-wise CPU offloading",
    from_string=int
)

# Server/client scale-out fields
server_urls = Field(
    "wan22_server_urls",
    description="Comma-separated list of WAN22 server URLs",
    from_string=lambda x: x.split(",") if x else []
)

gpus_per_server = Field(
    "wan22_gpus_per_server",
    description="Number of GPUs per WAN22 server instance",
    from_string=int
)

num_servers = Field(
    "wan22_num_servers",
    description="Number of WAN22 server instances to launch",
    from_string=int
)

use_server_client = Field(
    "wan22_use_server_client",
    description="Use server/client architecture for scale-out",
    from_string=lambda x: x.lower() in ('true', '1', 'yes')
)

# GPU selection
gpu_ids = Field(
    "wan22_gpu_ids",
    description="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3' or '4,5,6,7')",
    from_string=lambda x: x if x else None
)

# Warmup iterations
warmup_iters = Field(
    "wan22_warmup_iters",
    description="Number of warmup iterations before inference (default: 4)",
    from_string=int
)

# Total sample count override for accuracy testing
total_sample_count = Field(
    "wan22_total_sample_count",
    description="Override total sample count for QSL (default: 248 for full accuracy run)",
    from_string=int
)
