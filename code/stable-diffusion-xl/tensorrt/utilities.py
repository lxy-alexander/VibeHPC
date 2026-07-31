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
import torch
import numpy as np
import nvtx
import subprocess as sp

from PIL import Image


# map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,

    bool: torch.bool,
    np.bool_: torch.bool
}


class EmbeddingDims:
    PROMPT_LEN = 77
    CLIP = 768
    CLIP_PROJ = 1280
    UNETXL = 2048

# TODO: double check HF


class PipelineConfig:
    GUIDANCE = 8
    STEPS = 20
    VAE_SCALING_FACTOR = 0.13025
    IMAGE_SIZE = 1024
    LATENT_SIZE = 128


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for _, x in enumerate(memory_free_info)]
    return memory_free_values


def torch_to_image(tensor: torch.tensor):
    image = (tensor * 255).clamp(0, 255).detach().permute(1, 2, 0).round().type(torch.uint8).cpu().numpy()
    return Image.fromarray(image)


def calculate_max_engine_device_memory(engine_dict):
    max_device_memory = 0
    for engine in engine_dict.values():
        max_device_memory = max(max_device_memory, engine.engine.device_memory_size)
    return max_device_memory
