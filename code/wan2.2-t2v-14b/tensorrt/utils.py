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

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import mlperf_loadgen as lg

from code.common import logging


def find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def setup_distributed_env(rank: int, world_size: int, master_port: int):
    """Setup distributed environment variables for a worker process."""
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)



def load_prompts_from_file(prompt_file: Path) -> List[str]:
    """
    Load prompts from a text file, one prompt per line.

    Args:
        prompt_file: Path to the prompt file

    Returns:
        List of prompts
    """
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    logging.info(f"Loaded {len(prompts)} prompts from {prompt_file}")
    return prompts


# =============================================================================
# Video Output Utilities
# =============================================================================

def video_to_numpy(frames: Union[list, torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert frames to numpy array."""
    if isinstance(frames, list):
        return np.stack([np.array(f) for f in frames], axis=0)
    elif isinstance(frames, torch.Tensor):
        return frames.cpu().numpy()
    elif isinstance(frames, np.ndarray):
        return frames
    return np.array(frames)


def video_to_loadgen_response(sample_id: int, video_data: np.ndarray) -> lg.QuerySampleResponse:
    """Convert video data to LoadGen response."""
    video_data = np.ascontiguousarray(video_data)
    return lg.QuerySampleResponse(
        sample_id,
        video_data.__array_interface__["data"][0],
        video_data.nbytes,
    )


def create_placeholder_video(num_frames: int, height: int, width: int) -> np.ndarray:
    """Create placeholder video data for error cases."""
    return np.zeros((num_frames, height, width, 3), dtype=np.uint8)
