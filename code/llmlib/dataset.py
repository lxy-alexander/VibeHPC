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

import os
from pathlib import Path
from typing import List, Union

from code.common.mlcommons.runner import ScopedQSL
from code.fields import harness as harness_fields
from code.fields import loadgen as lg_fields
from code.common.utils import load_tensor
from .utils import prefix_logger as logging

from nvmitten.configurator import autoconfigure, bind

import numpy as np
import torch


@autoconfigure
@bind(harness_fields.tensor_path, "dataset_path")
@bind(lg_fields.test_mode)
class LLMDataLoader(ScopedQSL):
    def __init__(self,
                 fnames: List[os.PathLike],
                 *args,
                 dataset_path: os.PathLike = Path.cwd(),
                 test_mode: str = "PerformanceOnly",
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset_path = Path(dataset_path)
        assert self.dataset_path.exists(), f"Dataset path {self.dataset_path} does not exist"

        self.tensors = {}
        for fname in fnames:
            name = Path(fname).stem
            self.tensors[name] = load_tensor(self.dataset_path / fname)

    def __getattr__(self, name: str):
        if name in self.tensors:
            return self.tensors[name]
        raise AttributeError(f"Attribute {name} not found in {self.__class__.__name__}")

    def get_input_tokens(self, sample_indices: List[int]) -> List[List[int]]:
        """
        Get the input tokens for the given sample indices.

        Args:
            sample_indices (List[int]): The list of sample indices to retrieve tokens for.

        Returns:
            List[List[int]]: The input tokens as list of lists.
        """
        raise NotImplementedError

    def get_stop_tokens(self, sample_indices: List[int]) -> List[List[int]]:
        """
        Get the stop tokens for the given sample indices.

        Args:
            sample_indices (List[int]): The list of sample indices to retrieve tokens for.

        Returns:
            List[List[int]]: The stop tokens as list of lists.
        """
        return [None for _ in sample_indices]

    def __len__(self) -> int:
        """Get size of Dataset"""
        raise NotImplementedError
