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
"""Utilities for parsing and interpreting batch sizes, as the MLPerf Inference pipeline can have different batch size
configurations depending on device type and model component.
"""

from dataclasses import dataclass
from typing import Optional

from code.common.utils import get_e2e_batch_size
import code.fields.models as model_fields

from nvmitten.configurator import bind, autoconfigure


@autoconfigure
@bind(model_fields.gpu_batch_size)
@bind(model_fields.dla_batch_size)
@dataclass
class GeneralizedBatchSize:
    """A class for managing batch sizes across different components and devices.

    This class provides a unified interface for accessing and managing batch sizes
    for different model components (e.g., encoder, decoder) across different devices
    (e.g., GPU, DLA). It supports both GPU and DLA batch size configurations.

    Attributes:
        gpu_batch_size (Optional[ModelBatchSize]): Batch size configuration for GPU components
        dla_batch_size (Optional[ModelBatchSize]): Batch size configuration for DLA components
    """
    gpu_batch_size: Optional[model_fields.ModelBatchSize] = None
    dla_batch_size: Optional[model_fields.ModelBatchSize] = None

    def _dev(self, dev: str = "gpu"):
        return getattr(self, f"{dev}_batch_size")

    def get(self, component: str, dev: str = "gpu"):
        if self._dev(dev=dev) is None:
            return None
        return self._dev(dev=dev)[component]

    def iter_of(self, dev: str = "gpu"):
        return self._dev(dev=dev).items()

    def e2e(self, dev: str = "gpu"):
        if self._dev(dev=dev) is None:
            return None
        return get_e2e_batch_size(self._dev(dev=dev))

    def is_empty(self) -> bool:
        return self.gpu_batch_size is None and self.dla_batch_size is None
