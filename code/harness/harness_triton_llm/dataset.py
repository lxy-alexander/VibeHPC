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
import numpy as np
import logging
from typing import Optional


class LlmDataset():
    def __init__(self,
                 path_input_ids: str,
                 path_input_lens: str,
                 path_stop_ids: Optional[str] = None,
                 dtype: str = "int32"):
        self._tensor_input_ids = np.load(path_input_ids).astype(dtype)
        self._tensor_input_len = np.load(path_input_lens).astype(dtype).reshape([-1, 1])
        if path_stop_ids is None:
            self._tensor_stop_ids = None
        else:
            self._tensor_stop_ids = np.load(path_stop_ids).astype(dtype)
        assert self._tensor_input_ids.shape[0] == self._tensor_input_len.shape[0]
        self._dataset_size = self._tensor_input_ids.shape[0]

    def get_input(self, sample_idx: int):
        end_idx = sample_idx + 1
        x, y = self._tensor_input_ids[sample_idx:end_idx].reshape(1, -1), self._tensor_input_len[sample_idx:end_idx].reshape(1, 1)
        if self._tensor_stop_ids is None:
            z = None
        else:
            z = self._tensor_stop_ids[sample_idx:end_idx].reshape(1, -1)
        return x, y, z

    def get_size(self) -> int:
        return self._dataset_size

    def load_samples_to_ram(self, sample_list):
        pass

    def unload_samples_from_ram(self, sample_list):
        pass

    def __del__(self):
        logging.info("Finished destroying LlmDataset")
