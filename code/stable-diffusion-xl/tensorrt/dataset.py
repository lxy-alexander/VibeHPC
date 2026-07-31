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

from code.common.mlcommons.runner import ScopedQSL
from code.common.utils import load_tensor
from code.fields import harness as harness_fields

from nvmitten.configurator import autoconfigure, bind
import torch


@autoconfigure
@bind(harness_fields.tensor_path)
class Dataset(ScopedQSL):
    def __init__(self, *args, tensor_path: os.PathLike = Path.cwd(), **kwargs):
        super().__init__(*args, **kwargs)

        self.tensor_path = Path(tensor_path)
        assert self.tensor_path.exists(), f"Dataset path {self.tensor_path} does not exist"

        self.prompt_tokens_clip1 = load_tensor(self.tensor_path / "prompt_ids_clip1_padded_5k.npy", pin_memory=True)
        self.prompt_tokens_clip2 = load_tensor(self.tensor_path / "prompt_ids_clip2_padded_5k.npy", pin_memory=True)
        self.negative_prompt_tokens_clip1 = load_tensor(self.tensor_path / "negative_prompt_ids_clip1_padded_5k.npy", pin_memory=True)
        self.negative_prompt_tokens_clip2 = load_tensor(self.tensor_path / "negative_prompt_ids_clip2_padded_5k.npy", pin_memory=True)
        self.init_noise_latent = torch.load(self.tensor_path / "latents.pt")

        self.caption_count = self.prompt_tokens_clip1.shape[0]
        assert self.caption_count == self.total_sample_count, "Caption count does not match total sample count"

    def load_query_samples(self, sample_list):
        pass

    def unload_query_samples(self, sample_list):
        pass
