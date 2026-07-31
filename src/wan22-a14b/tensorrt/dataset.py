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
import torch
from pathlib import Path
from typing import Optional

from code.common.mlcommons.runner import ScopedQSL
from code.common import logging
from code.fields import harness as harness_fields
from nvmitten.configurator import autoconfigure, bind

from . import fields as wan22_fields
from .constants import DEFAULT_NEGATIVE_PROMPT
from .utils import load_prompts_from_file


@autoconfigure
@bind(harness_fields.tensor_path)
@bind(wan22_fields.prompt_file, "prompt_file")
@bind(wan22_fields.negative_prompt, "negative_prompt")
class Wan22Dataset(ScopedQSL):
    """
    Dataset for Wan2.2 T2V pipeline.
    Loads prompts from a file with one prompt per line.
    """

    def __init__(self,
                 *args,
                 tensor_path: os.PathLike = Path.cwd(),
                 prompt_file: str = "prompts.txt",
                 negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.tensor_path = Path(tensor_path)
        assert self.tensor_path.exists(), f"Dataset path {self.tensor_path} does not exist"

        # Load prompts from file
        prompt_path = self.tensor_path / prompt_file
        if not prompt_path.exists():
            # Try as absolute path
            prompt_path = Path(prompt_file)

        self.prompts = load_prompts_from_file(prompt_path)

        self.negative_prompt = negative_prompt
        # Ensure we have enough prompts for the dataset
        if len(self.prompts) < self.total_sample_count:
            logging.warning(
                f"Only {len(self.prompts)} prompts available, but {self.total_sample_count} "
                f"samples requested. Will cycle through prompts."
            )
            # Extend prompts by cycling
            while len(self.prompts) < self.total_sample_count:
                self.prompts.extend(self.prompts[:min(len(self.prompts),
                                                      self.total_sample_count - len(self.prompts))])

        self.prompts = self.prompts[:self.total_sample_count]
        logging.info(f"Done loading prompt. Total sample count {self.total_sample_count}")

        self.fixed_latent_path = self.tensor_path / "fixed_latent.pt"
        self.fixed_latent: Optional[torch.Tensor] = None

        if self.fixed_latent_path.exists():
            self.fixed_latent = torch.load(self.fixed_latent_path)
            logging.info(f"Loaded fixed latent from {self.fixed_latent_path} with shape {self.fixed_latent.shape}")
        else:
            logging.warning(f"Fixed latent file not found at {self.fixed_latent_path}")

        logging.info(f"Loaded {len(self.prompts)} prompts for Wan2.2 T2V dataset")
        logging.info(f"Negative prompt: {self.negative_prompt[:100]}...")

    def get_prompt(self, sample_idx: int) -> str:
        """Get prompt for a given sample index."""
        return self.prompts[sample_idx % len(self.prompts)]

    def get_negative_prompt(self) -> str:
        """Get the negative prompt."""
        return self.negative_prompt

    def get_fixed_latents(self) -> Optional[torch.Tensor]:
        """Get the fixed latent tensor for reproducible generation."""
        return self.fixed_latent

    def load_query_samples(self, sample_list):
        """Load query samples (no-op for text prompts)."""
        pass

    def unload_query_samples(self, sample_list):
        """Unload query samples (no-op for text prompts)."""
        pass
