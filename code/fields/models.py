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
from collections import UserDict
from typing import Union, Dict

from nvmitten.configurator import Field
from nvmitten.configurator.fields import AutoConfStrategy
from nvmitten.constants import Precision, InputFormats

import pathlib
import re
import logging


__doc__ = """Model flags

Used to control settings for benchmark models.
"""


class ModelBatchSize(UserDict):
    """
    A dictionary-like class for handling component-specific batch sizes.

    This class parses and stores batch sizes for different model components in the format:
    'component1:bs1,component2:bs2,...'

    Attributes:
        _str_pattern (str): Regular expression pattern for validating batch size strings.
    """

    _str_pattern = r'^.+:\d+(?:,.+:\d+)*$'  # required format is component1:component_bs1,component2:component_bs2,...

    @classmethod
    def from_string(cls, s: str):
        """
        Parse a batch size string into a ModelBatchSize object.

        Args:
            s (str): Batch size string in the format 'component1:bs1,component2:bs2,...'

        Returns:
            ModelBatchSize: A dictionary-like object containing component batch sizes.

        Raises:
            ValueError: If the input string does not match the required format.
        """
        if not re.fullmatch(cls._str_pattern, s):
            logging.error(f"Batch size {s} is not in the format component1:bs1,component2:bs2,...")
            raise ValueError(f"Invalid batch size format.")
        bs_pairs = [pair.split(':') for pair in s.split(',')]
        return ModelBatchSize({comp: int(val) for comp, val in bs_pairs})


def parse_precision(s: str) -> Union[Dict[str, Precision], Precision]:
    """
    Parse a precision string into a dictionary of component-specific precisions or a single precision.

    Args:
        s (str): Precision string for either a single precision or a dictionary of component-specific precisions
                 in the format 'component1:precision1,component2:precision2,...'

    Returns:
        Union[Dict[str, Precision], Precision]: A dictionary of component-specific precisions or a single precision.
    """
    if ':' not in s:
        return Precision.get_match(s)
    else:
        kv_pairs = [pair.split(':') for pair in s.split(',')]
        return {comp: Precision.get_match(val) for comp, val in kv_pairs}


# Batch size fields
gpu_batch_size = Field(
    "gpu_batch_size",
    description="Batch size(s) to use for models on GPU",
    from_string=ModelBatchSize.from_string,
    autoconf_strategy=AutoConfStrategy.DictUpdate)


dla_batch_size = Field(
    "dla_batch_size",
    description="Batch size(s) to use for models on DLA",
    from_string=ModelBatchSize.from_string,
    autoconf_strategy=AutoConfStrategy.DictUpdate)

# Precision and input configuration fields
precision = Field(
    "precision",
    description="Lowest allowed precision to use for network weights",
    from_string=parse_precision,
    autoconf_strategy=AutoConfStrategy.DictUpdate)

use_fp8 = Field(
    "use_fp8",
    description="Enable FP8 for the network where possible, otherwise use the original precision",
    from_string=bool)

input_dtype = Field(
    "input_dtype",
    description="Precision for input tensors.",
    from_string=Precision.get_match)

input_format = Field(
    "input_format",
    description="Format/layout for input tensors.",
    from_string=InputFormats.get_match)

# Model path and device configuration fields
model_path = Field(
    "model_path",
    description="Path to the model weights.",
    from_string=pathlib.Path)

dla_core = Field(
    "dla_core",
    description="DLA Core ID to use.",
    from_string=int)

energy_aware_kernels = Field(
    "energy_aware_kernels",
    description="Override layers with energy aware kernel selection",
    from_string=bool)
