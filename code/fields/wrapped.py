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
"""
This module provides wrapper functionality for TensorRT builder configurations and field bindings.

It sets up automatic configuration bindings for various TensorRT builder parameters and provides
utilities for creating auto-configured dataclasses with field bindings.
"""
from code.common.systems.system_list import DETECTED_SYSTEM
import code.fields.gen_engines as builder_fields
import code.fields.general as general_fields
import code.fields.models as model_fields
import code.fields.harness as harness_fields
import dataclasses as dcls
import nvmitten.nvidia.builder as MBuilder
import typing

from nvmitten.configurator import autoconfigure, bind


# Set up bindings for nvmitten classes
_builder_bindings = [bind(model_fields.precision),
                     bind(model_fields.input_dtype),
                     bind(model_fields.input_format),
                     bind(builder_fields.workspace_size),
                     # This is taken from the original fields.py - is this still true? This has not been changed since
                     # it was implemented, even after TRT was updated.
                     # Prior to TRT 8.6+, one optimization profile can only be linked to one set of bindings. In most
                     # harness cases, there is one binding per copy stream. Set num_profiles = gpu_copy_streams.
                     bind(harness_fields.gpu_copy_streams, "num_profiles"),
                     bind(general_fields.verbose),
                     bind(general_fields.verbose_nvtx)]
if "tags" in DETECTED_SYSTEM.extras and "is_soc" in DETECTED_SYSTEM.extras["tags"]:
    _builder_bindings.append(bind(model_fields.dla_core))

for binding in _builder_bindings:
    binding(MBuilder.TRTBuilder)


# Do not bind force_calibration, as we want that kwarg to only be controlled by CalibrateEngineOp.
for binding in [bind(builder_fields.force_calibration),
                bind(builder_fields.calib_batch_size),
                bind(builder_fields.calib_max_batches),
                bind(builder_fields.calib_data_map),
                bind(builder_fields.cache_file)]:
    binding(MBuilder.CalibratableTensorRTEngine)


TRTBuilder = autoconfigure(MBuilder.TRTBuilder)
"""
Auto-configured TensorRT builder class with predefined field bindings.

This class extends the base TensorRT builder with automatic configuration bindings for:
- Precision settings
- Input data type and format
- Workspace size
- GPU copy streams (used for optimization profiles)
- Verbosity settings
- DLA core configuration
"""

CalibratableTensorRTEngine = autoconfigure(MBuilder.CalibratableTensorRTEngine)
"""
Auto-configured TensorRT engine class with calibration support.

This class extends the base TensorRT engine with automatic configuration bindings for:
- Calibration batch size
- Maximum calibration batches
- Calibration data mapping
- Cache file configuration

Note: force_calibration is intentionally not bound to allow control by CalibrateEngineOp.
"""


def make_autoconf_dcls(name: str,
                       *fields,
                       **kwargs):
    """
    Creates an auto-configured dataclass with field bindings.

    This function creates a dataclass with the specified fields and automatically
    configures field bindings for each field. The resulting class can be used with
    the nvmitten configurator system.

    Args:
        name (str): The name of the dataclass to create.
        *fields: Variable number of field objects to include in the dataclass.
        **kwargs: Additional keyword arguments to pass to dataclasses.make_dataclass.

    Returns:
        Type: An auto-configured dataclass with the specified fields and bindings.

    Example:
        >>> MyClass = make_autoconf_dcls("MyClass", field1, field2)
        >>> instance = MyClass()
        >>> # Fields will be automatically configured based on the bindings
    """
    _c = dcls.make_dataclass(name,
                             [(field.name,
                               field.from_string if type(field.from_string) is type else typing.Any,
                               dcls.field(default=None))
                              for field in fields],
                             **kwargs)
    for field in fields:
        bind(field)(_c)
    return autoconfigure(_c)
