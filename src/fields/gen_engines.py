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
from nvmitten.configurator import Field

import pathlib


__doc__ = """TensorRT engine builder fields

Contains fields to forward to TensorRT builder and network config options.
"""


workspace_size = Field(
    "workspace_size",
    description="Max size (bytes) allocated for each layer by the TRT engine builder.",
    from_string=int)

no_child_process = Field(
    "no_child_process",
    description=("Generate engines in main process instead of child processes. "
                 "Useful when running profilers or gdb when debugging issues with "
                 "generate_engines."),
    from_string=bool)

active_sms = Field(
    "active_sms",
    description="Percentage of active SMs while generating engines.",
    from_string=int)

force_build_engines = Field(
    "force_build_engines",
    description="Force rebuild of TRT engines, even if they exist.",
    from_string=bool)

force_calibration = Field(
    "force_calibration",
    description="Always run quantization calibration, even if the cache exists.",
    from_string=bool)

calib_batch_size = Field(
    "calib_batch_size",
    description="Batch size to use when calibrating",
    from_string=int)

calib_max_batches = Field(
    "calib_max_batches",
    description="Number of batches to run for calibration.",
    from_string=int)

cache_file = Field(
    "cache_file",
    description="Path to calibration cache.",
    from_string=pathlib.Path)

calib_data_map = Field(
    "calib_data_map",
    description="Path to the data map of the calibration set.",
    from_string=pathlib.Path)

calib_data_dir = Field(
    "calib_data_dir",
    description="Path to the dataset of the calibration set.",
    from_string=pathlib.Path)

strongly_typed = Field(
    "strongly_typed",
    description="Use strongly typed mode for TRT engine builder.",
    from_string=bool)

skip_graphsurgeon = Field(
    "skip_graphsurgeon",
    description="If set, will skip the graphsurgeon step that processes the model before the TensorRT builder runs.",
    from_string=bool)
