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


# TODO: Should this file even exist? Can all of these files get moved to other places?
__doc__ = """Default fields

[DEPRECATION WARNING] General purpose fields. These will likely be relocated to other places.
"""


verbose = Field(
    "verbose",
    description="Enable verbose output",
    from_string=bool)

verbose_nvtx = Field(
    "verbose_nvtx",
    description="Enable verbose output for NVTX",
    from_string=bool)

verbose_nvsmi = Field(
    "verbose_nvsmi",
    description="Enable nvidia-smi monitoring with 200ms polling interval during harness execution",
    from_string=bool)

data_dir = Field(
    "data_dir",
    description="Directory for raw data from MLCommons instructions.",
    from_string=pathlib.Path,
    from_environ="DATA_DIR",
    disallow_argparse=True)

preprocessed_data_dir = Field(
    "preprocessed_data_dir",
    description="Directory for preprocessed data.",
    from_environ="PREPROCESSED_DATA_DIR",
    from_string=pathlib.Path,
    disallow_argparse=True)

log_dir = Field(
    "log_dir",
    description="Directory for all output logs.",
    from_string=pathlib.Path,
    from_environ="LOG_DIR")

results_dir = Field(
    "results_dir",
    description="Directory for submission results",
    from_string=pathlib.Path,
    from_environ="ARTIFACTS_DIR")

# TODO: Should this be in gen_engines?
engine_dir = Field(
    "engine_dir",
    description="Set the engine directory to load from or serialize to",
    from_string=pathlib.Path)

show_help = Field(
    "help",
    description="Show help message and exit",
    from_string=bool)

system_name = Field(
    "system_name",
    description="The name of the system to use for the run. If not set, will use the system detection logic.",
    from_environ="SYSTEM_NAME")

config_dir = Field(
    "config_dir",
    description="The directory containing the config files for the run.",
    from_string=pathlib.Path,
    from_environ="CONFIG_DIR")
