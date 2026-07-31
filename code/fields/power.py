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


__doc__ = """Power usage settings

Used to control power limits and system power usage.
"""


cpu_freq = Field(
    "cpu_freq",
    description="CPU frequency to set the system to.",
    from_string=int)

power_limit = Field(
    "power_limit",
    description="Upper power limit to set the system to in watts.",
    from_string=int)

soc_gpu_freq = Field(
    "soc_gpu_freq",
    description="Frequency to set the SoC iGPU to.",
    from_string=int)

soc_dla_freq = Field(
    "soc_dla_freq",
    description="Frequency to set the SoC DLA(s) to.",
    from_string=int)

soc_cpu_freq = Field(
    "soc_cpu_freq",
    description="Frequency to set the SoC CPU to.",
    from_string=int)

soc_emc_freq = Field(
    "soc_emc_freq",
    description="Frequency to set the SoC EMC to.",
    from_string=int)

soc_pva_freq = Field(
    "soc_pva_freq",
    description="Frequency to set the SoC PVA to.",
    from_string=int)

orin_num_cores = Field(
    "orin_num_cores",
    description="Number of CPU cores on Orin to have powered. Extra cores will be powered off.",
    from_string=int)

orin_skip_maxq_reset = Field(
    "orin_skip_maxq_reset",
    description="If set, will skip resetting board state via nvpmodel -m 0.",
    from_string=bool)
