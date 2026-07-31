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

__doc__ = """SDXL Fields"""


use_native_instance_norm = Field(
    "use_native_instance_norm",
    description="Use native instance normalization for SDXL",
    from_string=bool
)

batcher_time_limit = Field(
    "sdxl_batcher_time_limit",
    description="SDXL batcher time limit in seconds",
    from_string=float
)
