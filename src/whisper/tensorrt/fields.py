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

__doc__ = """Whisper Fields"""


whisper_encoder_build_flags = Field(
    "whisper_encoder_build_flags",
    description="decoder trtllm-build flags for Whisper",
    from_string=dict
)

whisper_decoder_build_flags = Field(
    "whisper_decoder_build_flags",
    description="decoder trtllm-build flags for Whisper",
    from_string=dict
)
