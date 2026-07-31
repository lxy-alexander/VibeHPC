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
from code.common.triton.base_harness import TritonLlmHarness


class TritonLlama3_1_8BHarness(TritonLlmHarness):
    def _setup_triton_model_repo(self, flag_dict):
        beam_width = 1
        decoupled = self.scenario.valstr.lower() == "server"
        super()._setup_triton_model_repo(flag_dict, beam_width=beam_width, decoupled=decoupled)
