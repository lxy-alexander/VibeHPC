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


from .constants import SDXLComponent as Component
from .builder import SDXLCLIPBuilder, SDXLCLIPWithProjBuilder, SDXLUNetXLBuilder, SDXLVAEBuilder
from .harness import SDXLHarnessOp

COMPONENT_MAP = {
    Component.CLIP1: SDXLCLIPBuilder,
    Component.CLIP2: SDXLCLIPWithProjBuilder,
    Component.UNETXL: SDXLUNetXLBuilder,
    Component.VAE: SDXLVAEBuilder,
}

VALID_COMPONENT_SETS = {"gpu": [{Component.CLIP1, Component.CLIP2, Component.UNETXL, Component.VAE}]}

BenchmarkHarnessOp = SDXLHarnessOp