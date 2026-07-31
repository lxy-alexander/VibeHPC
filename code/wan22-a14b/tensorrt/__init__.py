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

"""WAN2.2 T2V benchmark module for MLPerf Inference."""

__version__ = "1.0.0"
__all__ = ["Wan22HarnessOp", "BenchmarkHarnessOp"]

from .harness import Wan22HarnessOp

COMPONENT_MAP = {}

VALID_COMPONENT_SETS = {"gpu": [set()]}  # WAN22 uses a unified pipeline

# Export the harness operation for the benchmark framework
BenchmarkHarnessOp = Wan22HarnessOp

# Disable calibration and engine building ops since WAN22 uses PyTorch/VisionFly
CalibrateEngineOp = None
EngineBuilderOp = None
