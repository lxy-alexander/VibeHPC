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


from code.llmlib import TrtllmExecutorClientHarnessOp, TritonClientHarnessOp, TrtllmServeClientHarnessOp, TrtllmHLApiClientHarnessOp, CoreType
from code.llmlib.launch_server import GenerateTritonConfigOp, RunTritonServerOp, RunTrtllmServeOp  # noqa: F401
from code.llmlib.builder import LLMComponentEngine, TRTLLMBuilderOp
from .builder import Mixtral8x7bQuantizerOp
from .constants import Mixtral8x7BComponent as Component
from .dataset import MixtralDataset as DataLoader  # noqa: F401


COMPONENT_MAP = {
    Component.Mixtral8x7B: None,
}
VALID_COMPONENT_SETS = {"gpu": [{Component.Mixtral8x7B}]}
DEFAULT_CORE_TYPE = CoreType.TRTLLM_EXECUTOR

ComponentEngine = LLMComponentEngine
CalibrateEngineOp = Mixtral8x7bQuantizerOp
EngineBuilderOp = TRTLLMBuilderOp
TrtllmExecutorBenchmarkHarnessOp = TrtllmExecutorClientHarnessOp
TritonBenchmarkHarnessOp = TritonClientHarnessOp
