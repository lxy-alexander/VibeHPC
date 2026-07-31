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
from code.llmlib.launch_server import GenerateTritonConfigOp, RunTritonServerOp, RunTrtllmServeOp, GenerateTrtllmDisaggConfigOp, RunTrtllmServeDisaggOp  # noqa: F401 # noqa: F401
from code.llmlib.builder import LLMComponentEngine, TRTLLMBuilderOp, TRTLLMQuantizerOp, HFQuantizerOp  # noqa: F401
from .builder import LLAMA2QuantizerConfig as QuantizerConfig  # noqa: F401
from .constants import LLAMA2Component as Component
from .dataset import LlamaDataset as DataLoader  # noqa: F401

COMPONENT_MAP = {
    Component.LLAMA2: None,
}
VALID_COMPONENT_SETS = {"gpu": [{Component.LLAMA2}]}
DEFAULT_CORE_TYPE = CoreType.TRTLLM_EXECUTOR
HF_MODEL_REPO = {"meta-llama/Llama-2-70b-chat-hf": 'e9149a12809580e8602995856f8098ce973d1080'}

ComponentEngine = LLMComponentEngine
CalibrateEngineOp = TRTLLMQuantizerOp
EngineBuilderOp = TRTLLMBuilderOp
TrtllmExecutorBenchmarkHarnessOp = TrtllmExecutorClientHarnessOp
TritonBenchmarkHarnessOp = TritonClientHarnessOp
TrtllmServeBenchmarkHarnessOp = TrtllmServeClientHarnessOp
TrtllmHLApiBenchmarkHarnessOp = TrtllmHLApiClientHarnessOp
TrtllmDisaggServeBenchmarkHarnessOp = TrtllmServeClientHarnessOp
