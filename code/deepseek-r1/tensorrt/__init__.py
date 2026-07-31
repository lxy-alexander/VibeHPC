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

from code.llmlib import TrtllmServeClientHarnessOp, CoreType, TrtllmHLApiClientHarnessOp, DynamoEndpointHarnessOp
from code.llmlib.launch_server import RunTrtllmServeOp  # noqa: F401
from code.llmlib.builder import LLMComponentEngine, HFQuantizerOp  # noqa: F401
from .builder import DeepSeek_R1QuantizerConfig as QuantizerConfig  # noqa: F401
from .constants import DeepSeek_R1Component as Component
from .dataset import DeepseekDataset as DataLoader  # noqa: F401

COMPONENT_MAP = {
    Component.DeepSeek_R1: None,
}
VALID_COMPONENT_SETS = {"gpu": [{Component.DeepSeek_R1}]}
DEFAULT_CORE_TYPE = CoreType.TRTLLM_ENDPOINT
HF_MODEL_REPO = {"deepseek-ai/deepseek-r1": '56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad'}

ComponentEngine = LLMComponentEngine
TrtllmServeBenchmarkHarnessOp = TrtllmServeClientHarnessOp
TrtllmHLApiBenchmarkHarnessOp = TrtllmHLApiClientHarnessOp
DynamoEndpointBenchmarkHarnessOp = DynamoEndpointHarnessOp
