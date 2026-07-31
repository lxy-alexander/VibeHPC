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

from code.llmlib import TrtllmServeClientHarnessOp, CoreType, TrtllmHLApiClientHarnessOp, DummyHarnessOp
from code.llmlib.launch_server import RunTrtllmServeOp  # noqa: F401
from code.llmlib.builder import LLMComponentEngine, HFQuantizerOp  # noqa: F401
from .builder import GPT_OSS_120BQuantizerConfig as QuantizerConfig  # noqa: F401
from .constants import GPT_OSS_120BComponent as Component
from .dataset import GptOssDataset as DataLoader  # noqa: F401

COMPONENT_MAP = {
    Component.GPT_OSS_120B: None,
}
VALID_COMPONENT_SETS = {"gpu": [{Component.GPT_OSS_120B}]}
DEFAULT_CORE_TYPE = CoreType.TRTLLM_ENDPOINT
HF_MODEL_REPO = {"openai/gpt-oss-120b": 'main'}

ComponentEngine = LLMComponentEngine
TrtllmServeBenchmarkHarnessOp = TrtllmServeClientHarnessOp
TrtllmHLApiBenchmarkHarnessOp = TrtllmHLApiClientHarnessOp
DummyBenchmarkHarnessOp = DummyHarnessOp

