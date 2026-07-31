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

from typing import Type, Dict, List, Union
import importlib

from code.fields.harness import CoreType


class BackendRegistry:
    """Registry for LLM backend implementations"""
    _backends: Dict[str, Type['LLMCore']] = {}

    CORE_MODULES = {
        CoreType.DUMMY: ('dummy', 'DummyCore'),
        CoreType.TRTLLM_EXECUTOR: ('trtllm_executor_core', 'TrtllmExecutorCore'),
        CoreType.TRITON_GRPC: ('triton_grpc_core', 'TritonGrpcCore'),
        CoreType.TRTLLM_ENDPOINT: ('trtllm_endpoint_core', 'TrtllmEndpointCore'),
        CoreType.TRTLLM_DISAGG: ('trtllm_disagg_endpoint_core', 'TrtllmDisaggEndpointCore'),
        CoreType.TRTLLM_HLAPI: ('trtllm_hlapi_core', 'TrtllmHlApiCore'),
        CoreType.DYNAMO_ENDPOINT: ('dynamo_endpoint_core', 'DynamoEndpointCore'),
    }

    @classmethod
    def get(cls, name: Union[str, CoreType]) -> Type['LLMCore']:
        """Get a backend class by name or core_type """
        core_type = CoreType.from_string(name) if isinstance(name, str) else name
        assert core_type in cls.CORE_MODULES, f"Unknown core type: {core_type}"

        module_name, core_class = cls.CORE_MODULES[core_type]
        module = importlib.import_module(f'.{module_name}', package='code.llmlib.cores')
        return module.__dict__[core_class]

    @classmethod
    def list_backends(cls) -> List[str]:
        """List all registered backend names"""
        return map(str, list(cls.CORE_MODULES.keys()))


# Import and export core implementations
from .base import LLMCore, LLMRequest, LLMResponse

# Import concrete implementation that doesn't depend on trtllm
from .dummy import DummyCore

__all__ = [
    'BackendRegistry',
    'LLMCore',
    'LLMRequest',
    'LLMResponse',
    'DummyCore',
]
