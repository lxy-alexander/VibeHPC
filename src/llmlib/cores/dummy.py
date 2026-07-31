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

"""
Dummy LLM Core Implementation for Testing

This module provides a mock LLM core that immediately returns fixed responses
without any actual inference. Useful for:
- Testing the harness infrastructure
- Debugging loadgen integration
- Accuracy checker integration testing
- Performance baseline measurements (overhead testing)
"""

from typing import List, Dict, Any, Callable, Optional
import datetime

from ..config import HarnessConfig
from ..utils import LLMServerProgressDisplay
from .base import LLMCore, LLMRequest, LLMResponse


# Hardcoded 100-token response for dummy inference bypass mode
# This allows testing accuracy checker integration without actual inference
DUMMY_OUTPUT_TOKENS = list(range(100))


class DummyCore(LLMCore):
    """Dummy backend for testing harness functionality

    This core implementation:
    - Accepts any requests and tracks them
    - Immediately returns a fixed 100-token sequence [0, 1, 2, ..., 99]
    - Does not perform any actual inference
    - Always uses a single core regardless of available resources

    Use this to test accuracy checker integration or harness infrastructure
    without waiting for slow inference.
    """
    CONFIG_T = HarnessConfig

    def __init__(self, **kwargs):
        self.num_requests_received = 0
        super().__init__(**kwargs)
        # Initialize response thread as required by base class
        self._initialize_response_thread()

    def _enqueue_impl(self, queries: List[LLMRequest]) -> List[int]:
        """Accept requests and track them without processing

        Simply records the request IDs and counts for testing purposes.
        """
        num_new_requests = len(queries)
        self.num_requests_received += num_new_requests
        # Return same request_ids (no translation needed for dummy)
        return [q.request_id for q in queries]

    def _poll_responses_impl(self, timeout: Optional[datetime.timedelta] = None):
        """Return fixed responses for all pending requests

        Immediately returns a hardcoded 100-token sequence for each pending request.
        This simulates instant inference with zero latency.

        Args:
            timeout: Ignored - responses are returned immediately
        """
        # Create responses for all pending requests
        responses = []
        with self.pending_samples_lock:
            for executor_id, (request_id, enqueue_time) in list(self.pending_samples.items()):
                responses.append(LLMResponse(
                    request_id=executor_id,  # Use executor_id temporarily
                    output_tokens=[DUMMY_OUTPUT_TOKENS.copy()],  # Fixed 100-token response
                    is_final_token=True,
                    error=None,
                ))

        return responses

    def run_health_check(self):
        """No-op health check - dummy core is always healthy"""
        pass

    @classmethod
    def get_num_cores_for_workload(cls, **kwargs) -> int:
        """Always return 1 core for dummy backend

        The dummy core doesn't use any actual compute resources,
        so we only need one instance regardless of available GPUs.
        """
        return 1

    @classmethod
    def get_config_for_core(cls,
                            core_index: int,
                            progress_display: LLMServerProgressDisplay,
                            verbose: bool,
                            verbose_nvtx: bool,
                            complete_callback: Callable,
                            **kwargs) -> Dict[str, Any]:
        """Get dummy configuration

        Returns minimal configuration since dummy core doesn't need
        any special setup or resource allocation.
        """
        # Create config using class's CONFIG_T (same pattern as other cores)
        config = cls.CONFIG_T(**kwargs)

        return {
            # Common kwargs
            'name': f'DummyCore#{core_index}',
            'harness_config': config,
            'progress_display': progress_display,
            'verbose': verbose,
            'verbose_nvtx': verbose_nvtx,
            'complete_callback': complete_callback,
        }
