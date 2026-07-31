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

"""TensorRT-LLM high-level API core implementation."""

import asyncio
from copy import deepcopy
import datetime
from pathlib import Path
from pprint import pformat
import queue
import threading
import time
from typing import Any, Dict, List, Callable


from tensorrt_llm import SamplingParams
from tensorrt_llm.bench.benchmark.utils.general import get_settings
from tensorrt_llm.bench.dataclasses.configuration import RuntimeConfig
from tensorrt_llm.llmapi import LLM, CapacitySchedulerPolicy

from ..config import TrtllmHlApiConfig
from ..utils import LLMServerProgressDisplay
from .base import LLMCore, LLMRequest, LLMResponse


# Create a single event loop for the entire module
_module_loop = asyncio.new_event_loop()
_loop_thread = threading.Thread(target=_module_loop.run_forever, daemon=True)
_loop_thread.start()


class TrtllmHlApiCore(LLMCore):
    """LLMCore implementation using TensorRT-LLM high-level API"""
    CONFIG_T = TrtllmHlApiConfig

    def __init__(self, **kwargs):
        """Initialize the TRT-LLM high-level API core."""
        super().__init__(**kwargs)
        assert self.harness_config.gen_config.runtime_beam_width <= 1, "Beam > 1 not supported yet"

        # Initialize model paths
        self.model_repo = self.harness_config.get_model_repo()
        self.model_name, self.model_revision = list(self.model_repo.items())[0]
        self.model_path = Path(self.harness_config.model_path)
        assert Path(self.harness_config.model_path).exists(), f"{self.harness_config.model_path} does not exist"

        # Write extra config to log-dir
        self.extra_config_path = Path(self.harness_config.log_dir) / "trtllm_hlapi_extra_conf.yaml"
        with self.extra_config_path.open('w') as f:
            f.write(self.harness_config.extra_config_yaml)

        # Prepare optimization parameters
        params = {
            "backend": self.harness_config.runtime_flags['trtllm_backend'],
            "extra_llm_api_options": str(self.extra_config_path),
            "beam_width": self.harness_config.gen_config.runtime_beam_width,
            "tp": self.harness_config.tensor_parallelism,
            "pp": self.harness_config.pipeline_parallelism,
            "ep": self.harness_config.moe_expert_parallelism,
            "max_batch_size": self.harness_config.runtime_flags['max_batch_size'],
            "max_seq_len": self.harness_config.build_flags['max_seq_len'],
            "max_num_tokens": self.harness_config.build_flags['max_num_tokens'],
            "chunking": self.harness_config.runtime_flags['enable_chunked_context'],
            "kv_cache_percent": self.harness_config.runtime_flags['kvcache_free_gpu_mem_frac'],
            "kv_cache_reuse": False,
        }

        # get trtllm LLMApi kwargs
        exec_settings = get_settings(params, None, self.model_name, self.model_path)
        exec_settings["model"] = str(self.model_path)
        # exec_settings["settings_config"]["dynamic_max_batch_size"] = False

        scheduler_policy = {
            'MAX_UTILIZATION': CapacitySchedulerPolicy.MAX_UTILIZATION,
            'GUARANTEED_NO_EVICT': CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
            'STATIC_BATCH': CapacitySchedulerPolicy.STATIC_BATCH,
        }[self.harness_config.runtime_flags['batch_scheduler_policy']]
        exec_settings['settings_config']['scheduler_policy'] = scheduler_policy

        kwargs = RuntimeConfig(**exec_settings).get_llm_args()
        self.logger.info(f"TensorRT-LLM initialization parameters:\n{pformat(kwargs, compact=False)}")

        if self.harness_config.runtime_flags['trtllm_backend'] == 'pytorch':
            # TODO(vir):
            # cleanup some cpp backend flags
            # RuntimeConfig ret-val now requires override?
            kwargs.pop("extended_runtime_perf_knob_config")

            self.llm = LLM(**kwargs)
        else:
            # requires external engine build
            raise NotImplementedError("trt backend not enabled yet")
            self.llm = LLM(**kwargs)

        # Setup sampling parameters
        self.sampling_params = SamplingParams(
            temperature=self.harness_config.gen_config.temperature,
            min_tokens=self.harness_config.gen_config.min_output_len,
            max_tokens=self.harness_config.gen_config.max_output_len,
            top_k=self.harness_config.gen_config.top_k,
            top_p=self.harness_config.gen_config.top_p,
            seed=self.harness_config.random_seed,
            n=self.harness_config.gen_config.runtime_beam_width,
            end_id=self.harness_config.gen_config.eos_token_id,
            pad_id=self.harness_config.gen_config.eos_token_id,  # Use eos_token_id as pad_id
        )

        # Setup async infrastructure
        self._loop = _module_loop
        self._response_queue = queue.Queue()
        self._async_tasks = {}

        # start response completion thread after init
        self._initialize_response_thread()

    def _enqueue_impl(self, queries: List[LLMRequest]) -> List[int]:
        """Enqueue queries using the high-level API's generate_async method

        Args:
            queries: List of LLMRequest objects to process

        Returns:
            List of executor request IDs
        """
        executor_ids = []

        for query in queries:
            # Use request_id as executor_id
            executor_ids.append(query.request_id)

            # Create async task
            task = asyncio.run_coroutine_threadsafe(
                self._process_request_async(query),
                self._loop
            )
            self._async_tasks[query.request_id] = task

        return executor_ids

    async def _process_request_async(self, request: LLMRequest):
        """Process single request asynchronously"""
        try:
            # Create a copy of sampling params for this request
            sampling_params = deepcopy(self.sampling_params)

            # Set stop token IDs for this request if needed
            if request.stop_tokens and self.harness_config.gen_config.use_stop_tokens:
                sampling_params.stop_token_ids = request.stop_tokens

            # Use input tokens directly for the high-level API
            input_tokens = request.input_tokens

            # Submit request
            if self.harness_config.gen_config.streaming:
                # Handle streaming mode
                accumulated_tokens = []
                first_token_sent = False
                sent_token_count = 0  # Track how many tokens have been sent so far

                async for output in self.llm.generate_async(
                    input_tokens,
                    sampling_params,
                    streaming=True,
                ):
                    # Get the current generated tokens
                    current_tokens = output.outputs[0].token_ids
                    new_tokens = current_tokens[len(accumulated_tokens):]

                    if new_tokens:
                        accumulated_tokens = current_tokens

                    # Check if this is the final iteration
                    is_final = output.outputs[0].finish_reason is not None

                    # Send response for every iteration - base class needs this for accumulation
                    if not first_token_sent:
                        # First token response: send only new tokens
                        first_token_sent = True
                        sent_token_count = len(new_tokens) if new_tokens else 0
                        self._response_queue.put(LLMResponse(
                            request_id=request.request_id,
                            output_tokens=[new_tokens] if new_tokens else [[]],
                            is_final_token=is_final,
                            error=None
                        ))
                    elif is_final:
                        # Final response: send only remaining tokens (excluding already sent tokens)
                        remaining_tokens = accumulated_tokens[sent_token_count:]
                        self._response_queue.put(LLMResponse(
                            request_id=request.request_id,
                            output_tokens=[remaining_tokens],
                            is_final_token=True,
                            error=None
                        ))
            else:
                # Handle non-streaming mode
                output = await self.llm.generate_async(
                    input_tokens,
                    sampling_params,
                    streaming=False,
                )

                # Extract output tokens directly
                output_tokens = output.outputs[0].token_ids

                self._response_queue.put(LLMResponse(
                    request_id=request.request_id,
                    output_tokens=[output_tokens],
                    is_final_token=True,
                    error=None
                ))

        except Exception as e:
            self.logger.error(f"Error processing request {request.request_id}: {e}")
            self._response_queue.put(LLMResponse(
                request_id=request.request_id,
                output_tokens=[],
                is_final_token=True,
                error=e
            ))
        finally:
            # Clean up task reference
            if request.request_id in self._async_tasks:
                del self._async_tasks[request.request_id]

    def _poll_responses_impl(self, timeout: datetime.timedelta):
        """Poll responses from the response queue"""
        end_time = time.time() + timeout.total_seconds() if timeout is not None else 0
        responses = []

        # Get all available responses without blocking
        try:
            while True:
                responses.append(self._response_queue.get_nowait())
        except queue.Empty:
            pass

        remaining_time = end_time - time.time()
        if remaining_time <= 0:
            return responses

        # Block for remaining time to see if responses come in
        try:
            responses.append(self._response_queue.get(timeout=remaining_time))
            while True:
                responses.append(self._response_queue.get_nowait())
        except queue.Empty:
            pass

        return responses

    def _cleanup_resources(self):
        """Free GPU resources on core cleanup"""
        if self.llm is not None:
            try:
                self.llm.shutdown()
            except:
                pass

        super()._cleanup_resources()

    def run_health_check(self):
        """Health check for TRT-LLM high-level API"""
        try:
            # Try a simple generation to verify the model is loaded
            test_params = SamplingParams(
                max_tokens=5,
                temperature=0.0,
                end_id=self.harness_config.gen_config.eos_token_id,
                pad_id=self.harness_config.gen_config.eos_token_id,
            )

            # Use simple test tokens for the high-level API
            test_output = self.llm.generate(
                [1, 2, 3, 4, 5],  # Simple test tokens
                test_params
            )

            # Verify we got output
            assert len(test_output.outputs) > 0
            assert len(test_output.outputs[0].token_ids) > 0
        except Exception as e:
            raise RuntimeError(f"Health check failed: {e}")

    @classmethod
    def get_num_cores_for_workload(cls, **kwargs) -> int:
        """Calculate number of cores for workload"""
        # TODO(vir): LLMApi does not support device-ids, can't do DP
        return 1

    @classmethod
    def get_config_for_core(cls,
                            core_index: int,
                            progress_display: LLMServerProgressDisplay,
                            verbose: bool,
                            verbose_nvtx: bool,
                            complete_callback: Callable,
                            model_path: str = None,
                            **kwargs) -> Dict[str, Any]:
        """Get configuration for a core instance """
        config = TrtllmHlApiConfig(**kwargs)
        config.model_path = model_path

        return {
            'name': f'TrtllmHlApiCore#{core_index}',
            'harness_config': config,
            'progress_display': progress_display,
            'verbose': verbose,
            'verbose_nvtx': verbose_nvtx,
            'complete_callback': complete_callback,
        }
