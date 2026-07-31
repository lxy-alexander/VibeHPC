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
TensorRT-LLM Executor Core Implementation
"""

from __future__ import annotations
import datetime
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from code.common.systems.system_list import DETECTED_SYSTEM
from code.common.utils import nvtx_scope
from nvmitten.nvidia.accelerator import GPU
import tensorrt_llm
import tensorrt_llm.bindings.executor as trtllm

from ..config import TrtllmExecutorConfig, TrtllmHarnessConfig
from ..utils import LLMServerProgressDisplay
from .base import LLMCore, LLMRequest, LLMResponse


class TrtllmExecutorCore(LLMCore):
    """
    TrtllmExecutorCore is a wrapper for TRTLLM Executor, can own multiple gpus (orchestrator of its own domain).
    Sets up trtllm executor by loading trtllm engine.
    """
    CONFIG_T = TrtllmExecutorConfig
    EXECUTOR_WORKER_PATH = Path(tensorrt_llm.__file__).parent / 'bin' / 'executorWorker'

    def __init__(self, device_ids: List[int], **kwargs):
        """Initialize TRT-LLM executor with engine and device configuration

        Args:
            device_ids: List of GPU device IDs this executor will use
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(**kwargs)
        self.device_ids = device_ids
        self.engine_config = self.harness_config.engine_config

        # Add kvcache utilization tracking to progress display
        self.progress_display.add_additional_unit('%kvcache_util', 'value')

        scheduler_policy = {
            'max_util': trtllm.CapacitySchedulerPolicy.MAX_UTILIZATION,
            'no_evict': trtllm.CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
            'static': trtllm.CapacitySchedulerPolicy.STATIC_BATCH,
        }[self.harness_config.runtime_flags['batch_scheduler_policy']]

        context_chunking_policy = {
            'equal_progress': trtllm.ContextChunkingPolicy.EQUAL_PROGRESS,
            'first_come_first_served': trtllm.ContextChunkingPolicy.FIRST_COME_FIRST_SERVED,
        }[self.harness_config.runtime_flags['context_chunking_policy']]

        batching_type = {
            True: trtllm.BatchingType.INFLIGHT,
            False: trtllm.BatchingType.STATIC,
        }[self.harness_config.runtime_flags['use_inflight_batching']]

        executor_config = trtllm.ExecutorConfig(
            max_beam_width=self.harness_config.gen_config.runtime_beam_width,
            max_batch_size=int(self.harness_config.runtime_flags['max_batch_size']),
            max_num_tokens=int(self.harness_config.runtime_flags['max_num_tokens']),
            max_queue_size=-1,
            scheduler_config=trtllm.SchedulerConfig(
                capacity_scheduler_policy=scheduler_policy,
                context_chunking_policy=context_chunking_policy,
                dynamic_batch_config=trtllm.DynamicBatchConfig(
                    enable_batch_size_tuning=self.harness_config.runtime_flags['enable_batch_size_tuning'],
                    enable_max_num_tokens_tuning=self.harness_config.runtime_flags['enable_max_num_tokens_tuning'],
                    dynamic_batch_moving_average_window=self.harness_config.runtime_flags['dynamic_batch_moving_average_window']
                )
            ),
            kv_cache_config=trtllm.KvCacheConfig(
                enable_block_reuse=False,  # Block reuse not allowed in MLPerf
                free_gpu_memory_fraction=self.harness_config.runtime_flags['kvcache_free_gpu_mem_frac'],
            ),
            enable_chunked_context=self.harness_config.runtime_flags['enable_chunked_context'],
            batching_type=batching_type,
            parallel_config=trtllm.ParallelConfig(
                communication_type=trtllm.CommunicationType.MPI,
                communication_mode=trtllm.CommunicationMode.ORCHESTRATOR,
                device_ids=self.device_ids,
                orchestrator_config=trtllm.OrchestratorConfig(
                    is_orchestrator=True,
                    worker_executable_path=str(TrtllmExecutorCore.EXECUTOR_WORKER_PATH),
                )
            )
        )

        self.logger.info(f"Loading TensorRT-LLM engine: {self.engine_config.engine_dir}.")
        with nvtx_scope(f"{self.name}::trtllm_executor_init"):
            self.executor = trtllm.Executor(
                model_path=self.engine_config.engine_dir,
                model_type=trtllm.ModelType.DECODER_ONLY,
                executor_config=executor_config
            )
        self.logger.info(f"Executor Using Devices: #{self.device_ids}.")
        assert self.executor.can_enqueue_requests(), "Executor failed to initialize"

        self.output_config = trtllm.OutputConfig(
            exclude_input_from_output=self.harness_config.runtime_flags['exclude_input_from_output'],
        )

        self.sampling_config = trtllm.SamplingConfig(
            beam_width=self.harness_config.gen_config.runtime_beam_width,
            temperature=self.harness_config.gen_config.temperature,
            min_tokens=self.harness_config.gen_config.min_output_len,
            top_k=self.harness_config.gen_config.top_k,
            top_p=self.harness_config.gen_config.top_p,
            seed=self.harness_config.random_seed,
        )

        # start response completion thread after init
        self._initialize_response_thread()

    def _enqueue_impl(self, queries: List[LLMRequest]) -> List[int]:
        """
        Enqueue input samples to TRT-LLM Executor.

        Converts LLMRequest objects into TRT-LLM Request format and
        submits them to the executor. The executor handles batching
        and scheduling internally.

        Args:
            queries: List of LLMRequest objects containing request details

        Returns:
            List of request IDs assigned by the executor
        """
        assert not self.stop_work.is_set(), "Cannot issue queries after stop_work has been signalled to core"

        enqueue_batch = [
            trtllm.Request(input_token_ids=query.input_tokens,
                           max_tokens=self.harness_config.gen_config.max_output_len,
                           streaming=self.harness_config.gen_config.streaming,
                           sampling_config=self.sampling_config,
                           output_config=self.output_config,
                           end_id=self.harness_config.gen_config.eos_token_id,
                           stop_words=query.stop_tokens)
            for query in queries
        ]
        # Submit to executor and get assigned IDs
        trtllm_request_ids = self.executor.enqueue_requests(enqueue_batch)

        return trtllm_request_ids

    def _poll_responses_impl(self, timeout: datetime.timedelta):
        """Poll executor for completed responses

        The executor manages request scheduling and batching internally.
        This method retrieves any completed responses within the timeout.
        """
        timeout = 0.0001 if timeout is None else timeout  # timeout=None will actually block
        ready_responses = self.executor.await_responses(timeout)

        responses = [
            LLMResponse(
                request_id=response.request_id,
                output_tokens=response.result.output_token_ids,
                is_final_token=response.result.is_final,
                error=None,
            )
            for response in ready_responses
        ]

        return responses

    def _update_progress_display(self, num_completed, num_toks, ttfts, tpots):
        """Update progress display with executor-specific metrics

        In addition to standard metrics, this tracks:
        - KV cache utilization percentage
        - iteration stats from the executor
        """
        additional_unit_updates = {}

        # Get and record detailed iteration statistics
        if self.executor.can_enqueue_requests() and (stats := self.executor.get_latest_iteration_stats()):
            stats = [json.loads(stat.to_json_str()) for stat in stats]
            self.progress_display.record_iteration_stats(stats)

            # Extract KV cache utilization from latest iteration
            latest = stats[-1]
            kvcache_util = 100 * (float(latest["kvCacheStats"]["usedNumBlocks"]) / float(latest["kvCacheStats"]["maxNumBlocks"]))
            additional_unit_updates |= {'%kvcache_util': kvcache_util}

        super()._update_progress_display(num_completed, num_toks, ttfts, tpots, additional_unit_updates)

    def _cleanup_resources(self):
        """Shutdown the executor and release GPU resources"""
        self.executor.shutdown()
        super()._cleanup_resources()

    def run_health_check(self):
        """Simple health check for TensorRT-LLM executor"""
        assert self.executor.can_enqueue_requests(), "Executor not ready"

    @classmethod
    def get_num_cores_for_workload(cls, **kwargs) -> int:
        """Calculate cores based on GPU count and model parallelism"""
        harness_config = TrtllmHarnessConfig()
        model_world_size = harness_config.get_instance_size()
        devices = [gpu.gpu_index for gpu in DETECTED_SYSTEM.accelerators[GPU]]
        return len(devices) // model_world_size

    @classmethod
    def get_config_for_core(cls,
                            core_index: int,
                            progress_display: LLMServerProgressDisplay,
                            verbose: bool,
                            verbose_nvtx: bool,
                            complete_callback: Callable,
                            engine_dir: str,
                            **kwargs) -> Dict[str, Any]:
        """Get configuration for a core instance """
        config = TrtllmExecutorConfig(engine_dir=engine_dir, **kwargs)

        # device assignment
        devices = [gpu.gpu_index for gpu in DETECTED_SYSTEM.accelerators[GPU]]
        model_world_size = config.get_instance_size()  # only single node support by executor core
        start_idx = core_index * model_world_size
        end_idx = start_idx + model_world_size
        device_ids = devices[start_idx:end_idx]

        return {
            'name': f'TrtllmExecutorCore#{core_index}_devices-{device_ids}',
            'harness_config': config,
            'progress_display': progress_display,
            'verbose': verbose,
            'verbose_nvtx': verbose_nvtx,
            'complete_callback': complete_callback,
            'device_ids': device_ids,
        }
