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

from pprint import pformat
from typing import Callable, List

import numpy as np
import os

from code.common.workload import Workload
from code.fields.harness import CoreType
import mlperf_loadgen as lg

from .config import HarnessConfig
from .cores import BackendRegistry, LLMCore
from .server import LLMServer
from .utils import LLMServerProgressDisplay, prefix_logger as logging
from .warmup import WarmupManager


# NOTE(vir|ryan): WAR
#
# Non-Dynamo:
# ---------------------------
# FIRST | FINAL | Loadgen API
# ---------------------------
#   T   |   T   | QuerySamplesComplete
#   T   |   F   | FirstTokenComplete
#   F   |   T   | QuerySamplesComplete
#   F   |   F   | N/A
# ---------------------------
#
# Dynamo:
# ---------------------------
# FIRST | FINAL | Loadgen API
# ---------------------------
#   T   |   T   | FirstTokenComplete + QuerySamplesComplete
#   T   |   F   | FirstTokenComplete
#   F   |   T   | QuerySamplesComplete
#   F   |   F   | N/A
# ---------------------------
DYNAMO_OVERRIDE_EMPTY_TOKEN_RESPONSE = os.getenv('MLPINF_USE_DYNAMO', '0') == '1'


def complete_loadgen_request(
    request_id: int,
    is_first_token: bool,
    output_toks: np.ndarray,
    output_toks_len: int,
    is_final_token: bool
):
    """ Complete a Loadgen LLM request with first/final generated tokens. """

    if is_first_token:
        # original path: skip FirstTokenResponse (when also is-final)
        # OR
        # override via MLPINF_USE_DYNAMO
        if DYNAMO_OVERRIDE_EMPTY_TOKEN_RESPONSE or \
                (is_first_token and not is_final_token):
            lg.FirstTokenComplete([lg.QuerySampleResponse(
                request_id,
                output_toks.ctypes.data,
                output_toks.nbytes,
                output_toks_len
            )])

    if is_final_token:
        lg.QuerySamplesComplete([lg.QuerySampleResponse(
            request_id,
            output_toks.ctypes.data,
            output_toks.nbytes,
            output_toks_len
        )])


class LLMServerFactory:
    """Factory for creating and configuring all LLM server components"""

    @staticmethod
    def create_server(
        backend_type: CoreType,
        workload: Workload,
        disable_progress_display: bool = False,
        verbose: bool = False,
        verbose_nvtx: bool = False,
        **backend_kwargs
    ) -> LLMServer:
        """Create complete LLMServer with all components"""

        # Get backend class
        backend_class = BackendRegistry.get(backend_type)

        # Create base config class to initialize other components
        base_config = HarnessConfig()

        # Create progress display with backend-specific metrics
        progress_display = LLMServerFactory._create_progress_display(workload,
                                                                     base_config,
                                                                     disable_progress_display)

        # Create cores using backend's static methods
        num_cores = backend_class.get_num_cores_for_workload(**backend_kwargs)

        # Initialize cores sequentially
        cores = []
        for core_index in range(num_cores):
            core_kwargs = backend_class.get_config_for_core(
                core_index=core_index,
                progress_display=progress_display,
                verbose=verbose,
                verbose_nvtx=verbose_nvtx,
                complete_callback=complete_loadgen_request,
                **backend_kwargs
            )
            core = backend_class(**core_kwargs)
            cores.append(core)

            if core_index == 0:
                logging.info(f"Initialized {core.name} with HarnessConfig:\n{pformat(core.harness_config, compact=True)}")
            else:
                logging.info(f"Initialized {core.name}.")

        # Create scheduler based on traffic distribution policy
        scheduler = LLMServerFactory._create_scheduler(cores,
                                                       base_config.traffic_distribution_policy)

        logging.info(f"Factory created {len(cores)} {str(backend_type)} cores")

        # Create WarmupManager for parallel warmup
        warmup_manager = WarmupManager(readiness_timeout=base_config.readiness_timeout)

        # Create and return configured LLMServer
        return LLMServer(
            cores=cores,
            scheduler=scheduler,
            progress_display=progress_display,
            harness_config=base_config,
            workload=workload,
            warmup_manager=warmup_manager,
            complete_callback=complete_loadgen_request,
            verbose=verbose,
            verbose_nvtx=verbose_nvtx
        )

    @staticmethod
    def _create_progress_display(
        workload: Workload,
        harness_config: HarnessConfig,
        disable: bool,
    ) -> LLMServerProgressDisplay:
        """Create progress display with appropriate metrics"""
        # Base metrics
        additional_units = {'tokens/s': 'mean'}

        # Streaming metrics
        if harness_config.gen_config.streaming:
            additional_units |= {'TTFT(s)': '99%', 'TPOT(ms)': '99%'}

        if harness_config.show_steady_state_progress:
            additional_units |= {'steady_state_tokens/s': 'throughput_tracker'}

        return LLMServerProgressDisplay(
            total=0,
            enable_render=not disable,
            additional_units=additional_units,
            log_dir=workload.log_dir
        )

    @staticmethod
    def _create_scheduler(cores: List[LLMCore], policy: str) -> Callable[[], LLMCore]:
        """Create scheduler function (same as current LLMServer.reset_scheduler)"""
        state = {'round_robin_index': -1}

        def round_robin() -> LLMCore:
            state['round_robin_index'] = (state['round_robin_index'] + 1) % len(cores)
            return cores[state['round_robin_index']]

        def load_balancing() -> LLMCore:
            queue_sizes = {i: core.get_num_pending_samples() for i, core in enumerate(cores)}
            min_index = min(queue_sizes, key=queue_sizes.get)
            return cores[min_index]

        def isl_load_balancing() -> LLMCore:
            isl_sums = {i: core.get_isl_sum() for i, core in enumerate(cores)}
            min_index = min(isl_sums, key=isl_sums.get)
            return cores[min_index]

        schedulers = {
            'round_robin': round_robin,
            'load_balancing': load_balancing,
            'isl_load_balancing': isl_load_balancing,
        }

        return schedulers[policy]
