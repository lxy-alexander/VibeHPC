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

from __future__ import annotations
import os
import signal
import time
from typing import Callable, List, Optional

import numpy as np
import psutil

from code.common import logging
from code.common.utils import nvtx_scope
from code.common.workload import Workload

from .config import HarnessConfig
from .cores import LLMCore, LLMRequest
from .utils import LLMServerProgressDisplay, LatencyTracker, PrefixLogger
from .warmup import WarmupManager


def setup_interrupt_handler(server: Optional[LLMServer] = None):
    current_process = psutil.Process()

    def exit_fn(signum, frame):
        logging.info("Received SIGINT. Stop LLMServer and cleanup.")

        # Clean up server cores if available
        if server and hasattr(server, 'cores'):
            try:
                for core in server.cores:
                    core.notify_stop()
            except Exception as e:
                logging.error(f"Error during core cleanup: {e}")

        # Kill child processes
        children = current_process.children(recursive=True)
        for child in children:
            logging.debug(f"Sending SIGKILL to child process: {child.pid}")
            os.kill(child.pid, signal.SIGKILL)

    signal.signal(signal.SIGINT, exit_fn)


class LLMServer:
    """Minimal LLM server focused purely on query orchestration"""

    def __init__(
        self,
        cores: List[LLMCore],
        scheduler: Callable[[], LLMCore],
        progress_display: LLMServerProgressDisplay,
        harness_config: HarnessConfig,
        workload: Workload,
        warmup_manager: WarmupManager,
        complete_callback: Callable,
        verbose: bool = False,
        verbose_nvtx: bool = False
    ):
        """
        Initialize server with pre-configured components from factory

        Args:
            cores: List of LLMCore instances
            scheduler: Function to get next core for query
            progress_display: Shared progress display
            harness_config: Harness configuration
            workload: MLPerf workload
            warmup_manager: Manager for parallel warmup and health checks
            complete_callback: Callback function for completed requests
            verbose: Verbose logging flag
            verbose_nvtx: NVTX instrumentation flag
        """
        self.cores = cores
        self.get_next_core = scheduler
        self.progress_display = progress_display
        self.harness_config = harness_config
        self.wl = workload
        self.warmup_manager = warmup_manager
        self.complete_callback = complete_callback
        self.verbose = verbose
        self.verbose_nvtx = verbose_nvtx
        self.sample_count = 0
        self.logger = PrefixLogger(prefix=f"LLMServer-{os.getpid()}")

        if self.harness_config.enable_ttft_latency_tracker and self.harness_config.gen_config.streaming:
            self.latency_tracker = LatencyTracker()
            for c in cores:
                self.logger.info(f"tracking latency for core {c.name}")
                c.latency_tracker = self.latency_tracker

            self.max_concurrency_per_core = 0
            self.max_concurrency_core = None
        else:
            self.latency_tracker = None
            self.max_concurrency_per_core = -1
            self.max_concurrency_core = None

        setup_interrupt_handler(self)
        self.logger.info(f"LLMServer initialized with {len(self.cores)} cores")

        self.warmup_manager.run_health_checks_with_retry(self.cores)

    def warm_up(self, warmup_iters: Optional[int] = None):
        """
        Run warm-up iterations on all cores using WarmupManager.

        Args:
            warmup_iters: Number of warmup queries to generate
        """
        if warmup_iters is None:
            # some cores (eg: triton with multiple clients) may require extended warmups
            warmup_iters = max([core.get_num_warmup_iters() for core in self.cores])

        # Create warmup queries with random tokens (ids=[1, 100), input-len=[90, 300))
        warmup_queries = [
            LLMRequest(
                request_id=i,
                input_tokens=np.random.randint(1, 100, size=np.random.randint(90, 300)).tolist(),
                stop_tokens=None
            )
            for i in range(warmup_iters)
        ]

        with nvtx_scope("warm_up"):
            self.warmup_manager.warmup(self.cores, warmup_queries)

    def issue_queries(self, query_samples: List[LLMRequest]):
        """
        Issue queries to backend cores

        Args:
            query_samples: List of LLMRequest objects
        """
        for query in query_samples:
            core = self.get_next_core()
            self.sample_count += core.enqueue([query])
        
        # Update progress display
        self.progress_display.update_total(total=self.sample_count)

        # DEBUG-INFO: track runtime max concurrency for server scenario
        if self.verbose and self.harness_config.gen_config.streaming:
            queue_sizes = {core.name: core.get_num_pending_samples() for core in self.cores}
            self.logger.debug(f"Issued +{len(query_samples)} samples (ID0: {query_samples[0].request_id}). Core Load: {queue_sizes}")

            max_core_name, max_core_concurrency = max(queue_sizes.items(), key=lambda x: x[1])
            if max_core_concurrency > self.max_concurrency_per_core:
                self.max_concurrency_core = max_core_name
                self.max_concurrency_per_core = max_core_concurrency

    def flush_queries(self):
        """Block until all pending queries complete"""
        self.logger.debug("flush_queries() invoked.")
        with nvtx_scope("flush_queries"):
            for core in self.cores:
                core.flush()
        self.logger.debug("flush_queries() completed.")

    def stop_work(self):
        """Stop accepting new requests and cleanup"""
        self.logger.debug("stop_work() invoked.")
        with nvtx_scope("stop_work"):
            # Signal cores to stop
            for core in self.cores:
                core.notify_stop()

            # Wait for pending queries
            self.flush_queries()

            # Cleanup
            self.progress_display.finish()
            self.cores.clear()

        if self.latency_tracker is not None:
            self.latency_tracker.gen_csv(self.wl.log_dir / "latency.csv")

        self.logger.info(f"Total Samples Completed: {self.sample_count}")
        self.logger.debug(f"Core {self.max_concurrency_core} reached highest instantaneous concurrency of: {self.max_concurrency_per_core}")
        self.logger.debug("stop_work() completed.")
