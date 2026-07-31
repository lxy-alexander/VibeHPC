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

import threading
import time
from typing import List

from .cores import LLMCore, LLMRequest
from .utils import prefix_logger as logging


class WarmupManager:
    """Stateless manager for LLM core warmup and health checks.

    on warmup():
    - run health check untill all cores are ready
    - run warmup queries in parallel across all cores
    """

    def __init__(self, readiness_timeout: int = 300):
        """Initialize WarmupManager.

        Args:
            readiness_timeout: Maximum seconds to wait for all cores to be healthy
        """
        self.readiness_timeout = readiness_timeout

    def run_health_checks_with_retry(self, cores: List[LLMCore]):
        """Run health checks on all cores in parallel with exponential backoff retry.

        Args:
            cores: List of LLMCore instances to check
        """
        logging.info("Running health checks on all cores...")
        
        # Store exceptions from threads
        exceptions = []
        exceptions_lock = threading.Lock()

        def check_core(core: LLMCore):
            """Health check for a single core with retry logic."""
            try:
                start_time = time.time()
                poll_interval = 5
                exp_backoff = 1.3

                while time.time() - start_time < self.readiness_timeout:
                    try:
                        core.run_health_check()
                        logging.info(f"Core {core.name} is healthy.")
                        return  # Success for this core
                    except Exception as e:
                        logging.warning(f"Health check for {core.name} failed: {e}. Retrying in {poll_interval:.1f}s...")
                        time.sleep(poll_interval)
                        poll_interval *= exp_backoff

                raise TimeoutError(f"Core {core.name} did not become healthy within {self.readiness_timeout}s.")
            except Exception as e:
                # Capture any exception that occurs in this thread
                with exceptions_lock:
                    exceptions.append(e)

        # Run health checks in parallel
        threads = [threading.Thread(target=check_core, args=(core,)) for core in cores]
        for thread in threads:
            thread.start()

        # Wait for all to complete
        for thread in threads:
            thread.join()

        # Raise if any exceptions occurred
        # All cores must pass health check (alive-test) before warmup
        if exceptions:
            raise exceptions[0]

        logging.info("All cores are ready.")

    def warmup(self, cores: List[LLMCore], warmup_queries: List[LLMRequest]):
        """Run the full warmup sequence: health checks + parallel warmup.

        Args:
            cores: List of LLMCore instances to warmup
            warmup_queries: List of warmup queries to run on each core
        """
        # skip health check as well if no warmup queries provided
        if len(warmup_queries) == 0:
            return

        logging.info(f"Warming up {len(cores)} cores in parallel with {len(warmup_queries)} queries each...")

        # Track completion with threading event
        completed_count = 0
        completed_lock = threading.Lock()
        all_completed_event = threading.Event()

        def warmup_core_target(core: LLMCore):
            """Warmup target function for threading."""
            with core.warmup_mode():
                core.enqueue(warmup_queries)
            logging.debug(f"Core {core.name} warmup completed.")

            with completed_lock:
                nonlocal completed_count
                completed_count += 1
                if completed_count == len(cores):
                    all_completed_event.set()

        # Run warmup in parallel
        threads = [threading.Thread(target=warmup_core_target, args=(core,)) for core in cores]
        for thread in threads:
            thread.start()

        # Wait for all cores to complete warmup
        all_completed_event.wait()
        
        # Ensure all threads have finished
        for thread in threads:
            thread.join()

        logging.info("All cores warmed up successfully.")
