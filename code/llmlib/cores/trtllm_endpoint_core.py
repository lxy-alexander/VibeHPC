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
TensorRT-LLM HTTP Endpoint Core Implementation

This module provides integration with TensorRT-LLM servers via HTTP/OpenAI API.
It connects to trtllm-serve instances that expose an OpenAI-compatible endpoint.

Uses separate worker processes for Issue/Recv per LLMCore
"""

from __future__ import annotations
import asyncio
import atexit
import datetime
import logging
import threading
import os
from typing import Any, Callable, Dict, List, Optional

import httpx
from openai import AsyncOpenAI
import uvloop
uvloop.install()
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

from ..config import TrtllmEndpointConfig
from ..utils import LLMServerProgressDisplay
from .base import LLMCore, LLMRequest, LLMResponse
from .http_async_client import AsyncLLMHttpRequestManager
from .openai_client_utils import OpenAIConcurrentRequestManager
from .worker_metrics import aggregate_and_plot_worker_metrics

# shared event loop for all async operations in this module
_module_loop = None
_loop_thread = None
_shutdown_event = None


def _init_module_loop():
    """Initialize shared event loop."""
    global _module_loop, _loop_thread, _shutdown_event
    if _module_loop is None:
        _module_loop = uvloop.new_event_loop()
        _shutdown_event = threading.Event()
        _loop_thread = threading.Thread(target=_run_module_loop, daemon=True)
        _loop_thread.start()


def _run_module_loop():
    """Run the module event loop until shutdown is signaled."""
    global _module_loop, _shutdown_event
    asyncio.set_event_loop(_module_loop)

    async def _loop_runner():
        """coroutine that keeps the loop running until shutdown."""
        while not _shutdown_event.is_set():
            await asyncio.sleep(5)

    try:
        _module_loop.run_until_complete(_loop_runner())
    except Exception as e:
        logging.debug(f"Module loop error: {e}")
    finally:
        # Cancel all pending tasks to prevent warnings
        pending = asyncio.all_tasks(_module_loop)
        for task in pending:
            task.cancel()

        # Wait for tasks to be cancelled gracefully
        if pending:
            _module_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

        _module_loop.close()


def _shutdown_module_loop():
    """
    Shutdown the module event loop gracefully.

    Signals the loop thread to stop and waits for graceful shutdown.
    This function is called automatically on process exit via atexit.
    """
    global _shutdown_event
    if _shutdown_event:
        _shutdown_event.set()


# initialize module loop on import
_init_module_loop()

# register cleanup on exit to ensure proper resource cleanup
atexit.register(_shutdown_module_loop)


class TrtllmEndpointCore(LLMCore):
    """HTTP endpoint core using OpenAI client for trtllm-serve"""
    CONFIG_T = TrtllmEndpointConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.harness_config.gen_config.runtime_beam_width <= 1, "Beam > 1 not supported yet"

        # Reduce HTTP library log noise to focus on application-level logging
        logging.getLogger('openai').setLevel(logging.CRITICAL)
        logging.getLogger('httpx').setLevel(logging.CRITICAL)
        logging.getLogger('httpcore').setLevel(logging.CRITICAL)
        logging.getLogger('asyncio').setLevel(logging.CRITICAL)

        # Extract configuration parameters
        self.model_repo = self.harness_config.get_model_repo()
        self.model_name, self.model_revision = list(self.model_repo.items())[0]
        self.endpoint_url = self.harness_config.endpoint_url
        self.max_concurrency = self.harness_config.runtime_flags['max_concurrency']
        self.workers_per_core = self.harness_config.runtime_flags['workers_per_core']

        # Create concurrent request manager with pluggable implementation
        # Two implementations available:
        # 1. OpenAI Async client-based
        # 2. Lightweight HTTP implementation (aiohttp + ZMQ + Msgpack)
        http_backend = self.harness_config.runtime_flags['http_backend']
        http_provider_cls = {
            'openai_async': OpenAIConcurrentRequestManager,
            'custom_http': AsyncLLMHttpRequestManager,
        }[http_backend]
        self.logger.info(f"Using HTTP backend: {http_backend} ({http_provider_cls.__name__})")

        # Create endpoint_harness_logs subdirectory
        endpoint_logs_dir = os.path.join(self.harness_config.log_dir, "endpoint_harness_logs")
        os.makedirs(endpoint_logs_dir, exist_ok=True)

        self._request_manager = http_provider_cls(
            config=self.harness_config,
            max_concurrency=self.max_concurrency,
            workers_per_core=self.workers_per_core,
            log_dir=endpoint_logs_dir,
            verbose=self.verbose,
            enable_metrics=self.harness_config.enable_metrics
        )

        # Log initialization details for debugging and monitoring
        self.logger.info(f"Initialized TrtllmEndpointCore with {self.workers_per_core} workers (endpoint_url: {self.endpoint_url}, max_concurrency: {self.max_concurrency})")

        # start response completion thread after init
        self._initialize_response_thread()

    def run_health_check(self):
        """Check if underlying TRT-LLM server is healthy"""
        async def _health_check():
            """
            Async health check implementation.

            Returns:
                True if healthy, Exception if unhealthy
            """
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                    response = await client.get(f"http://{self.endpoint_url}/health")
                    response.raise_for_status()
                return True
            except httpx.ConnectError:
                return Exception("Connection error.")
            except httpx.TimeoutException:
                return Exception("Request timed out.")
            except Exception as e:
                return e

        # Ensure module loop is initialized and running
        global _module_loop
        _init_module_loop()

        # Run the health check using the module event loop with a timeout
        future = asyncio.run_coroutine_threadsafe(_health_check(), _module_loop)
        try:
            ok_or_error = future.result(timeout=15.0)  # 15 second total timeout
            if ok_or_error != True:
                raise ok_or_error
        except asyncio.TimeoutError:
            raise TimeoutError(f"Health check timed out for endpoint {self.endpoint_url}")
        except Exception as e:
            # Re-raise any other exception with context
            raise

    def _enqueue_impl(self, queries: List[LLMRequest]) -> List[int]:
        """
        Submit requests to the request manager for processing.

        This method implements the LLMCore interface for request submission.
        It delegates to the configured request manager, which handles the
        actual HTTP communication and response processing.

        Args:
            queries (List[LLMRequest]): List of requests to process

        Returns:
            List[int]: List of request IDs that were submitted
        """
        assert not self.stop_work.is_set()
        self._request_manager.submit_requests(queries)
        request_ids = [query.request_id for query in queries]
        return request_ids

    def _poll_responses_impl(self, timeout: Optional[datetime.timedelta] = None) -> List[LLMResponse]:
        """
        Get responses from the request manager within the specified timeout.

        This method implements the LLMCore interface for response polling.
        It delegates to the configured request manager, which collects
        responses from HTTP requests and returns them in the expected format.
        """
        responses = self._request_manager.get_responses(timeout)
        return responses

    def _cleanup_resources(self):
        """Clean up resources when response thread exits."""
        self._request_manager.shutdown()

        # Generate worker metrics plots if verbose mode and using custom HTTP backend
        if self.verbose and self.harness_config.runtime_flags['http_backend'] == 'custom_http':
            try:
                self.logger.info("Generating worker metrics plots...")
                endpoint_logs_dir = os.path.join(self.harness_config.log_dir, "endpoint_harness_logs")
                aggregate_and_plot_worker_metrics(endpoint_logs_dir)
            except Exception as e:
                self.logger.warning(f"Failed to generate worker metrics plots: {e}")

        super()._cleanup_resources()

    @classmethod
    def get_num_cores_for_workload(cls, **kwargs) -> int:
        """
        Calculate the number of LLM Cores cores needed for the workload.
        We do 1 LLMCore instance per endpoint.
        """
        return len(cls.CONFIG_T().trtllm_endpoint_urls)

    @classmethod
    def get_config_for_core(cls,
                            core_index: int,
                            progress_display: LLMServerProgressDisplay,
                            verbose: bool,
                            verbose_nvtx: bool,
                            complete_callback: Callable,
                            model_path: str,
                            **kwargs) -> Dict[str, Any]:
        """Get configuration for a core instance """
        config = cls.CONFIG_T(**kwargs)
        config.model_path = model_path
        config.endpoint_url = config.trtllm_endpoint_urls[core_index]

        return {
            'name': f'TrtllmEndpointCore#{core_index}',
            'harness_config': config,
            'progress_display': progress_display,
            'verbose': verbose,
            'verbose_nvtx': verbose_nvtx,
            'complete_callback': complete_callback,
        }
