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
OpenAI Client Utilities for TensorRT-LLM HTTP Endpoint

This module provides utility classes for managing concurrent requests to
OpenAI-compatible endpoints.
"""

from __future__ import annotations
import asyncio
import logging
import multiprocessing as mp
import queue
import time
import os
from pathlib import Path
from typing import List
import uvloop

import httpx
from openai import AsyncOpenAI
from tokenizers import Tokenizer as AutoTokenizer

from ..config import TrtllmEndpointConfig
from ..utils import PrefixLogger
from .base import LLMRequest, LLMResponse
from .mp_worker_utils import WorkerProcessManager


class OpenAIConcurrentRequestManager:
    """
    Manages concurrent request execution via multiprocess workers with Async OpenAI clients.
    This class serves as the main coordinator for distributing requests and collecting responses.

    Multiprocess Support:
        - Distributes requests across multiple worker processes
        - Each worker has its own event loop and tokenizer instance
        - Round-robin load balancing across workers

    Args:
        config (TrtllmEndpointConfig): Configuration for the TensorRT-LLM endpoint
        max_concurrency (int): Maximum number of concurrent requests per worker
        workers_per_core (int): Number of worker processes for multiprocess mode
    """

    def __init__(self,
                 config: TrtllmEndpointConfig,
                 max_concurrency: int,
                 workers_per_core: int,
                 log_dir: str,
                 verbose: bool = False):
        self.config = config
        self.max_concurrency = max_concurrency
        self.workers_per_core = workers_per_core
        self.log_dir = log_dir
        self.verbose = verbose
        self.model_name, self.model_revision = list(self.config.get_model_repo().items())[0]
        self.tokenizer_path = config.harness_tokenizer_path  # None means fall back to HF

        self._loop = None
        self._request_provider = None
        self._response_queue = None
        self._worker_manager = None
        self._initialize()

    def _initialize(self):
        """ Initialize multiprocess mode with worker processes. """
        self._response_queue = mp.Queue()

        # Create dedicated queues for request distribution (one per worker process)
        self.request_queues = [mp.Queue() for _ in range(self.workers_per_core)]
        self.current_worker_index = 0  # Round-robin counter for load balancing

        # Create worker manager instance
        self._worker_manager = WorkerProcessManager()

        # Start worker processes using manager instance
        init_args = [
            self.config,
            self.max_concurrency,
            self.request_queues,
            self._response_queue,
            self.model_name,
            self.model_revision,
            self.tokenizer_path,
            self.verbose
        ]
        self.worker_processes = self._worker_manager.start_worker_processes(
            worker_count=self.workers_per_core,
            worker_main_func=OpenAIConcurrentRequestManager._worker_process_main,
            init_args=init_args,
            process_name_prefix="OpenAIEndpointWorker",
            log_dir=self.log_dir
        )

    def submit_requests(self, queries: List[LLMRequest]) -> None:
        """
        Submit requests in multiprocess mode with round-robin distribution across workers.

        Distributes requests evenly across worker processes to balance load.
        Each request is serialized and sent via multiprocess queues.

        Args:
            queries (List[LLMRequest]): Requests to distribute across workers
        """
        for query in queries:
            # Prepare request data for inter-process communication (must be pickle-serializable)
            request_data = {
                'request_id': query.request_id,
                'input_tokens': query.input_tokens,
                'stop_tokens': query.stop_tokens,
            }

            # Round-robin distribution across worker processes for load balancing
            self.request_queues[self.current_worker_index].put(request_data)
            self.current_worker_index = (self.current_worker_index + 1) % self.workers_per_core

    def get_responses(self, timeout) -> List[LLMResponse]:
        """
        Get completed responses from the response queue (unified for both modes).

        Uses a two-phase approach:
        1. Non-blocking collection of all immediately available responses
        2. Blocking wait for additional responses within the timeout period

        Args:
            timeout (datetime.timedelta): Maximum time to wait for responses

        Returns:
            List[LLMResponse]: All responses collected within the timeout period
        """
        end_time = time.time() + timeout.total_seconds() if timeout is not None else 0
        responses = []

        # Phase 1: Get all immediately available responses without blocking
        try:
            while True:
                responses.append(self._response_queue.get_nowait())
        except queue.Empty:
            pass

        # Phase 2: Calculate remaining time and block for additional responses
        remaining_time = end_time - time.time()
        if remaining_time <= 0:
            return responses

        # Block for remaining time to capture any late-arriving responses
        try:
            responses.append(self._response_queue.get(timeout=remaining_time))
            # Collect any additional responses that arrived during the blocking wait
            while True:
                responses.append(self._response_queue.get_nowait())
        except queue.Empty:
            pass

        return responses

    def shutdown(self):
        """
        Clean up resources based on execution mode.
        Terminates worker processes and cleans up queues
        """
        # Use worker manager instance for consistent shutdown behavior
        self._worker_manager.shutdown_workers(
            self.worker_processes,
            self.request_queues
        )

    @classmethod
    def _worker_process_main(
        cls,
        config: TrtllmEndpointConfig,
        max_concurrency: int,
        request_queues: List[mp.Queue],
        response_queue: mp.Queue,
        model_name: str,
        model_revision: str,
        tokenizer_path: str,
        verbose: bool,
        worker_id: int,
        readiness_queue: mp.Queue,
    ):
        """ Main function for multiprocess worker. """
        # Get this worker's dedicated request queue
        request_queue = request_queues[worker_id]

        # NOTE(vir):
        # default is false, seems sufficient in my testing
        # can toggle here and capture nsys to measure any differences for very long OSL
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        # Create new event loop for this process (isolated from main process)
        loop = uvloop.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the worker loop until completion or shutdown signal
            loop.run_until_complete(cls._worker_loop(
                config,
                max_concurrency,
                request_queue,
                response_queue,
                readiness_queue,
                model_name,
                model_revision,
                tokenizer_path,
                verbose
            ))
        finally:
            # Always clean up the event loop
            loop.close()

    @classmethod
    async def _worker_loop(
        cls,
        config: TrtllmEndpointConfig,
        max_concurrency: int,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        readiness_queue: mp.Queue,
        model_name: str,
        model_revision: str,
        tokenizer_path: str,
        verbose: bool = False
    ):
        """ Worker loop for multiprocess mode. """
        try:
            # Initialize request provider for this worker process
            request_provider = OpenAIConcurrentRequestProvider(
                config,
                max_concurrency,
                model_name,
                model_revision,
                tokenizer_path,
                verbose
            )
            await request_provider.initialize()

            # Signal successful initialization to the main process
            readiness_queue.put(True)
        except Exception as e:
            # Signal failure so main process doesn't wait indefinitely
            readiness_queue.put(f"Worker initialization failed: {e}")
            return

        # Process requests with concurrent handling using asyncio
        loop = asyncio.get_event_loop()
        active_tasks = set()  # Track active tasks for graceful shutdown
        processed_queries = 0  # Track number of queries processed by this worker

        while True:
            try:
                # Non-blocking get with short timeout to allow for shutdown checks
                request_data = await loop.run_in_executor(None, request_queue.get, True, 0.1)
                if request_data is None:  # Shutdown signal from main process
                    break

                # Convert serialized data back to LLMRequest object
                request = LLMRequest(
                    request_id=request_data['request_id'],
                    input_tokens=request_data['input_tokens'],
                    stop_tokens=request_data['stop_tokens']
                )

                # Increment processed query counter
                processed_queries += 1

                # Create task for concurrent processing using the request provider
                task = asyncio.create_task(
                    request_provider.process_request(request, response_queue)
                )
                active_tasks.add(task)
                # Automatically remove completed tasks from the set
                task.add_done_callback(active_tasks.discard)

            except queue.Empty:
                # No requests available, continue the loop
                continue
            except Exception as e:
                logging.error(f"Worker error: {e}")

        # Wait for all active tasks to complete before shutting down
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)

        # Log total queries processed by this worker
        logging.info(f"Worker process (PID: {os.getpid()}) processed {processed_queries} queries")

        # Clean up request provider and its resources
        await request_provider.shutdown()


class OpenAIConcurrentRequestProvider:
    """
    Async OpenAI client for concurrent LLM request processing.

    Args:
        config (TrtllmEndpointConfig): Configuration containing endpoint URL and generation parameters
        max_concurrency (int): Maximum number of concurrent requests this provider can handle
    """

    def __init__(self,
                 config: TrtllmEndpointConfig,
                 max_concurrency: int,
                 model_name: str,
                 model_revision: str,
                 tokenizer_path: str,
                 verbose: bool = False):
        self.config = config
        self.max_concurrency = max_concurrency
        self.model_name = model_name
        self.model_revision = model_revision
        self.tokenizer_path = tokenizer_path
        self.verbose = verbose
        self.endpoint_url = config.endpoint_url

        # Set up logger with prefix like TrtllmEndpointCore does
        self.logger = PrefixLogger(prefix=f"OpenAIWorker-{os.getpid()}")
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)

        # Set up logger with prefix like TrtllmEndpointCore does
        self.logger = PrefixLogger(prefix=f"OpenAIWorker-{os.getpid()}")
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)

        # Initialize resources (will be set up in initialize() method)
        self.http_client = None
        self.async_client = None
        self.tokenizer = None
        self.concurrency_semaphore = None

    async def initialize(self):
        """ Initialize OpenAI client, tokenizer, and concurrency semaphore. """
        # Create semaphore for concurrency control (limits concurrent requests)
        if self.max_concurrency == -1:
            # Unlimited concurrency - no semaphore needed
            self.concurrency_semaphore = None
        else:
            self.concurrency_semaphore = asyncio.Semaphore(self.max_concurrency)

        # Set reasonable limits for HTTP client
        if self.max_concurrency == -1:
            # Use reasonable defaults for unlimited concurrency
            connection_limit = 1000
        else:
            connection_limit = self.max_concurrency

        # Setup HTTP client with proper connection limits for high concurrency
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(None),  # No timeout (handled by OpenAI client)
            limits=httpx.Limits(
                max_keepalive_connections=connection_limit * 2,  # Oversubscribe for better utilization
                max_connections=connection_limit * 2,
            ),
            http2=True
        )

        # Setup OpenAI client with the configured HTTP client
        self.async_client = AsyncOpenAI(
            api_key='dummy',  # TensorRT-LLM server doesn't require real API key
            base_url=f"http://{self.endpoint_url}/v1/",
            timeout=None,  # No timeout (let the server handle timing)
            max_retries=10,  # Retry failed requests up to 10 times
            http_client=self.http_client,  # Use our configured HTTP client
        )

        # Load tokenizer from local path or HuggingFace
        if self.tokenizer_path:
            tokenizer_file = Path(self.tokenizer_path) / "tokenizer.json"
            if not tokenizer_file.exists():
                raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")
            self.tokenizer = AutoTokenizer.from_file(str(tokenizer_file))
        else:
            # Load from HuggingFace using model_name
            logging.warning(f"Using HF repo for tokenizer: {self.model_name} {self.model_revision} - please use `hf auth login` to avoid being rate limited")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.model_revision)

    async def process_request(self, request: LLMRequest, response_queue) -> None:
        """
        Process a single request.
        Args:
            request (LLMRequest): The request to process
            response_queue: Queue to put responses
        """
        # Use semaphore for concurrency control if it exists
        if self.concurrency_semaphore:
            async with self.concurrency_semaphore:
                await self._process_request_impl(request, response_queue)
        else:
            # Unlimited concurrency - process directly
            await self._process_request_impl(request, response_queue)

    async def _process_request_impl(self, request: LLMRequest, response_queue) -> None:
        # Prepare generation parameters using standard OpenAI chat completions format
        gen_params = {
            "model": self.model_name,
            "max_tokens": self.config.gen_config.max_output_len,
            "temperature": self.config.gen_config.temperature,
            "top_p": self.config.gen_config.top_p,
            "stream": self.config.gen_config.streaming,
            "messages": [],  # Empty messages for token-based input
            "extra_body": {
                # TensorRT-LLM specific parameters passed in extra_body
                "prompt_token_ids": request.input_tokens,
                "min_tokens": self.config.gen_config.min_output_len,
                "top_k": self.config.gen_config.top_k,
            },
        }

        # Add stop tokens if configured
        if self.config.gen_config.use_stop_tokens:
            gen_params["stop_token_ids"] = request.stop_tokens

        # Route to appropriate handler based on streaming configuration
        if self.config.gen_config.streaming:
            await self._handle_streaming(request, gen_params, response_queue)
        else:
            await self._handle_non_streaming(request, gen_params, response_queue)

    async def _handle_streaming(self, request: LLMRequest, gen_params: dict, response_queue) -> None:
        """
        Handle streaming response processing.

        Streaming mode provides tokens as they are generated, allowing for lower latency
        and progressive response display. This method:
        1. Creates a streaming completion request
        2. Processes each chunk as it arrives
        3. Emits the first token immediately for TTFT measurement
        4. Accumulates the full response text
        5. Emits a final response with only the remaining tokens (excluding already sent tokens)

        Args:
            request (LLMRequest): The original request
            gen_params (dict): OpenAI API parameters
            response_queue: Queue for putting responses
        """
        output_text = ""
        first_token = True
        first_token_count = 0  # Track how many tokens were sent in first response

        try:
            # Create streaming completion
            stream = await self.async_client.chat.completions.create(**gen_params)
            # Process each chunk from the stream
            async for chunk in stream:
                if content := chunk.choices[0].delta.content:
                    output_text += content

                    # Emit first token immediately for TTFT (Time To First Token) measurement
                    if first_token:
                        first_token = False
                        encoded_toks = self.tokenizer.encode(content).ids
                        first_token_count = len(encoded_toks)  # Track how many tokens sent
                        response_queue.put(LLMResponse(
                            request_id=request.request_id,
                            output_tokens=[encoded_toks],
                            is_final_token=False,
                            error=None
                        ))

            # Emit final response with only the remaining tokens (excluding already sent tokens)
            output_tokens = self.tokenizer.encode(output_text).ids
            remaining_tokens = output_tokens[first_token_count:]  # Skip already sent tokens
            response_queue.put(LLMResponse(
                request_id=request.request_id,
                output_tokens=[remaining_tokens],
                is_final_token=True,
                error=None
            ))
        except Exception as e:
            # Handle any errors during streaming
            response_queue.put(LLMResponse(
                request_id=request.request_id,
                output_tokens=[],
                is_final_token=True,
                error=str(e)  # Convert exception to string for multiprocess serialization
            ))

    async def _handle_non_streaming(self, request: LLMRequest, gen_params: dict, response_queue) -> None:
        """
        Handle non-streaming response processing.

        Non-streaming mode waits for the complete response before returning.
        This method:
        1. Creates a standard completion request
        2. Waits for the complete response
        3. Tokenizes the full response text
        4. Emits a single final response

        Args:
            request (LLMRequest): The original request
            gen_params (dict): OpenAI API parameters
            response_queue: Queue for putting responses
        """
        try:
            # Create standard (non-streaming) completion
            completion = await self.async_client.chat.completions.create(**gen_params)
            response_text = completion.choices[0].message.content
            output_tokens = self.tokenizer.encode(response_text).ids

            # Emit single response with complete output
            response_queue.put(LLMResponse(
                request_id=request.request_id,
                output_tokens=[output_tokens],
                is_final_token=True,
                error=None
            ))
        except Exception as e:
            # Handle any errors during non-streaming request
            response_queue.put(LLMResponse(
                request_id=request.request_id,
                output_tokens=[],
                is_final_token=True,
                error=str(e)  # Convert exception to string for multiprocess serialization
            ))

    async def shutdown(self):
        """ Clean up resources. """
        if self.http_client:
            await self.http_client.aclose()

