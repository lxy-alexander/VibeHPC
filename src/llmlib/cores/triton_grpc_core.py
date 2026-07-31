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
Triton Inference Server gRPC Core Implementation

This module provides integration with Triton Inference Server via gRPC protocol.
It uses a multi-process architecture where:
- Main process handles request distribution and response collection
- Multiple client processes manage gRPC connections and async inference
- Communication between processes uses multiprocessing queues

"""

import datetime
import multiprocessing as mp
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tritonclient.grpc as grpcclient

from ..config import TritonHarnessConfig
from ..utils import LLMServerProgressDisplay
from .base import LLMCore, LLMRequest, LLMResponse


class TritonGrpcCore(LLMCore):
    """
    TritonGrpcCore is a LLMCore wrapper for Triton gRPC client.

    Architecture:
    - Main process: Distributes requests and collects responses
    - Client processes: Each maintains a gRPC connection to Triton server
    - Round-robin distribution: Requests are distributed across client processes
    - Async inference: Uses Triton's async_infer API for non-blocking requests
    - Streaming support: Handles both streaming and non-streaming inference

    For source code of tritonclient.grpc, see:
    https://github.com/triton-inference-server/client/blob/main/src/python/library/tritonclient/grpc/_client.py
    """
    CONFIG_T = TritonHarnessConfig

    def __init__(self, model_name: str, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.server_endpoint = self.harness_config.server_url
        self.num_clients = self.harness_config.clients_per_server
        self.logger.info(f"TritonGrpcCore initialized with {self.num_clients} clients")

        # Create separate request queues for each client process
        self.inference_request_queues = [mp.Queue() for _ in range(self.num_clients)]
        self.inference_response_queue = mp.Queue()

        # Create multiple client processes
        self.client_processes = []
        for i in range(self.num_clients):
            client_process = mp.Process(
                target=self._client_process_target,
                args=(
                    self.server_endpoint,
                    self.model_name,
                    self.harness_config.gen_config,
                    self.inference_request_queues[i],
                    self.inference_response_queue
                )
            )
            self.client_processes.append(client_process)

        # Round-robin counter for distributing queries
        self.current_client_index = 0
        for client_process in self.client_processes:
            client_process.start()

        # start response completion thread after init
        self._initialize_response_thread()

    @staticmethod
    def _client_process_target(
        server_endpoint: str,
        model_name: str,
        gen_config,
        request_queue: mp.Queue,
        response_queue: mp.Queue
    ):
        """Client process main loop - handles gRPC communication with Triton

        This method runs in a separate process and:
        1. Establishes gRPC connection to Triton server
        2. Processes requests from its dedicated queue
        3. Sends async inference requests to Triton
        4. Puts responses in shared queue for completion callback

        Args:
            server_endpoint: Triton server URL (host:port)
            model_name: Name of the model in Triton
            gen_config: Generation configuration with inference parameters
            request_queue: Queue to receive requests from main process
            response_queue: Queue to send responses back to main process
        """
        def is_nontrivial_stop_sequence(tensor_stop_ids: np.array, stop_id: int):
            """Check if stop sequence contains tokens other than the default EOS"""
            if tensor_stop_ids is None:
                return False
            return not np.all(tensor_stop_ids == stop_id)

        def create_stop_words_list_tensor(tensor_stop_ids: np.array):
            """Convert stop token IDs into Triton's expected tensor format

            Triton expects stop words in a specific format:
            - Shape: [1, 1, 2, max_stop_words]
            - First row contains the actual stop tokens
            - Second row contains the count of valid tokens
            """
            actual_stop_tok_len = 4
            stop_toks_shape = [1, 2, actual_stop_tok_len]
            stop_toks_buf = [-1] * (2 * actual_stop_tok_len)

            # Fill in the stop tokens
            for token_idx in range(actual_stop_tok_len):
                stop_toks_buf[token_idx] = tensor_stop_ids[0][token_idx]

            # Set the count of valid tokens
            stop_toks_buf[actual_stop_tok_len] = actual_stop_tok_len
            return np.array(stop_toks_buf, dtype="int32").reshape(1, 1, 2, -1)

        # Create gRPC client for this process
        client = grpcclient.InferenceServerClient(url=server_endpoint, verbose=False)

        # Setup streaming callback if needed
        if gen_config.streaming:
            client.start_stream(callback=lambda result, error: TritonGrpcCore._client_grpc_callback(
                result, error, response_queue, gen_config.eos_token_id, gen_config.min_output_len, gen_config.streaming))

        # Main request processing loop
        while True:
            request = request_queue.get()
            if request is None:  # Sentinel value to shutdown
                break

            request_id, input_tokens, stop_tokens = request
            input_len = len(input_tokens)

            # Prepare input tensors for Triton
            input_tokens = np.array(input_tokens, dtype=np.int32).reshape(1, -1)
            input_tokens = input_tokens[:, :input_len]
            input_len = np.array(input_len, dtype=np.int32).reshape(1, 1)

            # Create Triton input objects
            triton_input_tokens = grpcclient.InferInput("input_ids", input_tokens.shape, "INT32")
            triton_input_tokens.set_data_from_numpy(input_tokens)
            triton_input_len = grpcclient.InferInput("input_lengths", input_len.shape, "INT32")
            triton_input_len.set_data_from_numpy(input_len)

            # Handle custom stop sequences if provided
            triton_stop_tokens = None
            if is_nontrivial_stop_sequence(stop_tokens, gen_config.eos_token_id):
                stop_tokens = create_stop_words_list_tensor(stop_tokens)
                triton_stop_tokens = grpcclient.InferInput("stop_words_list", stop_tokens.shape, "INT32")
                triton_stop_tokens.set_data_from_numpy(stop_tokens)

            # Prepare generation configuration parameters
            output_len = np.array(gen_config.max_output_len, dtype=np.int32).reshape(1, 1)
            triton_output_len = grpcclient.InferInput("request_output_len", output_len.shape, "INT32")
            triton_output_len.set_data_from_numpy(output_len)

            beam_width = np.array(gen_config.runtime_beam_width, dtype=np.int32).reshape(1, 1)
            triton_beam_width = grpcclient.InferInput("beam_width", beam_width.shape, "INT32")
            triton_beam_width.set_data_from_numpy(beam_width)

            streaming = np.array(gen_config.streaming, dtype=bool).reshape(1, 1)
            triton_streaming = grpcclient.InferInput("streaming", streaming.shape, "BOOL")
            triton_streaming.set_data_from_numpy(streaming)

            end_id = np.array(gen_config.eos_token_id, dtype=np.int32).reshape(1, 1)
            triton_end_id = grpcclient.InferInput("end_id", end_id.shape, "INT32")
            triton_end_id.set_data_from_numpy(end_id)

            pad_id = np.array(gen_config.eos_token_id, dtype=np.int32).reshape(1, 1)
            triton_pad_id = grpcclient.InferInput("pad_id", pad_id.shape, "INT32")
            triton_pad_id.set_data_from_numpy(pad_id)

            # Collect all inputs
            triton_inputs = [triton_input_tokens, triton_input_len, triton_output_len,
                             triton_beam_width, triton_end_id, triton_pad_id, triton_streaming]
            if triton_stop_tokens is not None:
                triton_inputs.append(triton_stop_tokens)

            # Request output tensors
            outputs = [grpcclient.InferRequestedOutput("output_ids"),
                       grpcclient.InferRequestedOutput("sequence_length")]

            # Send async inference request
            if gen_config.streaming:
                # For streaming, use the pre-registered callback
                client.async_stream_infer(model_name=model_name, inputs=triton_inputs,
                                          request_id=str(request_id), outputs=outputs)
            else:
                # For non-streaming, register callback per request
                client.async_infer(model_name=model_name, inputs=triton_inputs,
                                   callback=lambda result, error: TritonGrpcCore._client_grpc_callback(
                                       result, error, response_queue, gen_config.eos_token_id,
                                       gen_config.min_output_len, gen_config.streaming),
                                   request_id=str(request_id), outputs=outputs)

        # Cleanup on exit
        if gen_config.streaming:
            client.stop_stream(cancel_requests=False)

        # Drain and close the queues after the loop exits
        # This ensures clean shutdown without leaving messages in queues
        for queue in [request_queue, response_queue]:
            while not queue.empty():
                try:
                    item = queue.get_nowait()
                    assert item is None, f"Expected None in {queue} queue during shutdown, got {item}"
                except mp.queues.Empty:
                    break
        request_queue.close()
        response_queue.close()

        client.close()

    @staticmethod
    def _client_grpc_callback(result, error, response_queue: mp.Queue, eos_token_id: int,
                              min_output_len: int, streaming: bool):
        """Callback invoked by Triton client when inference completes.

        This runs in the client process.

        This callback:
        1. Extracts output tokens from Triton response
        2. Ensures minimum output length by padding with EOS tokens
        3. Determines if this is the final response for streaming mode
        4. Puts the response in the queue for the main process (loadgen-rank)

        Args:
            result: Triton inference result object
            error: Any error that occurred during inference
            response_queue: Queue to send responses to main process
            eos_token_id: Token ID used for padding
            min_output_len: Minimum required output length
            streaming: Whether this is streaming inference
        """
        assert error is None, f"Error in Triton GRPC callback: {error}"
        response = result.get_response()
        result_id = int(response.id)

        # Extract output tensors
        output_ids_tensor = result.as_numpy("output_ids")
        output_len_tensor = result.as_numpy("sequence_length")

        n_tokens = output_len_tensor[0, 0]
        output_ids = output_ids_tensor[:, 0, :n_tokens].reshape(1, -1).tolist()

        # Pad output to meet minimum length requirement
        while n_tokens < min_output_len:
            output_ids[0].append(eos_token_id)  # only take care of first beam
            n_tokens += 1

        # Determine if this is the final response
        if streaming:
            # Check Triton's final response flag or if we hit EOS
            is_final_token = response.parameters.get("triton_final_response").bool_param
            is_final_token = is_final_token or (output_ids[0] == eos_token_id and len(output_ids) == 1)
        else:
            # Non-streaming always returns complete response
            is_final_token = True

        response_queue.put((result_id, output_ids, is_final_token))

    def _enqueue_impl(self, queries: List[LLMRequest]) -> List[int]:
        """
        Enqueue input samples to Triton server via client processes.

        Uses round-robin distribution to spread load across multiple
        client processes, avoiding bottlenecks on a single gRPC connection.

        Args:
            queries: List of LLMRequest objects containing request details

        Returns:
            List of request IDs (same as input request IDs)
        """
        assert not self.stop_work.is_set(), "Cannot issue queries after stop_work has been signalled to core"

        request_ids = []
        for query in queries:
            # Distribute queries in round-robin fashion
            request_data = (query.request_id, query.input_tokens, query.stop_tokens)
            self.inference_request_queues[self.current_client_index].put(request_data)
            self.current_client_index = (self.current_client_index + 1) % self.num_clients
            request_ids.append(query.request_id)

        return request_ids

    def _poll_responses_impl(self, timeout: datetime.timedelta):
        """Poll responses on the shared response queue"""
        end_time = time.time() + timeout.total_seconds() if timeout is not None else 0
        responses = []

        # get all available responses without blocking
        try:
            while True:
                result_id, output_ids, is_final_token = self.inference_response_queue.get_nowait()
                responses.append(LLMResponse(request_id=result_id, output_tokens=output_ids, is_final_token=is_final_token, error=None))
        except mp.queues.Empty:
            pass

        remaining_time = end_time - time.time()
        if remaining_time <= 0:
            return responses

        # block for remaining time to see if responses come in
        try:
            result_id, output_ids, is_final_token = self.inference_response_queue.get(timeout=remaining_time)
            responses.append(LLMResponse(request_id=result_id, output_tokens=output_ids, is_final_token=is_final_token, error=None))
            while True:
                result_id, output_ids, is_final_token = self.inference_response_queue.get_nowait()
                responses.append(LLMResponse(request_id=result_id, output_tokens=output_ids, is_final_token=is_final_token, error=None))
        except mp.queue.Empty:
            pass

        return responses

    def notify_stop(self):
        """Gracefully shutdown all client processes and clean up resources

        Sends sentinel values to all client queues to trigger shutdown,
        then waits for all processes to exit cleanly.
        """
        # Signal all client processes to stop
        for queue in self.inference_request_queues:
            queue.put(None)

        # Wait for all client processes to finish
        for process in self.client_processes:
            process.join()

        # Call parent class notify_stop
        super().notify_stop()

    def run_health_check(self):
        """Health check using Triton client server readiness check"""
        client = grpcclient.InferenceServerClient(url=self.server_endpoint, verbose=False)
        if not client.is_server_ready():
            raise ConnectionRefusedError(f"Triton server at {self.server_endpoint} is not ready")

    def get_num_warmup_iters(self):
        # each client requires independent warmup
        return self.num_clients * 10

    @classmethod
    def get_num_cores_for_workload(cls, **kwargs) -> int:
        """Calculate total number of client cores needed """
        harness_config = TritonHarnessConfig()
        return harness_config.get_num_endpoints()

    @classmethod
    def get_config_for_core(cls,
                            core_index: int,
                            progress_display: LLMServerProgressDisplay,
                            verbose: bool,
                            verbose_nvtx: bool,
                            complete_callback: Callable,
                            **kwargs) -> Dict[str, Any]:
        """Get configuration for a core instance """
        config = TritonHarnessConfig()
        config.server_url = config.triton_server_urls[core_index]
        model_name = f'model-{core_index}'

        return {
            'name': f'TritonGrpcCore#{core_index}_{config.server_url}_{config.model_name}',
            'harness_config': config,
            'progress_display': progress_display,
            'verbose': verbose,
            'verbose_nvtx': verbose_nvtx,
            'complete_callback': complete_callback,
            'model_name': model_name,
        }
