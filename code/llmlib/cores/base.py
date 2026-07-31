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
from abc import ABC, abstractmethod
from collections import defaultdict
import contextlib
from dataclasses import dataclass
import datetime
import threading
import time
import gc
from typing import Any, Callable, Dict, List, Tuple, Type, Optional

import numpy as np
from code.common.utils import nvtx_scope
import logging

from ..config import HarnessConfig
from ..utils import LLMServerProgressDisplay, LatencyTracker, PrefixLogger


@dataclass
class LLMRequest:
    """Represents a request to the LLM for inference.

    Attributes:
        request_id: Unique identifier for the request
        input_tokens: List of input token IDs to be processed
        stop_tokens: List of stop token IDs that terminate generation
    """
    request_id: int
    input_tokens: List[int]
    stop_tokens: List[int]


@dataclass
class LLMResponse:
    """Represents a response from the LLM after inference.

    Attributes:
        request_id: Unique identifier matching the original request
        output_tokens: List of generated token sequences (one per beam)
        is_final_token: Whether this is the final response for the request
        error: Optional exception if an error occurred
    """
    request_id: int
    output_tokens: Optional[List[List[int]]]  # List of responses for each beam
    is_final_token: Optional[bool]
    error: Optional[Exception]


class LLMCore(ABC):
    """Abstract base class for LLM inference cores.
    This class defines the interface that all LLM backend implementations must follow.

    Required Protocols (must be implemented by subclasses):
    - _enqueue_impl: Request enqueueing and processing
    - _poll_responses_impl: yield ready responses

    NOTE: subclasses must call _initialize_response_thread() at end of their __init__()

    LLMCore is the worker unit LLMServer can issue queries to.
    As such, it can be a in-process executor, external grpc server, or http endpoint, etc.
    """

    # Configuration type - can be overridden by subclasses
    CONFIG_T: Type[HarnessConfig] = HarnessConfig

    def __init__(
        self,
        name: str,
        harness_config: HarnessConfig,
        complete_callback: Callable,
        progress_display: LLMServerProgressDisplay,
        verbose: bool = False,
        verbose_nvtx: bool = False,
    ):
        """Initialize the LLM core.

        Args:
            name: Name of the core instance (e.g., "TrtllmExecutorCore#0")
            harness_config: Configuration for this core instance
            complete_callback: Callback function for completed requests
            progress_display: Shared progress display for metrics reporting
            verbose: Whether to enable verbose logging
            verbose_nvtx: Whether to enable verbose NVTX profiling markers
        """
        self.name = name
        self.harness_config = harness_config
        self.complete_callback = complete_callback
        self.progress_display = progress_display
        self.verbose = verbose
        self.verbose_nvtx = verbose_nvtx
        self.logger = PrefixLogger(prefix=self.name)

        self.processed_count = 0
        self.isl_sum = 0
        self.in_warmup_mode = threading.Event()
        self.latency_tracker: Optional[LatencyTracker] = None

        if self.verbose:
            self.logger.setLevel(logging.DEBUG)

        # Stop work signal
        self.stop_work = threading.Event()
        self.flush_signal = threading.Condition()

        # Track pending samples for proper cleanup and synchronization
        self.pending_samples_lock = threading.Lock()
        self.pending_samples: Dict[int, Tuple[int, int]] = {}  # executor_id -> (request_id, enqueue_time)

        # Response thread will be initialized by subclasses via _initialize_response_thread()
        self.response_thread = None
        self.response_thread_exit = threading.Event()

    @contextlib.contextmanager
    def warmup_mode(self):
        """A context manager to set the core in warmup mode.

        Prevents completion callbacks from being sent to LoadGen during warmup.
        """
        self.flush()
        self.in_warmup_mode.set()
        self.logger.debug("Entering warmup mode.")
        try:
            yield
        finally:
            self.flush()
            self.in_warmup_mode.clear()
            self.logger.debug("Exiting warmup mode.")

    def get_num_pending_samples(self) -> int:
        """
        Get (approx) number of in-flight samples for this core.
        Returns:
            Approximate number of pending samples
        """
        # NOTE(vir):
        # not making this thread safe since its used for load balancing and approx queue size is sufficient
        return len(self.pending_samples)

    def get_isl_sum(self) -> int:
        """
        Get the sum of input sequence lengths (total number of input tokens) enqueued.
        Returns:
            Total number of input tokens enqueued across all requests
        """
        return self.isl_sum

    def enqueue(self, queries: List[LLMRequest]) -> int:
        """
        Enqueue input samples for processing.

        Args:
            queries: List of LLMRequest objects containing request details

        Returns:
            Number of samples successfully enqueued
        """
        request_ids = [q.request_id for q in queries]

        with self.pending_samples_lock:
            enqueue_time = time.time()
            executor_request_ids = self._enqueue_impl(queries)
            self.pending_samples |= {
                executor_request_id: (request_id, enqueue_time)
                for request_id, executor_request_id in zip(request_ids, executor_request_ids)
            }
            # Track sum of input token lengths
            self.isl_sum += sum(len(q.input_tokens) for q in queries)
            if self.latency_tracker is not None and \
               self.harness_config.gen_config.streaming and \
               not self.in_warmup_mode.is_set():
                self.latency_tracker.add_sample(queries[0].request_id, len(queries[0].input_tokens))

        return len(executor_request_ids)

    def notify_stop(self):
        """
        Notify core to stop accepting new work and prepare for shutdown.

        This method signals the core to:
        1. Stop accepting new requests
        2. Complete all pending requests
        3. Shut down the response thread
        4. Clean up resources

        The core will not exit immediately but will wait for all in-flight
        requests to complete first.
        """
        # Signal response thread to stop
        self.stop_work.set()
        with self.pending_samples_lock:
            self.logger.debug(f"notified to stop work, pending: {len(self.pending_samples)}")

    def flush(self):
        """
        Block until all pending requests complete.
        """
        self.logger.debug(f"flush() invoked.")
        with nvtx_scope("flush"):
            # we only flush when not exiting (after all work completed)
            if not self.response_thread_exit.is_set():
                # wait for pending request queue to be empty
                with self.flush_signal:
                    self.flush_signal.wait()
        self.logger.debug(f"flush() completed.")

    def _initialize_response_thread(self):
        """
        Initialize the response thread that polls for completed requests.

        This should be called after the backend is fully initialized and ready
        to process responses.

        The response thread runs until notify_stop() is called.
        """
        if self.response_thread is not None:
            self.logger.warning("Response thread already initialized")
            return

        self.response_thread = threading.Thread(
            target=self._poll_responses,
            args=(),
            name=f"{self.name}_poll_responses"
        )
        self.response_thread.daemon = True
        self.response_thread.start()

    def _poll_responses(self):
        """Per-core response polling thread that processes responses."""
        # Response buffers, key: backend-request_id
        stream_output_toks: Dict[int, List[int]] = defaultdict(list)
        completed_output_toks: Dict[int, np.ndarray] = {}
        first_token_latencies: Dict[int, float] = {}
        last_token_latencies: Dict[int, float] = {}

        while True:
            with self.pending_samples_lock:
                num_pending = len(self.pending_samples)

            if num_pending == 0:
                # awake any pending flush calls
                with self.flush_signal:
                    self.flush_signal.notify_all()

                # core notified to stop work, exit the loop
                if self.stop_work.is_set():
                    break

            # Poll for responses ready responses within next 5ms
            timeout = datetime.timedelta(milliseconds=5)
            responses = self._poll_responses_impl(timeout=timeout)

            # Metrics for progress updates
            num_completed = 0
            num_toks = 0
            ttfts = []
            tpots = []

            for response in responses:
                if response.error:
                    self.logger.info(f"Response error for request {response.request_id}: {response.error}")
                    continue

                if self.harness_config.gen_config.streaming:
                    is_first_token = response.request_id not in stream_output_toks
                    is_final_token = response.is_final_token

                    for beam, chunk_output_toks in enumerate(response.output_tokens):
                        stream_output_toks[response.request_id].extend(chunk_output_toks)
                        response_num_output_toks = len(stream_output_toks[response.request_id])

                        # no signal for loadgen in this response
                        if not (is_first_token or is_final_token):
                            continue

                        # for 0-len response, we use <BOS> as RESP
                        if response_num_output_toks == 0:
                            stream_output_toks[response.request_id].append(self.harness_config.gen_config.bos_token_id)
                            response_num_output_toks = 1

                        # for 1-len response, we use RESP + <EOS>
                        if response_num_output_toks <= 1:
                            stream_output_toks[response.request_id].append(self.harness_config.gen_config.eos_token_id)
                            response_num_output_toks += 1

                        # update bookeeping for this response
                        with self.pending_samples_lock:
                            # NOTE(vir|nitin):
                            # Check if request is still being tracked
                            # log instead of raise error to handle stale warmup responses
                            # break since no need to process further chunks of this response
                            if response.request_id not in self.pending_samples:
                                self.logger.debug(f"Skipping response for request_id={response.request_id} (not in pending_samples)")
                                break

                            server_request_id, enqueue_time = self.pending_samples[response.request_id]
                            flight_time = time.time() - enqueue_time

                            if is_final_token:  # stop keeping track of this sample as pending
                                del self.pending_samples[response.request_id]
                                num_pending -= 1

                        if not self.in_warmup_mode.is_set():
                            # complete this request
                            completed_output_toks[response.request_id] = np.ascontiguousarray(stream_output_toks[response.request_id], dtype=np.uint32)
                            self.complete_callback(request_id=server_request_id,
                                                   output_toks=completed_output_toks[response.request_id],
                                                   output_toks_len=response_num_output_toks,
                                                   is_first_token=is_first_token,
                                                   is_final_token=is_final_token)

                            if is_first_token:
                                first_token_latencies[response.request_id] = flight_time
                                ttfts.append(flight_time)
                                if self.latency_tracker is not None:
                                    self.latency_tracker.add_TTFT(server_request_id, flight_time)

                            if is_final_token:
                                last_token_latencies[response.request_id] = flight_time
                                ttft = first_token_latencies[response.request_id]
                                tpot = ttft if response_num_output_toks <= 1 else ((flight_time - ttft) / (response_num_output_toks - 1))
                                tpots.append(tpot * 1000)

                                num_completed += 1
                                num_toks += response_num_output_toks

                        if self.verbose:
                            self.logger.debug(f"Completed request #{server_request_id} "
                                              f"(len={response_num_output_toks}, is_final={is_final_token}) [pending={num_pending}]")

                        # we only consier beam=0 since ordering is in descending order of cumLogProbs
                        # NOTE(vir): no model uses both streaming mode and >1 runtime_beams as of yet
                        break
                else:
                    # Non-streaming mode
                    assert response.is_final_token
                    for beam, response_output_toks in enumerate(response.output_tokens):
                        response_num_output_toks = len(response_output_toks)

                        # for 0-len response, we use <BOS> as RESP
                        if response_num_output_toks == 0:
                            response_output_toks.append(self.harness_config.gen_config.bos_token_id)
                            response_num_output_toks += 1

                        # for 1-len response, we use RESP + <EOS>
                        if response_num_output_toks <= 1:
                            response_output_toks.append(self.harness_config.gen_config.eos_token_id)
                            response_num_output_toks += 1

                        # update pending samples book-keeping
                        with self.pending_samples_lock:
                            # NOTE(vir|nitin):
                            # Check if request is still being tracked
                            # log instead of raise error to handle stale warmup responses
                            # break since no need to process further chunks of this response
                            if response.request_id not in self.pending_samples:
                                self.logger.debug(f"Skipping non-streaming response for request_id={response.request_id} (not in pending_samples)")
                                break

                            server_request_id, _ = self.pending_samples.pop(response.request_id)

                        if not self.in_warmup_mode.is_set():
                            # complete this request
                            completed_output_toks[response.request_id] = np.ascontiguousarray(response_output_toks, dtype=np.uint32)
                            self.complete_callback(request_id=server_request_id,
                                                   output_toks=completed_output_toks[response.request_id],
                                                   output_toks_len=response_num_output_toks,
                                                   is_first_token=False,
                                                   is_final_token=True)
                            num_completed += 1
                            num_toks += response_num_output_toks

                        num_pending -= 1

                        if self.verbose:
                            self.logger.debug(f"Completed request #{server_request_id} "
                                              f"(len={response_num_output_toks}, is_final=True) [pending={num_pending}]")

                        break  # Only consider beam=0

            if num_completed > 0:
                # Update progress display if we have completed requests
                self.processed_count += num_completed
                self._update_progress_display(num_completed, num_toks, ttfts, tpots)
                time.sleep(0)  # yield to other threads

            else:
                # sleep for 2ms to avoid busy-waiting
                time.sleep(0.002)

            # else:
            #     # opportunistically collect garbage when no responses
            #     if gc.get_count()[0] >= 10_000:
            #         gc.collect()
            pass

        self._cleanup_resources()
        with self.pending_samples_lock:
            assert len(self.pending_samples) == 0, f"Core stopped with self.pending_samples non-empty: {len(self.pending_samples)}"

        self.response_thread_exit.set()  # disable flushing
        with self.flush_signal:  # wake any pending flush
            self.flush_signal.notify_all()

        self.logger.debug("Response collection thread complete.")

    def _update_progress_display(self, num_completed, num_toks, ttfts, tpots, additional_unit_updates: Dict[str, Any] = {}):
        """
        Update the progress display with the latest batch of metrics.

        Subclasses can override this to add backend-specific metrics (e.g., KV cache utilization for TRT-LLM executor).
        Example:
        >>> self.progress_display.record_iteration_stats(<trtllm-executor-stats>)

        Args:
            num_completed: Number of requests completed in this batch
            num_toks: Total number of tokens generated in this batch
            ttfts: List of TTFT measurements (seconds) for streaming mode
            tpots: List of TPOT measurements (milliseconds) for streaming mode
            additional_unit_updates: Backend-specific metrics to report
        """
        additional_unit_updates |= {'tokens/s': num_toks}
        if self.harness_config.show_steady_state_progress:
            additional_unit_updates |= {'steady_state_tokens/s': num_toks}
        if self.harness_config.gen_config.streaming:
            additional_unit_updates |= {'TTFT(s)': ttfts, 'TPOT(ms)': tpots}

        self.progress_display.update(
            completed=num_completed,
            additional_unit_updates=additional_unit_updates,
        )

    def _cleanup_resources(self):
        """Clean up resources including the response thread."""
        pass

    def __del__(self):
        """Ensure completion of response thread on exit"""
        if not self.response_thread_exit.is_set():
            self.response_thread_exit.wait()
        self.logger.info(f"Completed {self.processed_count} samples.")

    def _enqueue_impl(self, queries: List[LLMRequest]) -> List[int]:
        """
        Backend-specific implementation for enqueueing samples.

        Subclasses must implement this to submit requests to their specific backend.

        Args:
            queries: List of LLMRequest objects containing request details

        Returns:
            List of backend-specific request IDs for tracking responses
        """
        raise NotImplementedError()

    def _poll_responses_impl(self, timeout: Optional[datetime.timedelta] = None):
        """
        Backend-specific implementation for fetching responses.

        Subclasses must implement this to poll for ready responses on demand.
        This method should return list of responses available, within given timeout.

        Args:
            timeout: Maximum time to wait for responses (default=None)

        Returns:
            Generator/iterator of LLMResponse objects
        """
        raise NotImplementedError()

    def get_num_warmup_iters(self):
        """ Get number of warmup iterations sufficient for this core. """
        return 10

    @abstractmethod
    def run_health_check(self):
        """
        Perform a health check on the core. Should raise an exception on failure,
        which will be caught by the WarmupManager for retries. For cores that
        do not connect to a server, this can be a no-op.

        Raises:
            Exception: Any exception indicating health check failure
        """
        pass

    @classmethod
    @abstractmethod
    def get_num_cores_for_workload(cls, **kwargs) -> int:
        """Calculate number of cores to spawn based on workload and configuration.

        Args:
            **kwargs: Backend-specific parameters (e.g., server URLs)

        Returns:
            Number of core instances to create
        """
        pass

    @classmethod
    @abstractmethod
    def get_config_for_core(cls,
                            core_index: int,
                            progress_display: LLMServerProgressDisplay,
                            verbose: bool,
                            verbose_nvtx: bool,
                            complete_callback: Callable,
                            **kwargs) -> Dict[str, Any]:
        """
        Get complete configuration dict for a specific core instance.
        This can be passed to __init__() to create a core instance.

        Args:
            core_index: Index of this core instance (0 to num_cores-1)
            progress_display: Shared progress display
            verbose: Verbose logging flag
            verbose_nvtx: NVTX profiling flag
            complete_callback: callback for completed requests
            **kwargs: Backend-specific parameters

        Returns:
            Dict containing all kwargs for core initialization
        """
        pass
