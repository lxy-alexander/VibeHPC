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
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
import threading
import queue
from abc import ABC, abstractmethod
import time
import mlperf_loadgen as lg
import numpy as np
import logging
import importlib.util
import multiprocessing.connection as mp_conn
import os
import signal
from functools import partial


try:
    import nvidia_smi
    nvidia_smi.nvmlInit()
except:
    pass

from code.harness.harness_triton_llm.utils import LlmConfig, get_triton_llm_inputs


def get_client(url: str, protocol: str = "http", concurrency: int = 192, verbose: bool = False):
    assert protocol in ["http", "grpc"]
    if protocol == "http":
        return httpclient.InferenceServerClient(url=url, concurrency=concurrency, verbose=verbose)
    else:
        return grpcclient.InferenceServerClient(url=url, verbose=verbose)


class ITritonSutFrontend(ABC):
    def __init__(self,
                 llm_config: LlmConfig,
                 triton_model_name: str,
                 frontend_name: str,
                 llm_batch_size: int,
                 num_frontends_per_model: int,
                 report_loadgen_conn: mp_conn.Connection,
                 url: str = "0.0.0.0:8001",
                 verbose: bool = False,
                 log_stats: bool = False,
                 num_clients: int = 1,
                 triton_batch_size: int = 1):
        assert num_clients >= 1, "Number of clients for a single frontend must be >= 1 (4 for Server, 1 for Offline)"
        assert triton_batch_size == 1, "BS>1 is not supported in Triton for now, check Jira TRTLLM-298"
        self.report_loadgen_conn = report_loadgen_conn  # report QuerySampleResponses to master process
        self.llm_batch_size = llm_batch_size

        if log_stats:
            logging.info("Stat logging run for triton harness detected. This may harm performance.")
        self.verbose = verbose
        self.num_frontends_per_model = num_frontends_per_model

        # log when a model has 1000 reqs/responses.
        self.logging_interval = 1000 // num_frontends_per_model

        self.log_stats = log_stats
        self.url = url
        self.clients = []
        self.dispatch_queues = []
        self.dispatchers = []
        self.num_dispatchers = num_clients
        self.llm_config = llm_config
        self.triton_batch_size = triton_batch_size
        self.num_queries_dispatched = 0
        self.num_queries_responded = 0
        self.triton_model_name = triton_model_name
        self.dispatching_lock = threading.Lock()  # acquired when dispatch_query_samples_called, to guard against premature termination

        for _ in range(self.num_dispatchers):
            self.clients.append(get_client(url=url, protocol="grpc", verbose=False))

        self.dispatch_idx = 0
        self.frontend_name = frontend_name

        self.wait_for_server_readiness()

        for dispatch_idx in range(self.num_dispatchers):
            self.dispatch_queues.append(queue.Queue())
            dispatcher = threading.Thread(target=self.dispatcher_target, args=(dispatch_idx,))
            self.dispatchers.append(dispatcher)
            dispatcher.start()

    def dispatcher_target(self, dispatch_idx: int):
        if self.verbose:
            logging.info(f"Starting dispatcher #{dispatch_idx} to model {self.triton_model_name}")
        client = self.clients[dispatch_idx]
        dispatch_queue = self.dispatch_queues[dispatch_idx]
        grpc_callback = partial(self.handle_queries_callback, dispatch_idx)
        if self.llm_config.streaming:
            client.start_stream(callback=grpc_callback)

        while True:
            sample = dispatch_queue.get()
            if sample is None:
                dispatch_queue.task_done()
                break
            sample_id, sample_input_ids, sample_input_lens, sample_stop_ids = sample
            inputs = get_triton_llm_inputs(llm_config=self.llm_config,
                                           tensor_input_ids=sample_input_ids,
                                           tensor_input_len=sample_input_lens,
                                           tensor_stop_ids=sample_stop_ids,
                                           protocol="grpc")
            outputs = []
            for output in ["output_ids", "sequence_length"]:
                outputs.append(grpcclient.InferRequestedOutput(output))

            if self.log_stats:
                self._update_disp_sample_stats(sample_id, sample_input_lens)
            if self.llm_config.streaming:
                client.async_stream_infer(model_name=self.triton_model_name, inputs=inputs, request_id=str(sample_id), outputs=outputs)
            else:
                client.async_infer(model_name=self.triton_model_name, inputs=inputs, callback=grpc_callback, request_id=str(sample_id), outputs=outputs)

            self.num_queries_dispatched += 1
            dispatch_queue.task_done()
        if self.verbose:
            logging.info(f"Stopping dispatcher #{dispatch_idx} to model {self.triton_model_name}")

        if self.llm_config.streaming:
            client.stop_stream(cancel_requests=False)

        dispatch_queue.join()
        client.close()

    def _dump_stats(self):
        pass

    def _update_disp_sample_stats(self, sample_id, isl):
        pass

    def dispatch_query_samples(self, query_samples):
        with self.dispatching_lock:
            if not type(query_samples[0]) is list:
                query_samples = [query_samples]
            for query_sample in query_samples:
                self.dispatch_queues[self.dispatch_idx].put(query_sample)
                self.dispatch_idx += 1
                self.dispatch_idx %= self.num_dispatchers
                if self.verbose:
                    if self.num_queries_dispatched % self.logging_interval == 0:
                        logging.info(f"Dispatched {self.num_queries_dispatched} samples from frontend {self.frontend_name}")

    def wait_for_server_readiness(self, poll_interval: int = 5, limit: int = 500):
        server_ready = False
        exp = 1.3
        while not server_ready:
            if poll_interval > limit:
                logging.info(f"Max time limit reached waiting for tritonserver, killing harness")
                raise ConnectionRefusedError
            client = get_client(url=self.url, protocol="grpc", verbose=False)
            try:
                server_ready = client.is_server_ready()
                if server_ready:
                    break
                else:
                    raise ConnectionRefusedError
            except (ConnectionRefusedError, InferenceServerException) as e:
                logging.info(f"{self.frontend_name} - server not ready. Retrying in {poll_interval}sec")
                time.sleep(poll_interval)
            poll_interval = exp * poll_interval

    def notify_dispatch_done(self):
        with self.dispatching_lock:
            for queue in self.dispatch_queues:
                queue.put(None)
            for dispatcher in self.dispatchers:
                dispatcher.join()
        if self.report_loadgen_conn is not None:
            self.send_result(None)

    def send_result(self, result):
        self.report_loadgen_conn.send([result, self.num_queries_dispatched, self.num_queries_responded])

    @abstractmethod
    def handle_queries_callback(self, dispatcher_idx, result, error):
        pass


class TritonSutGrpcFrontend(ITritonSutFrontend):
    def __init__(self,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

    def handle_queries_callback(self, dispatcher_idx, result, error):
        assert error is None, f"Inference not successful: {error}"

        sample_id = int(result.get_response().id)
        output_ids_tensor = result.as_numpy("output_ids")
        output_len_tensor = result.as_numpy("sequence_length")
        n_tokens = output_len_tensor[0, 0]
        output_ids_tensor = output_ids_tensor[:, 0, :n_tokens].reshape(1, -1)  # first beam

        while n_tokens <= 1:
            output_ids_tensor = np.append(output_ids_tensor, [[self.llm_config.end_token_id]], axis=1)
            n_tokens = output_ids_tensor.shape[1]

        self.num_queries_responded += 1
        if self.report_loadgen_conn is not None:
            self.send_result([False, sample_id, output_ids_tensor])
        else:
            curr_qsr = lg.QuerySampleResponse(sample_id, output_ids_tensor.ctypes.data, byte_size, n_tokens)
            lg.QuerySamplesComplete([curr_qsr])
        if self.verbose:
            if self.num_queries_responded % self.logging_interval == 0:
                logging.info(f"Frontend {self.frontend_name} completed inference of {self.num_queries_responded} samples")


class TritonSutGrpcStreamFrontend(ITritonSutFrontend):
    def __init__(self,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.inflight_responses = []
        self.inflight_responses_empty_idcs = []
        self.inflight_responses_metadata = []
        for dispatch_idx in range(self.num_dispatchers):
            ifr_register = np.empty((200, self.llm_config.max_output_len + 10), dtype=np.int32)
            self.inflight_responses.append(ifr_register)
            self.inflight_responses_empty_idcs.append(set(range(ifr_register.shape[0])))
            self.inflight_responses_metadata.append({})

        self.num_first_tokens_recvd = 0
        self.num_tokens_completed = 0

        if importlib.util.find_spec('nvidia_smi') is not None:
            self.nvsmi_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        else:
            self.nvsmi_handle = None

        if self.log_stats:
            self.token_stats_file = open(self.frontend_name, 'w')

        self.frontend_stopped_event = threading.Event()

    def handle_queries_callback(self, dispatcher_idx, result, error):
        assert error is None, f"Inference error caught: {error}"
        response = result.get_response()
        sample_id = int(response.id)

        is_first_token = sample_id not in self.inflight_responses_metadata[dispatcher_idx]

        if is_first_token:
            self._gather_first_token(result, dispatcher_idx)
        else:
            self._gather_interm_token(result, dispatcher_idx)

    def _gather_first_token(self, result, dispatcher_idx):
        inflight_responses = self.inflight_responses[dispatcher_idx]
        inflight_responses_empty_idcs = self.inflight_responses_empty_idcs[dispatcher_idx]
        inflight_responses_metadata = self.inflight_responses_metadata[dispatcher_idx]

        response = result.get_response()
        sample_id = int(response.id)

        output_ids_tensor = result.as_numpy("output_ids")
        output_len_tensor = result.as_numpy("sequence_length")
        n_tokens = output_len_tensor[0, 0]

        assert n_tokens <= 1, f"Received {n_tokens} number of tokens"
        assert sample_id not in inflight_responses_metadata

        if n_tokens == 0:
            # mixtral is known to send empty responses. Send [2, 2] to loadgen in this case.
            is_final = True
            recvd_token = self.llm_config.end_token_id
            output_ids_tensor = np.array([self.llm_config.end_token_id] * 2, dtype="int32").reshape(1, 2)
            output_lens_tensor = np.array([1], dtype="int32").reshape(1, 1)
        else:
            recvd_token = output_ids_tensor[0, 0]
            assert recvd_token != self.llm_config.end_token_id
            is_final = response.parameters.get("triton_final_response").bool_param
            is_final = is_final or output_ids_tensor[0, 0] == self.llm_config.end_token_id

        self.send_result([True, sample_id, output_ids_tensor])
        if is_final:
            self.num_queries_responded += 1
            self.send_result([False, sample_id, output_ids_tensor])
        else:
            try:
                empty_idx = inflight_responses_empty_idcs.pop()
            except KeyError:
                logging.info(f"Frontend {frontend_name}, dispatcher {dispatcher_idx} has reached maximum stream capacity.")
                os.kill(os.getpid(), signal.SIGINT)

            inflight_responses_metadata[sample_id] = (empty_idx, 1)  # register new single length intermediate output
            inflight_responses[empty_idx, 0] = recvd_token

        if self.log_stats:
            self._update_token_stats(sample_id=sample_id, tok_type='1', num_dispatched=self.num_queries_dispatched)

        self.num_first_tokens_recvd += 1

        if self.verbose:
            if self.num_first_tokens_recvd % self.logging_interval == 0:
                logging.info(f"Frontend {self.frontend_name} received {self.num_first_tokens_recvd} first tokens, in-flight reqs: {len(inflight_responses_metadata)}")

    def _gather_interm_token(self, result, dispatcher_idx):
        inflight_responses = self.inflight_responses[dispatcher_idx]
        inflight_responses_empty_idcs = self.inflight_responses_empty_idcs[dispatcher_idx]
        inflight_responses_metadata = self.inflight_responses_metadata[dispatcher_idx]

        output_len_tensor = result.as_numpy("sequence_length")
        n_tokens = output_len_tensor[0, 0]
        output_ids_tensor = result.as_numpy("output_ids")
        output_ids_tensor = output_ids_tensor[:, 0, :n_tokens].reshape(1, -1)  # first beam

        response = result.get_response()
        sample_id = int(response.id)
        is_final = response.parameters.get("triton_final_response").bool_param
        is_final = is_final or n_tokens == 0
        if n_tokens == 1:
            is_final = is_final or output_ids_tensor[0, 0] == self.llm_config.end_token_id

        assert n_tokens <= 1, "In streaming mode, we expect <= 1 tokens per response, got: {}".format(n_tokens)
        assert output_ids_tensor.shape[1] == n_tokens, "mismatching output_ids and sequence_length"
        assert sample_id in inflight_responses_metadata

        if is_final:
            ifr_metadata = inflight_responses_metadata.pop(sample_id)
            idx, length = ifr_metadata

            if n_tokens == 1:
                inflight_responses[idx, length] = output_ids_tensor[0, 0]
                length += 1

            inflight_responses[idx, length] = self.llm_config.end_token_id
            length += 1

            final_output_tensor = inflight_responses[idx, :length]
            self.num_queries_responded += 1
            self.send_result([False, sample_id, final_output_tensor])
            inflight_responses[idx, :] = self.llm_config.end_token_id
            inflight_responses_empty_idcs.add(idx)

            if self.log_stats:
                self._update_token_stats(sample_id=sample_id, tok_type='C', num_dispatched=self.num_queries_dispatched, out_len=seq_len)

            self.num_tokens_completed += length

            if self.verbose:
                if self.num_queries_responded % self.logging_interval == 0:
                    logging.info(f"Frontend {self.frontend_name} completed inference of {self.num_queries_responded} samples, {self.num_tokens_completed} tokens")

        else:  # intermediate token
            assert n_tokens == 1
            recvd_token = output_ids_tensor[0, 0]
            idx, length = inflight_responses_metadata[sample_id]
            assert idx not in inflight_responses_empty_idcs
            inflight_responses[idx, length] = recvd_token
            inflight_responses_metadata[sample_id] = (idx, length + 1)
            if self.log_stats:
                self._update_token_stats(sample_id=sample_id, tok_type='I', num_dispatched=self.num_queries_dispatched)

    def _update_token_stats(self, sample_id, tok_type, num_dispatched, **kwargs):
        util = -1
        out_len = -1
        if 'out_len' in kwargs:
            out_len = kwargs['out_len']
        if self.nvsmi_handle is not None:
            util = nvidia_smi.nvmlDeviceGetUtilizationRates(self.nvsmi_handle).gpu
        l = [sample_id, tok_type, num_dispatched, util, out_len, time.time()]
        self.token_stats_file.write(str(l) + '\n')

    def _update_disp_sample_stats(self, sample_id, isl):
        pass

    def _dump_stats(self):
        pass

    def notify_dispatch_done(self):
        super().notify_dispatch_done()
        with self.dispatching_lock:
            if self.log_stats:
                self.token_stats_file.close()
        self.frontend_stopped_event.set()
