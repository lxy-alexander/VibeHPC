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
from collections import namedtuple
from typing import Union, Optional
import numpy as np
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import json
import tqdm
import time
import threading

LlmConfig = namedtuple('LlmConfig', 'model_name, min_output_len, max_output_len, beam_width, end_token_id, pad_token_id, streaming, use_stop_tokens')
kMAX_STOP_TOKS_LEN = 4


def is_nontrivial_stop_sequence(tensor_stop_ids: np.array, stop_id: int):
    if tensor_stop_ids is None:
        return False
    return not np.all(tensor_stop_ids == stop_id)


def create_stop_words_list_tensor(tensor_stop_ids: np.array):
    global kMAX_STOP_TOKS_LEN
    actual_stop_tok_len = kMAX_STOP_TOKS_LEN
    stop_toks_shape = [1, 2, actual_stop_tok_len]
    stop_toks_buf = [-1] * (2 * actual_stop_tok_len)

    for token_idx in range(kMAX_STOP_TOKS_LEN):
        stop_toks_buf[token_idx] = tensor_stop_ids[0][token_idx]

    stop_toks_buf[kMAX_STOP_TOKS_LEN] = actual_stop_tok_len
    return np.array(stop_toks_buf, dtype="int32").reshape(1, 1, 2, -1)


def construct_triton_input_data(name: str, shape: Union[list, tuple, None] = None,
                                dtype: str = "INT32", tensor: Optional[np.array] = None, protocol: str = "http") -> Union[httpclient.InferInput, grpcclient.InferInput]:
    if shape is None:
        shape = tensor.shape

    assert protocol in ["http", "grpc"]
    if protocol == "http":
        infer_input = httpclient.InferInput(name=name, shape=list(shape), datatype=dtype)
        if tensor is not None:
            infer_input.set_data_from_numpy(tensor)
    else:
        infer_input = grpcclient.InferInput(name=name, shape=list(shape), datatype=dtype)
        if tensor is not None:
            infer_input.set_data_from_numpy(tensor)

    return infer_input


def get_triton_llm_inputs(llm_config: LlmConfig,
                          tensor_input_ids: np.array,
                          tensor_input_len: np.array,
                          tensor_stop_ids: Optional[np.array] = None,
                          protocol: str = "grpc"):
    """
    For a single sample, accepts (len, ids) and returns list of triton input objects
    """

    assert tensor_input_len.shape == (1, 1), "Triton only supports BS==1 for now (TRTLLM-298)"
    assert tensor_input_ids.shape[0] == 1, "Triton only supports BS==1 for now (TRTLLM-298)"

    ip_len = tensor_input_len[0, 0]
    tensor_input_ids = tensor_input_ids[:, :ip_len].reshape(1, -1)
    if not tensor_stop_ids is None:
        tensor_stop_ids = tensor_stop_ids.reshape(1, -1)
    tensor_output_len = np.array([llm_config.max_output_len], dtype="int32").reshape([1, 1])
    tensor_beam_width = np.array([llm_config.beam_width], dtype="int32").reshape([1, 1])
    tensor_streaming = np.array([llm_config.streaming], dtype="bool").reshape([1, 1])
    tensor_end_id = np.array([llm_config.end_token_id], dtype="int32").reshape([1, 1])
    tensor_pad_id = np.array([llm_config.pad_token_id], dtype="int32").reshape([1, 1])

    input_ids = construct_triton_input_data(name="input_ids", tensor=tensor_input_ids, protocol=protocol)
    input_lengths = construct_triton_input_data(name="input_lengths", tensor=tensor_input_len, protocol=protocol)
    request_output_len = construct_triton_input_data(name="request_output_len", tensor=tensor_output_len, protocol=protocol)
    beam_width = construct_triton_input_data(name="beam_width", tensor=tensor_beam_width, protocol=protocol)
    end_id = construct_triton_input_data(name="end_id", tensor=tensor_end_id, protocol=protocol)
    pad_id = construct_triton_input_data(name="pad_id", tensor=tensor_pad_id, protocol=protocol)
    streaming = construct_triton_input_data(name="streaming", tensor=tensor_streaming, dtype="BOOL", protocol=protocol)

    inputs = [input_ids, input_lengths, request_output_len, beam_width, end_id, pad_id, streaming]
    if is_nontrivial_stop_sequence(tensor_stop_ids, llm_config.end_token_id):
        tensor_stop_ids = create_stop_words_list_tensor(tensor_stop_ids)
        stop_ids = construct_triton_input_data(name="stop_words_list", tensor=tensor_stop_ids, protocol=protocol)
        inputs.append(stop_ids)

    return inputs


def get_llm_gen_config(model_name: str, scenario: str, llm_gen_config_path: str):
    with open(llm_gen_config_path) as f:
        vals = json.load(f)['generation_config']
        llm_config = LlmConfig(
            model_name=model_name,
            min_output_len=vals['min_output_len'],
            max_output_len=vals['max_output_len'],
            beam_width=vals['runtime_beam_width'],
            end_token_id=vals['eos_token_id'],
            pad_token_id=vals['eos_token_id'],
            streaming=vals['streaming'] and scenario != "Offline",
            use_stop_tokens=vals['use_stop_tokens'],
        )
        return llm_config


def split_string_into_chunks(st, n):
    # Split the string into a list of values
    values = st.split(',')

    # Calculate the number of chunks
    num_chunks = len(values) // n

    # Create the array of chunks
    arr = [','.join(values[i * n:(i + 1) * n]) for i in range(num_chunks)]

    return arr

class LoadingBarManager:
    def __init__(self):
        self.curr_position = 0
        self.tqdm_bars = []
        self.refresh_thread = threading.Thread(target=self.refresh_target, daemon=True)
        self.stopped = False

    def add_loading_bar(self, name: str) -> int:
        self.curr_position += 1
        self.tqdm_bars.append(
            tqdm.tqdm(total=0, desc=name, position=self.curr_position, leave=True)
        )
        return self.curr_position

    def start(self):
        assert not self.stopped
        self.refresh_thread.start()

    def stop(self):
        self.stopped = True
        self.refresh_thread.join()

    def update(self, idx: int, completed: int, total: int) -> bool:
        bar = self.tqdm_bars[idx]
        bar.n = completed
        bar.total = total
        return True

    def refresh_target(self):
        while not self.stopped:
            for bar in self.tqdm_bars:
                bar.refresh()
            time.sleep(1)
