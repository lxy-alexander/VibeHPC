# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import re
from collections import OrderedDict
from .tokenizer import get_tokenizer
from .whisper_utils import (log_mel_spectrogram, labels_dict, currency_to_text, numbers_to_text)
import threading
import queue
import array
import torch
from pathlib import Path
from code.common.utils import nvtx_scope
from nvmitten.nvidia.cupy import CUDARTWrapper as cudart
import mlperf_loadgen as lg
import numpy as np

import tensorrt_llm
from tensorrt_llm._utils import (str_dtype_to_torch
                                 )
from tensorrt_llm.bindings import GptJsonConfig
from tensorrt_llm.runtime import PYTHON_BINDINGS
from code.common import logging

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp


class WhisperCopyStream:
    def __init__(self,
                 device_id,
                 gpu_batch_size):
        cudart.cudaSetDevice(device_id)
        self.stream = cudart.cudaStreamCreate()
        self.h2d_event = cudart.cudaEventCreateWithFlags(cudart.cudaEventDefault | cudart.cudaEventDisableTiming)
        self.d2h_event = cudart.cudaEventCreateWithFlags(cudart.cudaEventDefault | cudart.cudaEventDisableTiming)

    def record_h2d_event(self):
        cudart.cudaEventRecord(self.h2d_event, self.stream)

    def record_d2h_event(self):
        cudart.cudaEventRecord(self.d2h_event, self.stream)

    def make_infer_await_h2d(self, infer_stream):
        cudart.cudaStreamWaitEvent(infer_stream, self.h2d_event, 0)

    def await_infer_done(self, infer_done):
        cudart.cudaStreamWaitEvent(self.stream, infer_done, 0)


class WhisperResponse:
    def __init__(self,
                 sample_ids,
                 predictions,
                 word_counts,
                 results_ready
                 ):
        self.sample_ids = sample_ids
        self.predictions = predictions
        self.word_counts = word_counts
        self.results_ready = results_ready


def remove_tensor_padding(input_tensor,
                          input_tensor_lengths=None,
                          pad_value=None):
    if pad_value:
        assert input_tensor_lengths is None, "input_tensor_lengths should be None when pad_value is provided"
        # Text tensor case: batch, seq_len
        assert torch.all(
            input_tensor[:, 0] !=
            pad_value), "First token in each sequence should not be pad_value"
        assert input_tensor_lengths is None

        # Create a mask for all non-pad tokens
        mask = input_tensor != pad_value

        # Apply the mask to input_tensor to remove pad tokens
        output_tensor = input_tensor[mask].view(1, -1)

    else:
        # Audio tensor case: batch, seq_len, feature_len
        # position_ids case: batch, seq_len
        assert input_tensor_lengths is not None, "input_tensor_lengths must be provided for 3D input_tensor"

        # Initialize a list to collect valid sequences
        valid_sequences = []

        for i in range(input_tensor.shape[0]):
            valid_length = input_tensor_lengths[i]
            valid_sequences.append(input_tensor[i, :valid_length])

        # Concatenate all valid sequences along the batch dimension
        output_tensor = torch.cat(valid_sequences, dim=0)
    return output_tensor


def read_config(component, engine_dir):
    config_path = engine_dir / component / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    model_config = OrderedDict()
    model_config.update(config['pretrained_config'])
    model_config.update(config['build_config'])
    return model_config


class WhisperTRTLLMCore(object):

    def __init__(self,
                 engine_dir,
                 dataset=None,
                 device_id=int,
                 debug_mode=False,
                 use_graphs=False,
                 assets_dir=None,
                 batch_size=256,
                 num_beams=1,
                 verbose=False):

        cudart.cudaSetDevice(device_id)
        self.total_samples = 0
        self.batch_size = batch_size
        self.device = "cuda"
        self.device_id = device_id
        self.dataset = dataset
        logging.info(f"[Device {self.device_id}] Initializing")

        torch.autograd.set_grad_enabled(False)

        world_size = 1
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)
        engine_dir = Path(engine_dir)
        encoder_config = read_config('encoder', engine_dir)
        decoder_config = read_config('decoder', engine_dir)
        self.n_mels = encoder_config['n_mels']
        self.num_languages = encoder_config['num_languages']
        is_multilingual = (decoder_config['vocab_size'] >= 51865)
        self.assets_dir = assets_dir
        if is_multilingual:
            tokenizer_name = "multilingual"
            assert (Path(assets_dir) / "multilingual.tiktoken").exists(
            ), "multilingual.tiktoken file is not existed in assets_dir"
        else:
            tokenizer_name = "gpt2"
            assert (Path(assets_dir) / "gpt2.tiktoken").exists(
            ), "gpt2.tiktoken file is not existed in assets_dir"
        self.tokenizer = get_tokenizer(name=tokenizer_name,
                                       num_languages=self.num_languages,
                                       tokenizer_dir=assets_dir)
        self.eot_id = self.tokenizer.encode(
            "<|endoftext|>",
            allowed_special=self.tokenizer.special_tokens_set)[0]

        json_config = GptJsonConfig.parse_file(engine_dir / 'decoder' /
                                               'config.json')
        assert json_config.model_config.supports_inflight_batching

        runner_kwargs = dict(engine_dir=engine_dir,
                             is_enc_dec=True,
                             max_batch_size=batch_size,
                             max_input_len=3000,
                             max_output_len=110,
                             max_beam_width=num_beams,
                             debug_mode=debug_mode,
                             kv_cache_free_gpu_memory_fraction=0.9,
                             cross_kv_cache_fraction=0.5,
                             device_ids=[self.device_id])
        logging.debug(f"runner_kwargs: {runner_kwargs}")
        self.model_runner_cpp = ModelRunnerCpp.from_dir(**runner_kwargs)

        # Initialize cuda graphs
#        if self.use_graphs:
#            self.engines.enable_cuda_graphs(self.buffers)
        # Runtime components
        self.context_memory = None
        self.infer_stream = cudart.cudaStreamCreate()
        self.infer_done = cudart.cudaEventCreateWithFlags(cudart.cudaEventDefault | cudart.cudaEventDisableTiming)
        self.copy_stream = WhisperCopyStream(device_id, self.batch_size)

        # QSR components
        self.response_queue = queue.Queue()
        self.response_thread = threading.Thread(target=self._process_response, args=(), daemon=True)
        # self.start_inference = threading.Condition()

        # Initialize QSR thread
        self.response_thread.start()

        logging.debug(f"[Device {self.device_id}] Done initialization")

    def process_batch(
            self,
            mel,
            mel_input_lengths,
            text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
            num_beams=1,
            max_new_tokens=150):

        prompt_id = self.tokenizer.encode(
            text_prefix, allowed_special=self.tokenizer.special_tokens_set)
        prompt_id = torch.tensor(prompt_id)
        batch_size = len(mel)
        decoder_input_ids = prompt_id.repeat(batch_size, 1)
        with torch.no_grad():
            if isinstance(mel, list):
                mel = [
                    m.transpose(1, 2).type(
                        str_dtype_to_torch("float16")).squeeze(0)
                    for m in mel
                ]
            else:
                mel = mel.transpose(1, 2)
            outputs = self.model_runner_cpp.generate(
                batch_input_ids=decoder_input_ids,
                encoder_input_features=mel,
                encoder_output_lengths=mel_input_lengths // 2,
                max_new_tokens=max_new_tokens,
                end_id=self.eot_id,
                pad_id=self.eot_id,
                num_beams=num_beams,
                output_sequence_lengths=True,
                return_dict=True)
            # TODO remove torch.cuda sync and sync later
            # torch.cuda.synchronize()
            output_ids = outputs['output_ids'].cpu().numpy().tolist()

        return output_ids

    def post_process(self, output_ids):
        texts = []
        for i in range(len(output_ids)):
            text = self.tokenizer.decode(output_ids[i][0]).strip()
            texts.append(text)
        return texts

    def decode_warmup(
            self,
            dataset,
            batch_size=256,
            sample_rate=16000,
            padding_strategy="max",
            iters=-1):

        logging.debug(f"[Device {self.device_id}] start decode_dataset for warm up")

        text_prefix = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"

        num_beams = 1

        mel_filters_dir = self.assets_dir
        count = 0
        total_duration = 0
        results = []
        for i, batch in enumerate(dataset.create_batch(batch_size)):

            count += batch_size
            waveforms, durations, texts, ids = batch
            features, durations, texts, ids = batch
            total_duration += sum(durations) / sample_rate

            for wave in waveforms:
                assert wave.is_pinned()

            if padding_strategy == "longest":
                longest_duration = max(durations)
            elif padding_strategy == "nopad":
                longest_duration = 0
            else:  # padding_strategy = "max":
                longest_duration = int(16000 * 30)

            features = [
                log_mel_spectrogram(wave,
                                    self.n_mels,
                                    padding=longest_duration - wave.shape[-1],
                                    device='cuda',
                                    mel_filters_dir=mel_filters_dir).unsqueeze(0)
                for wave in waveforms
            ]

            # pad to the even number of features, for remove_padding option, conv layer padding corner case
            for i, feature in enumerate(features):
                if feature.shape[2] % 2:
                    features[i] = torch.nn.functional.pad(feature, (0, 1))

            features_input_lengths = torch.tensor([f.shape[2] for f in features],
                                                  dtype=torch.int32,
                                                  device='cuda')

            output_ids = self.process_batch(features, features_input_lengths,
                                            text_prefix, num_beams)
            # move post process
            predictions = self.post_process(output_ids)
            for wav_id, label, prediction in zip(ids, texts, predictions):
                # remove all special tokens in the prediction
                prediction = re.sub(r'<\|.*?\|>', '', prediction)
                prediction = prediction.lower().strip()
                label = label.split()
                prediction = prediction.split()
                results.append((wav_id, label, prediction))
            # early stop if iters is set
            if iters != -1:
                iters * batch_size >= count - 1
                return results, total_duration

        return results, total_duration

    def decode_samples(
            self,
            samples,
            padding_strategy="max"):

        cudart.cudaSetDevice(self.device_id)
        actual_batch_size = len(samples)
        sample_indices = [q.index for q in samples]
        sample_ids = [q.id for q in samples]

        # assert False
        sample_rate = 16000
        num_beams = 1
        batch_size = actual_batch_size
        text_prefix = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"

        mel_filters_dir = self.assets_dir
        count = 0
        total_duration = 0
        results = []
        word_counts = []

        waveforms = [self.dataset.audio[i] for i in sample_indices]
        durations = [self.dataset.duration[i] for i in sample_indices]
        total_duration += sum(durations)

        for wave in waveforms:
            assert wave.is_pinned()

        if padding_strategy == "longest":
            longest_duration = max(durations)
        elif padding_strategy == "nopad":
            longest_duration = 0
        else:  # padding_strategy == max
            longest_duration = int(16000 * 30)

        features = [
            log_mel_spectrogram(wave,
                                self.n_mels,
                                padding=longest_duration - wave.shape[-1],  # longest_duration - wave.shape[-1],
                                device='cuda',
                                mel_filters_dir=mel_filters_dir).unsqueeze(0)
            for wave in waveforms
        ]

        # pad to the even number of features, for remove_padding option, conv layer padding corner case
        for i, feature in enumerate(features):
            if feature.shape[2] % 2:
                features[i] = torch.nn.functional.pad(feature, (0, 1))

        features_input_lengths = torch.tensor([f.shape[2] for f in features],
                                              dtype=torch.int32,
                                              device='cuda')
        # try:
        output_ids = self.process_batch(features, features_input_lengths,
                                        text_prefix, num_beams)
        predictions = self.post_process(output_ids)
        for prediction in predictions:
            # remove all special tokens in the prediction
            prediction = re.sub(r'<\|.*?\|>', '', prediction)
            prediction = prediction.lower().strip()

            # TODO for matching accuracy_eval decoding rules. Tended to remove in 6.0
            prediction = currency_to_text(prediction)
            prediction = numbers_to_text(prediction)

            word_counts.append(len(prediction.split()))
            transcript = []
            for s in prediction:
                if s in labels_dict:
                    transcript.append(labels_dict[s])
            transcript = [transcript]
            assert len(transcript) == 1

            predictions_array = array.array('B', np.array(transcript[0], np.int8).tobytes())
            results.append(predictions_array)

        response = WhisperResponse(sample_ids=sample_ids,
                                   predictions=results,
                                   word_counts=word_counts,
                                   results_ready=cudart.cudaEventCreateWithFlags(cudart.cudaEventDefault | cudart.cudaEventDisableTiming))
        self.response_queue.put(response)

    def __del__(self):
        # exit all threads
        self.response_queue.put(None)
        self.response_queue.join()
        self.response_thread.join()

    def _process_response(self):
        while True:
            response = self.response_queue.get()
            if response is None:
                # None in the queue indicates the parent want us to exit
                self.response_queue.task_done()
                break
            qsr = []
            actual_batch_size = len(response.sample_ids)
            cudart.cudaEventSynchronize(response.results_ready)
            torch.cuda.synchronize()

            with nvtx_scope("report_qsl", color='yellow'):

                for idx, sample_id in enumerate(response.sample_ids):

                    buffer_address, buffer_length = response.predictions[idx].buffer_info()
                    qsr.append(lg.QuerySampleResponse(sample_id,
                                                      buffer_address,
                                                      buffer_length * response.predictions[idx].itemsize, response.word_counts[idx]))
                    torch.cuda.synchronize()

                lg.QuerySamplesComplete(qsr)
                self.total_samples += actual_batch_size
                self.response_queue.task_done()
            cudart.cudaEventSynchronize(response.results_ready)
            torch.cuda.synchronize()

        logging.debug(f"[Device {self.device_id}] Reporting back {self.total_samples} in samples [Done]")
