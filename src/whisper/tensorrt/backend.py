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

import threading
import queue
import time
from typing import List

from code.common import logging
from code.common.workload import ComponentEngine
from code.fields import general as general_fields
from code.fields import harness as harness_fields

from nvmitten.configurator import autoconfigure, bind

from .constants import WhisperComponent
from .whisper_models import WhisperTRTLLMCore

import os
loggers = ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"]
os.environ['TLLM_LOG_LEVEL'] = loggers[4]


@autoconfigure
@bind(general_fields.verbose)
@bind(harness_fields.use_graphs)
# Enable these when Whisper supports multiple cores per device
# @bind(harness_fields.gpu_inference_streams)
# @bind(harness_fields.gpu_copy_streams)
class WhisperServer:
    def __init__(self,
                 devices: List[int],
                 dataset: Dataset,
                 engines: List[Tuple[ComponentEngine, str]],
                 assets_dir: str,
                 use_graphs: bool = False,
                 verbose: bool = False,
                 enable_batcher: bool = False,
                 batch_timeout_threshold: float = -1):

        self.dataset = dataset
        self.devices = devices
        self.verbose = verbose
        self.enable_batcher = enable_batcher and batch_timeout_threshold > 0
        self.assets_dir = assets_dir
        # Server components
        self.sample_queue = queue.Queue()  # sample sync queue
        self.sample_count = 0
        self.whisper_cores = {}
        self.core_threads = []

        assert len(engines) == 2, "Whisper requires 2 engine per component (Components: Encoder, Decoder)"
        self.component_info = dict()
        for c_eng, fpath in engines:
            assert isinstance(c_eng.component, WhisperComponent), f"Whisper component of unexpected type {type(c_eng.component)} - Is the EngineIndex parsing correctly?"
            self.component_info[c_eng.component] = {
                "engine_path": fpath,
                "batch_size": c_eng.batch_size,
            }
            self.engine_dir = fpath
        # Validate batch size
        assert self.component_info[WhisperComponent.ENCODER]["batch_size"] == self.component_info[WhisperComponent.DECODER]["batch_size"]

        self.batch_size = self.component_info[WhisperComponent.ENCODER]["batch_size"]
        self.component_info["batch_size"] = self.batch_size  # Passthrough to WhisperCore

        # Initialize the cores
        for device_id in self.devices:

            self.whisper_cores[device_id] = WhisperTRTLLMCore(
                engine_dir=self.engine_dir,
                device_id=device_id,
                dataset=self.dataset,
                use_graphs=use_graphs,
                debug_mode=False,
                assets_dir=self.assets_dir,
                batch_size=self.component_info[WhisperComponent.ENCODER]["batch_size"],
                verbose=self.verbose)

        # Start the cores
        for device_id in self.devices:
            thread = threading.Thread(target=self.process_samples, args=(device_id,))
            thread.daemon = True
            self.core_threads.append(thread)
            thread.start()
        if self.enable_batcher:
            self.batcher_threshold = batch_timeout_threshold  # maximum seconds to form a batch
            self.batcher_queue = queue.SimpleQueue()  # batcher sync queue
            self.batcher_thread = threading.Thread(target=self.batch_samples, args=())
            self.batcher_thread.start()

    def warm_up(self):

        logging.info(f"Start warmp up {self.devices}")
        for device_id in self.devices:
            _, _ = self.whisper_cores[device_id].decode_warmup(
                dataset=self.dataset,
                batch_size=self.batch_size,
                padding_strategy="max",
                iters=10)

    def process_samples(self, device_id):
        count = 0
        while True:
            samples = self.sample_queue.get()
            if samples is None:
                # None in the queue indicates the SUT want us to exit
                self.sample_queue.task_done()
                break
            count = count + len(samples)
            logging.debug(f"Process_samples:{len(samples)}")
            self.whisper_cores[device_id].decode_samples(
                samples=samples,
                padding_strategy="max")
            self.sample_queue.task_done()

    def batch_samples(self):
        batched_samples = self.batcher_queue.get()
        timeout_stamp = time.time()
        while True:
            if len(batched_samples) != 0 and (len(batched_samples) >= self.batch_size or time.time() - timeout_stamp >= self.batcher_threshold):  # max batch or time limit exceed
                logging.info(f"Formed batch of {len(batched_samples[:self.batch_size])} samples")
                self.sample_queue.put(batched_samples[:self.batch_size])
                batched_samples = batched_samples[self.batch_size:]
                timeout_stamp = time.time()

            try:
                samples = self.batcher_queue.get(timeout=self.batcher_threshold)
            except queue.Empty:
                continue

            if samples is None:  # None in the queue indicates the SUT want us to exit
                break
            batched_samples += samples

    def issue_queries(self, query_samples):
        num_samples = len(query_samples)
        logging.info(f"[Server] issue_queries {num_samples} samples")

        self.sample_count += num_samples

        for i in range(0, num_samples, self.batch_size):
            # Construct batches
            actual_batch_size = self.batch_size if num_samples - i > self.batch_size else num_samples - i
            if self.enable_batcher:
                self.batcher_queue.put(query_samples[i: i + actual_batch_size])
            else:
                self.sample_queue.put(query_samples[i: i + actual_batch_size])

    def flush_queries(self):
        pass

    def finish_test(self):
        # exit all threads
        logging.info(f"SUT finished!")
        logging.info(f"[Server] Received {self.sample_count} total samples")
        for _ in self.core_threads:
            self.sample_queue.put(None)
        self.sample_queue.join()
        if self.enable_batcher:
            self.batcher_queue.put(None)
            self.batcher_thread.join()
        for device_id in self.devices:
            logging.info(f"[Device {device_id}] Reported {self.whisper_cores[device_id].total_samples} samples")
        for thread in self.core_threads:
            thread.join()
