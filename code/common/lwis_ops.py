# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
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

from typing import Any, Iterable, Dict
from datetime import timedelta

import time
import ctypes

from cuda import cudart
from code.common import run_command, logging
from nvmitten.mlcommons.inference.ops import LoadgenSUT, LoadgenQSL, LoadgenWorkload
from nvmitten.importer import import_from

lwis_api = import_from(["build/harness/lib"], "lwis_api")

try:
    import mlperf_loadgen as lg
except:
    logging.warning("Loadgen Python bindings are not installed. Installing Loadgen Python bindings!")
    run_command("make build_loadgen")
    import mlperf_loadgen as lg


class LwisWorkload(LoadgenWorkload):
    """LWIS Loadgen Workload. Used for RN50 and RetinaNet."""

    def __init__(self, flag_dict: Dict[str, Any]):
        super().__init__(flag_dict)
        self.flag_dict = flag_dict
        self.flag_dict["devices"] = flag_dict.get("devices", "all")
        self.flag_dict["dla_core"] = flag_dict.get("dla_core", -1)
        self.flag_dict["use_spin_wait"] = flag_dict.get("use_spin_wait", False)
        self.flag_dict["use_device_schedule_spin"] = flag_dict.get("use_device_schedule_spin", False)
        self.flag_dict["coalesced_tensor"] = flag_dict.get("coalesced_tensor", False)
        self.flag_dict["use_direct_host_access"] = flag_dict.get("use_direct_host_access", False)
        self.flag_dict["use_deque_limit"] = flag_dict.get("use_deque_limit", False)
        self.flag_dict["deque_timeout_usec"] = flag_dict.get("deque_timeout_usec", 10000)
        self.flag_dict["use_batcher_thread_per_device"] = flag_dict.get("use_batcher_thread_per_device", False)
        self.flag_dict["use_cuda_thread_per_device"] = flag_dict.get("use_cuda_thread_per_device", False)
        self.flag_dict["assume_contiguous"] = flag_dict.get("assume_contiguous", False)
        self.flag_dict["use_same_context"] = flag_dict.get("use_same_context", False)
        self.flag_dict["gpu_copy_streams"] = flag_dict.get("gpu_copy_streams", 4)
        self.flag_dict["gpu_inference_streams"] = flag_dict.get("gpu_inference_streams", 1)
        self.flag_dict["gpu_batch_size"] = flag_dict.get("gpu_batch_size", 256)
        self.flag_dict["gpu_engine_batch_size"] = flag_dict.get("gpu_engine_batch_size", "")
        self.flag_dict["dla_copy_streams"] = flag_dict.get("dla_copy_streams", 4)
        self.flag_dict["dla_inference_streams"] = flag_dict.get("dla_inference_streams", 1)
        self.flag_dict["dla_batch_size"] = flag_dict.get("dla_batch_size", 32)
        self.flag_dict["dla_engine_batch_size"] = flag_dict.get("dla_engine_batch_size", "")
        self.flag_dict["max_dlas"] = flag_dict.get("max_dlas", 2)
        self.flag_dict["run_infer_on_copy_streams"] = flag_dict.get("run_infer_on_copy_streams", False)
        self.flag_dict["complete_threads"] = flag_dict.get("complete_threads", 1)
        self.flag_dict["warmup_duration"] = flag_dict.get("warmup_duration", 5.0)
        self.flag_dict["response_postprocess"] = flag_dict.get("response_postprocess", "")
        self.flag_dict["performance_sample_count"] = flag_dict.get("performance_sample_count", 0)
        self.flag_dict["end_on_device"] = flag_dict.get("end_on_device", False)
        self.flag_dict["eviction_last"] = flag_dict.get("eviction_last", 0.0)
        self.flag_dict["model"] = flag_dict.get("model", "resnet50")
        self.flag_dict["logfile_prefix_with_datetime"] = flag_dict.get("logfile_prefix_with_datetime", False)
        self.flag_dict["log_copy_detail_to_stdout"] = flag_dict.get("log_copy_detail_to_stdout", False)
        self.flag_dict["disable_log_copy_summary_to_stdout"] = flag_dict.get("disable_log_copy_summary_to_stdout", False)
        self.flag_dict["log_mode_async_poll_interval_ms"] = flag_dict.get("log_mode_async_poll_interval_ms", 1000)

        # Initialize Google Logging for C++ classes
        lwis_api.init_glog(flag_dict["verbose"])

        # Load all the needed shared objects for plugins.
        for s in self.flag_dict["plugins"].split(","):
            try:
                ctypes.CDLL(s)
            except OSError as e:
                logging.error(f"Error loading plugin library {s}: {e}")
                raise RuntimeError

    def setup(self):
        super().setup()
        self.test_settings.server_coalesce_queries = True

        self.sut_wrapper = LwisSUT()
        self.qsl_wrapper = LwisQSL(flag_dict=self.flag_dict, test_settings=self.test_settings, sut=self.sut_wrapper.sut)

        self.sut_wrapper.setup(flag_dict=self.flag_dict, test_settings=self.test_settings)
        self.qsl_wrapper.setup()

    def start_test(self):
        # Perform the inference testing
        logging.info("Starting running actual test.")
        cudart.cudaProfilerStart()
        lwis_api.start_test(self.sut_wrapper.sut, self.qsl_wrapper.qsl, self.test_settings, self.log_settings)
        cudart.cudaProfilerStop()
        logging.info("Finished running actual test.")

        lwis_api.log_device_stats(self.sut_wrapper.sut)

    def run(self):
        self.setup()
        self.sut_wrapper.start(self.flag_dict)
        self.start_test()
        self.sut_wrapper.stop()

        lwis_api.reset(self.qsl_wrapper.qsl)
        lwis_api.reset(self.sut_wrapper.sut)


class LwisQSL(LoadgenQSL):
    """LWIS QSL Python wrapper. Used for RN50 and RetinaNet."""

    def __init__(self, flag_dict: Dict[str, Any], test_settings: lg.TestSettings, sut: lwis_api.Server):

        # Instantiate our QSL
        logging.info("Creating QSL.")
        tensor_paths = flag_dict["tensor_path"].split(",")
        start_from_device = [flag_dict["start_from_device"]] * len(tensor_paths)
        numa_config = lwis_api.parse_numa_config(flag_dict["numa_config"])

        if test_settings.scenario == lg.TestScenario.MultiStream:
            padding = test_settings.multi_stream_samples_per_query - 1
        else:
            padding = 0

        if len(numa_config) == 0:
            one_qsl = lwis_api.SampleLibrary("LWIS_SampleLibrary",
                                             flag_dict["map_path"],
                                             flag_dict["tensor_path"].split(","),
                                             flag_dict["performance_sample_count"] if flag_dict["performance_sample_count"]
                                             else max(flag_dict["gpu_batch_size"], flag_dict["dla_batch_size"]),
                                             padding,
                                             flag_dict["coalesced_tensor"],
                                             start_from_device)
            sut.add_sample_library(one_qsl)
            self.qsl = one_qsl
        else:
            # When NUMA is used, create one QSL per NUMA node.
            logging.info("Using NUMA. Config: " + flag_dict["numa_config"])
            self.qsl = lwis_api.create_qsl_per_numa_node(sut,
                                                         numa_config,
                                                         flag_dict["map_path"],
                                                         flag_dict["tensor_path"].split(","),
                                                         flag_dict["performance_sample_count"] if flag_dict["performance_sample_count"]
                                                         else max(flag_dict["gpu_batch_size"], flag_dict["dla_batch_size"]),
                                                         padding,
                                                         flag_dict["coalesced_tensor"],
                                                         start_from_device)
        logging.info("Finished Creating QSL.")

    def setup(self):
        pass

    def load_query_samples(self, sample_list: Iterable[int]):
        return self.qsl.load_samples_to_ram(sample_list)

    def unload_query_samples(self, sample_list: Iterable[int]):
        return self.qsl.unload_samples_from_ram(sample_list)


class LwisSUT(LoadgenSUT):
    """LWIS SUT Python wrapper. Used for RN50 and RetinaNet."""

    def __init__(self):

        # Instantiate and configure our SUT
        self.sut = lwis_api.Server("LWIS_Server")
        self.sut_settings = lwis_api.ServerSettings()
        self.sut_params = lwis_api.ServerParams()

    def issue_query(self, query_samples: Iterable[Any]):
        self.sut.issue_query(query_samples)

    def flush_queries(self):
        self.sut.flush_queries()

    def setup(self, flag_dict: Dict[str, Any], test_settings: lg.TestSettings):

        self.sut_settings.enable_cuda_graphs = flag_dict["use_graphs"]
        self.sut_settings.gpu_copy_streams = flag_dict["gpu_copy_streams"]
        self.sut_settings.gpu_infer_streams = flag_dict["gpu_inference_streams"]

        # gpu batch size settings
        gpu_e2e_batch_size = flag_dict["gpu_batch_size"]
        gpu_engine_batch_size_list = flag_dict["gpu_engine_batch_size"]
        if len(gpu_engine_batch_size_list) == 0:
            logging.error("Empty gpu_engine_batch_size or missing gpu_engine_batch_size input argument")
            raise RuntimeError

        gpu_batch_sizes = []
        gpu_loop_counts = []
        for gpu_engine_batch_size_str in gpu_engine_batch_size_list.split(","):
            gpu_engine_batch_size = int(gpu_engine_batch_size_str)
            if gpu_e2e_batch_size % gpu_engine_batch_size:
                logging.error(f"E2E GPU batch size {gpu_e2e_batch_size} is not \
                              divisible by GPU engine batch size {gpu_engine_batch_size}")
                raise RuntimeError
            gpu_batch_sizes.append(gpu_engine_batch_size)
            gpu_loop_counts.append(gpu_e2e_batch_size // gpu_engine_batch_size)

        self.sut_settings.gpu_batch_sizes = gpu_batch_sizes
        self.sut_settings.gpu_loop_counts = gpu_loop_counts

        if flag_dict["dla_core"] != -1:
            self.sut_settings.max_gpus = 0
            self.sut_settings.max_dlas = 1  # no interface to specify which DLA
        else:
            self.sut_settings.max_dlas = flag_dict["max_dlas"]

        if self.sut_settings.max_dlas > 0:
            self.sut_settings.dla_copy_streams = flag_dict["dla_copy_streams"]
            self.sut_settings.dla_infer_streams = flag_dict["dla_inference_streams"]

            dla_e2e_batch_size = flag_dict["dla_batch_size"]
            dla_engine_batch_size_list = flag_dict["dla_engine_batch_size"]
            if len(dla_engine_batch_size_list) == 0:
                logging.error("Empty dla_engine_batch_size or missing dla_engine_batch_size input argument")
                raise RuntimeError

            dla_batch_sizes = []
            dla_loop_counts = []
            for dla_engine_batch_size_str in dla_engine_batch_size_list.split(","):
                dla_engine_batch_size = int(dla_engine_batch_size_str)
                if dla_e2e_batch_size % dla_engine_batch_size:
                    logging.error(f"E2E DLA batch size {dla_e2e_batch_size} is not \
                                  divisible by DLA engine batch size {dla_engine_batch_size}")
                    raise RuntimeError
                dla_batch_sizes.append(dla_engine_batch_size)
                dla_loop_counts.append(dla_e2e_batch_size // dla_engine_batch_size)
            self.sut_settings.dla_batch_sizes = dla_batch_sizes
            self.sut_settings.dla_loop_counts = dla_loop_counts

        self.sut_settings.enable_spin_wait = flag_dict["use_spin_wait"]
        self.sut_settings.enable_device_schedule_spin = flag_dict["use_device_schedule_spin"]
        self.sut_settings.run_infer_on_copy_streams = flag_dict["run_infer_on_copy_streams"]
        self.sut_settings.enable_direct_host_access = flag_dict["use_direct_host_access"]
        self.sut_settings.enable_deque_limit = flag_dict["use_deque_limit"]
        self.sut_settings.timeout = timedelta(microseconds=flag_dict["deque_timeout_usec"])
        self.sut_settings.enable_batcher_thread_per_device = flag_dict["use_batcher_thread_per_device"]
        self.sut_settings.enable_cuda_thread_per_device = flag_dict["use_cuda_thread_per_device"]
        self.sut_settings.enable_start_from_device_mem = flag_dict["start_from_device"]
        self.sut_settings.force_contiguous \
            = flag_dict["assume_contiguous"] or (test_settings.scenario == lg.TestScenario.MultiStream)
        self.sut_settings.complete_threads = flag_dict["complete_threads"]
        self.sut_settings.use_same_context = flag_dict["use_same_context"]
        self.sut_settings.numa_config = lwis_api.parse_numa_config(flag_dict["numa_config"])
        self.sut_settings.gpu_to_numa_map = lwis_api.get_gpu_to_numa_map(self.sut_settings.numa_config)
        self.sut_settings.verbose_nvtx = flag_dict["verbose_nvtx"]
        self.sut_settings.end_on_device = flag_dict["end_on_device"]
        self.sut_settings.el_ratio = flag_dict["eviction_last"]

        self.sut_params.device_names = flag_dict["devices"]

        engine_names = [[], []]
        for engine_name in flag_dict["gpu_engines"].split(","):
            if engine_name == "":
                continue
            engine_names[0].append(engine_name)

        for engine_name in flag_dict["dla_engines"].split(","):
            if engine_name == "":
                continue
            engine_names[1].append(engine_name)
        self.sut_params.engine_names = engine_names

        logging.info("Setting up SUT.")
        self.sut.setup(self.sut_settings, self.sut_params)  # Pass the requested sut settings and params to our SUT
        self.sut.set_response_callback(lwis_api.get_callback_map(flag_dict["response_postprocess"]))    # Set QuerySampleResponse post-processing callback

        logging.info("Finished setting up SUT.")

    def start(self, flag_dict: Dict[str, Any]):
        logging.info(f'Starting warmup. Running for a minimum of {flag_dict["warmup_duration"]} seconds.')
        t_start = time.perf_counter()
        self.sut.warmup(flag_dict["warmup_duration"])
        elapsed = time.perf_counter() - t_start
        logging.info(f"Finished warmup. Ran for {elapsed} s.")

    def stop(self):
        self.sut.done()
