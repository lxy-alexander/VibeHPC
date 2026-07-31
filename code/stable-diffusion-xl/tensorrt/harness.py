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

import contextlib
from typing import List

from nvmitten.configurator import autoconfigure
from nvmitten.nvidia.accelerator import GPU
import mlperf_loadgen as lg

from code.common import logging
from code.common.constants import Scenario
from code.common.systems.system_list import DETECTED_SYSTEM
from code.ops.harness import PyHarnessOp
from code.ops.loadgen import LoadgenConfFilesOp
from code.ops.generate_engines import EngineBuilderOp

from .dataset import Dataset
from .backend import SDXLServer


@autoconfigure
class SDXLHarnessOp(PyHarnessOp):
    """SDXL harness."""

    @classmethod
    def immediate_dependencies(cls):
        return {LoadgenConfFilesOp, EngineBuilderOp}

    @classmethod
    def output_keys(cls):
        return ["log_dir", "result_metadata"]

    def __init__(self, *args, **kwargs):
        super().__init__(Dataset, *args, **kwargs)

        self._server_inst = None

    def issue_queries(self, query_samples: List[lg.QuerySample]):
        self._server_inst.issue_queries(query_samples)

    def flush_queries(self):
        self._server_inst.flush_queries()

    @contextlib.contextmanager
    def wrap_lg_test(self, scratch_space, dependency_outputs):
        engine_index = dependency_outputs[EngineBuilderOp]["engine_index"]
        devices = [gpu.gpu_index for gpu in DETECTED_SYSTEM.accelerators[GPU]]

        engines = [(c_eng, engine_index.full_path(c_eng)) for c_eng in engine_index.engines]

        try:
            self._server_inst = SDXLServer(
                devices=devices,
                dataset=self._qsl_inst,
                engines=engines,
                gpu_inference_streams=1,  # Change this when SDXL supports multiple cores per device
                gpu_copy_streams=1,
                enable_batcher=(engine_index.wl.scenario == Scenario.Server))
            logging.info("Start Warm Up!")
            self._server_inst.warm_up()
            logging.info("Warm Up Done!")
            yield None
        finally:
            self._server_inst.finish_test()
