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


from typing import List

from code.ops.harness import PyHarnessOp
from code.ops.loadgen import LoadgenConfFilesOp
from code.common.mlcommons.runner import ScopedQSL
from code.common.gbs import GeneralizedBatchSize
from code.common.systems.system_list import DETECTED_SYSTEM

from code.fields import harness as harness_fields

# pylint: disable=c-extension-no-member
import mlperf_loadgen as lg
from nvmitten.configurator import autoconfigure, bind
from nvmitten.nvidia.accelerator import GPU

from .rgat import RGATOfflineServer
from .config import RGATConfig
from .fields import rgat_buffer_size


#pylint: disable=invalid-name
EngineBuilderOp = None
CalibrateEngineOp = None


@autoconfigure
@bind(harness_fields.complete_threads, "num_complete_threads")
@bind(rgat_buffer_size, "buffer_size")
class RGATHarnessOp(PyHarnessOp):
    """RGAT Harness Operation"""

    @classmethod
    def immediate_dependencies(cls):
        """Get the immediate dependencies of this operation.

        Returns:
            set: Set of operation classes that this operation depends on.
        """
        return {LoadgenConfFilesOp}

    @classmethod
    def output_keys(cls):
        """Get the output keys produced by this operation.

        Returns:
            list: List of output keys.
        """
        return ["log_dir", "result_metadata"]

    def __init__(self,
                 *args,
                 # This number is obtained from: torch.load("/path/to/val_idx.pt").size()
                 total_sample_count: int = 788379,
                 buffer_size: int = 788379 * 2,
                 num_complete_threads: int = 128,
                 **kwargs):
        super().__init__(ScopedQSL, *args, total_sample_count=total_sample_count, **kwargs)

        self.devices = [gpu.gpu_index for gpu in DETECTED_SYSTEM.accelerators[GPU]]
        bs_info = GeneralizedBatchSize()
        self.batch_size = bs_info.get("rgat")
        self.server = RGATOfflineServer(self.devices,
                                        RGATConfig(batch_size=self.batch_size),
                                        ds_size=total_sample_count,
                                        delegator_max_size=buffer_size,
                                        n_complete_threads=num_complete_threads)

    def issue_queries(self, query_samples: List[lg.QuerySample]):
        """Issue queries to the SUT.

        Args:
            query_samples (List[lg.QuerySample]): List of query samples to issue.
        """
        self.server.issue_queries(query_samples)

    def flush_queries(self):
        """Flush queries from the SUT.

        Args:
            query_samples (List[lg.QuerySample]): List of query samples to flush.
        """
        self.server.flush_queries()


BenchmarkHarnessOp = RGATHarnessOp
