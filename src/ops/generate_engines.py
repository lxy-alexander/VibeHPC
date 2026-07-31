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

# Standard library imports
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Type

# Local imports
from ..common import logging
from ..common.constants import Scenario
from ..common.workload import EngineIndex
from ..fields import gen_engines as gen_fields

# Third-party imports
from nvmitten.aliased_name import AliasedNameEnum
from nvmitten.configurator import bind, autoconfigure, Field, HelpInfo
from nvmitten.nvidia.builder import TRTBuilder, CalibratableTensorRTEngine
from nvmitten.pipeline import Operation
from nvmitten.utils import run_command


class MPS:
    """Context manager for NVIDIA Multi-Process Service (MPS) control."""

    def __init__(self, active_sms: int = 100):
        """Initialize MPS controller with specified active SM percentage.

        Args:
            active_sms (int): Percentage of SMs to make active (1-100).
        """
        assert active_sms > 0 and active_sms <= 100
        self.active_sms = active_sms

    def is_enabled(self):
        """Check if MPS service is currently running.

        Returns:
            bool: True if MPS service is running, False otherwise.
        """
        cmd = "ps -ef | grep nvidia-cuda-mps-control | grep -c -v grep"
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.wait()
        output = p.stdout.readlines()
        return int(output[0]) >= 1

    def disable(self):
        """Stop the MPS service if it is running."""
        if self.is_enabled():
            cmd = "echo quit | nvidia-cuda-mps-control"
            run_command(cmd)

    def enable(self):
        """Start the MPS service with configured active SM percentage."""
        self.disable()
        if self.active_sms == 100:
            return

        cmd = f"export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={self.active_sms} && nvidia-cuda-mps-control -d"
        run_command(cmd)

    def __enter__(self):
        """Enable MPS when entering context."""
        return self.enable()

    def __exit__(self, *args):
        """Disable MPS when exiting context."""
        self.disable()


@autoconfigure
class CalibrateEngineOp(Operation):
    """Operation for calibrating TensorRT engines that require calibration."""

    @classmethod
    def immediate_dependencies(cls):
        """Get the immediate dependencies of this operation.

        Returns:
            set: Set of operation classes that this operation depends on.
        """
        return None

    def __init__(self, *args, **kwargs):
        """Initialize the calibration operation."""
        super().__init__(*args, **kwargs)

        self.engine_index = EngineIndex()

    def run(self, scratch_space, dependency_outputs):
        """Run the calibration operation.

        Args:
            scratch_space: The scratch space for temporary files.
            dependency_outputs: Outputs from dependency operations.
        """
        for c_eng, builder_cls in self.engine_index.iter_components():
            if not issubclass(builder_cls, CalibratableTensorRTEngine):
                continue
            builder = builder_cls(batch_size=c_eng.batch_size)
            if builder.need_calibration:
                builder.set_calibrator(scratch_space.path.joinpath("preprocessed_data",
                                                                   builder.calib_data_dir))
                builder.create_profiles = builder.calibration_profiles
                network = builder.create_network(builder.builder)
                builder(1, Path("/dev/null"), network)
            else:
                logging.info(f"Calibration not needed for {c_eng.component}, skipping.")

HelpInfo.add_configurator_dependency(CalibrateEngineOp, EngineIndex)


@autoconfigure
@bind(gen_fields.active_sms)
@bind(gen_fields.no_child_process, "disallow_mps")
@bind(gen_fields.force_build_engines)
class EngineBuilderOp(Operation):
    """Operation for building TensorRT engines with optional MPS (Multi-Process Service) support."""

    @classmethod
    def immediate_dependencies(cls):
        """Get the immediate dependencies of this operation.

        Returns:
            set: Set of operation classes that this operation depends on.
        """
        return {CalibrateEngineOp}

    @classmethod
    def output_keys(cls):
        """Get the output keys produced by this operation.

        Returns:
            list: List of output keys.
        """
        return ["engine_index"]

    def __init__(self,
                 *args,
                 active_sms: int = 100,
                 force_build_engines: bool = False,
                 disallow_mps: bool = False,
                 **kwargs):
        """Initialize the engine builder operation.

        Args:
            active_sms (int): Percentage of SMs to make active in MPS mode.
            force_build_engines (bool): Whether to rebuild existing engines.
            disallow_mps (bool): Whether to disable MPS functionality.
        """
        super().__init__(*args, **kwargs)

        self.active_sms = active_sms
        self.engine_index = EngineIndex()
        self.use_mps = (self.engine_index.wl.scenario == Scenario.Server) and (active_sms < 100) and not disallow_mps
        self.force_build_engines = force_build_engines

    def run(self, scratch_space, dependency_outputs):
        """Run the engine building operation.

        Args:
            scratch_space: The scratch space for temporary files.
            dependency_outputs: Outputs from dependency operations.

        Returns:
            dict: Dictionary containing the engine index.
        """
        if self.use_mps:
            mps_scope = MPS(active_sms=self.active_sms)
        else:
            mps_scope = nullcontext()

        ts = [(time.time(), None)]
        with mps_scope:
            for c_eng, builder_cls in self.engine_index.iter_components():
                eng_fpath = self.engine_index.full_path(c_eng)
                if eng_fpath.exists():
                    if self.force_build_engines:
                        logging.info(f"{c_eng.component}: Existing engine will be overwritten at {eng_fpath}")
                    else:
                        logging.info(f"{c_eng.component}: Engine already exists at {eng_fpath}. Skipping.")
                        continue

                builder = builder_cls(batch_size=c_eng.batch_size)
                if isinstance(builder, CalibratableTensorRTEngine):
                    builder.set_calibrator(scratch_space.path.joinpath("preprocessed_data",
                                                                       builder.calib_data_dir))
                network = builder.create_network(builder.builder)
                builder(c_eng.batch_size, eng_fpath, network)
                ts.append((time.time(), c_eng.component))

        n_engines = len(ts) - 1
        total_duration = ts[-1][0] - ts[0][0]
        logging.info(f"Built {n_engines} component engines in {total_duration:.2f} seconds.")
        for i in range(n_engines):
            end_event = ts[i+1][1]
            duration = ts[i+1][0] - ts[i][0]
            logging.info(f"  {end_event}: {duration:.2f}s")

        return {"engine_index": self.engine_index}

HelpInfo.add_configurator_dependency(EngineBuilderOp, EngineIndex)
