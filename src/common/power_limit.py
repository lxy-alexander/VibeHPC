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


import subprocess
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass
from nvmitten.configurator import bind, autoconfigure, Configuration
from nvmitten.nvidia.accelerator import GPU, DLA
from pathlib import Path

import code.fields.power as powerfields

from code.common import logging, run_command
from code.common.systems.system_list import DETECTED_SYSTEM
from code.common.nvpmodel_orin import nvpmodel_template_orin, nvpmodel_template_orin_nx, cpu_clock_str


class PowerState:
    def activate(self): pass
    def deactivate(self): pass

    @contextmanager
    def ctx(self):
        try:
            self.activate()
            yield None
        finally:
            self.deactivate()


@autoconfigure
@bind(powerfields.power_limit, "power_limit")
@bind(powerfields.cpu_freq, "cpu_freq")
class ServerPowerState(PowerState):
    def __init__(self,
                 power_limit: Optional[int] = None,
                 cpu_freq: Optional[int] = None):
        self.power_limit = power_limit
        self.cpu_freq = cpu_freq
        self._saved_state = None

    def set_cpufreq(self):
        # Record current cpu governor
        cmd = "sudo cpupower -c all frequency-set -g userspace"
        logging.info(f"Set cpu power governor: userspace")
        run_command(cmd)

        # Set cpu freq
        cmd = f"sudo cpupower -c all frequency-set -f {self.cpu_freq}"
        logging.info(f"Setting cpu frequency: {cmd}")
        run_command(cmd)

    def reset_cpufreq(self):
        # Record current cpu governor
        cmd = "sudo cpupower -c all frequency-set -g ondemand"
        logging.info(f"Set cpu power governor: ondemand")
        run_command(cmd)

    def activate(self):
        if self._saved_state is not None:
            logging.info("Power state is already active.")
            return

        # Record current power limits.
        self._saved_state = dict()
        if self.power_limit:
            cmd = "nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits"
            logging.info(f"Getting current GPU power limits: {cmd}")
            output = run_command(cmd, get_output=True, tee=False)
            self._saved_state["gpu_limits"] = [float(line) for line in output]

            # Set power limit to the specified value.
            cmd = f"sudo nvidia-smi -pl {self.power_limit}"
            logging.info(f"Setting current GPU power limits: {cmd}")
            run_command(cmd)

        if self.cpu_freq:
            self.set_cpufreq()

    def deactivate(self):
        if self._saved_state is None:
            logging.info("Power state is not active.")
            return

        # Reset power limit to the specified value.
        if "gpu_limits" in self._saved_state:
            power_limits = self._saved_state["gpu_limits"]
            for i in range(len(power_limits)):
                cmd = f"sudo nvidia-smi -i {i} -pl {power_limits[i]}"
                logging.info(f"Resetting power limit for GPU {i}: {cmd}")
                run_command(cmd)
                time.sleep(1)
        self._saved_state = None


@autoconfigure
@bind(powerfields.soc_gpu_freq, "gpu_freq")
@bind(powerfields.soc_dla_freq, "dla_freq")
@bind(powerfields.soc_cpu_freq, "cpu_freq")
@bind(powerfields.soc_emc_freq, "emc_freq")
@bind(powerfields.soc_pva_freq, "pva_freq")
@bind(powerfields.orin_num_cores, "n_cpu_cores")
@bind(powerfields.orin_skip_maxq_reset, "no_reset")
class OrinPowerState(PowerState):
    def __init__(self,
                 gpu_freq: Optional[int] = None,
                 dla_freq: Optional[int] = None,
                 cpu_freq: Optional[int] = None,
                 emc_freq: Optional[int] = None,
                 pva_freq: Optional[int] = None,
                 n_cpu_cores: Optional[int] = None,
                 no_reset: bool = False):
        def _assert_is_set(x):
            assert x is not None
            return x

        self.gpu_freq = _assert_is_set(gpu_freq)
        self.dla_freq = _assert_is_set(dla_freq)
        self.cpu_freq = _assert_is_set(cpu_freq)
        self.emc_freq = _assert_is_set(emc_freq)
        self.pva_freq = _assert_is_set(pva_freq)
        self.n_cpu_cores = _assert_is_set(n_cpu_cores)
        self.no_reset = no_reset

    def activate(self):
        logging.info(f"Setting power state on Orin")

        if DETECTED_SYSTEM.extras["id"] == "Orin":
            num_cores_total = 12
            template = nvpmodel_template_orin
        elif DETECTED_SYSTEM.extras["id"] == "Orin_NX":
            num_cores_total = 8
            template = nvpmodel_template_orin_nx
        else:
            raise ValueError(f"Unrecognized Orin System {DETECTED_SYSTEM.extras['id']}!")

        cores_map = [1] * num_cores_total  # all cores on by default
        if self.n_cpu_cores:
            assert num_cores_total >= self.n_cpu_cores, f"{num_cores_total} available, but {self.n_cpu_cores} requested"
            diff = num_cores_total - self.n_cpu_cores
            cores_map[-diff:] = [0] * diff

        kwargs = {"gpu_clock": self.gpu_freq,
                  "dla_clock": self.dla_freq,
                  "dla_falcon_clock": self.dla_freq // 2,
                  "cpu_clock_str": cpu_clock_str(self.n_cpu_cores, self.cpu_freq),
                  "emc_clock": self.emc_freq,
                  "pva_axi_clock": self.pva_freq // 2,
                  "pva_clock": self.pva_freq,
                  "cpu_core": cores_map}

        with Path("build/nvpmodel.temp.conf").open(mode='w') as f:
            f.write(template.format(**kwargs))

        cmd = "sudo /usr/sbin/nvpmodel -f build/nvpmodel.temp.conf -m 0"
        # Ignore error because disabling CPU cores makes nvpmodel command return 255, even though they
        # do get disabled
        logging.info(f"Setting current nvpmodel conf: {cmd}")
        run_command(cmd)

    def deactivate(self):
        if not self.no_reset:
            cmd = "sleep 5; sudo /usr/sbin/nvpmodel -f /etc/nvpmodel.conf -m 0"
            logging.info(f"Resetting nvpmodel conf: {cmd}")
            run_command(cmd)


def get_power_context():
    if "is_orin" in DETECTED_SYSTEM.extras["tags"]:
        return OrinPowerState().ctx()

    gpus = DETECTED_SYSTEM.accelerators[GPU]
    if len(gpus) > 0 and not gpus[0].is_integrated:
        return ServerPowerState().ctx()

    return nullcontext
