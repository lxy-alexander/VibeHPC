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

import os

from nvmitten.configurator import autoconfigure, bind, Configuration
from nvmitten.constants import ByteSuffix
from nvmitten.interval import NumericRange
from nvmitten.json_utils import load

try:
    from nvmitten.nvidia.accelerator import GPU, DLA
except:
    # This will only happen if CUDA isn't installed on the system / if no GPUs are present
    import dataclasses
    @dataclasses.dataclass
    class GPU: pass

    @dataclasses.dataclass
    class DLA: pass

from nvmitten.system.component import Description
from nvmitten.system.system import System, NUMANode
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Set

from code.common import logging
from code.fields import general as general_fields

from .known_hardware import *


# Dynamically build Enum for known systems
_system_confs = dict()


def add_systems(name_format_string: str,
                id_format_string: str,
                cpu: KnownCPU,
                accelerator: KnownGPU,
                accelerator_counts: List[int],
                mem_requirement: Memory,
                target_dict: Dict[str, Description] = _system_confs,
                tags: List[str] = None,
                n_dlas: int = 0):
    """Adds a Description to a dictionary.

    Args:
        name_format_string (str): A Python format to generate the name for the Enum member. Can have a single format
                                  item to represent the count.
        id_format_string (str): A Python format to generate the system ID to use. The system ID is used for the systems/
                                json file. Can contain a single format item to represent the count.
        cpu (KnownCPU): The CPU that the system uses
        accelerator (KnownGPU): The Accelerator that the system uses
        accelerator_counts (List[int]): The list of various counts to use for accelerators.
        mem_requirement (Memory): The minimum memory requirement to have been tested for the hardware configuration.
        target_dict (Dict[str, Description]): The dictionary to add the Description to.
                                              (Default: _system_confs)
        tags (List[str]): A list of strings denoting certain tags used for classifying the system. (Default: None)
        n_dlas (int): The number of DLAs present on the system (Default: 0)
    """
    def _mem_cmp(m):
        thresh = NumericRange(mem_requirement._num_bytes * 0.95)
        return thresh.contains_numeric(m.capacity._num_bytes)

    for count in accelerator_counts:
        def _accelerator_cmp(count=0):
            def _f(d):
                # Check GPUs
                if GPU not in d:
                    return False

                if len(d[GPU]) != count:
                    return False

                for i in range(count):
                    if not accelerator.matches(d[GPU][i]):
                        return False

                # Check DLAs
                if len(d[DLA]) != n_dlas:
                    return False
                return True
            return _f

        k = name_format_string.format(count)
        v = Description(System,
                        _match_ignore_fields=["extras"],
                        cpu=cpu,
                        host_memory=_mem_cmp,
                        accelerators=_accelerator_cmp(count=count),
                        extras={"id": id_format_string.format(count),
                                "tags": set(tags) if tags else set()})

        target_dict[k] = v


# Blackwell Systems
add_systems("B200_SXM_180GBx{}",
            "B200-SXM-180GBx{}",
            KnownCPU.x86_64_Generic,
            KnownGPU.B200_SXM_180GB,
            [1, 2, 4, 8],
            Memory(100, ByteSuffix.GiB),
            tags=["start_from_device_enabled"])
add_systems("B300_SXM_270GBx{}",
            "B300-SXM-270GBx{}",
            KnownCPU.x86_64_Generic,
            KnownGPU.B300_SXM_270GB,
            [1, 2, 4, 8],
            Memory(100, ByteSuffix.GiB),
            tags=["start_from_device_enabled"])

# Grace-Blackwell NVL
add_systems("GB200_NVL72_186GB_ARMx{}",
            "GB200-NVL72_GB200-186GB_aarch64x{}",
            KnownCPU.ARMGeneric,
            KnownGPU.GB200_GraceBlackwell_186GB,
            [1, 2, 4],
            Memory(500, ByteSuffix.GiB),
            tags=["start_from_device_enabled", "end_on_device_enabled"])
add_systems("GB300_NVL72_288GB_ARMx{}",
            "GB300-NVL72_GB300-288GB_aarch64x{}",
            KnownCPU.ARMGeneric,
            KnownGPU.GB300_GraceBlackwell_288GB,
            [1, 2, 4],
            Memory(500, ByteSuffix.GiB),
            tags=["start_from_device_enabled", "end_on_device_enabled"])

# Grace-Hopper Superchip Systems
add_systems("GH200_96GB_ARMx{}",
            "GH200-GraceHopper-Superchip_GH200-96GB_aarch64x{}",
            KnownCPU.ARMGeneric,
            KnownGPU.GH200_GraceHopper_96GB,
            [1],
            Memory(500, ByteSuffix.GiB),
            tags=["start_from_device_enabled", "end_on_device_enabled"])
add_systems("GH200_144GB_ARMx{}",
            "GH200-GraceHopper-Superchip_GH200-144GB_aarch64x{}",
            KnownCPU.ARMGeneric,
            KnownGPU.GH200_GraceHopper_144GB,
            [1, 2],
            Memory(600, ByteSuffix.GiB),
            tags=["start_from_device_enabled", "end_on_device_enabled"])

# Hopper systems
add_systems("H200_SXM_141GBx{}",
            "H200-SXM-141GBx{}",
            KnownCPU.x86_64_Intel_Generic,
            KnownGPU.H200_SXM_141GB,
            [1, 2, 4, 8],
            Memory(100, ByteSuffix.GiB),
            tags=["start_from_device_enabled"])
add_systems("H200_SXM_141GB_CTSx{}",
            "H200-SXM-141GB-CTSx{}",
            KnownCPU.x86_64_Intel_Generic,
            KnownGPU.H200_SXM_141GB_CTS,
            [1, 2, 4, 8],
            Memory(2, ByteSuffix.TB),
            tags=["start_from_device_enabled"])
add_systems("H100_NVL_94GBx{}",
            "H100-NVL-94GBx{}",
            KnownCPU.x86_64_Intel_Generic,
            KnownGPU.H100_NVL,
            [1, 2, 4, 8],
            Memory(100, ByteSuffix.GiB))
add_systems("H100_SXM_80GBx{}",
            "DGX-H100_H100-SXM-80GBx{}",
            KnownCPU.x86_64_Intel_Generic,
            KnownGPU.H100_SXM_80GB,
            [1, 2, 4, 8],
            Memory(100, ByteSuffix.GiB),
            tags=["start_from_device_enabled"])
add_systems("H100_PCIe_80GBx{}",
            "H100-PCIe-80GBx{}",
            KnownCPU.x86_64_AMD_Generic,
            KnownGPU.H100_PCIe_80GB,
            [1, 2, 4, 8],
            Memory(100, ByteSuffix.GiB))
add_systems("H100_PCIe_80GB_ARMx{}",
            "H100-PCIe-80GB_aarch64x{}",
            KnownCPU.ARMGeneric,
            KnownGPU.H100_PCIe_80GB,
            [1, 2, 4, 8],
            Memory(100, ByteSuffix.GiB))

# Embedded systems
add_systems("Orin",
            "Orin",
            KnownCPU.ARMGeneric,
            KnownGPU.Orin,
            [1],
            Memory(7, ByteSuffix.GiB),
            n_dlas=2)
add_systems("Orin_NX",
            "Orin_NX",
            KnownCPU.ARMGeneric,
            KnownGPU.OrinNX,
            [1],
            Memory(7, ByteSuffix.GiB),
            n_dlas=2)


def classification_tags(system: System) -> Set[str]:
    tags = set()

    # This may break for non-homogeneous systems.
    gpus = system.accelerators.get(GPU, list())
    if len(gpus) > 0:
        tags.add("gpu_based")

        primary_sm = int(gpus[0].compute_sm)
        if primary_sm in (100, 103):
            tags.add("is_blackwell")
        if primary_sm == 90:
            tags.add("is_hopper")
        if primary_sm == 89:
            tags.add("is_ada")
        if primary_sm in (80, 86, 87, 89):
            tags.add("is_ampere")
        if primary_sm == 75:
            tags.add("is_turing")

        if gpus[0].name.startswith("Orin") and primary_sm == 87:
            tags.add("is_orin")
            tags.add("is_soc")

    if len(gpus) > 1:
        tags.add("multi_gpu")

    if system.cpu.architecture == CPUArchitecture.aarch64:
        tags.add("is_aarch64")

    return tags


@autoconfigure
@bind(general_fields.system_name, "system_name_override")
class SystemIndex:
    def __init__(self,
                 system_dict: Dict[str, System],
                 system_name_override: Optional[str] = None):
        self.system_dict = system_dict
        self.system_name_override = system_name_override

    def __getattr__(self, name: str):
        if name in self.system_dict:
            return self.system_dict[name]
        raise AttributeError(f"System {name} not found")

    def get_runtime_system(self) -> System:
        sys = System.detect()[0]

        if self.system_name_override:
            sys.extras["id"] = self.system_name_override
            sys.extras["tags"] = {"custom"}
        else:
            found = False
            for name, sys_desc in self.system_dict.items():
                if sys_desc.matches(sys):
                    found = True
                    sys.extras["id"] = sys_desc.mapping["extras"]["id"]
                    sys.extras["tags"] = sys_desc.mapping["extras"]["tags"]
                    break

            if not found:
                logging.error("Runtime system not found in built-ins. Please specify a system name using the SYSTEM_NAME environment variable.")

                try:
                    tmp_id = "UNREGISTERED"
                    tmp_id += f"_{sys.cpu.architecture.valstr}"

                    gpus = sys.accelerators.get(GPU, list())
                    if len(gpus) > 0:
                        clean_gpu_name = gpus[0].name.strip().replace(' ', '_')
                        tmp_id += f"_{clean_gpu_name}x{len(gpus)}"

                    sys.extras["id"] = tmp_id
                except:
                    sys.extras["id"] = "UNREGISTERED"

                sys.extras["tags"] = {"unregistered"}
        sys.extras["tags"] |= classification_tags(sys)

        if len(sys.accelerators.get(GPU, list())) > 0:
            sys.extras["primary_compute_sm"] = sys.accelerators[GPU][0].compute_sm
        else:
            sys.extras["primary_compute_sm"] = None

        return sys


def numa_config_string(L: List[NUMANode]):
    nodes = sorted(L, key=lambda n: n.index)

    def _node_str(n):
        gpus = ','.join(str(gpu.gpu_index) for gpu in n.accelerators[GPU])
        cpus = ','.join(str(cpu_set) for cpu_set in n.cpu_cores)
        return f"{gpus}:{cpus}"
    return '&'.join(_node_str(n) for n in nodes)


with Configuration().autoapply():
    KnownSystem = SystemIndex(_system_confs)
    DETECTED_SYSTEM = KnownSystem.get_runtime_system()
