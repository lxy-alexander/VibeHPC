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


from __future__ import annotations

from collections.abc import Iterable
from nvmitten.constants import CPUArchitecture
from nvmitten.memory import Memory
from nvmitten.interval import NumericRange
try:
    from nvmitten.nvidia.accelerator import GPU
except:
    # This will only happen if CUDA isn't installed on the system / if no GPUs are present
    import dataclasses
    @dataclasses.dataclass
    class GPU: pass
from nvmitten.nvidia.constants import ComputeSM
from nvmitten.system.component import Description
from nvmitten.system.cpu import CPU


def PCIVendorID(s):
    return f"0x{s}10DE"


def GPUDescription(**kwargs):
    if "pci_id" in kwargs:
        if isinstance(kwargs["pci_id"], str):
            kwargs["pci_id"] = PCIVendorID(kwargs["pci_id"])
        elif isinstance(kwargs["pci_id"], Iterable):
            kwargs["pci_id"] = list(PCIVendorID(s) for s in kwargs["pci_id"]).__contains__
        else:
            raise TypeError("Unexpected type for PCI ID")
    if "vram" in kwargs:
        kwargs["vram"] = NumericRange(Memory.from_string(kwargs["vram"]), rel_tol=0.05)
    if "max_power_limit" in kwargs:
        kwargs["max_power_limit"] = NumericRange(kwargs["max_power_limit"], rel_tol=0.05)
    if "is_integrated" not in kwargs:
        kwargs["is_integrated"] = False

    return Description(GPU, **kwargs)


class KnownGPU:
    """By convention, we use the PCI ID but not the device name, as the name is prone and subject to change while PCI
    vendor IDs are not.
    """
    B200_SXM_180GB = GPUDescription(
        pci_id="2901",
        compute_sm=ComputeSM(10, 0),
        vram="180 GiB",
        max_power_limit=1000.0)
    B300_SXM_270GB = GPUDescription(
        pci_id="3182",
        compute_sm=ComputeSM(10, 3),
        vram="268 GiB",
        max_power_limit=1100.0)
    GB200_GraceBlackwell_186GB = GPUDescription(
        pci_id="2941",
        compute_sm=ComputeSM(10, 0),
        vram="186 GiB",
        max_power_limit=1200.0)
    GB300_GraceBlackwell_288GB = GPUDescription(
        pci_id="31C2",
        compute_sm=ComputeSM(10, 3),
        vram="277 GiB",
        max_power_limit=1400.0)
    GH200_GraceHopper_96GB = GPUDescription(
        pci_id="2342",
        compute_sm=ComputeSM(9, 0),
        vram="96 GiB",
        max_power_limit=900.0)
    GH200_GraceHopper_144GB = GPUDescription(
        pci_id="2348",
        compute_sm=ComputeSM(9, 0),
        vram="144 GiB",
        max_power_limit=900.0)
    L4 = GPUDescription(
        pci_id="27B8",
        compute_sm=ComputeSM(8, 9),
        vram="24 GiB",  # This was 23 in the original file, but is 24 in the technical spec.
        max_power_limit=72.0)
    L40 = GPUDescription(
        pci_id="26B5",
        compute_sm=ComputeSM(8, 9),
        vram="45 GiB",
        max_power_limit=300.0)
    L40S = GPUDescription(
        pci_id="26B9",
        compute_sm=ComputeSM(8, 9),
        vram="45 GiB",
        max_power_limit=350.0)
    H100_SXM_80GB = GPUDescription(
        pci_id=("2330", "233F"),
        compute_sm=ComputeSM(9, 0),
        vram="80 GiB",
        max_power_limit=700.0)
    H100_PCIe_80GB = GPUDescription(
        pci_id="2331",
        compute_sm=ComputeSM(9, 0),
        vram="80 GiB",
        max_power_limit=350.0)
    H100_NVL = GPUDescription(
        pci_id="2321",
        compute_sm=ComputeSM(9, 0),
        vram="94 GiB",
        max_power_limit=400.0)
    H200_SXM_141GB = GPUDescription(
        pci_id="2335",
        compute_sm=ComputeSM(9, 0),
        vram="141 GiB",
        max_power_limit=700.0)
    H200_SXM_141GB_CTS = GPUDescription(
        pci_id="2335",
        compute_sm=ComputeSM(9, 0),
        vram="141 GiB",
        max_power_limit=1000.0)
    A100_PCIe_40GB = GPUDescription(
        pci_id=("20F1", "20BF"),
        compute_sm=ComputeSM(8, 0),
        vram="40 GiB",
        max_power_limit=250.0)
    A100_PCIe_80GB = GPUDescription(
        pci_id="20B5",
        compute_sm=ComputeSM(8, 0),
        vram="80 GiB",
        max_power_limit=300.0)
    A100_SXM4_40GB = GPUDescription(
        pci_id="20B0",
        compute_sm=ComputeSM(8, 0),
        vram="40 GiB",
        max_power_limit=400.0)
    A100_SXM_80GB = GPUDescription(
        pci_id="20B2",
        compute_sm=ComputeSM(8, 0),
        vram="80 GiB",
        max_power_limit=400.0)
    A100_SXM_80GB_RO = GPUDescription(
        pci_id="20B2",
        compute_sm=ComputeSM(8, 0),
        vram="80 GiB",
        max_power_limit=275.0)
    GeForceRTX_3080 = GPUDescription(
        pci_id="2206",
        compute_sm=ComputeSM(8, 6),
        vram="10 GiB",
        max_power_limit=320.0)
    GeForceRTX_3090 = GPUDescription(
        pci_id=("2204", "2230"),
        compute_sm=ComputeSM(8, 6),
        vram="24 GiB",
        max_power_limit=350.0)
    A10 = GPUDescription(
        pci_id="2236",
        compute_sm=ComputeSM(8, 6),
        vram="24 GiB",
        max_power_limit=150.0)
    A30 = GPUDescription(
        pci_id="20B7",
        compute_sm=ComputeSM(8, 0),
        vram="24 GiB",
        max_power_limit=165.0)
    A2 = GPUDescription(
        pci_id="25B6",
        compute_sm=ComputeSM(8, 6),
        vram="16 GiB",
        max_power_limit=60.0)
    DRIVE_A100_PCIE = GPUDescription(
        pci_id="20BB",
        compute_sm=ComputeSM(8, 0),
        vram="32 GiB")
    T4_16GB = GPUDescription(
        pci_id=("1EB8", "1EB9"),
        compute_sm=ComputeSM(7, 5),
        vram="16 GiB",
        max_power_limit=70.0)
    T4_32GB = GPUDescription(
        pci_id=("1EB8", "1EB9"),
        compute_sm=ComputeSM(7, 5),
        vram="32 GiB",
        max_power_limit=70.0)
    Orin = GPUDescription(
        name="Orin",
        compute_sm=ComputeSM(8, 7),
        vram="64 GiB",
        is_integrated=True)
    OrinNX = GPUDescription(
        name="Orin NX",
        compute_sm=ComputeSM(8, 7),
        vram="16 GiB",
        is_integrated=True)


class KnownCPU:
    ARMGeneric = Description(CPU,
                             architecture=CPUArchitecture.aarch64,
                             vendor="ARM")
    x86_64_AMD_Generic = Description(CPU,
                                     architecture=CPUArchitecture.x86_64,
                                     vendor="AuthenticAMD")
    x86_64_Intel_Generic = Description(CPU,
                                       architecture=CPUArchitecture.x86_64,
                                       vendor="GenuineIntel")
    x86_64_Generic = Description(CPU,
                                 architecture=CPUArchitecture.x86_64)
