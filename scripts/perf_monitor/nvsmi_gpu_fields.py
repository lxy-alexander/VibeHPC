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

nvidia_smi_csv_keys = {
    "timestamp": "The timestamp of when the query was made in format 'YYYY/MM/DD HH:MM:SS.msec'",
    "driver_version": "The version of the installed NVIDIA display driver. This is an alphanumeric string.",
    "gpu_name": " The official product name of the GPU. This is an alphanumeric string. For all products.",
    "gpu_serial": " This number matches the serial number physically printed on each board. It is a globally unique immutable alphanumeric value.",
    "gpu_uuid": " This value is the globally unique immutable alphanumeric identifier of the GPU. It does not correspond to any physical label on the board.",
    "pci.bus_id": " PCI bus id as 'domain:bus:device.function', in hex.",
    "pci.domain": " PCI domain number, in hex.",
    "pci.bus": " PCI bus number, in hex.",
    "pci.device": " PCI device number, in hex.",
    "pci.baseClass": " PCI Base Classcode, in hex.",
    "pci.subClass": " PCI Sub Classcode, in hex.",
    "pci.device_id": " PCI vendor device id, in hex",
    "pci.sub_device_id": " PCI Sub System id, in hex",
    "pstate": " The current performance state for the GPU. States range from P0 (maximum performance) to P12 (minimum performance).",
    "clocks_throttle_reasons.supported": " Bitmask of supported clock event reasons. See nvml.h for more details.",
    "clocks_throttle_reasons.active": " Bitmask of active clock event reasons. See nvml.h for more details.",
    "clocks_throttle_reasons.gpu_idle": " Nothing is running on the GPU and the clocks are dropping to Idle state. This limiter may be removed in a later release.",
    "clocks_throttle_reasons.applications_clocks_setting": " GPU clocks are limited by applications clocks setting. E.g. can be changed by nvidia-smi --applications-clocks=",
    "clocks_throttle_reasons.sw_power_cap": " SW Power Scaling algorithm is reducing the clocks below requested clocks because the GPU is consuming too much power. E.g. SW power cap limit can be changed with nvidia-smi --power-limit=",
    "clocks_throttle_reasons.hw_slowdown": " HW Slowdown (reducing the core clocks by a factor of 2 or more) is engaged.",
    "clocks_throttle_reasons.hw_thermal_slowdown": " HW Thermal Slowdown (reducing the core clocks by a factor of 2 or more) is engaged. This is an indicator of temperature being too high",
    "clocks_throttle_reasons.hw_power_brake_slowdown": " HW Power Brake Slowdown (reducing the core clocks by a factor of 2 or more) is engaged. This is an indicator of External Power Brake Assertion being triggered (e.g. by the system power supply)",
    "clocks_throttle_reasons.sw_thermal_slowdown": " SW Thermal capping algorithm is reducing clocks below requested clocks because GPU temperature is higher than Max Operating Temp.",
    "clocks_throttle_reasons.sync_boost": " Sync Boost This GPU has been added to a Sync boost group with nvidia-smi or DCGM",
    "memory.total": " Total installed GPU memory.",
    "memory.reserved": " Total memory reserved by the NVIDIA driver and firmware.",
    "memory.used": " Total memory allocated by active contexts.",
    "memory.free": " Total free memory.",
    "utilization.gpu": " Percent of time over the past sample period during which one or more kernels was executing on the GPU.",
    "utilization.memory": " Percent of time over the past sample period during which global (device) memory was being read or written.",
    "utilization.encoder": " Percent of time over the past sample period during which one or more kernels was executing on the Encoder Engine.",
    "utilization.decoder": " Percent of time over the past sample period during which one or more kernels was executing on the Decoder Engine.",
    "utilization.jpeg": " Percent of time over the past sample period during which one or more kernels was executing on the Jpeg Engine.",
    "utilization.ofa": " Percent of time over the past sample period during which one or more kernels was executing on the Optical Flow Accelerator Engine.",
    "temperature.gpu": " Core GPU temperature. in degrees C.",
    "temperature.gpu.tlimit": " GPU T.Limit temperature. in degrees C.",
    "temperature.memory": " HBM memory temperature. in degrees C.",
    "power.draw": " The last measured power draw for the entire board, in watts. On Ampere or newer devices, returns average power draw over 1 sec. On older devices, returns instantaneous power draw. Only available if power management is supported. This reading is accurate to within +/- 5 watts.",
    "power.draw.average": " The last measured average power draw for the entire board, in watts. Only available if power management is supported and Ampere (except GA100) or newer devices. This reading is accurate to within +/- 5 watts.",
    "power.draw.instant": " The last measured instant power draw for the entire board, in watts. Only available if power management is supported. This reading is accurate to within +/- 5 watts.",
    "clocks.gr": " Current frequency of graphics (shader) clock.",
    "clocks.sm": " Current frequency of SM (Streaming Multiprocessor) clock.",
    "clocks.mem": " Current frequency of memory clock.",
    "clocks.video": " Current frequency of video encoder/decoder clock.",
    "clocks.applications.gr": " User specified frequency of graphics (shader) clock.",
    "clocks.applications.mem": " User specified frequency of memory clock.",
    "clocks.default_applications.gr": " Default frequency of applications graphics (shader) clock.",
    "clocks.default_applications.mem": " Default frequency of applications memory clock.",
}
