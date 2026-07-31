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

"""B200-SXM-180GBx8 configuration for WAN22-A14B (Wan2.2 T2V) Offline scenario."""

import code.common.constants as C
from importlib import import_module
wan22_a14b_fields = import_module("code.wan22-a14b.tensorrt.fields")
import code.fields.harness as harness_fields
import code.fields.loadgen as loadgen_fields

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        # Data paths
        harness_fields.tensor_path: 'build/preprocessed_data/wan22-a14b/',

        # WAN22-A14B specific settings
        wan22_a14b_fields.prompt_file: 'prompts.txt',
        wan22_a14b_fields.model_path: 'Wan-AI/Wan2.2-T2V-A14B-Diffusers',
        wan22_a14b_fields.num_frames: 81,
        wan22_a14b_fields.height: 720,
        wan22_a14b_fields.width: 1280,
        wan22_a14b_fields.num_inference_steps: 20,
        wan22_a14b_fields.guidance_scale: 4.0,
        wan22_a14b_fields.guidance_scale_2: 3.0,
        wan22_a14b_fields.attn_type: 'te-fp8',
        wan22_a14b_fields.linear_type: 'te-fp8-per-tensor',

        # Multi-device parallelism
        wan22_a14b_fields.ulysses_size: 1,
        wan22_a14b_fields.cfg_size: 1,
        wan22_a14b_fields.dp_size: 8,
        wan22_a14b_fields.enable_visual_gen_cpu_offload: False,
        wan22_a14b_fields.cpu_offload_stride: 1,

        # LoadGen settings
        loadgen_fields.offline_expected_qps: 0.04,
    },
}
