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
import argparse

import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from onnx_utils.export import generate_fp8_scales, modelopt_export_sd
from utils import filter_func_unet, quantize_lvl, set_fmha

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq


def main():
    parser = argparse.ArgumentParser()
    # Model hyperparameters
    parser.add_argument("--onnx-dir", default=None)
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0"
    )
    parser.add_argument(
        "--quantized-ckpt",
        type=str,
        default="./base.unet.state_dict.fp8.0.25.384.percentile.all.pt",
    )
    args = parser.parse_args()

    pipe = DiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

    # Lets restore the quantized model
    if args.quant_level == 4.0:
        assert args.format != "int8", "We only support FP8 for Level 4 Quantization"
        set_fmha(pipe.unet)
    mto.restore(pipe.unet, args.quantized_ckpt)

    mtq.disable_quantizer(pipe.unet, filter_func_unet)

    # QDQ needs to be in FP32
    pipe.unet.to("cuda")
    if args.format == "fp8":
        generate_fp8_scales(pipe.unet)
    modelopt_export_unet(pipe, f"{str(args.onnx_dir)}", args.model)


if __name__ == "__main__":
    main()
