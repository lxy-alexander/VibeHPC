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
from diffusers import DiffusionPipeline
from onnx_utils.export import generate_fp8_scales, modelopt_export_sd
from utils import filter_func, quantize_lvl
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq


def main():
    parser = argparse.ArgumentParser()
    # Model hyperparameters
    parser.add_argument("--onnx-dir", default="./vae_int8.onnx")
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    parser.add_argument(
        "--quantized-ckpt",
        type=str,
        default="./vae_int8.pt",
    )
    parser.add_argument("--format", default="int8", choices=["int8", "fp8"])
    parser.add_argument(
        "--quant-level",
        default=3.0,
        type=float,
        choices=[1.0, 2.0, 2.5, 3.0, 4.0],
        help="Quantization level, 1: CNN, 2: CNN+FFN, 2.5: CNN+FFN+QKV, 3: CNN+FC",
    )
    args = parser.parse_args()

    pipe = DiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

    mto.restore(pipe.vae, args.quantized_ckpt)

    quantize_lvl(pipe.vae, args.quant_level)
    mtq.disable_quantizer(pipe.vae, filter_func)

    # QDQ needs to be in FP32
    pipe.vae.to(torch.float32).to("cpu")

    if args.format == "fp8":
        generate_fp8_scales(pipe.vae.decoder)
    pipe.vae.forward = pipe.vae.decode
    with torch.inference_mode(), torch.autocast("cuda"):
        modelopt_export_sd(pipe, f"{str(args.onnx_dir)}", args.model)


if __name__ == "__main__":
    main()
