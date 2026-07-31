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
from pathlib import Path
import torch
from diffusers import DiffusionPipeline
from utils import load_calib_prompts, set_fmha
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq

from utils import SDXL_FP8_CFG, filter_func_unet
from onnx_utils.export import generate_fp8_scales, modelopt_export_unet

# do the calibration with the given MLPerf calibration prompts and fixed latent


def do_calibrate(pipe, calibration_prompts, **kwargs):
    for i_th, prompts in enumerate(calibration_prompts):
        if i_th >= kwargs["calib_size"]:
            return
        pipe(
            prompt=prompts,
            num_inference_steps=kwargs["n_steps"],
            negative_prompt=[
                "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
            ]
            * len(prompts),
            latents=kwargs["latent"],
            guidance_scale=8.0,
        ).images


def main():
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    parser.add_argument("--exp_name", default="./unet_fp8.pt")
    parser.add_argument("--onnx_dir", default="./test.onnx")
    parser.add_argument(
        "--n_steps",
        type=int,
        default=20,
        help="Number of denoising steps, default: 20",
    )

    # Calibration and quantization parameters
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--calib-size", type=int, default=64)
    parser.add_argument("--calib-data", type=str, default="./captions.tsv")

    # MLPerf fixed latent
    parser.add_argument("--latent", type=str, default="./latents.pt")

    args = parser.parse_args()

    args.calib_size = args.calib_size // args.batch_size

    pipe = DiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")

    # load the MLPerf prompts
    cali_prompts = load_calib_prompts(args.batch_size, args.calib_data)

    # initialize the fixed latent
    init_latent = None
    if args.latent is not None:
        init_latent = torch.load(args.latent).to(torch.float16)

    def forward_loop(unet):
        pipe.unet = unet
        do_calibrate(
            pipe=pipe,
            calibration_prompts=cali_prompts,
            calib_size=args.calib_size,
            n_steps=args.n_steps,
            latent=init_latent,
        )
    set_fmha(pipe.unet)

    mtq.quantize(pipe.unet, SDXL_FP8_CFG, forward_loop)

    # export quantized PyTorch model
    quantized_unet_path = Path(args.exp_name)
    mto.save(pipe.unet, quantized_unet_path)
    mtq.disable_quantizer(pipe.unet, filter_func_unet)
    generate_fp8_scales(pipe.unet)
    modelopt_export_unet(pipe, f"{str(args.onnx_dir)}")


if __name__ == "__main__":
    main()
