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
import pandas as pd
import re
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from utils import get_int8_config, load_calib_prompts
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq

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
    parser.add_argument("--format", type=str, default="int8", choices=["int8", "fp8"])
    parser.add_argument("--exp_name", default="./vae_int8.pt")
    parser.add_argument(
        "--n_steps",
        type=int,
        default=20,
        help="Number of denoising steps, default: 20",
    )

    # Calibration and quantization parameters
    parser.add_argument("--percentile", type=float, default=1.0, required=False)
    parser.add_argument(
        "--collect-method",
        type=str,
        required=False,
        default="default",
        choices=["global_min", "min-max", "min-mean", "mean-max", "default"],
        help=(
            "Ways to collect the amax of each layers, for example, min-max means min(max(step_0),"
            " max(step_1), ...)"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--calib-size", type=int, default=64)
    parser.add_argument("--calib-data", type=str, default="./captions.tsv")
    parser.add_argument("--alpha", type=float, default=1.0, help="SmoothQuant Alpha")
    parser.add_argument(
        "--quant-level",
        default=3.0,
        type=float,
        choices=[1.0, 2.0, 2.5, 3.0, 4.0],
        help="Quantization level, 1: CNN, 2: CNN+FFN, 2.5: CNN+FFN+QKV, 3: CNN+FC, 4: CNN+FC+fMHA",
    )

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

    # determine quant config based on input format
    if args.format == "int8":
        quant_config = get_int8_config(pipe.vae)  # new config (both CNNs & linear)
    elif args.format == "fp8":
        quant_config = mtq.FP8_DEFAULT_CFG
    else:
        raise ValueError(f"Unsupported quantization format: {args.format}")

    def forward_loop(vae):
        pipe.vae = vae
        do_calibrate(
            pipe=pipe,
            calibration_prompts=cali_prompts,
            calib_size=args.calib_size,
            n_steps=args.n_steps,
            latent=init_latent,
        )

    # quantize the VAE given int8 or fp8 config
    mtq.quantize(pipe.vae, quant_config, forward_loop)

    # export quantized PyTorch model
    quantized_vae_path = Path(args.exp_name)
    mto.save(pipe.vae, quantized_vae_path)


if __name__ == "__main__":
    main()
