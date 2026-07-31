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
import tensorrt_llm
import modelopt
from pathlib import Path

# Because hyphens break the traditional way to import
from importlib import import_module
quant_module = import_module("code.mixtral-8x7b.modelopt")
load_model = quant_module.load_model
get_calib_data_loader = quant_module.get_calib_data_loader
quantize = quant_module.quantize
export_trt_llm = quant_module.export_trt_llm

import random
import numpy as np
import torch


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('true', '1'):
            return True
        elif v.lower() in ('false', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=Path,
        default="/work/build/models/Mixtral/Mixtral-8x7B-Instruct-v0.1/",
        help="Path to the Mixtral-8x7B-Instruct-v0.1 HF model.",
    )
    parser.add_argument(
        "--quantized_checkpoint_path",
        type=Path,
        required=True,
        help="Path to export the TRTLLM quantized checkpoint",
    )
    parser.add_argument(
        "--calib_dataset_path",
        type=Path,
        default="/work/build/data/moe/mlperf_mixtral8x7b_moe_calibration_dataset_1k.pkl",
        help="Path to calibration dataset",
    )
    parser.add_argument(
        "--calib_batch_size",
        type=int,
        default=4,
        help="Batch size for calibration data loader",
    )
    parser.add_argument(
        "--effective_bits",
        type=float,
        required=True,
        help="The effective number of bits in the quantized checkpoint, controls degree of quantization",
    )
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--pp_size", type=int, default=1)
    parser.add_argument(
        "--num_calib_steps",
        type=int,
        default=16,
        help="Number of calibration steps to use in the auto_quantize algorithm",
    )
    parser.add_argument(
        "--num_score_steps",
        type=int,
        default=4,
        help="Number of scoring steps to use in the auto_quantize algorithm",
    )
    parser.add_argument(
        "--fp4",
        type=str2bool,
        default=False,
        help="specify true/false. If true, quantization will be fp4+fp8+fp16. If false, then quantization will be fp8+fp16",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose run"
    )
    args = parser.parse_args()

    assert args.model_path.exists(), f"{args.model_path} does not exist"
    assert args.calib_dataset_path.exists()

    l = 4 if args.fp4 else 8
    assert (
        l < args.effective_bits < 16
    ), f"set effective_bits in range ({l}, 16), {args.effective_bits} is invalid for selected quantization recipe"

    print(args)
    return args


if __name__ == "__main__":
    print(f"TensorRT-LLM version {tensorrt_llm.__version__}")
    print(f"TensorRT ModelOpt version: {modelopt.__version__}")

    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Below may cause some nondeterministic operations to throw RuntimeErrors, please check doc:
    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    torch.use_deterministic_algorithms(mode=True)

    args = parse_args()
    model, tokenizer = load_model(args.model_path)
    calib_dataloader = get_calib_data_loader(args.calib_dataset_path, args.calib_batch_size, tokenizer)
    model = quantize(
        model=model,
        calib_dataloader=calib_dataloader,
        num_calib_steps=args.num_calib_steps,
        num_score_steps=args.num_score_steps,
        effective_bits=args.effective_bits,
        use_fp4=args.fp4,
    )
    export_trt_llm(
        model,
        tokenizer,
        args.quantized_checkpoint_path,
        args.tp_size,
        args.pp_size,
    )
