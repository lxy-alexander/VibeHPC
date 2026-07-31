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
import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline
import re
from PIL import Image
import numpy as np
import pandas as pd
from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq


SDXL_FP8_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Half"},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Half"},
        "*output_quantizer": {"enable": False},
        "*q_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Half"},
        "*k_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Half"},
        "*v_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Half"},
        "*softmax_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "trt_high_precision_dtype": "Half",
        },
        "*lm_head*": {"enable": False},
        "*block_sparse_moe.gate*": {"enable": False},  # Skip the MOE router
        "*output_layer*": {"enable": False},
        "default": {"num_bits": (4, 3), "axis": None},
    },
    "algorithm": "max",
}

def set_fmha(unet):
    for name, module in unet.named_modules():
        if isinstance(module, Attention):
            module.set_processor(AttnProcessor())


# used to disable certain blocks that are producing NaNs
def filter_func(name):
    pattern = re.compile(
        r".*(conv_out|conv_in|quant_conv|to_v|to_k|to_q|mid_block|decoder.up_blocks.3|up_blocks.2.resnets.2|up_blocks.2.resnets.1).*"
    )
    return pattern.match(name) is not None

def filter_func_unet(name):
    pattern = re.compile(
        r".*(time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding|up_blocks.2.resnets.2.conv1|up_blocks.2.resnets.2.conv2).*"
    )
    return pattern.match(name) is not None

# create new int8 quant config for SDXL (considering both CNN & Linear layers)
def get_int8_config(model, alpha=0.8):
    quant_config = {
        "quant_cfg": {
            "*lm_head*": {"enable": False},
            "*output_layer*": {"enable": False},
            "default": {"num_bits": 8, "axis": None},
        },
        "algorithm": {"method": "smoothquant", "alpha": alpha},
    }
    for name, module in model.named_modules():
        w_name = f"{name}*weight_quantizer"
        i_name = f"{name}*input_quantizer"

        if w_name in quant_config["quant_cfg"].keys() or i_name in quant_config["quant_cfg"].keys():
            continue
        if isinstance(module, torch.nn.Linear):
            quant_config["quant_cfg"][w_name] = {"num_bits": 8, "axis": 0}
            quant_config["quant_cfg"][i_name] = {"num_bits": 8, "axis": -1}
        elif isinstance(module, torch.nn.Conv2d):
            quant_config["quant_cfg"][w_name] = {"num_bits": 8, "axis": 0}
            quant_config["quant_cfg"][i_name] = {"num_bits": 8, "axis": None}
    return quant_config

# convert from TSV format

def load_calib_prompts(batch_size, calib_data_path):
    df = pd.read_csv(calib_data_path, sep="\t")
    lst = df["caption"].tolist()
    return [lst[i: i + batch_size] for i in range(0, len(lst), batch_size)]


def quantize_lvl(unet, quant_level=2.5, linear_only=False):
    """
    We should disable the unwanted quantizer when exporting the onnx
    Because in the current modelopt setting, it will load the quantizer amax for all the layers even
    if we didn't add that unwanted layer into the config during the calibration
    """
    for name, module in unet.named_modules():
        if isinstance(module, (torch.nn.Conv2d, LoRACompatibleConv)):
            if linear_only:
                module.input_quantizer.disable()
                module.weight_quantizer.disable()
            else:
                module.input_quantizer.enable()
                module.weight_quantizer.enable()
        elif isinstance(module, (torch.nn.Linear, LoRACompatibleLinear)):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (quant_level >= 2.5 and ("to_q" in name or "to_k" in name or "to_v" in name))
                or quant_level >= 3
            ):
                module.input_quantizer.enable()
                module.weight_quantizer.enable()
            else:
                module.input_quantizer.disable()
                module.weight_quantizer.disable()
        elif isinstance(module, Attention):
            if quant_level >= 4:
                module.q_bmm_quantizer.enable()
                module.k_bmm_quantizer.enable()
                module.v_bmm_quantizer.enable()
                module.softmax_quantizer.enable()
            else:
                module.q_bmm_quantizer.disable()
                module.k_bmm_quantizer.disable()
                module.v_bmm_quantizer.disable()
                module.softmax_quantizer.disable()
