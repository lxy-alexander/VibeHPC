# Adapted https://github.com/huggingface/optimum/blob/15a162824d0c5d8aa7a3d14ab6e9bb07e5732fb6/optimum/exporters/onnx/convert.py#L573-L614

# Copyright 2022 The HuggingFace Team. All rights reserved.
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
from pathlib import Path

import onnx
import torch
from optimum.onnx.utils import _get_onnx_external_data_tensors, check_model_uses_external_data
from torch.onnx import export as onnx_export

from diffusers.models.attention_processor import Attention


def generate_fp8_scales(model):
    # temporary solution due to a known bug in torch.onnx._dynamo_export
    for _, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)) and (
            hasattr(module.input_quantizer, "_amax") and module.input_quantizer is not None
        ):
            module.input_quantizer._num_bits = 8
            module.weight_quantizer._num_bits = 8
            module.input_quantizer._amax = module.input_quantizer._amax * (127 / 448.0)
            module.weight_quantizer._amax = module.weight_quantizer._amax * (127 / 448.0)
        elif isinstance(module, Attention) and (
            hasattr(module.q_bmm_quantizer, "_amax") and module.q_bmm_quantizer is not None
        ):
            module.q_bmm_quantizer._num_bits = 8
            module.q_bmm_quantizer._amax = module.q_bmm_quantizer._amax * (127 / 448.0)
            module.k_bmm_quantizer._num_bits = 8
            module.k_bmm_quantizer._amax = module.k_bmm_quantizer._amax * (127 / 448.0)
            module.v_bmm_quantizer._num_bits = 8
            module.v_bmm_quantizer._amax = module.v_bmm_quantizer._amax * (127 / 448.0)
            module.softmax_quantizer._num_bits = 8
            module.softmax_quantizer._amax = module.softmax_quantizer._amax * (127 / 448.0)


def generate_dummy_inputs(device, unet=False):
    if unet:
        dummy_input = {}
        dummy_input["sample"] = torch.ones(2, 4, 128, 128).to(device).cuda().half()
        dummy_input["timestep"] = torch.ones(1).to(device).cuda().half()
        dummy_input["encoder_hidden_states"] = torch.ones(2, 77, 2048).to(device).cuda().half()
        dummy_input["added_cond_kwargs"] = {}
        dummy_input["added_cond_kwargs"]["text_embeds"] = torch.ones(2, 1280).to(device).cuda().half()
        dummy_input["added_cond_kwargs"]["time_ids"] = torch.ones(2, 6).to(device).cuda().half()
        return dummy_input
    return torch.ones(1, 4, 128, 128).to(device)


def modelopt_export_sd(base, exp_name, model_name):
    os.makedirs(f"{exp_name}", exist_ok=True)
    dummy_inputs = generate_dummy_inputs(device=base.vae.device)

    output = Path(f"{exp_name}/vae.onnx")
    input_names = ["latent"]
    output_names = ["images"]

    dynamic_axes = {"latent": {0: "B", 2: "H", 3: "W"}, "images": {0: "B", 2: "8H", 3: "8W"}}

    do_constant_folding = True
    opset_version = 17

    # Copied from Huggingface's Optimum
    # pipe.vae.forward = pipe.vae.decode
    onnx_export(
        base.vae,
        (dummy_inputs,),
        f=output.as_posix(),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=do_constant_folding,
        opset_version=opset_version,
    )

    onnx_model = onnx.load(str(output), load_external_data=False)
    model_uses_external_data = check_model_uses_external_data(onnx_model)

    if model_uses_external_data:
        tensors_paths = _get_onnx_external_data_tensors(onnx_model)
        onnx_model = onnx.load(str(output), load_external_data=True)
        onnx.save(
            onnx_model,
            str(output),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=output.name + "_data",
            size_threshold=1024,
        )
        for tensor in tensors_paths:
            os.remove(output.parent / tensor)

def modelopt_export_unet(base, exp_name):
    os.makedirs(f"{exp_name}", exist_ok=True)
    dummy_inputs = generate_dummy_inputs(device=base.unet.device, unet=True)

    output = Path(f"{exp_name}/unet.onnx")
    input_names = ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"]
    output_names = ["latent"]

    dynamic_axes = {
        "sample": {0: "B", 1: "num_channels", 2: "height", 3: "width"},
        "timestep": {0: "steps"},
        "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
        "text_embeds": {0: "batch_size"},
        "time_ids": {0: "batch_size"},
        "latent": {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"},
    }
    do_constant_folding = True
    opset_version = 17

    # Copied from Huggingface's Optimum
    onnx_export(
        base.unet,
        (dummy_inputs,),
        f=output.as_posix(),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=do_constant_folding,
        opset_version=opset_version,
    )

    onnx_model = onnx.load(str(output), load_external_data=False)
    model_uses_external_data = check_model_uses_external_data(onnx_model)

    if model_uses_external_data:
        tensors_paths = _get_onnx_external_data_tensors(onnx_model)
        onnx_model = onnx.load(str(output), load_external_data=True)
        onnx.save(
            onnx_model,
            str(output),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=output.name + "_data",
            size_threshold=1024,
        )
        for tensor in tensors_paths:
            os.remove(output.parent / tensor)
