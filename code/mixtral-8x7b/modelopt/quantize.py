#!/usr/bin/env python3
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

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
import torch
import pandas as pd
import time
from pathlib import Path

from accelerate.hooks import remove_hook_from_module
from torch.utils.data import DataLoader
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.conversion import set_quantizer_by_cfg
from modelopt.torch.quantization.model_calib import max_calibrate
from modelopt.torch.export import export_tensorrt_llm_checkpoint


class _CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.flat_encodings = []
        for idx in range(len(encodings["input_ids"])):
            self.flat_encodings.append(
                {key: torch.tensor(val[idx]) for key, val in encodings.items()}
            )

    def __getitem__(self, idx):
        return self.flat_encodings[idx]

    def __len__(self):
        return len(self.flat_encodings)


def load_model(
    model_path: Path,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding_side="left", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model.to(torch.float16)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def get_calib_data_loader(
    calib_dataset_path: Path,
    calib_batch_size: int,
    tokenizer: PreTrainedTokenizerBase,
) -> DataLoader:
    calib_dataset = pd.read_pickle(calib_dataset_path)
    calib_dataset = calib_dataset["input"].tolist()
    batch_encoded = tokenizer.batch_encode_plus(
        calib_dataset,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    )
    batch_encoded = batch_encoded.to("cuda")
    batch_encoded["labels"] = torch.where(
        batch_encoded["attention_mask"] > 0.5, batch_encoded["input_ids"], -100
    )
    tokenized_dataset = _CustomDataset(batch_encoded)
    calib_dataloader = DataLoader(
        tokenized_dataset, batch_size=calib_batch_size, shuffle=False
    )
    return calib_dataloader


def quantize(
    model: PreTrainedModel,
    calib_dataloader: DataLoader,
    num_calib_steps: int,
    num_score_steps: int,
    effective_bits: int,
    use_fp4: bool,
    verbose: bool = False
) -> PreTrainedModel:
    q_format = ["FP8_DEFAULT_CFG", None]
    if use_fp4:
        q_format = ["NVFP4_DEFAULT_CFG", "FP8_DEFAULT_CFG", None]
    model, _ = mtq.auto_quantize(
        model,
        constraints={"effective_bits": effective_bits},
        data_loader=calib_dataloader,
        forward_step=lambda model, batch: model(**batch),
        loss_func=lambda output, data: output.loss,
        quantization_formats=q_format,
        num_calib_steps=num_calib_steps,
        num_score_steps=num_score_steps,
        verbose=True,
        disabled_layers=["*lm_head*"],
    )

    KV_CACHE_CFG = {
        "*.k_proj.output_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True,
        },
        "*.v_proj.output_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True,
        },
    }

    print("Quantization summary after auto quantize")
    mtq.print_quant_summary(model)
    set_quantizer_by_cfg(model, KV_CACHE_CFG)

    # Do one more round of calibration
    def calibrate_loop(model):
        for idx, data in enumerate(calib_dataloader):
            print(f"Calibrating {idx}")
            model(**data)

    max_calibrate(model, calibrate_loop)
    print("Quantization summary after KV Cache calibration")
    mtq.print_quant_summary(model)
    return model


def export_trt_llm(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    quantized_checkpoint_path: Path,
    tp_size: int,
    pp_size: int,
):
    with torch.inference_mode():
        start_time = time.time()

        # Move meta tensor back to device before exporting.
        remove_hook_from_module(model, recurse=True)

        export_tensorrt_llm_checkpoint(
            model,
            "llama",
            export_dir=quantized_checkpoint_path,
            inference_tensor_parallel=tp_size,
            inference_pipeline_parallel=pp_size,
        )

        # Export the tokenizer as well.
        tokenizer.save_pretrained(quantized_checkpoint_path)

        end_time = time.time()
        print(
            f"Quantized model exported to :{quantized_checkpoint_path}. Total time used {end_time - start_time}s"
        )
