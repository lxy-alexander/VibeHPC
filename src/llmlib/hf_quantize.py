# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Quantize a transformer model by ModelOpt. The code is adapted from tensorrt-llm's examples/quantization/quantize.py as of Jun, 2025.
"""

import argparse
import copy
import json
import os
import random
import sys
import time
import numpy as np
import torch
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint

from tensorrt_llm.models.convert_utils import infer_dtype


from importlib.metadata import version
from datasets import load_dataset
from modelopt.torch.utils import print_rank_0
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModelForCausalLM, AutoProcessor,
                          AutoTokenizer)

from tensorrt_llm._utils import release_gc, str_dtype_to_torch
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

EMPTY_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "enable": False,
        },
        "*input_quantizer": {
            "enable": False
        },
        "*lm_head*": {
            "enable": False
        },
        "*output_layer*": {
            "enable": False
        },
        "default": {
            "enable": False
        },
    },
    "algorithm": "max",
}

KV_CACHE_CFG = {
    "*.query_key_value.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.Wqkv.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.W_pack.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.c_attn.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.k_proj.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.v_proj.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.k.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.v.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
}

KV_QUANT_CFG_CHOICES = {
    "fp8": "FP8_KV_CFG",
    "nvfp4": "NVFP4_KV_CFG",
}


def quant_cfg_choices():
    import modelopt.torch.quantization as mtq
    QUANT_CFG_CHOICES = {
        "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
        "fp8": mtq.FP8_DEFAULT_CFG,
        "int4_awq": mtq.INT4_AWQ_CFG,
        "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,
        "int8_wo": EMPTY_CFG,
        "int4_wo": EMPTY_CFG,
        "full_prec": EMPTY_CFG,
    }
    if hasattr(mtq, "NVFP4_DEFAULT_CFG"):
        QUANT_CFG_CHOICES["nvfp4"] = mtq.NVFP4_DEFAULT_CFG
    return QUANT_CFG_CHOICES


MODEL_NAME_PATTERN_MAP = {
    "Llama": "llama",
    "Mistral": "llama",
    "GPTJ": "gptj",
    "MixtralForCausalLM": "llama",
}

MULTIMODAL_DATASETS = ['scienceqa', 'science_qa']


class _CustomDataset(torch.utils.data.Dataset):

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {
            key: val[idx].clone().detach().requires_grad_(False)
            for key, val in self.encodings.items()
        }
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def get_tokenizer(ckpt_path, max_seq_length=2048, model_type=None):
    logger.info(f"Initializing tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path,
        model_max_length=max_seq_length,
        padding_side="left",
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token is not None, f"Pad token for {model_type} cannot be set!"

    return tokenizer


def get_model(ckpt_path: str,
              dtype: str = 'bfloat16',
              device: str = 'cuda',
              device_map: str = "auto"):
    logger.info(f"Initializing model from {ckpt_path}")
    torch_dtype = str_dtype_to_torch(dtype)

    model_cls = AutoModelForCausalLM

    model = model_cls.from_pretrained(
        ckpt_path,
        device_map=device_map if device != "cpu" else "cpu",
        torch_dtype="auto",
        trust_remote_code=True)

    model.eval()

    model_dtype = next(model.parameters()).dtype
    if torch_dtype != model_dtype:
        logger.info(
            f"[WARNING] The manually set model data type is {dtype}, "
            f"but the data type of the HuggingFace model is {model_dtype}.")

    return model


def get_model_type(model):
    if type(model).__name__ in MODEL_NAME_PATTERN_MAP:
        return MODEL_NAME_PATTERN_MAP[type(model).__name__]
    for k, v in MODEL_NAME_PATTERN_MAP.items():
        if k.lower() in type(model).__name__.lower():
            return v
    return None


def get_calib_dataloader(dataset_name_or_dir="cnn_dailymail",
                         tokenizer=None,
                         batch_size=1,
                         calib_size=512,
                         block_size=512,
                         device=None,
                         include_labels=False):
    logger.info("Loading calibration dataset")
    if dataset_name_or_dir == "pileval":
        dataset = load_dataset(
            "json",
            data_files="https://the-eye.eu/public/AI/pile/val.jsonl.zst",
            split="train",
            trust_remote_code=True)
        dataset = dataset["text"][:calib_size]
    elif "scienceqa" in dataset_name_or_dir.lower(
    ) or "science_qa" in dataset_name_or_dir.lower():
        if os.path.isdir(dataset_name_or_dir):
            dataset = load_dataset(dataset_name_or_dir,
                                   split="train",
                                   trust_remote_code=True)
        else:
            dataset = load_dataset("derek-thomas/ScienceQA",
                                   split="train",
                                   trust_remote_code=True)
        dataset = dataset.select(range(calib_size))
    elif "cnn_dailymail" in dataset_name_or_dir:
        dataset = load_dataset(
            dataset_name_or_dir,
            name="3.0.0",
            split="train",
            trust_remote_code=True,
        )
        dataset = dataset["article"][:calib_size]
    elif os.path.isdir(dataset_name_or_dir):
        logger.info(
            f"Recognized local dataset repo {dataset_name_or_dir} for calibration; "
            "assuming the calibration data are in the train split and text column."
        )
        dataset = load_dataset(dataset_name_or_dir,
                               split="train",
                               trust_remote_code=True)
        dataset = dataset["text"][:calib_size]
    else:
        raise NotImplementedError(
            f"Unsupported dataset name or local repo directory: {dataset_name_or_dir}."
        )

    is_multimodal = False
    for dataset_name in MULTIMODAL_DATASETS:
        if dataset_name in dataset_name_or_dir:
            is_multimodal = True
    if is_multimodal:
        # Apply the preprocessing function to the dataset
        processed_dataset = dataset.map(tokenizer.preprocess_function,
                                        batched=False,
                                        remove_columns=dataset.column_names)

        # Create DataLoader with the custom collate function
        calib_dataloader = DataLoader(processed_dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      collate_fn=tokenizer.collate_function)
    else:
        batch_encoded = tokenizer.batch_encode_plus(dataset,
                                                    return_tensors="pt",
                                                    padding=True,
                                                    truncation=True,
                                                    max_length=block_size)
        if device:
            batch_encoded = batch_encoded.to(device)

        if include_labels:
            # Labels are needed when backward is called in the model.
            # The labels should be a shifted version of the input_ids.
            # However, we should not shift the input_ids here since the labels are shifted by
            # Huggingface models during loss calculation as shown here -
            # https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/llama/modeling_llama.py#L1093-L1095
            batch_encoded["labels"] = torch.where(
                batch_encoded["attention_mask"] > 0.5,
                batch_encoded["input_ids"], -100)
            batch_encoded = _CustomDataset(batch_encoded)
        else:
            # For backward compatibility, if labels are not needed, we only return input_ids.
            batch_encoded = _CustomDataset(
                {"input_ids": batch_encoded["input_ids"]})

        calib_dataloader = DataLoader(batch_encoded,
                                      batch_size=batch_size,
                                      shuffle=False)

    return calib_dataloader


def quantize_model(model, quant_cfg, calib_dataloader, batch_size, qformat,
                   auto_quantize_bits):
    import modelopt.torch.quantization as mtq

    # NOTE: for ModelOpt v0.19 release
    # calibrate_loop = dataset_utils.create_forward_loop(
    #     calib_dataloader, dataloader=calib_dataloader)

    def calibrate_loop():
        if calib_dataloader is None:
            return
        with torch.no_grad():
            low_mem_mode = False
            for idx, data in enumerate(calib_dataloader):
                logger.debug(f"Calibrating batch {idx}")
                batch_size = data[list(data.keys())[0]].shape[0]
                if batch_size == 1:
                    model(**data)
                elif not low_mem_mode:
                    # Try running the forward once.
                    # If output memory, we try running inference with split input tensors
                    try:
                        model(**data)
                    except torch.OutOfMemoryError:
                        print(
                            "Warning: torch.OutOfMemoryError detected, try reducing the batch size..."
                        )
                        low_mem_mode = True

                if low_mem_mode:
                    split_data_1 = {
                        key: data[key][:batch_size // 2, ...]
                        for key in data
                    }
                    model(**split_data_1)

                    split_data_2 = {
                        key: data[key][batch_size // 2:, ...]
                        for key in data
                    }
                    model(**split_data_2)

    QUANT_CFG_CHOICES = {
        "int8": "INT8_DEFAULT_CFG",
        "int8_sq": "INT8_SMOOTHQUANT_CFG",
        "fp8": "FP8_DEFAULT_CFG",
        "int4_awq": "INT4_AWQ_CFG",
        "w4a8_awq": "W4A8_AWQ_BETA_CFG",
    }

    logger.info("Starting quantization...")
    start_time = time.time()
    if auto_quantize_bits:
        logger.info("Starting mixed precision quantization...")

        from packaging import version as v
        opt_kwargs = {}
        modelopt_version = version('nvidia-modelopt')
        if v.parse(modelopt_version) > v.parse("0.21"):
            opt_kwargs['disabled_layers'] = ["*lm_head*"]

        model, search_history = mtq.auto_quantize(
            model,
            data_loader=calib_dataloader,
            loss_func=lambda output, batch: output.loss,
            constraints={"effective_bits": auto_quantize_bits},
            forward_step=lambda model, batch: model(**batch),
            quantization_formats=[
                QUANT_CFG_CHOICES[item] for item in qformat.split(",")
            ] + [None],
            num_calib_steps=len(calib_dataloader),
            num_score_steps=min(
                len(calib_dataloader), 128 // batch_size
            ),  # Limit the number of score steps to avoid long calibration time
            verbose=True,
            **opt_kwargs)
        mtq.print_quant_summary(model)

        # We need to explicitly calibrate for kv cache quantization
        enable_kv_cache_quantization = "int8" not in qformat
        if enable_kv_cache_quantization:
            mtq.set_quantizer_by_cfg(
                model,
                quant_cfg={
                    "*output_quantizer": {
                        "num_bits": (4, 3),
                        "axis": None,
                        "enable": True
                    }
                },
            )
            # Lets calibrate only the output quantizer this time. Let's disable all other quantizers.
            with mtq.set_quantizer_by_cfg_context(model, {
                    "*": {
                        "enable": False
                    },
                    "*output_quantizer": {
                        "enable": True
                    }
            }):
                mtq.calibrate(model,
                              algorithm="max",
                              forward_loop=calibrate_loop)
    else:
        mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    end_time = time.time()
    logger.info(
        "Quantization done. Total time used: {:.2f} s.".format(end_time -
                                                               start_time))
    return model


def quantize_and_export(*,
                        model_dir,
                        device,
                        calib_dataset,
                        dtype,
                        qformat,
                        kv_cache_dtype,
                        calib_size,
                        batch_size,
                        calib_max_seq_length,
                        awq_block_size,
                        output_dir,
                        seed,
                        tokenizer_max_seq_length,
                        auto_quantize_bits=None,
                        device_map="auto",
                        quantize_lm_head=False):
    '''
        Load model from the model_dir, call Modelopt to quantize the model, and then export
        the quantized model as TRT-LLM checkpoint
    '''

    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is required for inference.")

    random.seed(seed)
    np.random.seed(seed)

    # Check that only one quantization format is provided for non auto_quant case
    if not auto_quantize_bits:
        assert (len(qformat.split(",")) == 1
                ), "Quantization supports only one quantization format."

    hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    dtype = infer_dtype(dtype, getattr(hf_config, 'torch_dtype', None))

    model = get_model(model_dir, dtype, device=device, device_map=device_map)
    model_type = get_model_type(model)
    tokenizer = get_tokenizer(model_dir,
                              max_seq_length=tokenizer_max_seq_length,
                              model_type=model_type)

    if qformat in ["full_prec", "int8_wo", "int4_wo"
                   ] and kv_cache_dtype is None:
        logger.info(f"No quantization applied, export {dtype} model")
    else:
        if "awq" in qformat:
            if calib_size > 32:
                logger.info(
                    f"AWQ calibration could take longer with calib_size = {calib_size}, Using"
                    " calib_size=32 instead")
                calib_size = 32
            logger.info(
                "\nAWQ calibration could take longer than other calibration methods. Please"
                " increase the batch size to speed up the calibration process. Batch size can be"
                " set by adding the argument --batch_size <batch_size> to the command line.\n"
            )

        quant_cfg = None
        if not auto_quantize_bits:
            if qformat in quant_cfg_choices():
                quant_cfg = quant_cfg_choices()[qformat]
            else:
                raise ValueError(f"Unsupported quantization format: {qformat}")

            if "awq" in qformat:
                quant_cfg = copy.deepcopy(quant_cfg_choices()[qformat])
                weight_quantizer = quant_cfg["quant_cfg"]["*weight_quantizer"]
                if isinstance(weight_quantizer, list):
                    weight_quantizer = weight_quantizer[0]
                if awq_block_size:
                    weight_quantizer["block_sizes"][-1] = awq_block_size

            if kv_cache_dtype is not None:
                if kv_cache_dtype == "fp8" or kv_cache_dtype == "nvfp4":
                    kv_cache_quant_cfg = getattr(
                        mtq, KV_QUANT_CFG_CHOICES[kv_cache_dtype])["quant_cfg"]
                    quant_cfg["quant_cfg"].update(kv_cache_quant_cfg)
                else:
                    quant_cfg["quant_cfg"].update(KV_CACHE_CFG)  # type: ignore

            if qformat == 'fp8' and quantize_lm_head:
                print_rank_0("Quantizing lm_head layer")
                del quant_cfg["quant_cfg"]["*lm_head*"]

        calib_dataloader = get_calib_dataloader(
            dataset_name_or_dir=calib_dataset,
            tokenizer=tokenizer,
            batch_size=batch_size,
            calib_size=calib_size,
            block_size=calib_max_seq_length,
            device=model.device,
            include_labels=auto_quantize_bits is not None,
        )

        model = quantize_model(model, quant_cfg, calib_dataloader, batch_size,
                               qformat, auto_quantize_bits)

    with torch.inference_mode():
        if model_type is None:
            logger.info(
                f"Unknown model type {type(model).__name__}. Continue exporting..."
            )
            model_type = f"unknown:{type(model).__name__}"

        architecture = type(model).__name__

        export_path = output_dir
        start_time = time.time()

        export_hf_checkpoint(
            model,
            export_dir=export_path,
        )

        # save tokenizer
        tokenizer.save_pretrained(export_path)

        end_time = time.time()
        logger.info(
            "Quantized model exported to {} \nTotal time used {:.2f} s.".format(
                export_path, end_time - start_time))

        # Need to delete the model and release memory explicitly;
        # otherwise torch may retain its GPU memory until a delayed GC running,
        # which reduces the available GPU memory for subsequent stages.
        del model
        release_gc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", "--model-dir",
                        help="Specify where the HuggingFace model is",
                        default=None)
    parser.add_argument(
        '--decoder_type',
        '--decoder-type',
        type=str,
        default='gptnext',
        choices=['gptnext', 'llama'],
        help="Decoder type; effective for NeMo checkpoint only.")
    parser.add_argument(
        '--device',
        help="The device to run calibration; effective for HuggingFace model only.",
        default='cuda',
        choices=['cuda', 'cpu'])
    parser.add_argument(
        '--device_map',
        '--device-map',
        help="How to map the model on the devices",
        default="auto",
        choices=["auto", "sequential", "cpu", "gpu"],
    )
    parser.add_argument(
        '--calib_dataset',
        '--calib-dataset',
        type=str,
        default='cnn_dailymail',
        help="The huggingface dataset name or the local directory of the dataset for calibration."
    )
    parser.add_argument(
        '--calib_tp_size',
        '--calib-tp-size',
        type=int,
        default=1,
        help="Tensor parallel size for calibration; effective for NeMo checkpoint only."
    )
    parser.add_argument(
        '--calib_pp_size',
        '--calib-pp-size',
        type=int,
        default=1,
        help="Pipeline parallel size for calibration; effective for NeMo checkpoint only."
    )

    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'float16', 'bfloat16', 'float32'],
        help="The data type for the model weights and activations of the non-quantized part, e.g., embedding and lm_head. "
        "If 'auto', the data type is automatically inferred from the source model; "
        "however, if the source dtype is float32, it is converted to float16.")
    parser.add_argument(
        "--qformat",
        help="Quantization format.",
        default="full_prec",
        choices=[
            "nvfp4",
            "fp8",
            "int8_sq",
            "int4_awq",
            "w4a8_awq",
            "int8_wo",
            "int4_wo",
            "full_prec",
        ],
    )
    parser.add_argument(
        "--seed",
        help="Seed the generate random numbers, the value will be used to call"
        "random.seed(value) and numpy.random.seed(value)",
        type=int,
        default=1234)
    parser.add_argument("--tokenizer_max_seq_length",
                        '--tokenizer-max-seq-length',
                        help="Max sequence length to init the tokenizers",
                        type=int,
                        default=2048)

    parser.add_argument("--batch_size",
                        '--batch-size',
                        help="Batch size for calibration.",
                        type=int,
                        default=1)
    parser.add_argument("--calib_size",
                        '--calib-size',
                        help="Number of samples for calibration.",
                        type=int,
                        default=512)
    parser.add_argument("--calib_max_seq_length",
                        '--calib-max-seq-length',
                        help="Max sequence length for calibration",
                        type=int,
                        default=512)
    parser.add_argument("--output_dir", "--output-dir", default="exported_model")
    parser.add_argument("--awq_block_size", "--awq-block-size", type=int, default=128)
    parser.add_argument("--kv_cache_dtype", "--kv-cache-dtype",
                        help="KV Cache dtype.",
                        default=None,
                        choices=["int8", "fp8", "nvfp4", None])
    parser.add_argument("--quantize_lm_head", "--quantize-lm-head",
                        action='store_true',
                        default=False)

    # auto quantization
    parser.add_argument(
        '--autoq_format',
        '--autoq-format',
        default=None,
        type=str,
        help="Specific quantization algorithms will be searched in auto quantization."
        "The algorithm must in ['fp8', 'int4_awq', 'w4a8_awq', 'int8_sq']."
        "You can use ',' to separate more than one quantization algorithms(e.g. --autoq_format fp8,int4_awq,w4a8_awq)."
        "Notice: fp8 and int8_sq can't be used at the same time.")
    parser.add_argument(
        '--auto_quantize_bits',
        '--auto-quantize-bits',
        type=float,
        default=None,
        help="Effective bits constraint for auto quantization. If not set, "
        "regular quantization without auto quantization search will be applied."
        "You can't set it lower than the num_bits of most aggressive quantization format."
        "For example, if 'int4_awq' is in autoq_format, it can't be lower than 4.0."
    )

    args = parser.parse_args()

    # auto_quantize_bits check
    if args.autoq_format:
        lower_bound, upper_bound = 4 if '4' in args.autoq_format else 8, 16
        if args.auto_quantize_bits is None or args.auto_quantize_bits < lower_bound or args.auto_quantize_bits > upper_bound:
            print(
                f"invalid auto_quantize_bits value, will be set to {lower_bound}"
            )
            args.auto_quantize_bits = lower_bound

    if args.model_dir is not None:
        quantize_and_export(
            model_dir=args.model_dir,
            device=args.device,
            calib_dataset=args.calib_dataset,
            dtype=args.dtype,
            qformat=args.qformat
            if args.auto_quantize_bits is None else args.autoq_format,
            kv_cache_dtype=args.kv_cache_dtype,
            calib_size=args.calib_size,
            batch_size=args.batch_size,
            calib_max_seq_length=args.calib_max_seq_length,
            awq_block_size=args.awq_block_size,
            output_dir=args.output_dir,
            seed=args.seed,
            tokenizer_max_seq_length=args.tokenizer_max_seq_length,
            auto_quantize_bits=args.auto_quantize_bits,
            device_map=args.device_map,
            quantize_lm_head=args.quantize_lm_head)
    else:
        raise ValueError(
            "Source checkpoint model_dir must be specified"
        )
