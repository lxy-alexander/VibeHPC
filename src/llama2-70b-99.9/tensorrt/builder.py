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

from __future__ import annotations
import dataclasses
import os
from pathlib import Path

from nvmitten.constants import Precision
from nvmitten.configurator import autoconfigure, bind

import code.common.paths as paths
from code.fields import models as model_fields
from code.fields import gen_engines as builder_fields
from code.llmlib import fields as llm_fields
from code.llmlib.builder import QuantizerConfig
from code.llmlib.cores import BackendRegistry
from code.llmlib.config import CheckpointType
from code.llmlib.config import HarnessConfig


@autoconfigure
@bind(model_fields.model_path)
@bind(model_fields.precision, "dtype_out")
@bind(builder_fields.calib_data_dir, "dataset_path")
@bind(llm_fields.quantizer_outdir, "output_path")
@bind(llm_fields.tensor_parallelism, "tp_size")
@bind(llm_fields.pipeline_parallelism, "pp_size")
@dataclasses.dataclass(init=False)
class LLAMA2QuantizerConfig(QuantizerConfig):
    pp_size: int = 1
    tp_size: int = 1

    def __init__(self,
                 *args,
                 model_name: str = "llama2-70b-chat-hf",
                 model_path: os.PathLike = paths.MODEL_DIR / "Llama2/Llama-2-70b-chat-hf",
                 output_path: os.PathLike = paths.MODEL_DIR / "Llama2",
                 dataset_path: os.PathLike = paths.BUILD_DIR / "preprocessed_data/llama2-70b/mlperf_llama2_openorca_calibration_1k",
                 dtype_out: Precision = Precision.FP8,
                 tp_size: int = 1,
                 pp_size: int = 1,
                 **kwargs):
        self.tp_size = tp_size
        self.pp_size = pp_size
        ckpt_dir_map = {Precision.FP8: 'fp8-quantized-modelopt',
                        Precision.FP4: 'fp4-quantized-modelopt',
                        Precision.NVFP4: 'fp4-quantized-modelopt'}
        if dtype_out not in ckpt_dir_map:
            raise ValueError(f"Unsupported Precision for LLAMA2: {dtype_out.valstr}")

        # To support testing multiple variants, we used the following naming schema to store each of them
        variant_dir = f"{model_name}-tp{self.tp_size}pp{self.pp_size}-{dtype_out.valstr}"
        self.hf_output_path = output_path / ckpt_dir_map[dtype_out] / f"{model_name}-torch-{dtype_out.valstr}"
        output_path = output_path / ckpt_dir_map[dtype_out] / variant_dir

        super().__init__(*args,
                         model_path=model_path,
                         dataset_path=dataset_path,
                         dtype_out=dtype_out,
                         output_path=output_path,
                         **kwargs)
