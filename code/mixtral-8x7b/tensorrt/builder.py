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
from code.llmlib.builder import QuantizerConfig, TRTLLMQuantizerOp


@autoconfigure
@bind(model_fields.model_path)
@bind(builder_fields.calib_data_dir, "dataset_path")
@bind(llm_fields.tensor_parallelism, "tp_size")
@bind(llm_fields.pipeline_parallelism, "pp_size")
@dataclasses.dataclass(init=False)
class Mixtral8x7bQuantizerConfig(QuantizerConfig):
    tp_size: int = 1
    pp_size: int = 1

    def __init__(self,
                 *args,
                 model_name: str = "mixtral-8x7b-instruct-v0.1",
                 model_path: os.PathLike = paths.MODEL_DIR / "Mixtral/Mixtral-8x7B-Instruct-v0.1",
                 dataset_path: os.PathLike = paths.BUILD_DIR / "data/moe/mlperf_mixtral8x7b_moe_calibration_dataset_1k.pkl",
                 tp_size: int = 1,
                 pp_size: int = 1,
                 **kwargs):
        super().__init__(*args,
                         model_path=model_path,
                         dataset_path=dataset_path,
                         output_path=paths.MODEL_DIR / "Mixtral",
                         **kwargs)
        self.tp_size = tp_size
        self.pp_size = pp_size

        assert "effective_bits" in self.flags, "effective_bits must be specified in TRTLLM checkpoint flags"
        self.effective_bits = self.flags["effective_bits"]

        ckpt_dir_map = {Precision.FP8: 'fp8-quantized-modelopt',
                        Precision.FP4: 'fp4-quantized-modelopt',
                        Precision.NVFP4: 'fp4-quantized-modelopt'}
        assert self.dtype_out in ckpt_dir_map, f"Unsupported Precision for Mixtral-8x7B: {self.dtype_out.valstr}"
        if self.dtype_out in [Precision.FP4, Precision.NVFP4]:
            # Override for manually generated quantized checkpoint
            ckpt_name = "mixtral-8x7b-instruct-v0.1-tp1pp1-fp4-e7.25-passing_acuracies"
        else:
            ckpt_name = f"{model_name}-tp{self.tp_size}pp{self.pp_size}-{self.dtype_out.valstr}-e{self.effective_bits}"
        self.output_path = self.output_path / ckpt_dir_map[self.dtype_out] / ckpt_name
        if not self.output_path.exists():
            if Path("/opt/fp4-quantized-modelopt/").exists():
                logging.error("Please extract the tarball from /opt/fp4-quantized-modelopt/ into build/models/Mixtral/fp4-quantized-modelopt/.")
            else:
                logging.error("Could not find model tarball in /opt/fp4-quantized-modelopt/.")
            raise FileNotFoundError("Could not find the Mixtral-8x7B checkpoint.")


class Mixtral8x7bQuantizerOp(TRTLLMQuantizerOp):
    @classmethod
    def immediate_dependencies(cls):
        return None

    @classmethod
    def output_keys(cls):
        return ["quantized_checkpoint_path"]

    def __init__(self,
                 *args,
                 quantizer_config: Optional[Mixtral8x7bQuantizerConfig] = None,
                 script_path: os.PathLike = Path("code/mixtral-8x7b/modelopt/main.py"),
                 lib_path: os.PathLike = Path.cwd(),
                 **kwargs):
        if quantizer_config is None:
            quantizer_config = Mixtral8x7bQuantizerConfig()

        super().__init__(*args,
                         quantizer_config=quantizer_config,
                         script_path=script_path,
                         lib_path=lib_path,
                         **kwargs)

        self.custom_env = {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}

    def get_cli_flags(self) -> List[str]:
        _d = {
            'model_path': str(self.quantizer_config.model_path.absolute()),
            'quantized_checkpoint_path': str(self.quantizer_config.output_path.absolute()),
            'calib_dataset_path': self.quantizer_config.dataset_path,
            'calib_batch_size': self.quantizer_config.batch_size,
            'effective_bits': self.quantizer_config.effective_bits,
            'tp_size': self.trtllm_loader.tp_size,
            'pp_size': self.trtllm_loader.pp_size,
            'num_calib_steps': self.quantizer_config.flags['num_calib_steps'],
            'num_score_steps': self.quantizer_config.flags['num_score_steps'],
            'fp4': self.quantizer_config.dtype_out == Precision.FP4,
        }
        return [f"--{k}={v}" for k, v in _d.items() if v is not None]
