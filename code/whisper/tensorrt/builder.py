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

import os
from pathlib import Path

from nvmitten.constants import Precision
from nvmitten.configurator import autoconfigure, bind

import code.common.paths as paths
from . import fields as whisper_fields
from code.fields import models as model_fields
from code.fields import gen_engines as builder_fields
from code.llmlib.builder import TRTLLMBuilderOp, QuantizerConfig, TRTLLMQuantizerOp
from code.llmlib.utils import prefix_logger as logging
from code.common.workload import EngineIndex


@autoconfigure
@bind(model_fields.model_path)
@bind(builder_fields.calib_data_dir, "dataset_path")
@bind(model_fields.precision, "dtype_out")
class WhisperQuantizerConfig(QuantizerConfig):

    def __init__(self,
                 *args,
                 model_name: str = "whisper-large-v3",
                 model_path: os.PathLike = paths.MODEL_DIR / "whisper-large-v3",
                 dataset_path: os.PathLike = paths.BUILD_DIR / "whisper-large-v3",
                 dtype_out: Precision = Precision.FP16,
                 **kwargs):

        ckpt_dir_map = {Precision.FP16: 'whisper_large_v3_float16_weight_ckt'}
        if dtype_out not in ckpt_dir_map:
            raise ValueError(f"Unsupported Precision for Whisper: {dtype_out.valstr}")

        output_path = paths.MODEL_DIR / "whisper-large-v3" / ckpt_dir_map[dtype_out]

        super().__init__(*args,
                         model_path=model_path,
                         dataset_path=dataset_path,
                         dtype_out=dtype_out,
                         output_path=output_path,
                         **kwargs)


class WhisperQuantizerOp(TRTLLMQuantizerOp):
    @classmethod
    def immediate_dependencies(cls):
        return None

    @classmethod
    def output_keys(cls):
        return ["quantized_checkpoint_path"]

    def dtype_full(self, dtype):
        if dtype == 'fp16':
            return 'float16'
        if dtype == 'bf16':
            return 'bfloat16'
        if dtype == 'fp32':
            return 'float32'

    def __init__(self,
                 *args,
                 quantizer_config: Optional[WhisperEncoderQuantizerConfig] = None,
                 script_path: os.PathLike = Path("/work/code/whisper/tensorrt/convert_checkpoint.py"),
                 lib_path: os.PathLike = Path.cwd(),
                 **kwargs):
        if quantizer_config is None:
            quantizer_config = WhisperQuantizerConfig()

        super().__init__(*args,
                         quantizer_config=quantizer_config,
                         script_path=script_path,
                         lib_path=lib_path,
                         **kwargs)

    def get_cli_flags(self) -> List[str]:
        _d = {
            "model_dir": str(self.quantizer_config.model_path.absolute()),
            "output_dir": str(self.quantizer_config.output_path.absolute()),
            "dtype": self.dtype_full(self.quantizer_config.dtype_in.valstr.lower()),
            "model_name": "large-v3"
        }
        _d |= self.quantizer_config.flags

        return [f"--{k}={v}" for k, v in _d.items() if v is not None]

    def run(self, scratch_space, dependency_outputs):
        out_dir = self.quantizer_config.output_path.absolute()
        if not self.force and out_dir.exists():
            logging.info(f"Quantized checkpoint already exists at: {out_dir}. Set --force_calibration to overwrite.")
            return {"quantized_checkpoint_path": out_dir}

        if not out_dir.exists():
            out_dir.mkdir(parents=True)
        script_module = '.'.join(self.script_path.with_suffix('').parts)
        ret, duration = self.trtllm_loader(str(self.script_path),
                                           self.get_cli_flags(),
                                           python_bin=self.wl.benchmark.python_path,
                                           log_dir=out_dir,
                                           custom_env=self.custom_env)

        if ret.returncode != 0:
            logging.error(ret.stderr)
            raise RuntimeError(f"Quantization failed. Logs dumped to: {out_dir}")

        logging.info(f"Quantization complete in {duration}s. Saved to: {out_dir}")
        return {"quantized_checkpoint_path": out_dir}


@autoconfigure
@bind(whisper_fields.whisper_encoder_build_flags, "build_flags")
@bind(whisper_fields.whisper_decoder_build_flags, "build_flags_dec")
@bind(builder_fields.force_build_engines, "force")
class WhisperBuilderOp(TRTLLMBuilderOp):

    @classmethod
    def immediate_dependencies(cls):
        return {WhisperQuantizerOp}

    @classmethod
    def output_keys(cls):
        return ["engine_dir", "engine_index"]

    def __init__(self,
                 *args,
                 lib_path: os.PathLike = paths.TRTLLM_DIR,
                 script_path: os.PathLike = Path("tensorrt_llm/commands/build.py"),
                 build_flags: Optional[Dict[str, Any]] = None,
                 build_flags_dec: Optional[Dict[str, Any]] = None,
                 force: bool = True,
                 **kwargs):
        self.build_flags_dec = build_flags_dec
        super().__init__(*args,
                         lib_path=lib_path,
                         script_path=script_path,
                         build_flags=build_flags,
                         force=force,
                         **kwargs)

        self.engine_index = EngineIndex()
        self.wl = self.engine_index.wl

    def run(self, scratch_space, dependency_outputs):
        ckpt_path = paths.MODEL_DIR / "whisper-large-v3/whisper_large_v3_float16_weight_ckt/encoder"
        assert ckpt_path.exists(), f"Quantized checkpoint not found at: {ckpt_path}"
        assert (ckpt_path / 'rank0.safetensors').exists(), f"rank0.safetensors not found at: {ckpt_path}"

        c_eng = self.engine_index.engines[0]
        out_dir = self.engine_index.full_path(c_eng)
        out_dir_enc = out_dir / "encoder"
        if not self.force and out_dir_enc.exists():
            logging.info(f"Engine already exists at: {out_dir_enc}. Set --force_build_engines to overwrite.")
            return {"engine_dir": out_dir, "engine_index": self.engine_index}
        elif not out_dir_enc.exists():
            out_dir_enc.mkdir(parents=True)

        logging.debug(f"Encoder build_flags: {self.build_flags}")

        script_module = '.'.join(self.script_path.with_suffix('').parts)
        argv = [f'--checkpoint_dir={str(ckpt_path.absolute())}',
                f'--output_dir={str(out_dir_enc.absolute())}'] + self.get_cli_flags()
        ret, duration = self.trtllm_loader(str(self.script_path),
                                           argv,
                                           python_bin=self.wl.benchmark.python_path,
                                           log_dir=out_dir_enc)

        if ret.returncode != 0:
            logging.error(ret.stderr)
            raise RuntimeError(f"Engine build failed. Logs dumped to: {out_dir_enc}")

        logging.info(f"Engine build complete in {duration}s. Saved to: {out_dir_enc}")

        # Decoder
        ckpt_path = paths.MODEL_DIR / "whisper-large-v3/whisper_large_v3_float16_weight_ckt/decoder"
        assert ckpt_path.exists(), f"Quantized checkpoint not found at: {ckpt_path}"
        assert (ckpt_path / 'rank0.safetensors').exists(), f"rank0.safetensors not found at: {ckpt_path}"
        # update build_flags
        self.build_flags = self.build_flags_dec
        logging.debug(f"Update build_flags for decoder: {self.build_flags}")
        c_eng = self.engine_index.engines[1]
        out_dir = self.engine_index.full_path(c_eng)
        out_dir_dec = out_dir / "decoder"

        if not self.force and out_dir_dec.exists():
            logging.info(f"Engine already exists at: {out_dir_dec}. Set --force_build_engines to overwrite.")
            return {"engine_dir": out_dir, "engine_index": self.engine_index}
        elif not out_dir_dec.exists():
            out_dir_dec.mkdir(parents=True)

        script_module = '.'.join(self.script_path.with_suffix('').parts)
        argv = [f'--checkpoint_dir={str(ckpt_path.absolute())}',
                f'--output_dir={str(out_dir_dec.absolute())}'] + self.get_cli_flags()
        ret, duration = self.trtllm_loader(str(self.script_path),
                                           argv,
                                           python_bin=self.wl.benchmark.python_path,
                                           log_dir=out_dir_dec)

        if ret.returncode != 0:
            logging.error(ret.stderr)
            raise RuntimeError(f"Engine build failed. Logs dumped to: {out_dir_dec}")

        logging.info(f"Engine build complete in {duration}s. Saved to: {out_dir_dec}")

        return {"engine_dir": out_dir, "engine_index": self.engine_index}
