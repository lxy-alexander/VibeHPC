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

from __future__ import annotations
from os import PathLike
from pathlib import Path
import onnx
import os
import tempfile
import tensorrt as trt
import shutil
import polygraphy.logger

from typing import Dict, List, Optional
from importlib import import_module

from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    engine_from_network,
    engine_from_bytes,
    modify_network_outputs,
    save_engine,

)
from polygraphy.util import (
    load_file
)

from nvmitten.configurator import autoconfigure, bind
from nvmitten.constants import Precision
from nvmitten.pipeline import Operation
from nvmitten.utils import dict_get, logging

from code.common import paths
from code.fields.wrapped import TRTBuilder, CalibratableTensorRTEngine
from code.common.systems.system_list import DETECTED_SYSTEM
from code.fields import gen_engines as builder_fields
from code.fields import models as model_fields

from . import fields as sdxl_fields
from .network import AbstractModel, CLIP, CLIPWithProj, UNetXL, VAE
from .sdxl_graphsurgeon import SDXLGraphSurgeon
from .constants import SDXLComponent

polygraphy.logger.G_LOGGER.module_severity = polygraphy.logger.G_LOGGER.ERROR


class ScopedTimingCache:
    def __init__(self, builder):
        self.builder = builder
        if self.builder.use_timing_cache and self.builder.timing_cache_path:
            self.original_timing_cache_path = self.builder.timing_cache_path

    def __enter__(self):
        if self.builder.use_timing_cache and self.builder.timing_cache_path:
            shutil.copyfile(self.builder.timing_cache_path, f'{self.builder.timing_cache_path}_in_use')
            self.builder.timing_cache_path = self.builder.timing_cache_path + "_in_use"
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.builder.use_timing_cache and self.builder.timing_cache_path:
            os.remove(f'{self.original_timing_cache_path}_in_use')
            self.builder.timing_cache_path = self.original_timing_cache_path
        return False


@autoconfigure
@bind(model_fields.model_path)
@bind(builder_fields.workspace_size)
@bind(builder_fields.strongly_typed)
@bind(builder_fields.skip_graphsurgeon)
@bind(sdxl_fields.use_native_instance_norm)
@bind(model_fields.precision)
class SDXLComponentBuilder(TRTBuilder):
    def __init__(self,
                 component_subpath: os.PathLike,
                 model: AbstractModel,
                 *args,
                 model_path: os.PathLike = paths.MODEL_DIR / "SDXL",
                 batch_size: int = 1,
                 precision: Precision = Precision.FP32,
                 workspace_size: int = 80 << 30,
                 strongly_typed: bool = False,
                 use_native_instance_norm: bool = False,
                 skip_graphsurgeon: bool = False,
                 include_hidden_states: bool = False,
                 device_type: str = "gpu",
                 **kwargs):
        precision = precision if isinstance(precision, Precision) else precision[model.name]

        # Force num_profiles to 1 for SDXL, not sure if multiple execution context can help heavy benchmarks
        super().__init__(*args,
                         num_profiles=1,
                         workspace_size=workspace_size,
                         precision=precision,
                         **kwargs)

        self.component_name = model.name
        self.model = model
        self.model_path = Path(model_path) / component_subpath
        if not self.model_path.exists():
            raise FileNotFoundError(f"Component model file {self.model_path} does not exist")

        self.batch_size = batch_size
        self.device_type = device_type
        self.strongly_typed = strongly_typed
        self.use_native_instance_norm = use_native_instance_norm
        self.include_hidden_states = include_hidden_states

        self.skip_gs = skip_graphsurgeon
        self.use_timing_cache = False

    def create_network(self, builder: trt.Builder = None) -> trt.INetworkDefinition:
        flags = 0
        if self.strongly_typed:
            flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        network = super().create_network(builder=builder, flags=flags)

        parser = trt.OnnxParser(network, self.logger)

        # set instance norm flag for better perf of SDXL
        if self.use_native_instance_norm:
            parser.set_flag(trt.OnnxParserFlag.NATIVE_INSTANCENORM)

        if self.skip_gs:
            success = parser.parse_from_file(str(self.model_path))
            if not success:
                err_desc = parser.get_error(0).desc()
                raise RuntimeError(f"Parse SDXL graphsurgeon onnx model failed! Error: {err_desc}")
        else:
            sdxl_gs = SDXLGraphSurgeon(self.model_path,
                                       self.precision,
                                       self.device_type,
                                       self.model.name,
                                       add_hidden_states=self.include_hidden_states)
            model = sdxl_gs.create_onnx_model()

            if model.ByteSize() >= SDXLGraphSurgeon.ONNX_LARGE_FILE_THRESHOLD:
                # onnx._serialize cannot take input proto >= 2 BG
                # We need to save proto larger than 2GB into separate files and parse from files
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir)
                    tmp_path.mkdir(exist_ok=True)
                    onnx_tmp_path = tmp_path / "tmp_model.onnx"
                    onnx.save_model(model,
                                    str(onnx_tmp_path),
                                    save_as_external_data=True,
                                    all_tensors_to_one_file=True,
                                    convert_attribute=False)
                    success = parser.parse_from_file(str(onnx_tmp_path))
                    if not success:
                        err_desc = parser.get_error(0).desc()
                        raise RuntimeError(f"Parse SDXL graphsurgeon onnx model failed! Error: {err_desc}")
            else:
                # Parse from ONNX file
                success = parser.parse(model.SerializeToString())
                if not success:
                    err_desc = parser.get_error(0).desc()
                    raise RuntimeError(f"Parse SDXL graphsurgeon onnx model failed! Error: {err_desc}")

        logging.info(f"Updating network outputs to {self.model.get_output_names()}")
        _, network, _ = modify_network_outputs((self.builder, network, parser), self.model.get_output_names())

        if not self.strongly_typed:
            self.apply_network_io_types(network)
        return network

    def apply_network_io_types(self, network: trt.INetworkDefinition):
        """Applies I/O dtype and formats for network inputs and outputs to the tensorrt.INetworkDefinition.
        Args:
            network (tensorrt.INetworkDefinition): The network generated from the builder.
        """
        # Set input dtype
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            if self.precision == Precision.FP32:
                input_tensor.dtype = trt.float32
            elif self.precision == Precision.FP16 or self.precision == Precision.INT8:
                input_tensor.dtype = trt.float16

        # Set output dtype
        for i in range(network.num_outputs):
            output_tensor = network.get_output(i)
            if self.precision == Precision.FP32:
                output_tensor.dtype = trt.float32
            elif self.precision == Precision.FP16 or self.precision == Precision.INT8:
                output_tensor.dtype = trt.float16

    # Overwrites mitten function with the same signature, `network` is unused
    def gpu_profiles(self,
                     network: trt.INetworkDefinition,
                     batch_size: int):
        profile = Profile()
        input_profile = self.model.get_input_profile(batch_size)
        for name, dims in input_profile.items():
            assert len(dims) == 3
            profile.add(name, min=dims[0], opt=dims[1], max=dims[2])
        return [profile]

    def create_builder_config(self,
                              workspace_size: Optional[int] = None,
                              profiles: Optional[List[Profile]] = None,
                              precision: Optional[Precision] = None,
                              **kwargs) -> trt.IBuilderConfig:
        # TODO: yihengz explore if builder_optimization_level = 5 can get better perf, disabling for making engine build time too long
        if precision is None:
            precision = self.precision
        if workspace_size is None:
            workspace_size = self.workspace_size

        if self.strongly_typed and self.use_timing_cache:
            logging.info(f"Using timing cache file.")
            builder_config = CreateConfig(
                profiles=profiles,
                tf32=True,
                profiling_verbosity=trt.ProfilingVerbosity.DETAILED if (self.verbose or self.verbose_nvtx) else trt.ProfilingVerbosity.LAYER_NAMES_ONLY,
                load_timing_cache=self.timing_cache_path,
                memory_pool_limits={trt.MemoryPoolType.WORKSPACE: workspace_size}
            )
        elif self.strongly_typed:
            builder_config = CreateConfig(
                profiles=profiles,
                tf32=True,
                profiling_verbosity=trt.ProfilingVerbosity.DETAILED if (self.verbose or self.verbose_nvtx) else trt.ProfilingVerbosity.LAYER_NAMES_ONLY,
                memory_pool_limits={trt.MemoryPoolType.WORKSPACE: workspace_size}
            )
        else:
            builder_config = CreateConfig(
                profiles=profiles,
                int8=precision == Precision.INT8,
                fp16=precision == Precision.FP16 or precision == Precision.INT8,
                tf32=precision == Precision.FP32,
                profiling_verbosity=trt.ProfilingVerbosity.DETAILED if (self.verbose or self.verbose_nvtx) else trt.ProfilingVerbosity.LAYER_NAMES_ONLY,
                memory_pool_limits={trt.MemoryPoolType.WORKSPACE: workspace_size}
            )

        return builder_config

    def build_engine(self,
                     network: trt.INetworkDefinition,
                     builder_config: trt.IBuilderConfig,  # created inside
                     save_to: PathLike):
        save_to = Path(save_to)
        if save_to.is_file():
            logging.warning(f"{save_to} already exists. This file will be overwritten")
        save_to.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Building TensorRT engine for {self.model_path}: {save_to}")

        if self.use_timing_cache:
            if self.timing_cache_path is None:
                logging.warning("No cache path is set.")

            trt_config = builder_config.call_impl(self.builder, network)

            timing_cache = trt_config.get_timing_cache()
            prev_cache = trt_config.create_timing_cache(load_file(self.timing_cache_path))

            if self.verbose:
                if builder_config:
                    if timing_cache is None:
                        logging.verbose("No local timing cache is used")
                    else:
                        logging.info("timing cache is set!")

            if timing_cache:
                combine_success = timing_cache.combine(
                    prev_cache, ignore_mismatch=True
                )
                if not combine_success:
                    logging.info("Could not combine old timing cache into current timing cache")
                else:
                    logging.info("Successfully combine engine!")
                assert combine_success

            serialized_engine = self.builder.build_serialized_network(network, trt_config)
            engine = engine_from_bytes(serialized_engine)
        else:
            engine = engine_from_network(
                (self.builder, network),
                config=builder_config,
            )

            if self.verbose:
                engine_inspector = engine.create_engine_inspector()
                layer_info = engine_inspector.get_engine_information(trt.LayerInformationFormat.ONELINE)
                logging.info("========= TensorRT Engine Layer Information =========")
                logging.info(layer_info)

                # [https://nvbugs/3965323] Need to delete the engine inspector to release the refcount
                del engine_inspector

        save_engine(engine, path=save_to)

    def __call__(self, *args, **kwargs):
        with ScopedTimingCache(self) as scoped_timing_cache:
            return super().__call__(*args, **kwargs)


class SDXLCLIPBuilder(SDXLComponentBuilder):
    """SDXL CLIP builder class.
    """

    def __init__(self, *args, batch_size: int = 2, **kwargs):
        _precision = Precision.FP32 if int(DETECTED_SYSTEM.extras["primary_compute_sm"]) >= 100 else Precision.FP16
        super().__init__(*args,
                         component_subpath="onnx_models/clip1/model.onnx",
                         model=CLIP(name=SDXLComponent.CLIP1.valstr,
                                    max_batch_size=batch_size,
                                    precision=_precision,
                                    device='cuda'),
                         batch_size=batch_size,
                         precision=_precision,
                         include_hidden_states=True,
                         **kwargs)
        if int(DETECTED_SYSTEM.extras["primary_compute_sm"]) >= 100:
            self.use_timing_cache = True
            self.timing_cache_path = str(paths.WORKING_DIR / "scripts/cache/clip1.cache")

    def apply_network_io_types(self, network: trt.INetworkDefinition):
        """Applies I/O dtype and formats for network inputs and outputs to the tensorrt.INetworkDefinition.
        CLIP keeps int32 input (tokens)
        Args:
            network (tensorrt.INetworkDefinition): The network generated from the builder.
        """
        # Set output dtype
        for i in range(network.num_outputs):
            output_tensor = network.get_output(i)
            if self.precision == Precision.FP32:
                output_tensor.dtype = trt.float32
            elif self.precision == Precision.FP16:
                output_tensor.dtype = trt.float16


class SDXLCLIPWithProjBuilder(SDXLComponentBuilder):
    """SDXL CLIPWithProj builder class.
    """

    def __init__(self,
                 *args,
                 batch_size: int = 2,
                 **kwargs):
        _precision = Precision.FP32 if int(DETECTED_SYSTEM.extras["primary_compute_sm"]) >= 100 else Precision.FP16
        super().__init__(*args,
                         component_subpath="onnx_models/clip2/model.onnx",
                         model=CLIPWithProj(name=SDXLComponent.CLIP2.valstr,
                                            max_batch_size=batch_size,
                                            precision=_precision,
                                            device='cuda'),
                         batch_size=batch_size,
                         precision=_precision,
                         include_hidden_states=True,
                         **kwargs)

        if int(DETECTED_SYSTEM.extras["primary_compute_sm"]) >= 100:
            self.use_timing_cache = True
            self.timing_cache_path = str(paths.WORKING_DIR / "scripts/cache/clip2.cache")

    def apply_network_io_types(self, network: trt.INetworkDefinition):
        """Applies I/O dtype and formats for network inputs and outputs to the tensorrt.INetworkDefinition.
        CLIPWithProj keeps int32 input (tokens)
        Args:
            network (tensorrt.INetworkDefinition): The network generated from the builder.
        """
        # Set output dtype
        for i in range(network.num_outputs):
            output_tensor = network.get_output(i)
            if self.precision == Precision.FP32:
                output_tensor.dtype = trt.float32
            elif self.precision == Precision.FP16:
                output_tensor.dtype = trt.float16


@autoconfigure
@bind(model_fields.precision)
class SDXLUNetXLBuilder(SDXLComponentBuilder):
    """SDXL UNetXL builder class.
    """

    def __init__(self,
                 *args,
                 batch_size: int = 2,
                 precision: str = Precision.INT8,
                 **kwargs):
        if isinstance(precision, dict):
            precision = precision[SDXLComponent.UNETXL]

        _precision_map = {Precision.FP8: ("modelopt_models/unetxl.fp8/unet.onnx", True),
                          Precision.INT8: ("modelopt_models/unetxl.int8/unet.onnx", False),
                          Precision.FP16: ("onnx_models/unetxl/model.onnx", False)}
        if precision not in _precision_map:
            raise ValueError("Unsupported UNetXL precision")
        unetxl_path, strongly_typed = _precision_map[precision]

        # UNET will hit OOM error in GraphSurgeon on Orin
        skip_gs = "is_soc" in DETECTED_SYSTEM.extras["tags"]

        super().__init__(*args,
                         component_subpath=unetxl_path,
                         model=UNetXL(name=SDXLComponent.UNETXL.valstr,
                                     max_batch_size=batch_size,
                                     precision=precision,
                                     device='cuda'),
                         batch_size=batch_size,
                         strongly_typed=strongly_typed,
                         skip_graphsurgeon=skip_gs,
                         precision=precision,
                         **kwargs)

        if int(DETECTED_SYSTEM.extras["primary_compute_sm"]) >= 100:
            self.use_timing_cache = True
            self.timing_cache_path = str(paths.WORKING_DIR / "scripts/cache/unet.cache")


class SDXLVAEBuilder(SDXLComponentBuilder):
    """SDXL VAE builder class.
    """

    def __init__(self,
                 *args,
                 batch_size: int = 1,
                 **kwargs):

        if "is_soc" in DETECTED_SYSTEM.extras["tags"]:
            precision = Precision.FP32
            component_subpath = "onnx_models/vae/model.onnx"
            strongly_typed = False
        elif int(DETECTED_SYSTEM.extras["primary_compute_sm"]) >= 100:
            precision = Precision.FP32
            component_subpath = "onnx_models/vae/model.onnx"
            strongly_typed = False
        else:
            precision = Precision.INT8
            component_subpath = "modelopt_models/vae.int8/vae.onnx"
            strongly_typed = True

        super().__init__(*args,
                         component_subpath=component_subpath,
                         model=VAE(name=SDXLComponent.VAE.valstr,
                                 max_batch_size=batch_size,
                                 precision=precision,
                                 device='cuda'),
                         batch_size=batch_size,
                         precision=precision,
                         strongly_typed=strongly_typed,
                         use_native_instance_norm=True,
                         # GraphSurgeon causes accuracy issues on B200
                         skip_graphsurgeon=(int(DETECTED_SYSTEM.extras["primary_compute_sm"]) >= 100),
                         **kwargs)

        if int(DETECTED_SYSTEM.extras["primary_compute_sm"]) >= 100:
            self.use_timing_cache = True
            self.timing_cache_path = str(paths.WORKING_DIR / "scripts/cache/vae.cache")
