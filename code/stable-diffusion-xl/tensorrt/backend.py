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

import threading
import queue
import array
import os
import time

import numpy as np
import tensorrt as trt
import torch

from pathlib import Path
from typing import Dict, List, Union
from importlib import import_module

from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes
from code.common import logging, run_command
from code.common.gbs import GeneralizedBatchSize
from code.common.utils import nvtx_scope
from code.common.workload import ComponentEngine
from code.fields import general as general_fields
from code.fields import harness as harness_fields

from nvmitten.configurator import autoconfigure, bind
from nvmitten.nvidia.cupy import CUDARTWrapper as cudart
import mlperf_loadgen as lg

from .constants import SDXLComponent
from .dataset import Dataset
from .utilities import PipelineConfig, numpy_to_torch_dtype_dict, calculate_max_engine_device_memory
from .network import CLIP, CLIPWithProj, UNetXL, VAE
from . import scheduler as sdxl_scheduler
from . import fields as sdxl_fields


class SDXLEngine:
    """
    Sub-Engine/Network within SDXL pipeline.
    reads engine file, loads and activates execution context
    """

    def __init__(self,
                 engine_name: str,
                 engine_path: os.PathLike):
        self.engine_name = engine_name
        self.engine_path = Path(engine_path)

        assert self.engine_path.exists(), f"Engine file does not exist: {self.engine_path}"

        logging.info(f"Loading TensorRT engine: {self.engine_path}.")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, device_memory: int):
        self.context = self.engine.create_execution_context_without_device_memory()
        self.context.device_memory = device_memory

        # NOTE(vir): need to call enable_cuda_graphs to switch to graph mode
        self.use_graphs = False

    def enable_cuda_graphs(self,
                           buffers: SDXLBufferManager,
                           stream: int = 2):
        '''
        enable cuda graphs for SDXLEngine.
        will capture graphs for all valid batch sizes.
        all subsequent calls to infer will now use cuda-graphs

        assumptions:
            - assumes activate() has been called already
            - buffers are staged already
        '''

        assert self.context is not None, "need to activate engine first"

        self.use_graphs = True
        self.cuda_graphs = {}
        logging.info(f'Enabling cuda graphs for {self.engine_name}')

        # get tensor names
        names = [self.engine.get_tensor_name(index) for index in range(self.engine.num_io_tensors)]
        num_inputs = sum([1 for name in names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT])
        input_names, output_names = names[:num_inputs], names[num_inputs:]

        # get opt profiles
        input_profiles = [list(self.engine.get_tensor_profile_shape(name, 0)) for name in input_names]
        max_shapes = [list(profile[-1]) for profile in input_profiles]
        min_shapes = [list(profile[0]) for profile in input_profiles]
        opt_shapes = [list(profile[1]) for profile in input_profiles]

        # set engine BS bounds
        self.max_batch_size = max([shape[0] for shape in max_shapes])
        self.min_batch_size = max([shape[0] for shape in min_shapes])
        self.opt_batch_size = max([shape[0] for shape in opt_shapes])

        # helper func. flatten out input dict
        def yield_inputs(table: Union[torch.Tensor, dict, list]):
            for entry in table:
                if type(entry) is torch.Tensor:
                    yield entry

                elif type(entry) is dict:
                    for sub_entry in yield_inputs(list(entry.values())):
                        yield sub_entry

                elif type(entry) is list:
                    for sub_entry in entry:
                        yield sub_entry

                else:
                    assert False, "table not supported"

        # capture graph for each even in BS: [2 ... max_batch_size]
        for actual_batch_size in range(self.min_batch_size, self.max_batch_size + 1, 2):
            # create and stage sample input
            sample_inputs = yield_inputs(UNetXL(name=self.engine_name,
                                                max_batch_size=self.max_batch_size,
                                                precision=Precision.FP16).get_sample_input(actual_batch_size))
            for name, buffer in zip(input_names, sample_inputs):
                full_name = f'{self.engine_name}_{name}'
                buffers[full_name] = buffer
            for tensor_name, tensor_shape in UNetXL(name=self.engine_name,
                                                    max_batch_size=self.max_batch_size,
                                                    precision=Precision.FP16).get_shape_dict(actual_batch_size).items():
                self.stage_tensor(tensor_name, buffers[f'{self.engine_name}_{tensor_name}'], tensor_shape)
            # first run after reshape
            noerror = self.context.execute_async_v3(stream)
            if not noerror:
                raise ValueError(f"ERROR: inference failed.")

            # capture graph
            cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal)
            self.context.execute_async_v3(stream)
            graph = cudart.cudaStreamEndCapture(stream)
            self.cuda_graphs[actual_batch_size] = cudart.cudaGraphInstantiate(graph, 0)

            # test first run of graph
            cudart.cudaGraphLaunch(self.cuda_graphs[actual_batch_size], stream)
            logging.info(f'captured graph for {self.engine_name} BS={actual_batch_size}')

    def stage_tensor(self,
                     name: str,
                     buffer: torch.Tensor,
                     shape_override: List[int] = None):
        if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            assert self.context.set_input_shape(name, shape_override or list(buffer.shape))

        assert self.context.set_tensor_address(name, buffer.data_ptr())

    def infer(self,
              stream: Union[int, cudart.cudaStream_t],
              batch_size: int = None):
        if self.use_graphs:
            actual_batch_size = self.opt_batch_size if batch_size is None else batch_size * 2
            assert self.min_batch_size <= actual_batch_size <= self.max_batch_size

            # run using appropriate cuda graph
            cudart.cudaGraphLaunch(self.cuda_graphs[actual_batch_size], stream)

        else:
            # run without cuda graph
            noerror = self.context.execute_async_v3(stream)
            if not noerror:
                raise ValueError(f"ERROR: {self.engine_path} inference failed.")

        return True


class SDXLBufferManager:
    """
    Buffer Manager for sdxl pipeline.
    manages sdxl engine buffers
    """

    def __init__(self,
                 engines_dict: Dict[str, SDXLEngine],
                 device: str = 'cuda'):
        self.engines_dict = engines_dict
        self.device = device

        self.buffers: Dict[str, torch.Tensor] = {}

        # inputs [
        #     'clip1_input_ids',
        #     'clip2_input_ids',
        #     'unet_sample',
        #     'unet_timestep',
        #     'unet_encoder_hidden_states',
        #     'unet_text_embeds',
        #     'unet_time_ids',
        #     'vae_latent',
        # ]
        self.input_tensors: Dict[str, List[int]] = {}

        # outputs [
        #     'clip1_hidden_states',
        #     'clip1_text_embeddings',
        #     'clip2_hidden_states',
        #     'clip2_text_embeddings',
        #     'unet_latent',
        #     'vae_images',
        # ]
        self.output_tensors: Dict[str, List[int]] = {}

    def to(self, device: str):
        assert device in ['cpu', 'cuda']
        self.device = device

        # put buffers on device
        for _, buffer in self.buffers.items():
            buffer.to(self.device)

    def initialize(self, shape_dict: Dict[str, List[int]] = {}):
        # allocate all buffers, bookkeep shapes and setup with contexts
        for network_name, sdxl_engine in self.engines_dict.items():
            if isinstance(network_name, SDXLComponent):
                network_name = network_name.valstr

            trt_engine = sdxl_engine.engine

            # [-1]: max opt profile [0]: batch dimension
            max_batch_size = trt_engine.get_tensor_profile_shape(trt_engine[0], 0)[-1][0]
            for idx in range(trt_engine.num_io_tensors):
                tensor_name = trt_engine[idx]
                full_name = f'{network_name}_{tensor_name}'

                tensor_shape = list(shape_dict.get(full_name, trt_engine.get_tensor_shape(tensor_name)))
                # set dynamic dimension
                if tensor_shape[0] == -1:
                    tensor_shape[0] = max_batch_size

                dtype = numpy_to_torch_dtype_dict[trt.nptype(trt_engine.get_tensor_dtype(tensor_name))]

                # NOTE(vir): use torch as storage/allocation backend for now
                self.buffers[full_name] = torch.zeros(tensor_shape, dtype=dtype).to(device=self.device)

                if trt_engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    self.input_tensors[full_name] = tensor_shape

                if trt_engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                    self.output_tensors[full_name] = tensor_shape

    def get_dummy_feed_dict(self):
        feed_dict = {
            name: torch.rand(shape)
            for name, shape in self.input_tensors.items()
        }

        return feed_dict

    def get_outputs(self):
        return {output: self.buffers[output] for output in self.output_tensors.keys()}

    def get_input_names(self):
        return list(self.input_tensors.keys())

    def get_output_names(self):
        return list(self.output_tensors.keys())

    def __getitem__(self, buffer_name: str):
        assert buffer_name in self.buffers, f"invalid buffer identifier, no such buffer: {buffer_name}"
        return self.buffers[buffer_name]

    def __setitem__(self, buffer_name: str, tensor: torch.Tensor):
        assert buffer_name in self.buffers, f"invalid buffer identifier, got {buffer_name}, expected one of {self.buffers.keys()}"

        max_batch_size = self.buffers[buffer_name].shape[0]
        actual_batch_size = 1 if len(tensor.shape) < 2 else tensor.shape[0]
        assert max_batch_size >= actual_batch_size, f"BS={actual_batch_size} must be <={max_batch_size} for {buffer_name}"

        # copy submatrix in
        self.buffers[buffer_name][:actual_batch_size].copy_(tensor)

        engine, tensor_name = self.engines_dict[buffer_name.split('_')[0]], '_'.join(buffer_name.split('_')[1:])
        tensor_mode = engine.engine.get_tensor_mode(tensor_name)

        # capture shape changes
        if tensor_mode == trt.TensorIOMode.INPUT:
            self.input_tensors[buffer_name] = list(tensor.shape)

        if tensor_mode == trt.TensorIOMode.OUTPUT:
            self.output_tensors[buffer_name] = list(tensor.shape)


class SDXLResponse:
    def __init__(self,
                 sample_ids,
                 generated_images,
                 results_ready):
        self.sample_ids = sample_ids
        self.generated_images = generated_images
        self.results_ready = results_ready


class SDXLCopyStream:
    def __init__(self,
                 device_id,
                 gpu_batch_size):
        cudart.cudaSetDevice(device_id)
        self.stream = cudart.cudaStreamCreate()
        self.h2d_event = cudart.cudaEventCreateWithFlags(cudart.cudaEventDefault | cudart.cudaEventDisableTiming)
        self.d2h_event = cudart.cudaEventCreateWithFlags(cudart.cudaEventDefault | cudart.cudaEventDisableTiming)
        self.vae_outputs = torch.zeros((gpu_batch_size, PipelineConfig.IMAGE_SIZE, PipelineConfig.IMAGE_SIZE, 3), dtype=torch.uint8)  # output cpu buffer

    def save_buffer_to_cpu_images(self, vae_ouput_buffer):
        # Normalize TRT (output + 1) * 0.5
        # Post process following the reference: https://github.com/mlcommons/inference/blob/master/text_to_image/coco.py
        vae_output_post_processed = ((vae_ouput_buffer + 1) * 255 * 0.5).clamp(0, 255).round().permute(0, 2, 3, 1).to(torch.uint8).contiguous()

        cudart.cudaMemcpyAsync(self.vae_outputs.data_ptr(),
                               vae_output_post_processed.data_ptr(),
                               vae_output_post_processed.nelement() * vae_output_post_processed.element_size(),
                               cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                               self.stream)

    def record_h2d_event(self):
        cudart.cudaEventRecord(self.h2d_event, self.stream)

    def record_d2h_event(self):
        cudart.cudaEventRecord(self.d2h_event, self.stream)

    def make_infer_await_h2d(self, infer_stream):
        cudart.cudaStreamWaitEvent(infer_stream, self.h2d_event, 0)

    def await_infer_done(self, infer_done):
        cudart.cudaStreamWaitEvent(self.stream, infer_done, 0)


class SDXLCore:
    def __init__(self,
                 device_id: int,
                 dataset: Dataset,
                 component_info: Dict[SDXLComponent, Dict[str, Any]],
                 gpu_copy_streams: int = 1,  # TODO copy stream number limit to 1
                 use_graphs: bool = False,
                 verbose: bool = False):

        cudart.cudaSetDevice(device_id)
        torch.autograd.set_grad_enabled(False)
        self.device = "cuda"
        self.device_id = device_id
        self.use_graphs = use_graphs
        self.verbose = verbose

        logging.debug(f"[Device {self.device_id}] Initializing")
        self.component_info = component_info
        self.e2e_batch_size = self.component_info["e2e_batch_size"]
        # Dataset
        self.dataset = dataset
        self.total_samples = 0

        # Pipeline components
        self.models = {
            SDXLComponent.CLIP1: CLIP(name='clip1',
                                      max_batch_size=self.component_info[SDXLComponent.CLIP1]["batch_size"],
                                      device=self.device),
            SDXLComponent.CLIP2: CLIPWithProj(name='clip2',
                                              max_batch_size=self.component_info[SDXLComponent.CLIP2]["batch_size"],
                                              device=self.device),
            SDXLComponent.UNETXL: UNetXL(name='unet',
                                         max_batch_size=self.component_info[SDXLComponent.UNETXL]["batch_size"],
                                         device=self.device),
            SDXLComponent.VAE: VAE(name='vae',
                                   max_batch_size=self.component_info[SDXLComponent.VAE]["batch_size"],
                                   device=self.device),
        }

        self.buffers = None
        self.scheduler = sdxl_scheduler.EulerDiscreteScheduler()
        self.latent_dtype = torch.float16
        self.vae_loop_count = self.e2e_batch_size // self.component_info[SDXLComponent.VAE]["batch_size"]

        # Runtime components
        self.context_memory = None
        self.infer_stream = cudart.cudaStreamCreate()
        self.infer_done = cudart.cudaEventCreateWithFlags(cudart.cudaEventDefault | cudart.cudaEventDisableTiming)
        self.copy_stream = SDXLCopyStream(device_id, self.e2e_batch_size)

        # QSR components
        self.response_queue = queue.Queue()
        self.response_thread = threading.Thread(target=self._process_response, args=(), daemon=True)
        # self.start_inference = threading.Condition()

        # Initialize scheduler
        self.scheduler.set_timesteps(PipelineConfig.STEPS)

        # Initialize engines
        self.engines = dict()
        for c_eng, c_info in self.component_info.items():
            if isinstance(c_eng, SDXLComponent):
                self.engines[c_eng] = SDXLEngine(engine_name=c_eng.valstr, engine_path=c_info["engine_path"])
            else:
                assert c_eng == "e2e_batch_size", f"Unexpected key in component_info: {c_eng}"

        # Initialize engine runtime
        max_device_memory = calculate_max_engine_device_memory(self.engines)
        shared_device_memory = cudart.cudaMalloc(max_device_memory)
        self.context_memory = shared_device_memory

        for engine in self.engines.values():
            logging.debug(f"Activating engine: {engine.engine_path}")
            engine.activate(self.context_memory)

        # Initialize buffers
        self.buffers = SDXLBufferManager(self.engines, device=self.device)
        vae_shape_override = {}
        shape_dict = self.models[SDXLComponent.VAE].get_shape_dict(self.component_info[SDXLComponent.VAE]["batch_size"])
        for engine_tensor_name, engine_tensor_shape in shape_dict.items():  # VAE buffers are allocated according to loop count
            buffer_shape = list(engine_tensor_shape)
            buffer_shape[0] = buffer_shape[0] * self.vae_loop_count
            vae_shape_override[f"{self.models[SDXLComponent.VAE].name}_{engine_tensor_name}"] = buffer_shape
        self.buffers.initialize(vae_shape_override)
        self.add_time_ids = torch.tensor(
            [PipelineConfig.IMAGE_SIZE, PipelineConfig.IMAGE_SIZE, 0, 0, PipelineConfig.IMAGE_SIZE, PipelineConfig.IMAGE_SIZE],
            dtype=torch.float16,
            device=self.device).repeat(self.e2e_batch_size * 2, 1)
        self.init_noise_latent = self.dataset.init_noise_latent.to(self.device)
        self.init_noise_latent = torch.concat([self.init_noise_latent] * self.e2e_batch_size) * self.scheduler.init_noise_sigma()

        # Initialize cuda graphs
        if self.use_graphs:
            self.engines[SDXLComponent.UNETXL].enable_cuda_graphs(self.buffers)

        # Initialize QSR thread
        self.response_thread.start()

    def __del__(self):
        # exit all threads
        self.response_queue.put(None)
        self.response_queue.join()
        self.response_thread.join()

    def _process_response(self):
        while True:
            response = self.response_queue.get()
            if response is None:
                # None in the queue indicates the parent want us to exit
                self.response_queue.task_done()
                break
            qsr = []
            actual_batch_size = len(response.sample_ids)
            cudart.cudaEventSynchronize(response.results_ready)
            logging.debug(f"[Device {self.device_id}] Reporting back {actual_batch_size} samples")

            with nvtx_scope("report_qsl", color='yellow'):
                for idx, sample_id in enumerate(response.sample_ids):
                    qsr.append(lg.QuerySampleResponse(sample_id,
                                                      response.generated_images[idx].data_ptr(),
                                                      response.generated_images[idx].nelement() * response.generated_images[idx].element_size()))

                lg.QuerySamplesComplete(qsr)
                self.total_samples += actual_batch_size
                self.response_queue.task_done()

    def _transfer_to_clip_buffer(self,
                                 prompt_tokens_clip1,
                                 prompt_tokens_clip2,
                                 negative_prompt_tokens_clip1,
                                 negative_prompt_tokens_clip2):
        # TODO: yihengz support copy stream
        # cudart.cudaMemcpy(self.buffers['clip1'].get_tensor('input_ids').data_ptr(),
        #                   prompt_tokens_clip1.ctypes.data,
        #                   prompt_tokens_clip1.size * prompt_tokens_clip1.itemsize,
        #                   cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        # [ negative prompt, prompt ]
        concat_prompt_clip1 = torch.concat([negative_prompt_tokens_clip1, prompt_tokens_clip1], dim=0)
        concat_prompt_clip2 = torch.concat([negative_prompt_tokens_clip2, prompt_tokens_clip2], dim=0)
        self.buffers['clip1_input_ids'] = concat_prompt_clip1
        self.buffers['clip2_input_ids'] = concat_prompt_clip2

    def _encode_tokens(self, actual_batch_size):
        for clip in [SDXLComponent.CLIP1, SDXLComponent.CLIP2]:
            for tensor_name, tensor_shape in self.models[clip].get_shape_dict(actual_batch_size * 2).items():
                self.engines[clip].stage_tensor(tensor_name, self.buffers[f'{clip.valstr}_{tensor_name}'], tensor_shape)
            self.engines[clip].infer(self.infer_stream)

    def _denoise_latent(self, actual_batch_size):
        # Prepare predetermined input tensors
        with nvtx_scope("prepare_denoise", color='yellow'):
            latents = self.init_noise_latent[:actual_batch_size]
            encoder_hidden_states = torch.concat([
                self.buffers['clip1_hidden_states'],
                self.buffers['clip2_hidden_states'].to(self.latent_dtype)
            ], dim=-1)
            text_embeds = self.buffers['clip2_text_embeddings'].to(self.latent_dtype)

            self.buffers['unet_encoder_hidden_states'] = encoder_hidden_states
            self.buffers['unet_text_embeds'] = text_embeds
            self.buffers['unet_time_ids'] = self.add_time_ids[:actual_batch_size * 2]

        for step_index, timestep in enumerate(self.scheduler.timesteps):
            with nvtx_scope("stage_denoise", color='pink'):
                # Expand the latents because we have prompt and negative prompt guidance
                latents_expanded = self.scheduler.scale_model_input(torch.concat([latents] * 2), step_index, timestep)

                # Prepare runtime dependent input tensors
                self.buffers['unet_sample'] = latents_expanded.to(self.latent_dtype)
                self.buffers['unet_timestep'] = timestep.to(self.latent_dtype).to("cuda")

                for tensor_name, tensor_shape in self.models[SDXLComponent.UNETXL].get_shape_dict(actual_batch_size * 2).items():
                    self.engines[SDXLComponent.UNETXL].stage_tensor(tensor_name, self.buffers[f'unet_{tensor_name}'], tensor_shape)

            with nvtx_scope("denoise_infer", color='green'):
                self.engines[SDXLComponent.UNETXL].infer(self.infer_stream, batch_size=actual_batch_size)

                # TODO: yihengz check if we actually need sync the stream
                cudart.cudaStreamSynchronize(self.infer_stream)  # make sure Unet kernel execution are finished

            with nvtx_scope("scheduler", color='pink'):
                # Perform guidance
                noise_pred = self.buffers['unet_latent']

                # negative prompt in batch dimension [0:BS]
                noise_pred_negative_prompt = noise_pred[0:actual_batch_size]

                # prompt in batch dimension [BS:]
                noise_pred_prompt = noise_pred[actual_batch_size:actual_batch_size * 2]

                noise_pred = noise_pred_negative_prompt + PipelineConfig.GUIDANCE * (noise_pred_prompt - noise_pred_negative_prompt)
                latents = self.scheduler.step(noise_pred, latents, step_index)

        latents = 1. / PipelineConfig.VAE_SCALING_FACTOR * latents
        # Transfer the Unet output to vae buffer
        self.buffers['vae_latent'] = latents

    def _decode_latent(self, actual_batch_size):
        vae_max_batch_size = self.component_info[SDXLComponent.VAE]["batch_size"]
        with nvtx_scope("vae_decode", color='blue'):
            # Loop over VAE engine
            vae_loop_count = (actual_batch_size + vae_max_batch_size - 1) // vae_max_batch_size
            for i in range(vae_loop_count):
                vae_actual_batch_size = vae_max_batch_size if i < vae_loop_count - 1 else actual_batch_size - (vae_loop_count - 1) * vae_max_batch_size
                # Stage VAE buffer
                for tensor_name, tensor_shape in self.models[SDXLComponent.VAE].get_shape_dict(vae_actual_batch_size).items():
                    self.engines[SDXLComponent.VAE].stage_tensor(tensor_name, self.buffers[f'vae_{tensor_name}'][i * vae_max_batch_size: i * vae_max_batch_size + vae_actual_batch_size], tensor_shape)
                self.engines[SDXLComponent.VAE].infer(self.infer_stream)
            cudart.cudaEventRecord(self.infer_done, self.infer_stream)

    def _save_buffer_to_images(self):
        with nvtx_scope("post_process", color='yellow'):
            self.copy_stream.await_infer_done(self.infer_done)
            self.copy_stream.save_buffer_to_cpu_images(self.buffers['vae_images'])
            self.copy_stream.record_d2h_event()

    def generate_images(self, samples):
        cudart.cudaSetDevice(self.device_id)
        with nvtx_scope("read_tokens", color='yellow'):
            actual_batch_size = len(samples)
            sample_indices = [q.index for q in samples]
            sample_ids = [q.id for q in samples]
            logging.debug(f"[Device {self.device_id}] Running inference on sample {sample_indices} with batch size {actual_batch_size}")

            # TODO add copy stream support
            prompt_tokens_clip1 = self.dataset.prompt_tokens_clip1[sample_indices, :].to(self.device)
            prompt_tokens_clip2 = self.dataset.prompt_tokens_clip2[sample_indices, :].to(self.device)
            negative_prompt_tokens_clip1 = self.dataset.negative_prompt_tokens_clip1[sample_indices, :].to(self.device)
            negative_prompt_tokens_clip2 = self.dataset.negative_prompt_tokens_clip2[sample_indices, :].to(self.device)

        with nvtx_scope("stage_clip_buffers", color='pink'):
            self._transfer_to_clip_buffer(
                prompt_tokens_clip1,
                prompt_tokens_clip2,
                negative_prompt_tokens_clip1,
                negative_prompt_tokens_clip2
            )

        self._encode_tokens(actual_batch_size)
        self._denoise_latent(actual_batch_size)  # runs self.denoising_steps inside
        self._decode_latent(actual_batch_size)

        self._save_buffer_to_images()

        # Report back to loadgen use sample_ids
        response = SDXLResponse(sample_ids=sample_ids,
                                generated_images=self.copy_stream.vae_outputs,
                                results_ready=self.copy_stream.d2h_event)
        self.response_queue.put(response)

    def warm_up(self, warm_up_iters):
        cudart.cudaSetDevice(self.device_id)
        logging.debug(f"[Device {self.device_id}] Running warm up with batch size {self.e2e_batch_size}x{warm_up_iters}")

        for _ in range(warm_up_iters):
            prompt_tokens_clip1 = self.dataset.prompt_tokens_clip1[:self.e2e_batch_size, :].to(self.device)
            prompt_tokens_clip2 = self.dataset.prompt_tokens_clip2[:self.e2e_batch_size, :].to(self.device)
            negative_prompt_tokens_clip1 = self.dataset.negative_prompt_tokens_clip1[:self.e2e_batch_size, :].to(self.device)
            negative_prompt_tokens_clip2 = self.dataset.negative_prompt_tokens_clip2[:self.e2e_batch_size, :].to(self.device)

            self._transfer_to_clip_buffer(
                prompt_tokens_clip1,
                prompt_tokens_clip2,
                negative_prompt_tokens_clip1,
                negative_prompt_tokens_clip2
            )

            self._encode_tokens(self.e2e_batch_size)
            self._denoise_latent(self.e2e_batch_size)
            self._decode_latent(self.e2e_batch_size)

            self._save_buffer_to_images()


@autoconfigure
@bind(general_fields.verbose)
@bind(harness_fields.use_graphs)
# Enable these when SDXL supports multiple cores per device
# @bind(harness_fields.gpu_inference_streams)
# @bind(harness_fields.gpu_copy_streams)
@bind(sdxl_fields.batcher_time_limit, "batch_timeout_threshold")
class SDXLServer:
    def __init__(self,
                 devices: List[int],
                 dataset: Dataset,
                 engines: List[Tuple[ComponentEngine, str]],
                 gpu_inference_streams: int = 1,  # TODO support multiple SDXLCore per device
                 gpu_copy_streams: int = 1,  # TODO copy stream number limit to 1
                 use_graphs: bool = False,
                 verbose: bool = False,
                 enable_batcher: bool = False,
                 batch_timeout_threshold: float = -1):

        self.devices = devices
        self.verbose = verbose
        self.enable_batcher = enable_batcher and batch_timeout_threshold > 0

        # Server components
        self.sample_queue = queue.Queue()  # sample sync queue
        self.sample_count = 0
        self.sdxl_cores = {}
        self.core_threads = []

        assert len(engines) == 4, "SDXL requires 1 engine per component (Components: CLIP1, CLIP2, UNet, VAE)"
        self.component_info = dict()
        for c_eng, fpath in engines:
            assert isinstance(c_eng.component, SDXLComponent), f"SDXL component of unexpected type {type(c_eng.component)} - Is the EngineIndex parsing correctly?"
            self.component_info[c_eng.component] = {
                "engine_path": fpath,
                "batch_size": c_eng.batch_size,
            }
        # Validate batch size
        assert self.component_info[SDXLComponent.CLIP1]["batch_size"] == self.component_info[SDXLComponent.UNETXL]["batch_size"]

        # UNETXL and CLIP need to be double E2E BS for positive and negative prompt
        self.e2e_batch_size = self.component_info[SDXLComponent.CLIP1]["batch_size"] // 2
        self.component_info["e2e_batch_size"] = self.e2e_batch_size  # Passthrough to SDXLCore

        # Initialize the cores
        for device_id in self.devices:
            self.sdxl_cores[device_id] = SDXLCore(device_id=device_id,
                                                  dataset=dataset,
                                                  component_info=self.component_info,
                                                  gpu_copy_streams=gpu_copy_streams,
                                                  use_graphs=use_graphs,
                                                  verbose=self.verbose)

        # Start the cores
        for device_id in self.devices:
            thread = threading.Thread(target=self.process_samples, args=(device_id,))
            thread.daemon = True
            self.core_threads.append(thread)
            thread.start()

        if self.enable_batcher:
            self.batcher_threshold = batch_timeout_threshold  # maximum seconds to form a batch
            self.batcher_queue = queue.SimpleQueue()  # batcher sync queue
            self.batcher_thread = threading.Thread(target=self.batch_samples, args=())
            self.batcher_thread.start()

    def warm_up(self):
        for device_id in self.devices:
            self.sdxl_cores[device_id].warm_up(warm_up_iters=2)

    def process_samples(self, device_id):
        while True:
            samples = self.sample_queue.get()
            if samples is None:
                # None in the queue indicates the SUT want us to exit
                self.sample_queue.task_done()
                break
            self.sdxl_cores[device_id].generate_images(samples)
            self.sample_queue.task_done()

    def batch_samples(self):
        batched_samples = self.batcher_queue.get()
        timeout_stamp = time.time()
        while True:
            if len(batched_samples) != 0 and (len(batched_samples) >= self.e2e_batch_size or time.time() - timeout_stamp >= self.batcher_threshold):  # max batch or time limit exceed
                logging.debug(f"Formed batch of {len(batched_samples[:self.e2e_batch_size])} samples")
                self.sample_queue.put(batched_samples[:self.e2e_batch_size])
                batched_samples = batched_samples[self.e2e_batch_size:]
                timeout_stamp = time.time()

            try:
                samples = self.batcher_queue.get(timeout=self.batcher_threshold)
            except queue.Empty:
                continue

            if samples is None:  # None in the queue indicates the SUT want us to exit
                break
            batched_samples += samples

    def issue_queries(self, query_samples):
        num_samples = len(query_samples)
        logging.debug(f"[Server] Received {num_samples} samples")
        self.sample_count += num_samples
        for i in range(0, num_samples, self.e2e_batch_size):
            # Construct batches
            actual_batch_size = self.e2e_batch_size if num_samples - i > self.e2e_batch_size else num_samples - i
            if self.enable_batcher:
                self.batcher_queue.put(query_samples[i: i + actual_batch_size])
            else:
                self.sample_queue.put(query_samples[i: i + actual_batch_size])

    def flush_queries(self):
        pass

    def finish_test(self):
        # exit all threads
        logging.debug(f"SUT finished!")
        logging.info(f"[Server] Received {self.sample_count} total samples")
        for _ in self.core_threads:
            self.sample_queue.put(None)
        self.sample_queue.join()
        if self.enable_batcher:
            self.batcher_queue.put(None)
            self.batcher_thread.join()
        for device_id in self.devices:
            logging.info(f"[Device {device_id}] Reported {self.sdxl_cores[device_id].total_samples} samples")
        for thread in self.core_threads:
            thread.join()
