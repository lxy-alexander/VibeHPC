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

"""
WAN22 T2V Harness for MLPerf Inference.

Design:
- dp_size workers, each processing BS=1
- Each worker can use sp_size GPUs internally (ulysses/cfg parallelism)
- Offline: all queries distributed round-robin across workers
- SingleStream: one query at a time
"""

import contextlib
import json
import os
import pickle
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import mlperf_loadgen as lg
from PIL import Image
from diffusers.utils import export_to_video

from nvmitten.configurator import autoconfigure, bind
from code.common import logging
from code.common.constants import AuditTest
from code.fields import general as general_fields
from code.fields import harness as harness_fields
from code.ops.harness import PyHarnessOp
from code.ops.loadgen import LoadgenConfFilesOp

from . import fields as wan22_fields
from .dataset import Wan22Dataset
from .utils import video_to_loadgen_response, create_placeholder_video


@dataclass
class Wan22Response:
    """Response object for WAN22 video generation."""
    sample_id: int
    sample_index: int
    video: np.ndarray


@autoconfigure
@bind(general_fields.verbose)
@bind(wan22_fields.model_path, "model_path")
@bind(wan22_fields.num_frames, "num_frames")
@bind(wan22_fields.height, "height")
@bind(wan22_fields.width, "width")
@bind(wan22_fields.num_inference_steps, "num_inference_steps")
@bind(wan22_fields.guidance_scale, "guidance_scale")
@bind(wan22_fields.guidance_scale_2, "guidance_scale_2")
@bind(wan22_fields.attn_type, "attn_type")
@bind(wan22_fields.linear_type, "linear_type")
@bind(wan22_fields.ulysses_size, "ulysses_size")
@bind(wan22_fields.cfg_size, "cfg_size")
@bind(wan22_fields.cp_size, "cp_size")
@bind(wan22_fields.ring_size, "ring_size")
@bind(wan22_fields.tp_size, "tp_size")
@bind(wan22_fields.dp_size, "dp_size")
@bind(wan22_fields.fsdp_size, "fsdp_size")
@bind(wan22_fields.t5_fsdp_size, "t5_fsdp_size")
@bind(wan22_fields.disable_parallel_vae, "disable_parallel_vae")
@bind(wan22_fields.parallel_vae_split_dim, "parallel_vae_split_dim")
@bind(wan22_fields.enable_visual_gen_cpu_offload, "enable_visual_gen_cpu_offload")
@bind(wan22_fields.cpu_offload_stride, "cpu_offload_stride")
@bind(wan22_fields.warmup_iters, "warmup_iters")
@bind(wan22_fields.gpu_ids, "gpu_ids")
class Wan22Server:
    """Server for WAN2.2 T2V inference."""

    def __init__(
        self,
        dataset: Wan22Dataset,
        model_path: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        num_frames: int = 81,
        height: int = 720,
        width: int = 1280,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.0,
        guidance_scale_2: float = 3.0,
        attn_type: str = "sage-attn",
        linear_type: str = "default",
        ulysses_size: int = 1,
        cfg_size: int = 1,
        cp_size: int = 1,
        ring_size: int = 1,
        tp_size: int = 1,
        dp_size: int = 1,
        fsdp_size: int = 1,
        t5_fsdp_size: int = 1,
        disable_parallel_vae: bool = False,
        parallel_vae_split_dim: str = "width",
        enable_visual_gen_cpu_offload: bool = False,
        cpu_offload_stride: int = 1,
        warmup_iters: int = 2,
        gpu_ids: Optional[str] = None,
        verbose: bool = False,
        accuracy_mode: bool = False,
    ):
        self.dataset = dataset
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.accuracy_mode = accuracy_mode

        # Number of workers = dp_size (each worker handles sp_size GPUs)
        self.dp_size = dp_size
        sp_size = ulysses_size * cfg_size * cp_size * ring_size * tp_size * fsdp_size
        self.sp_size = sp_size if sp_size > 0 else 1

        logging.info(f"Creating {self.dp_size} workers, each with {self.sp_size} GPUs")

        # Pipeline config for workers
        self.pipeline_kwargs = {
            "model_path": model_path,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "guidance_scale_2": guidance_scale_2,
            "attn_type": attn_type,
            "linear_type": linear_type,
            "ulysses_size": ulysses_size,
            "cfg_size": cfg_size,
            "cp_size": cp_size,
            "ring_size": ring_size,
            "tp_size": tp_size,
            "fsdp_size": fsdp_size,
            "t5_fsdp_size": t5_fsdp_size,
            "disable_parallel_vae": disable_parallel_vae,
            "parallel_vae_split_dim": parallel_vae_split_dim,
            "enable_cpu_offload": enable_visual_gen_cpu_offload,
            "cpu_offload_stride": cpu_offload_stride,
            "warmup_iters": warmup_iters,
        }

        # Parse GPU IDs
        if gpu_ids:
            self.gpu_list = [g.strip() for g in gpu_ids.split(',')]
        else:
            total_gpus = self.dp_size * self.sp_size
            self.gpu_list = [str(i) for i in range(total_gpus)]

        # Worker state
        self.workers = []
        self.queue_dir = None
        self.accuracy_videos: Dict[int, np.ndarray] = {}

        self._init_workers()

    def _init_workers(self):
        """Initialize dp_size worker processes."""
        self.queue_dir = tempfile.mkdtemp(prefix="wan22_")

        # Write pipeline config
        config_file = os.path.join(self.queue_dir, "pipeline_config.json")
        with open(config_file, 'w') as f:
            json.dump(self.pipeline_kwargs, f)

        # Launch workers
        for dp_rank in range(self.dp_size):
            # Each worker gets sp_size GPUs
            start_gpu = dp_rank * self.sp_size
            end_gpu = start_gpu + self.sp_size
            worker_gpus = ','.join(self.gpu_list[start_gpu:end_gpu])

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = worker_gpus

            if self.sp_size > 1:
                # Use torchrun for multi-GPU SP
                cmd = [
                    sys.executable, "-m", "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node", str(self.sp_size),
                    "-m", "code.wan22-a14b.tensorrt.worker",
                    "--dp_rank", str(dp_rank),
                    "--sp_size", str(self.sp_size),
                    "--queue_dir", self.queue_dir,
                    "--pipeline_config", config_file,
                ]
            else:
                # Single GPU - direct launch
                cmd = [
                    sys.executable, "-m", "code.wan22-a14b.tensorrt.worker",
                    "--dp_rank", str(dp_rank),
                    "--sp_size", str(self.sp_size),
                    "--queue_dir", self.queue_dir,
                    "--pipeline_config", config_file,
                ]

            logging.info(f"Starting worker {dp_rank} on GPUs {worker_gpus}")
            proc = subprocess.Popen(cmd, env=env)
            self.workers.append(proc)

        # Wait for workers to be ready
        logging.info("Waiting for workers to initialize...")
        timeout = 1800
        start = time.time()

        while True:
            ready_count = sum(
                1 for r in range(self.dp_size)
                if os.path.exists(os.path.join(self.queue_dir, f"ready_{r}"))
            )
            if ready_count == self.dp_size:
                break
            if time.time() - start > timeout:
                missing = [r for r in range(self.dp_size)
                          if not os.path.exists(os.path.join(self.queue_dir, f"ready_{r}"))]
                raise RuntimeError(f"Workers {missing} failed to initialize")
            time.sleep(5)

        logging.info("All workers initialized")

    def _send_work(self, dp_rank: int, prompt: str, sample_id: int, sample_index: int):
        """Send work to a specific worker."""
        work_file = os.path.join(self.queue_dir, f"work_{dp_rank}")
        tmp_file = work_file + ".tmp"
        negative_prompt = self.dataset.get_negative_prompt()
        latents = self.dataset.get_fixed_latents()

        with open(tmp_file, 'wb') as f:
            pickle.dump((prompt, negative_prompt, latents, sample_id, sample_index), f)
        os.rename(tmp_file, work_file)

    def _collect_result(self, dp_rank: int, timeout: float = 600) -> Optional[Wan22Response]:
        """Collect result from a specific worker."""
        result_file = os.path.join(self.queue_dir, f"result_{dp_rank}")
        start = time.time()

        while not os.path.exists(result_file):
            if time.time() - start > timeout:
                return None
            time.sleep(0.01)

        try:
            with open(result_file, 'rb') as f:
                sample_id, sample_index, video = pickle.load(f)
            os.remove(result_file)
            return Wan22Response(sample_id, sample_index, video)
        except Exception as e:
            logging.error(f"Error reading result from worker {dp_rank}: {e}")
            return None

    def issue_queries(self, query_samples):
        """Process queries - distribute round-robin across workers."""
        num_samples = len(query_samples)

        # Distribute queries round-robin
        worker_queues = [[] for _ in range(self.dp_size)]
        for i, sample in enumerate(query_samples):
            dp_rank = i % self.dp_size
            worker_queues[dp_rank].append(sample)

        # Process in rounds - each worker handles one query per round
        max_per_worker = max(len(q) for q in worker_queues) if worker_queues else 0

        # Progress tracking
        completed = 0
        start_time = time.time()

        for round_idx in range(max_per_worker):
            # Send work to all workers that have queries this round
            active_workers = []
            for dp_rank in range(self.dp_size):
                if round_idx < len(worker_queues[dp_rank]):
                    sample = worker_queues[dp_rank][round_idx]
                    prompt = self.dataset.get_prompt(sample.index)
                    self._send_work(dp_rank, prompt, sample.id, sample.index)
                    active_workers.append(dp_rank)

            # Collect results from all active workers
            for dp_rank in active_workers:
                response = self._collect_result(dp_rank)
                if response:
                    video_data = response.video
                    if isinstance(video_data, torch.Tensor):
                        video_data = video_data.cpu().numpy()

                    lg.QuerySamplesComplete([
                        video_to_loadgen_response(response.sample_id, video_data)
                    ])

                    if self.accuracy_mode:
                        self.accuracy_videos[response.sample_index] = video_data
                else:
                    sample = worker_queues[dp_rank][round_idx]
                    raise RuntimeError(f"Worker {dp_rank} failed to generate video for sample {sample.id}")

                completed += 1

            # Log progress after each round
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            logging.info(f"Progress: {completed}/{num_samples} samples, {rate:.3f} samples/s")

    def flush_queries(self):
        pass

    def finish_test(self):
        """Shutdown workers and cleanup."""
        # Signal shutdown
        shutdown_file = os.path.join(self.queue_dir, "shutdown")
        with open(shutdown_file, 'w') as f:
            f.write("shutdown")

        # Wait for workers
        for proc in self.workers:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        # Cleanup
        if self.queue_dir and os.path.exists(self.queue_dir):
            shutil.rmtree(self.queue_dir, ignore_errors=True)

    def save_accuracy_videos(self, log_dir: Path):
        if not self.accuracy_mode:
            return

        video_dir = log_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)

        for sample_idx, video_data in self.accuracy_videos.items():
            prompt = self.dataset.get_prompt(sample_idx)
            video_path = video_dir / f"{prompt}-0.mp4"

            try:
                if video_data.dtype != np.uint8:
                    if video_data.max() <= 1.0:
                        video_data = (video_data * 255).clip(0, 255).astype(np.uint8)
                    else:
                        video_data = video_data.clip(0, 255).astype(np.uint8)

                frames = [Image.fromarray(frame) for frame in video_data]
                export_to_video(frames, str(video_path), fps=16)
            except Exception as e:
                logging.error(f"Failed to save video {sample_idx}: {e}")


@autoconfigure
@bind(harness_fields.audit_test)
@bind(wan22_fields.total_sample_count, "total_sample_count")
class Wan22HarnessOp(PyHarnessOp):
    """WAN22 T2V harness operation for MLPerf LoadGen."""

    @classmethod
    def immediate_dependencies(cls):
        return {LoadgenConfFilesOp}

    @classmethod
    def output_keys(cls):
        return ["log_dir", "result_metadata"]

    def __init__(self, *args, audit_test: Optional[AuditTest] = None,
                 total_sample_count: Optional[int] = None, **kwargs):
        super().__init__(Wan22Dataset, *args, total_sample_count=total_sample_count, **kwargs)
        self._server_inst = None
        self.audit_test = audit_test

    def issue_queries(self, query_samples: List):
        self._server_inst.issue_queries(query_samples)

    def flush_queries(self):
        self._server_inst.flush_queries()

    @contextlib.contextmanager
    def wrap_lg_test(self, scratch_space, dependency_outputs):
        accuracy_mode = (self.test_mode == "AccuracyOnly")
        dump_videos = accuracy_mode or (self.audit_test == AuditTest.TEST01)
        self._server_inst = Wan22Server(dataset=self._qsl_inst, accuracy_mode=dump_videos)

        yield None
        self._server_inst.finish_test()

        if dump_videos:
            self._server_inst.save_accuracy_videos(self.wl.log_dir)
