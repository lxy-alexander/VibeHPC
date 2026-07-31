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
Worker process for WAN22 inference.

Each worker:
- Has dp_rank (data parallel rank)
- Uses sp_size GPUs for sequence parallelism
- Processes BS=1 queries
"""

import argparse
import datetime
import json
import os
import pickle
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp_rank", type=int, required=True)
    parser.add_argument("--sp_size", type=int, required=True)
    parser.add_argument("--queue_dir", type=str, required=True)
    parser.add_argument("--pipeline_config", type=str, required=True)
    args = parser.parse_args()

    dp_rank = args.dp_rank
    sp_size = args.sp_size
    queue_dir = args.queue_dir

    # Suppress tqdm on non-main ranks (must be set before imports)
    local_rank_early = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank_early != 0:
        os.environ["TQDM_DISABLE"] = "1"

    # Load pipeline config
    with open(args.pipeline_config, 'r') as f:
        pipeline_kwargs = json.load(f)

    # Now import torch (CUDA_VISIBLE_DEVICES already set by parent)
    import torch
    import torch.distributed as dist
    from code.common import logging
    from .utils import video_to_numpy

    # Import visual_gen
    from visual_gen import setup_configs
    from visual_gen.models.transformers.wan_transformer import ditWanTransformer3DModel
    from visual_gen.models.vaes.wan_vae import ditWanAutoencoderKL
    from visual_gen.pipelines.wan_pipeline import ditWanPipeline
    from visual_gen.utils import create_default_dit_configs

    # Get rank from torchrun environment (if launched via torchrun)
    # LOCAL_RANK is set by torchrun for multi-GPU SP
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main_rank = (local_rank == 0)

    # Initialize distributed only for multi-GPU SP (sp_size > 1)
    # For DP mode (sp_size = 1), each worker is independent - no distributed needed
    if sp_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    torch.autograd.set_grad_enabled(False)

    # Initialize pipeline
    pipe = init_pipeline(pipeline_kwargs, setup_configs, ditWanTransformer3DModel,
                         ditWanAutoencoderKL, ditWanPipeline, create_default_dit_configs, logging)

    if is_main_rank:
        logging.info(f"[Worker {dp_rank}] Ready")

        # Signal ready (only rank 0)
        ready_file = os.path.join(queue_dir, f"ready_{dp_rank}")
        with open(ready_file, 'w') as f:
            f.write("ready")

    # Synchronize all ranks before entering work loop
    if dist.is_initialized():
        dist.barrier()

    # Work loop
    work_file = os.path.join(queue_dir, f"work_{dp_rank}")
    result_file = os.path.join(queue_dir, f"result_{dp_rank}")
    shutdown_file = os.path.join(queue_dir, "shutdown")

    while True:
        # Check shutdown (all ranks)
        should_shutdown = False
        if is_main_rank:
            should_shutdown = os.path.exists(shutdown_file)
        if dist.is_initialized() and world_size > 1:
            should_shutdown_tensor = torch.tensor([1 if should_shutdown else 0], device=f"cuda:{local_rank}")
            dist.broadcast(should_shutdown_tensor, src=0)
            should_shutdown = should_shutdown_tensor.item() == 1
        if should_shutdown:
            break

        # Check for work (rank 0 reads, broadcasts to others)
        has_work = False
        prompt, negative_prompt, latents, sample_id, sample_index = None, None, None, None, None

        if is_main_rank:
            has_work = os.path.exists(work_file)
            if has_work:
                try:
                    with open(work_file, 'rb') as f:
                        prompt, negative_prompt, latents, sample_id, sample_index = pickle.load(f)
                    os.remove(work_file)
                except Exception as e:
                    logging.error(f"[Worker {dp_rank}] Error reading work: {e}")
                    has_work = False

        # Broadcast has_work to all ranks
        if dist.is_initialized() and world_size > 1:
            has_work_tensor = torch.tensor([1 if has_work else 0], device=f"cuda:{local_rank}")
            dist.broadcast(has_work_tensor, src=0)
            has_work = has_work_tensor.item() == 1

        if has_work:
            # Broadcast work data to all ranks
            if dist.is_initialized() and world_size > 1:
                work_data = [prompt, negative_prompt, latents, sample_id, sample_index] if is_main_rank else [None] * 5
                dist.broadcast_object_list(work_data, src=0)
                prompt, negative_prompt, latents, sample_id, sample_index = work_data

            try:
                # Generate video (BS=1) - all ranks participate
                output = pipe(
                    prompt=[prompt],
                    negative_prompt=[negative_prompt] if negative_prompt else None,
                    num_frames=pipeline_kwargs["num_frames"],
                    height=pipeline_kwargs["height"],
                    width=pipeline_kwargs["width"],
                    guidance_scale=pipeline_kwargs["guidance_scale"],
                    guidance_scale_2=pipeline_kwargs["guidance_scale_2"],
                    num_inference_steps=pipeline_kwargs["num_inference_steps"],
                    latents=latents,
                )

                # Only rank 0 writes result
                if is_main_rank:
                    video = video_to_numpy(output.frames[0])
                    tmp_file = result_file + ".tmp"
                    with open(tmp_file, 'wb') as f:
                        pickle.dump((sample_id, sample_index, video), f)
                    os.rename(tmp_file, result_file)

            except Exception as e:
                if is_main_rank:
                    logging.error(f"[Worker {dp_rank}] Error: {e}")
                    tmp_file = result_file + ".tmp"
                    with open(tmp_file, 'wb') as f:
                        pickle.dump((sample_id, sample_index, None), f)
                    os.rename(tmp_file, result_file)
        else:
            time.sleep(0.01)

    # Cleanup
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except:
            pass


def init_pipeline(kwargs, setup_configs, transformer_cls, vae_cls, pipeline_cls, create_configs_fn, logging):
    """Initialize the VisionFly pipeline."""
    import torch

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_rank = (local_rank == 0)

    if is_main_rank:
        logging.info("Initializing WAN2.2 pipeline...")

    # Configure visual_gen
    vfly_configs = create_configs_fn()
    vfly_configs["pipeline"]["enable_torch_compile"] = True
    vfly_configs["pipeline"]["torch_compile_models"] = ["transformer", "transformer_2"]
    vfly_configs["pipeline"]["torch_compile_mode"] = "max-autotune-no-cudagraphs"
    vfly_configs["pipeline"]["fuse_qkv"] = True
    vfly_configs["attn"]["type"] = kwargs["attn_type"]
    vfly_configs["linear"]["type"] = kwargs["linear_type"]

    # Parallelism settings
    vfly_configs["parallel"]["dit_ulysses_size"] = kwargs["ulysses_size"]
    vfly_configs["parallel"]["dit_cfg_size"] = kwargs["cfg_size"]
    vfly_configs["parallel"]["dit_cp_size"] = kwargs["cp_size"]
    vfly_configs["parallel"]["dit_ring_size"] = kwargs["ring_size"]
    vfly_configs["parallel"]["dit_tp_size"] = kwargs["tp_size"]
    vfly_configs["parallel"]["dit_dp_size"] = 1  # Each worker is independent
    vfly_configs["parallel"]["dit_fsdp_size"] = kwargs["fsdp_size"]
    vfly_configs["parallel"]["t5_fsdp_size"] = kwargs["t5_fsdp_size"]
    vfly_configs["parallel"]["disable_parallel_vae"] = kwargs["disable_parallel_vae"]
    vfly_configs["parallel"]["parallel_vae_split_dim"] = kwargs["parallel_vae_split_dim"]

    # Refiner settings
    vfly_configs["parallel"]["refiner_dit_ulysses_size"] = kwargs["ulysses_size"]
    vfly_configs["parallel"]["refiner_dit_cfg_size"] = kwargs["cfg_size"]
    vfly_configs["parallel"]["refiner_dit_cp_size"] = kwargs["cp_size"]
    vfly_configs["parallel"]["refiner_dit_ring_size"] = kwargs["ring_size"]
    vfly_configs["parallel"]["refiner_dit_tp_size"] = kwargs["tp_size"]
    vfly_configs["parallel"]["refiner_dit_dp_size"] = 1
    vfly_configs["parallel"]["refiner_dit_fsdp_size"] = kwargs["fsdp_size"]

    setup_configs(**vfly_configs)

    # Load components
    vae = vae_cls.from_pretrained(
        kwargs["model_path"], subfolder="vae", torch_dtype=torch.float32
    )
    transformer = transformer_cls.from_pretrained(
        kwargs["model_path"], subfolder="transformer", torch_dtype=torch.bfloat16
    )

    pipe = pipeline_cls.from_pretrained(
        kwargs["model_path"], vae=vae, transformer=transformer,
        torch_dtype=torch.bfloat16, **vfly_configs,
    )

    # Device placement - use local_rank for multi-GPU SP
    if kwargs["enable_cpu_offload"]:
        model_wise = ["text_encoder"]
        block_wise = ["transformer"]
        if "Wan2.2" in kwargs["model_path"]:
            block_wise.append("transformer_2")
        pipe.enable_async_cpu_offload(
            model_wise=model_wise, block_wise=block_wise,
            offloading_stride=kwargs["cpu_offload_stride"],
        )
    else:
        pipe.to(f"cuda:{local_rank}")

    # Warmup (tqdm suppressed on non-main ranks via TQDM_DISABLE env var set at startup)
    warmup_iters = kwargs.get("warmup_iters", 0)
    if warmup_iters > 0:
        if is_main_rank:
            logging.info(f"Running {warmup_iters} warmup iterations...")

        # Create dummy latents to match real inference code path
        latent_frames = (kwargs["num_frames"] - 1) // 4 + 1
        dummy_latents = torch.randn(
            1, 16, latent_frames, kwargs["height"] // 8, kwargs["width"] // 8,
            dtype=torch.bfloat16, device=f"cuda:{local_rank}"
        )

        for _ in range(warmup_iters):
            try:
                pipe(
                    prompt=["A fox walking in a garden"],
                    negative_prompt=["low quality, blurry"],
                    num_frames=kwargs["num_frames"],
                    height=kwargs["height"],
                    width=kwargs["width"],
                    guidance_scale=kwargs["guidance_scale"],
                    guidance_scale_2=kwargs["guidance_scale_2"],
                    num_inference_steps=kwargs["num_inference_steps"],
                    latents=dummy_latents,
                )
            except Exception as e:
                if is_main_rank:
                    logging.warning(f"Warmup failed: {e}")

        del dummy_latents
        torch.cuda.empty_cache()

    if is_main_rank:
        logging.info("Pipeline initialized")
    return pipe


if __name__ == "__main__":
    main()
