"""The CLI definition for the NVIDIA-optimized Qwen3-VL (Q3VL) benchmark."""

from __future__ import annotations

from typing import Annotated, Optional

import wandb
from loguru import logger
from mlperf_inf_mm_q3vl.benchmark import run_benchmark
from pydantic import PositiveInt

from .benchmark import run_benchmark_aiohttp
from mlperf_inf_mm_q3vl.schema import (
    Dataset,
    Settings,
    Verbosity,
)
from pydantic_typer import Typer
from typer import Option

from .schema import DynamoEndpoint, VllmProfileEndpoint, Wandb


def register_nv() -> tuple[Typer, str]:
    """Register the NVIDIA-optimized Qwen3-VL (Q3VL) benchmark."""
    benchmark_nv_app = Typer(help="NVIDIA-optimized Qwen3-VL (Q3VL) benchmark.")

    @benchmark_nv_app.command(name="mpi-dynamo-vllm")
    def benchmark_nv_mpi_dynamo_vllm(
        *,
        settings: Settings,
        dataset: Dataset,
        dynamo: DynamoEndpoint,
        random_seed: Annotated[
            int,
            Option(
                help="The seed for the random number generator used by the benchmark.",
            ),
        ] = 12345,
        use_http_client: Annotated[
            bool,
            Option(
                help="Use the aiohttp-based HTTP client with pre-serialized request "
                "bodies instead of the default OpenAI SDK client.",
            ),
        ] = False,
        max_concurrency: Annotated[
            Optional[PositiveInt],
            Option(
                help="Maximum number of concurrent in-flight requests when using "
                "the aiohttp HTTP client (--use-http-client). "
                "If not set, no concurrency throttling is applied. "
                "Ignored when --no-use-http-client.",
            ),
        ] = None,
        verbosity: Annotated[
            Verbosity,
            Option(help="The verbosity level of the logger."),
        ] = Verbosity.INFO,
        wandb_config: Wandb,
    ) -> None:
        """Deploy and benchmark the MPI-Dynamo-vLLM implementation."""
        from mpi4py import MPI

        from .deploy import MpiDynamoVllmEndpointDeployer
        from .log import setup_loguru_for_benchmark_nv
        from .wandb_utils import read_and_log_mlperf_detail_to_wandb

        comm = MPI.COMM_WORLD
        setup_loguru_for_benchmark_nv(settings=settings, verbosity=verbosity)

        # Initialize wandb on rank 0
        if wandb_config.is_configured():
            if comm.Get_rank() == 0:
                wandb.login(key=wandb_config.api_key)
                dynamo_config = dynamo.model_dump(exclude={"model": {"token"}})
                wandb.init(
                    name=wandb_config.name,
                    entity=wandb_config.entity,
                    project=wandb_config.project,
                    config={
                        "random_seed": random_seed,
                        "verbosity": verbosity.value,
                        "settings": settings.model_dump(),
                        "dataset": dataset.model_dump(),
                        "dynamo": dynamo_config,
                    },
                )

        with MpiDynamoVllmEndpointDeployer(
            endpoint=dynamo,
            settings=settings,
            warmup_dataset=dataset,
            base_random_seed=random_seed,
        ):
            if comm.Get_rank() == 0:
                if use_http_client:
                    run_benchmark_aiohttp(
                        settings=settings,
                        dataset=dataset,
                        endpoint=dynamo,
                        random_seed=random_seed,
                        max_concurrency=max_concurrency,
                    )
                else:
                    if max_concurrency is not None:
                        logger.warning("--max-concurrency is ignored without --use-http-client.")
                    run_benchmark(
                        settings=settings,
                        dataset=dataset,
                        endpoint=dynamo,
                        random_seed=random_seed,
                    )
                if wandb_config.is_configured():
                    # Also read and log the mlperf_log_detail.txt file
                    read_and_log_mlperf_detail_to_wandb(settings)
                    wandb.finish()
            comm.Barrier()

    @benchmark_nv_app.command(name="vllm-profiler")
    def profiler_nv(
        *,
        settings: Settings,
        dataset: Dataset,
        vllm: VllmProfileEndpoint,
        random_seed: Annotated[
            int,
            Option(
                help="The seed for the random number generator used by the benchmark.",
            ),
        ] = 12345,
        verbosity: Annotated[
            Verbosity,
            Option(help="The verbosity level of the logger."),
        ] = Verbosity.INFO,
    ) -> None:
        """Deploy the endpoint using vLLM into a healthy state and then benchmark it.

        This is suitable when you have access to the `vllm serve` command in the local
        environment where this benchmarking CLI is running.
        """
        from .deploy import VllmProfileEndpointDeployer
        from .log import setup_loguru_for_benchmark_nv

        setup_loguru_for_benchmark_nv(settings=settings, verbosity=verbosity)
        with VllmProfileEndpointDeployer(endpoint=vllm, settings=settings):
            run_benchmark(
                settings=settings,
                dataset=dataset,
                endpoint=vllm,
                random_seed=random_seed,
            )

    return (benchmark_nv_app, "nv")
