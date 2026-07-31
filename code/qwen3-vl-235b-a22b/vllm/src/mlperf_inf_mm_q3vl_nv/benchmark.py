"""Core benchmark execution logic for the NVIDIA-optimized Qwen3-VL (Q3VL) benchmark."""

from __future__ import annotations

from typing import Optional

import mlperf_loadgen as lg
from loguru import logger

from mlperf_inf_mm_q3vl.schema import Dataset, Endpoint, Settings

from .task import AioHttpTask


def run_benchmark_aiohttp(
    settings: Settings,
    dataset: Dataset,
    endpoint: Endpoint,
    random_seed: int,
    max_concurrency: Optional[int] = None,
) -> None:
    """Run benchmark with direct aiohttp POST and pre-serialized request bodies.

    Uses AioHttpTask which pre-serializes request bodies and posts them
    directly via aiohttp.  When *max_concurrency* is set, an
    asyncio.Semaphore limits the number of in-flight requests.

    Args:
        settings: Benchmark settings.
        dataset: Dataset configuration.
        endpoint: Endpoint configuration.
        random_seed: Random seed for reproducibility.
        max_concurrency: Maximum concurrent in-flight requests.
            ``None`` means no limit.
    """
    logger.info("Running benchmark with AioHttpTask")
    logger.info("Settings: {}", settings)
    logger.info("Dataset: {}", dataset)
    logger.info("Endpoint: {}", endpoint)
    logger.info("Random seed: {}", random_seed)
    logger.info("Max concurrency: {}", max_concurrency)

    test_settings, log_settings = settings.to_lgtype()
    task = AioHttpTask(
        dataset=dataset,
        endpoint=endpoint,
        settings=settings.test,
        random_seed=random_seed,
        max_concurrency=max_concurrency,
    )
    sut = task.construct_sut()
    qsl = task.construct_qsl()
    logger.info("Starting benchmark with LoadGen...")
    lg.StartTestWithLogSettings(sut, qsl, test_settings, log_settings)
    logger.info("Benchmark completed.")
    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)
