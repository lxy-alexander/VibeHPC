"""Logging utilities for the NVIDIA-optimized Qwen3-VL (Q3VL) benchmark."""

from __future__ import annotations

import sys

from loguru import logger
from mlperf_inf_mm_q3vl.log import get_log_file_path
from mlperf_inf_mm_q3vl.schema import Settings, Verbosity
from mpi4py import MPI


def setup_loguru_for_benchmark_nv(settings: Settings, verbosity: Verbosity) -> None:
    """Setup the loguru logger for running the benchmark."""
    logger.remove()
    logger.add(sys.stdout, level=verbosity.value.upper())
    logger.add(
        get_log_file_path(
            key=f"benchmark.rank{MPI.COMM_WORLD.Get_rank()}",
            settings=settings,
        ),
        level=verbosity.value.upper(),
    )
