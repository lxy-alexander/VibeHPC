"""Qwen3-VL 235B benchmark module for MLPerf Inference."""

__version__ = "6.0.2"
__all__ = ["Qwen3VL235BHarnessOp", "BenchmarkHarnessOp"]

from .harness import Qwen3VL235BHarnessOp

COMPONENT_MAP = {}

VALID_COMPONENT_SETS = {"gpu": [set()]}  # Q3VL uses a unified pipeline

# Export the harness operation for the benchmark framework
BenchmarkHarnessOp = Qwen3VL235BHarnessOp

# Disable calibration and engine building ops since Q3VL uses vLLM/Dynamo
CalibrateEngineOp = None
EngineBuilderOp = None
