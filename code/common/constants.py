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
import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass, asdict, field
from enum import unique
from typing import Any, Dict, Final

# Local imports
from code.common import __MLPERF_INF_VERSION__

# Third-party imports
# pylint: disable=unused-import
from nvmitten.memory import QUANTITY_UNIT_FORMAT, ByteSuffix, Memory  # noqa: F401
from nvmitten.aliased_name import AliasedName, AliasedNameEnum  # noqa: F401
from nvmitten.constants import Precision, InputFormats, CPUArchitecture as CPUArch  # noqa: F401


# Conditional imports
if importlib.util.find_spec("tensorrt") is not None:
    import tensorrt as trt

    TRT_LOGGER: Final[trt.Logger] = trt.Logger(trt.Logger.INFO)
    """trt.Logger: TRT Logger Singleton. TRT Runtime requires its logger to be a singleton object, so if we create
    multiple runtimes with different trt.Logger objects, a segmentation fault will occur with 'The logger passed into
    createInferRuntime differs from one already assigned'"""

if importlib.util.find_spec("mlperf_loadgen") is not None:
    import mlperf_loadgen as lg


__doc__ = """Stores constants and Enums related to MLPerf Inference"""


VERSION: Final[str] = __MLPERF_INF_VERSION__
"""str: Current version of MLPerf Inference"""


IS_EXTERNAL_USER: Final[bool] = os.environ.get("EXTERNAL_USER", "1").lower() == "1"


@unique
class HarnessType(AliasedNameEnum):
    """Possible harnesses a benchmark can use."""
    Custom: AliasedName = AliasedName("custom", ('c',))
    Triton: AliasedName = AliasedName("triton", ('t',))
    HeteroMIG: AliasedName = AliasedName("hetero", ('h',))

    @property
    def short(self):
        return self.value.aliases[0]


class AccuracyTarget:
    """
    Represents an accuracy target for a benchmark.
    """

    def __init__(self, value: float = 0.99):
        self.value = value

    def __str__(self):
        return f"{self.value * 100:.1f}%"

    @property
    def short(self):
        """
        Returns a short string representation of the accuracy target.
        """
        d = int(round(self.value * 1000))
        return str(d)

    @classmethod
    def get_match(cls, v):
        """
        Returns an AccuracyTarget instance based on a string or float value.
        """
        if isinstance(v, float):
            return AccuracyTarget(v)
        elif isinstance(v, str):
            return AccuracyTarget(float(v) / 1000)
        return None

    def __eq__(self, o):
        if not isinstance(o, AccuracyTarget):
            return NotImplemented

        return self.value == o.value

    def __hash__(self):
        return hash(self.value)


@unique
class PowerSetting(AliasedNameEnum):
    """Possible power settings the system can be set in when running a benchmark."""
    MaxP: AliasedName = AliasedName("MaxP", ('p',))
    MaxQ: AliasedName = AliasedName("MaxQ", ('q',))

    @property
    def short(self):
        return self.value.aliases[0]


@dataclass(frozen=True)
class WorkloadSetting:
    """
    Describes the various settings used when running a benchmark workload. These are usually for different use cases that
    MLPerf Inference allows (i.e. power submission), or running the same workload with different software (i.e. Triton).
    """
    harness_type: HarnessType = HarnessType.Custom
    """HarnessType: Harness to use for this workload. Default: HarnessType.Custom"""

    accuracy_target: AccuracyTarget = field(default_factory=AccuracyTarget)
    """AccuracyTarget: Accuracy target for the benchmark. Default: .99 of FP32 accuracy"""

    power_setting: PowerSetting = PowerSetting.MaxP
    """PowerSetting: Power setting for the system during this workload. Default: PowerSetting.MaxP"""

    def __str__(self) -> str:
        return f"WorkloadSetting({self.harness_type}, {self.accuracy_target}, {self.power_setting})"

    def as_dict(self) -> Dict[str, Any]:
        """
        Convenience wrapper around dataclasses.asdict to convert this WorkloadSetting to a dict().

        Returns:
            Dict[str, Any]: This WorkloadSetting as a dict
        """
        return asdict(self)

    @property
    def short(self) -> str:
        """
        Returns a short string representation of the WorkloadSetting.
        """
        return ''.join((self.harness_type.short,
                        self.power_setting.short,
                        self.accuracy_target.short))

    @classmethod
    def from_short(cls, s: str) -> WorkloadSetting:
        """
        Creates a WorkloadSetting from a short string.
        """
        return cls(harness_type=HarnessType.get_match(s[0]),
                   accuracy_target=AccuracyTarget.get_match(s[2:]),
                   power_setting=PowerSetting.get_match(s[1]))


@unique
class Benchmark(AliasedNameEnum):
    """Names of supported Benchmarks in MLPerf Inference."""

    BERT: AliasedName = AliasedName("bert", ("bert-99", "bert-99.9"))
    DLRMv2: AliasedName = AliasedName("dlrm-v2", ("dlrm_v2", "dlrmv2", "dcnv2", "dlrm_dcn", "dlrm-v2-99", "dlrm-v2-99.9"))
    GPTJ: AliasedName = AliasedName("gptj", ("gptj6b", "gpt-j", "gptj-99", "gptj-99.9"))
    LLAMA2: AliasedName = AliasedName("llama2-70b", ("llama2", "llama-v2", "llama-v2-70b", "llama2-70b-99", "llama2-70b-99.9"))
    LLAMA3_1_8B: AliasedName = AliasedName("llama3_1-8b", ("llama3.1-8b", "llama-v3.1-8b", "llama3.1-8b-99"))
    LLAMA3_1_405B: AliasedName = AliasedName("llama3_1-405b", ("llama3.1-405b", "llama-v3.1-405b", "llama3.1-405b-99", "llama3.1-405b-99.9"))
    Mixtral8x7B: AliasedName = AliasedName("mixtral-8x7b", ("mixtral", "mixtral8x7b", "moe", "mixtral-8x7b-99", "mixtral-8x7b-99.9"))
    DeepSeek_R1: AliasedName = AliasedName("deepseek-r1", ("deepseek_r1", "deeseek-r1-99", "deepseek_r1-99"))
    GPT_OSS_120B: AliasedName = AliasedName("gpt-oss-120b", ("gpt_oss_120b", "gptoss120b", "gpt-oss-120b-99"))
    ResNet50: AliasedName = AliasedName("resnet50", ("resnet",))
    Retinanet: AliasedName = AliasedName("retinanet", ("ssd-retinanet", "resnext", "ssd-resnext"))
    SDXL: AliasedName = AliasedName("stable-diffusion-xl", ("sdxl-base", "diffusion", "stable-diffusion", "sdxl"))
    WAN22_A14B: AliasedName = AliasedName("wan22-a14b", ("wan-2.2-t2v-a14b", "wan2.2-a14b", "wan-t2v-a14b", "wan22-t2v-a14b", "wan2.2-t2v-a14b", "wan2.2-t2v-14b"))
    WHISPER: AliasedName = AliasedName("whisper", ("whisper-large-v3"))
    Q3VL: AliasedName = AliasedName("qwen3-vl-235b-a22b", ("q3vl", "qwen3-vl", "qwen3-vl-235b-a22b"))
    RGAT: AliasedName = AliasedName("rgat", ("r-gat",))

    @property
    def supports_high_acc(self):
        return self in (Benchmark.BERT,
                        Benchmark.DLRMv2,
                        Benchmark.GPTJ,
                        Benchmark.LLAMA2)

    @property
    def supports_datacenter(self):
        return self in (Benchmark.GPTJ,
                        Benchmark.LLAMA2,
                        Benchmark.LLAMA3_1_8B,
                        Benchmark.LLAMA3_1_405B,
                        Benchmark.Mixtral8x7B,
                        Benchmark.DeepSeek_R1,
                        Benchmark.GPT_OSS_120B,
                        Benchmark.DLRMv2,
                        Benchmark.ResNet50,
                        Benchmark.Retinanet,
                        Benchmark.SDXL,
                        Benchmark.WAN22_A14B,
                        Benchmark.Q3VL,
                        Benchmark.RGAT)

    @property
    def supports_edge(self):
        return self in (Benchmark.BERT,
                        Benchmark.GPTJ,
                        Benchmark.ResNet50,
                        Benchmark.Retinanet,
                        Benchmark.SDXL)

    @property
    def default_harness_type(self):
        # If default needs to be changed on per-benchmark basis, define the mapping here.
        return HarnessType.Custom

    @property
    def is_llm(self):
        """Returns whether the given benchmark is an LLM benchmark.

        Returns:
            bool: True if benchmark is an LLM benchmark, False otherwise
        """
        return self in (Benchmark.GPTJ,
                        Benchmark.LLAMA2,
                        Benchmark.LLAMA3_1_8B,
                        Benchmark.LLAMA3_1_405B,
                        Benchmark.Mixtral8x7B,
                        Benchmark.DeepSeek_R1,
                        Benchmark.GPT_OSS_120B,
                        Benchmark.WHISPER)

    @property
    def supports_triton(self):
        """Returns whether the given benchmark supports Triton."""
        return self in (Benchmark.LLAMA2,
                        Benchmark.LLAMA3_1_8B,
                        Benchmark.LLAMA3_1_405B,
                        Benchmark.Mixtral8x7B)

    @property
    def python_path(self):
        return sys.executable


def submission_benchmark_name(b: Benchmark, acc_target: AccuracyTarget) -> str:
    if b == Benchmark.ResNet50:
        s = "resnet"
    elif b == Benchmark.LLAMA3_1_8B:
        s = "llama3.1-8b"
    elif b == Benchmark.LLAMA3_1_405B:
        s = "llama3.1-405b"
    elif b == Benchmark.WAN22_A14B:
        s = "wan-2.2-t2v-a14b"
    else:
        s = b.valstr
    if b.supports_high_acc:
        _v = acc_target.value * 100
        suffix = f"-{_v:.1f}".rstrip('0').rstrip('.')  # Remove trailing 0
        return s + suffix
    return s


@unique
class Scenario(AliasedNameEnum):
    """Names of supported workload scenarios in MLPerf Inference."""

    Offline: AliasedName = AliasedName("Offline")
    Server: AliasedName = AliasedName("Server")
    Interactive: AliasedName = AliasedName("Interactive")
    SingleStream: AliasedName = AliasedName("SingleStream", ("single-stream", "single_stream"))
    MultiStream: AliasedName = AliasedName("MultiStream", ("multi-stream", "multi_stream"))

    def to_lg_obj(self):
        # Temporary fix till loadgen makes Interactive a scenario instead of a Benchmark.
        if self.name == "Interactive":
            return getattr(lg.TestScenario, self.Server.name)
        return getattr(lg.TestScenario, self.name)


@unique
class Action(AliasedNameEnum):
    """Names of actions performed by our MLPerf Inference pipeline."""

    GenerateConfFiles: AliasedName = AliasedName("generate_conf_files")
    GenerateEngines: AliasedName = AliasedName("generate_engines")
    Calibrate: AliasedName = AliasedName("calibrate")
    RunHarness: AliasedName = AliasedName("run_harness")
    RunAuditHarness: AliasedName = AliasedName("run_audit_harness")
    GenerateTritonConfig: AliasedName = AliasedName("generate_triton_config")
    GenerateDisaggConfig: AliasedName = AliasedName("generate_disagg_config")
    RunLLMServer: AliasedName = AliasedName("run_llm_server")


@unique
class AuditTest(AliasedNameEnum):
    TEST01: AliasedName = AliasedName("TEST01")
    TEST04: AliasedName = AliasedName("TEST04")
    TEST06: AliasedName = AliasedName("TEST06")
    TEST07: AliasedName = AliasedName("TEST07")
    TEST09: AliasedName = AliasedName("TEST09")


def test_mode_to_lg_obj(test_mode_str: str):
    test_mode_map = {"PerformanceOnly": lg.TestMode.PerformanceOnly,
                     "AccuracyOnly": lg.TestMode.AccuracyOnly,
                     "SubmissionRun": lg.TestMode.SubmissionRun}
    return test_mode_map[test_mode_str]


def log_mode_to_lg_obj(mode_str: str):
    log_mode_map = {"AsyncPoll": lg.LoggingMode.AsyncPoll,
                    "EndOfTestOnly": lg.LoggingMode.EndOfTestOnly,
                    "Synchronous": lg.LoggingMode.Synchronous}
    return log_mode_map[mode_str]
