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


from code import G_BENCHMARK_MODULES
from code.common.gbs import GeneralizedBatchSize
from code.common.systems.system_list import DETECTED_SYSTEM
from dataclasses import dataclass
from nvmitten.aliased_name import AliasedNameEnum
from nvmitten.configurator import bind, autoconfigure, Field
from nvmitten.nvidia.builder import TRTBuilder
from nvmitten.system import System
from pathlib import Path
from typing import ClassVar, List, Optional

import code.common.constants as C
import code.common.paths as paths
import code.fields.general as general_fields
import code.fields.models as model_fields
import code.fields.meta as metafields
import json
import os


@autoconfigure
@bind(metafields.device_types)
@bind(general_fields.log_dir)
class Workload:
    """
    Represents a workload configuration for MLPerf inference benchmarks.

    This class manages the configuration and settings for running MLPerf inference workloads,
    including benchmark type, scenario, system settings, and device types.

    Attributes:
        FIELD (ClassVar[Field]): Configuration field for workload injection.
        benchmark (C.Benchmark): The benchmark type to run.
        scenario (C.Scenario): The inference scenario to use.
        system (System): The system configuration.
        setting (C.WorkloadSetting): Workload-specific settings.
        device_types (List[str]): List of device types to use (e.g., ["gpu", "dla"]).
        log_dir (Path): Directory for storing workload logs.
    """

    # Convenience Field so that Workload can be injected into the Configuration by MainRunner
    FIELD: ClassVar[Field] = Field("workload",
                                   disallow_default=True,
                                   disallow_argparse=True)

    def __init__(self,
                 benchmark: C.Benchmark,
                 scenario: C.Scenario,
                 system: System = DETECTED_SYSTEM,
                 setting: C.WorkloadSetting = C.WorkloadSetting(),
                 device_types: List[str] = None,
                 log_dir: os.PathLike = paths.BUILD_DIR / "logs" / "default"):
        """
        Initialize a Workload instance.

        Args:
            benchmark (C.Benchmark): The benchmark type to run.
            scenario (C.Scenario): The inference scenario to use.
            system (System, optional): The system configuration. Defaults to DETECTED_SYSTEM.
            setting (C.WorkloadSetting, optional): Workload-specific settings. Defaults to C.WorkloadSetting().
            device_types (List[str], optional): List of device types to use. Defaults to ["gpu"] or ["gpu", "dla"] for SoC.
            log_dir (os.PathLike, optional): Directory for storing workload logs. Defaults to BUILD_DIR/logs/default.
        """
        self.benchmark = benchmark
        self.scenario = scenario
        self.system = system
        self.setting = setting

        if device_types is None or device_types == ["all"]:
            self.device_types = ["gpu"]
            if "is_soc" in self.system.extras["tags"]:
                self.device_types.append("dla")
        else:
            self.device_types = device_types

        self.base_log_dir = Path(log_dir)
        self.log_dir = self.base_log_dir / self.submission_system / self.submission_benchmark / scenario.valstr
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.audit_test01_fallback_mode = False

    def __eq__(self, other):
        """
        Check if two Workload instances are equal.

        Args:

            other (Workload): The other Workload instance to compare with.

        Returns:
            bool: True if the Workloads are equal, False otherwise.
        """
        if not isinstance(other, Workload):
            return NotImplemented
        return (self.benchmark == other.benchmark and
                self.scenario == other.scenario and
                self.system == other.system and
                self.setting == other.setting and
                self.device_types == other.device_types and
                self.log_dir == other.log_dir)

    def __str__(self):
        """
        Return a string representation of the Workload.

        Returns:
            str: String representation in the format "Workload(benchmark, scenario, setting, log_dir)"
        """
        return f"Workload({self.benchmark}, {self.scenario}, {self.setting.short}, {self.log_dir})"

    @property
    def submission_benchmark(self) -> str:
        """
        Get the submission benchmark name based on benchmark and accuracy target.

        Returns:
            str: The submission benchmark name.
        """
        return C.submission_benchmark_name(self.benchmark, self.setting.accuracy_target)

    @property
    def submission_system(self) -> str:
        """
        Get the submission system name based on system ID and power settings.

        Returns:
            str: The submission system name in the format "system_id_TRT[_MaxQ]".
        """
        parts = [self.system.extras["id"], "TRT"]
        if self.setting.power_setting == C.PowerSetting.MaxQ:
            parts.append("MaxQ")
        return '_'.join(parts)


@dataclass
class ComponentEngine:
    """
    Represents a component engine configuration for MLPerf inference.

    This class holds the configuration for a specific component engine, including
    its component type, precision, batch sizes, and device type.

    Attributes:
        component (AliasedNameEnum): The component type.
        precision (C.Precision): The precision setting for the engine.
        batch_size (int): The batch size for this component.
        e2e_batch_size (int): The end-to-end batch size.
        setting (C.WorkloadSetting): The workload settings.
        device_type (str): The device type (default: "gpu").
    """

    component: AliasedNameEnum
    precision: C.Precision
    batch_size: int
    e2e_batch_size: int
    setting: C.WorkloadSetting
    device_type: str = "gpu"

    @property
    def fname(self) -> str:
        """
        Generate the filename for the engine plan.

        Returns:
            str: The filename in the format "device_type-component-precision-batch_size.setting.plan"
        """
        base = '-'.join([self.device_type,
                         self.component.valstr,
                         self.precision.valstr,
                         f"b{self.batch_size}"])
        return f"{base}.{self.setting.short}.plan"


@autoconfigure
@bind(Workload.FIELD)
@bind(general_fields.engine_dir)
@bind(model_fields.precision)
class EngineIndex:
    """
    Manages the indexing and enumeration of engine components for MLPerf inference.

    This class handles the organization and validation of engine components,
    including batch size configurations and component combinations.

    Attributes:
        wl (Workload): The associated workload.
        base_dir (Path): Base directory for engine files.
        precision (C.Precision): The precision setting for engines.
        bs_info (GeneralizedBatchSize): Batch size information.
        module: The benchmark module.
        component_type: The type of components.
        component_map: Mapping of components.
        valid_component_sets: Valid combinations of components.
        _ComponentEngineCls: The class to use for component engines.
        engines (List[ComponentEngine]): List of configured engines.
    """

    def __init__(self,
                 workload: Optional[Workload] = None,
                 engine_dir: Optional[os.PathLike] = None,
                 bs_info: Optional[GeneralizedBatchSize] = None,
                 precision: C.Precision = C.Precision.FP32):
        """
        Initialize an EngineIndex instance.

        Args:
            workload (Optional[Workload]): The associated workload. Required if engine_dir is not provided.
            engine_dir (Optional[os.PathLike]): Directory for engine files.
            bs_info (Optional[GeneralizedBatchSize]): Batch size information.
            precision (C.Precision): The precision setting for engines. Defaults to FP32.

        Raises:
            ValueError: If Triton harness is used with multi-component inference.
        """
        self.wl = workload
        if engine_dir:
            self.base_dir = Path(engine_dir)
        else:
            self.base_dir = paths.BUILD_DIR.joinpath("engines",
                                                     self.wl.system.extras["id"],
                                                     self.wl.scenario.valstr,
                                                     self.wl.benchmark.valstr)

        self.precision = precision
        self.bs_info = bs_info if bs_info else GeneralizedBatchSize()
        # Triton does not support multi-component inference
        if self.wl.setting.harness_type == C.HarnessType.Triton and \
            len(self.bs_info.gpu_batch_size) != 1:
            raise ValueError("Triton does not support multi-component inference. Expected 1 component in batch size, "
                             f"got {len(self.bs_info.gpu_batch_size)}.")

        self.module = G_BENCHMARK_MODULES[self.wl.benchmark]
        self.component_type = self.module.component_type
        self.component_map = self.module.component_map
        self.valid_component_sets = self.module.valid_component_sets

        if hasattr(self.module.load(), "ComponentEngine"):
            self._ComponentEngineCls = self.module.load().ComponentEngine
        else:
            self._ComponentEngineCls = ComponentEngine

        self.engines = self.enumerate_engines()

    def enumerate_engines(self) -> List[ComponentEngine]:
        """
        Enumerate and validate all engine components for the workload.

        Returns:
            List[ComponentEngine]: List of configured engine components.

        Raises:
            KeyError: If an invalid component is specified.
            ValueError: If an invalid component combination is used.
        """
        engines = list()
        for dev_type in self.wl.device_types:
            e2e = self.bs_info.e2e(dev=dev_type)

            component_set = set()
            for component, bs in self.bs_info.iter_of(dev_type):
                _c = self.component_type.get_match(component)
                if not _c:
                    raise KeyError(f"{component} is not a valid component in {self.component_type}")
                component_set.add(_c)

                precision = self.precision if isinstance(self.precision, C.Precision) else self.precision[component]
                engines.append(
                    self._ComponentEngineCls(_c,
                                             precision,
                                             bs,
                                             e2e,
                                             self.wl.setting,
                                             device_type=dev_type))
            if component_set not in self.valid_component_sets[dev_type]:
                raise ValueError(f"{component_set} is not a valid component combination. Valid combinations are "
                                 f"{self.valid_component_sets[dev_type]}")
        return engines

    def full_path(self, comp_eng: ComponentEngine) -> Path:
        """
        Get the full path for a component engine file.

        Args:
            comp_eng (ComponentEngine): The component engine.

        Returns:
            Path: The full path to the engine file.
        """
        return self.base_dir / comp_eng.fname

    def iter_components(self):
        """
        Iterate over all component engines and their mappings.

        Yields:
            Tuple[ComponentEngine, Any]: Pairs of component engines and their mappings.
        """
        # Assume all keys are of the same type
        for c_eng in self.engines:
            yield c_eng, self.component_map[c_eng.component]
