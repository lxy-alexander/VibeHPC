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
from functools import partial

from nvmitten.configurator import Field
from code.common.constants import Action, Benchmark, Scenario, HarnessType, AccuracyTarget, PowerSetting


__doc__ = """Meta-Fields

Used to control codepaths and determine which other fields are relevant.
"""


def parse_aliased_name_enum_strict(s, enum_cls, as_list: bool = False):
    if as_list:
        L = list()
        for tok in s.split(','):
            if enum_member := enum_cls.get_match(tok):
                L.append(enum_member)
            else:
                raise ValueError(f"Invalid {enum_cls.__name__}: {tok}")
        return L
    else:
        if enum_member := enum_cls.get_match(s):
            return enum_member
        else:
            raise ValueError(f"Invalid {enum_cls.__name__}: {s}")


action = Field(
    "action",
    description="The phase in the benchmarking or submission process to run.",
    disallow_default=True,
    from_string=partial(parse_aliased_name_enum_strict, enum_cls=Action))

benchmarks = Field(
    "benchmarks",
    description=("The names of benchmarks to run the current phase on. "
                 "If specifying multiple benchmarks, use a comma,separated,list of names "
                 "(i.e. resnet50,bert,dlrm) "
                 "Default: Runs all benchmarks with valid configurations for the current system."),
    from_string=partial(parse_aliased_name_enum_strict, enum_cls=Benchmark, as_list=True))

scenarios = Field(
    "scenarios",
    description=("The names of scenarios to run the current phase for. "
                 "If specifying multiple scenarios, use a comma,separated,list of names "
                 "(i.e. offline,server) "
                 "Default: Runs all scenarios with valid configurations for the current system."),
    from_string=partial(parse_aliased_name_enum_strict, enum_cls=Scenario, as_list=True))

device_types = Field(
    "device_types",
    description=("A comma,separated,list of strings denoting which of the available "
                 "accelerators to use. By default, will use all available accelerators, "
                 "which is denoted by the string 'all'. "
                 "For example, 'gpu' will use only gpus, and 'dla' will only use DLAs."),
    from_string=lambda s: s.split(','))

enable_power_meter = Field(
    "use_power_meter",
    description="Measure power during this harness run. Note that this does NOT set power_setting to MaxQ.",
    from_string=bool)

harness_type = Field(
    "harness_type",
    description=("Selects which harness to use during the run_harness phase. If not set, will use "
                 "the default harness."),
    from_string=partial(parse_aliased_name_enum_strict, enum_cls=HarnessType))

accuracy_target = Field(
    "accuracy_target",
    description=("Selects which accuracy target to compare the accuracy results against. "
                 "Note that using the .999 accuracy target is only available on certain "
                 "benchmarks."),
    from_string=lambda s: AccuracyTarget(float(s)))

power_setting = Field(
    "power_setting",
    description=("Selects which power setting to run the harness on. Note that this flag is only "
                 "used to select the appropriate configuration. It does NOT actually apply the "
                 "system settings."),
    from_string=partial(parse_aliased_name_enum_strict, enum_cls=PowerSetting))
