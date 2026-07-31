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

# Standard library imports
import datetime
import dataclasses as dcls
import glob
import json
import logging
import os
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import ClassVar, Any, Optional

# Third-party imports
from nvmitten.configurator import bind, autoconfigure, HelpInfo

# Local imports
from .. import constants as C
from .. import paths
from ..utils import safe_divide
from ..workload import Workload
from ...fields import general as general_fields
from ...fields import loadgen as lg_fields


@autoconfigure
@bind(general_fields.log_dir)
@bind(lg_fields.logfile_suffix)
@bind(lg_fields.logfile_prefix_with_datetime)
@bind(lg_fields.log_copy_detail_to_stdout)
@bind(lg_fields.disable_log_copy_summary_to_stdout)
@bind(lg_fields.log_mode)
@bind(lg_fields.log_mode_async_poll_interval_ms)
@bind(lg_fields.log_enable_trace)
@dcls.dataclass
class LoadgenLogSettings:
    """Configuration settings for MLPerf loadgen logging.

    This class holds configuration parameters for MLPerf loadgen logging,
    including log directory, file naming, and output settings.
    """
    log_dir: os.PathLike = paths.BUILD_DIR / "logs" / "default"
    logfile_prefix: str = "mlperf_log_"
    logfile_suffix: str = ""
    logfile_prefix_with_datetime: bool = False
    log_copy_detail_to_stdout: bool = False
    disable_log_copy_summary_to_stdout: bool = False
    log_mode: str = "AsyncPoll"
    log_mode_async_poll_interval_ms: int = 1000
    log_enable_trace: bool = False


def timestamp_to_dt(timestamp: str, local_time: bool = False) -> datetime.datetime:
    """Convert a timestamp string to a datetime object.

    Args:
        timestamp (str): Timestamp string in format "MM-DD-YYYY HH:MM:SS.microseconds"
        local_time (bool, optional): If True, converts to local time. Defaults to False.

    Returns:
        datetime.datetime: Converted datetime object
    """
    result = datetime.datetime.strptime(timestamp, "%m-%d-%Y %H:%M:%S.%f")
    if local_time:
        timezone = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
        result -= timezone.utcoffset(None)
    return result


def result_key(benchmark: C.Benchmark, scenario: C.Scenario) -> str:
    """Get the appropriate result key based on benchmark and scenario.

    Args:
        benchmark (C.Benchmark): The benchmark being run
        scenario (C.Scenario): The scenario being tested

    Returns:
        str: The key to use for retrieving results from the log

    Raises:
        ValueError: If the scenario is not valid
    """
    if scenario == C.Scenario.SingleStream:
        return "result_90.00_percentile_latency_ns"
    elif scenario == C.Scenario.MultiStream:
        return "result_99.00_percentile_per_query_latency_ns"
    elif scenario == C.Scenario.Offline:
        if benchmark is not None and (benchmark.is_llm or benchmark is C.Benchmark.WHISPER):
            return "result_tokens_per_second"
        else:
            return "result_samples_per_second"
    elif scenario in (C.Scenario.Server, C.Scenario.Interactive):
        if benchmark is not None and benchmark.is_llm:
            return "result_completed_tokens_per_second"
        else:
            return "result_completed_samples_per_sec"
    else:
        raise ValueError(f"{scenario} is not a valid scenario")


LogPaths = namedtuple("LogPaths", ["base", "detail", "summary", "spl"])


class LoadgenLogReader:
    """A class for reading and parsing MLPerf loadgen log files.

    This class provides functionality to read and parse MLPerf loadgen log files,
    extract performance metrics, and handle power measurements.
    """

    LINE_PREFIX: ClassVar[str] = ":::MLLOG"

    def __init__(self,
                 workload: Optional[Workload] = None,
                 benchmark: Optional[C.Benchmark] = None,
                 scenario: Optional[C.Scenario] = None,
                 power_setting: Optional[C.PowerSetting] = None,
                 log_dir: Optional[os.PathLike] = None):
        """Initialize the LoadgenLogReader with a workload.

        Args:
            workload (Workload): The workload configuration to use for log reading
            benchmark (Benchmark): The benchmark to use. Ignored if workload is provided.
            scenario (Scenario): The scenario to use. Ignored if workload is provided.
            power_setting (PowerSetting): The power setting to use. Ignored if workload is provided.
            log_dir (os.PathLike): The base path of log files. Ignored if workload is provided.
        """
        if workload:
            self.benchmark = workload.benchmark
            self.scenario = workload.scenario
            self.power_setting = workload.setting.power_setting
            base_path = workload.log_dir
            detail = base_path / "mlperf_log_detail.txt"
            if not detail.exists():
                detail = Path(glob.glob(base_path / "**" / "mlperf_log_detail.txt", recursive=True)[0])
        else:
            self.benchmark = benchmark
            self.scenario = scenario
            self.power_setting = power_setting
            base_path = Path(log_dir)
            detail = base_path / "mlperf_log_detail.txt"
            if not detail.exists():
                raise FileNotFoundError("log_dir was explicitly provided, but detail log does not exist.")

        self.log_settings = LoadgenLogSettings()
        self.log_paths = LogPaths(base_path,
                                  detail,
                                  base_path / "mlperf_log_summary.txt",
                                  base_path / "spl.txt")

        self.data = self.load_log()
        self.found_scenario = C.Scenario.get_match(self.get("requested_scenario"))

    @property
    def perf_metric(self) -> str:
        """Get the performance metric key based on power setting and scenario.

        Returns:
            str: The key to use for performance metrics
        """
        if self.power_setting == C.PowerSetting.MaxQ:
            if self.scenario in [C.Scenario.Offline, C.Scenario.Server]:
                return "qps_per_avg_watt"
            else:
                return "joules_per_stream"
        else:
            return result_key(self.benchmark, self.scenario)

    @property
    def perf_value(self) -> float:
        """Get the performance value from the log data.

        Returns:
            float: The performance value, or 0.0 if not found
        """
        k = result_key(self.benchmark, self.found_scenario)
        v = self.get(k)
        if v:
            return float(v)
        else:
            logging.warning("Could not find perf value for %s in %s", k, self.log_paths.detail)
            return 0.0

    def load_log(self):
        """Load and parse the log file.

        Returns:
            defaultdict: A dictionary containing log entries grouped by key
        """
        path = self.log_paths.detail
        with path.open(mode='r') as f:
            lines = f.read().strip().split('\n')

        log_entries = list()
        for line in lines:
            if line.startswith(LoadgenLogReader.LINE_PREFIX):
                buf = line[len(LoadgenLogReader.LINE_PREFIX) + 1:]
                try:
                    log_entries.append(json.loads(buf))
                except:
                    logging.error("Could not parse %s: %s", path, buf)
                    raise

        results = defaultdict(list)
        for entry in log_entries:
            key = entry["key"]
            results[key].append(entry["value"])
        return results

    def get_keys(self, *keys, latest_value: bool = True) -> dict:
        """Get values for specified keys from the log data.

        Args:
            *keys: Variable number of keys to retrieve
            latest_value (bool, optional): If True, returns only the latest value. Defaults to True.

        Returns:
            dict: Dictionary mapping keys to their values
        """
        d = dict()
        for k in keys:
            if k not in self.data:
                d[k] = None
            elif latest_value:
                d[k] = self.data[k][-1]
            else:
                d[k] = self.data[k].copy()
        return d

    def get(self, k: str, latest_value: bool = True) -> Any:
        """Get a single value from the log data.

        Args:
            k (str): The key to retrieve
            latest_value (bool, optional): If True, returns only the latest value. Defaults to True.

        Returns:
            Any: The value associated with the key
        """
        return self.get_keys(k, latest_value=latest_value)[k]

    def power_summary(self, local_time: bool = False) -> tuple:
        """Get power measurements from the log.

        Args:
            local_time (bool, optional): If True, uses local time for power measurements. Defaults to False.

        Returns:
            tuple: List of power measurements in watts
        """
        if not self.log_paths.spl.exists():
            return tuple()

        power_times = self.get_keys("power_begin", "power_end")
        power_begin = timestamp_to_dt(power_times["power_begin"], local_time=local_time)
        power_end = timestamp_to_dt(power_times["power_end"], local_time=local_time)

        with self.log_paths.spl.open(mode='r') as f:
            lines = f.read().strip().split('\n')

        power_vals = []
        for line in lines:
            data = line.split(",")
            if len(data) < 4:
                continue

            timestamp = data[1]
            watts = float(data[3])
            curr_time = timestamp_to_dt(timestamp)

            if power_begin <= curr_time and curr_time <= power_end:
                power_vals.append(watts)

        if not local_time and len(power_vals) == 0:
            logging.warning("No power samples found. Re-parsing with timezone shift.")
            return self.power_summary(local_time=True)
        return power_vals

    def avg_power(self) -> float:
        """Calculate the average power consumption.

        Returns:
            float: Average power in watts, or None if no measurements found
        """
        L = self.power_summary()
        if len(L) == 0:
            return None
        return sum(L) / len(L)

    def result_summary(self, strict_match: bool = True) -> tuple[float, str]:
        """Get a summary of the results including performance metrics.

        Args:
            strict_match (bool, optional): If True, raises an error for MaxQ runs without power measurements. Defaults to True.

        Returns:
            tuple[float, str]: A tuple containing the performance value and metric name

        Raises:
            RuntimeError: If strict_match is True and MaxQ run has no power measurements
            NotImplementedError: If conversion between scenarios is not supported
        """
        convert = (self.found_scenario != self.scenario)
        if convert:
            logging.info("Converting perf metric from %s to %s", self.found_scenario, self.scenario)

        v = self.perf_value
        if v != 0.0 and convert:
            if self.scenario == C.Scenario.Offline and self.found_scenario == C.Scenario.SingleStream:
                logging.info("Converting SingleStream latency -> Offline QPS")
                v = 1 / (v / (10 ** 9))
            elif self.scenario == C.Scenario.Interactive and self.found_scenario == C.Scenario.Server:
                pass
            else:
                raise NotImplementedError(f"Cannot convert {self.found_scenario} result to {self.scenario}")

        if self.power_setting == C.PowerSetting.MaxP:
            return v, self.perf_metric
        else:
            avg_power = self.avg_power()
            if avg_power is None:
                # User is running MaxQ config, but not measuring power.
                if strict_match:
                    raise RuntimeError(f"{self.log_paths.base} is MaxQ run, but has no power measurements.")
                return v, result_key(self.benchmark, self.scenario)
            elif self.scenario in [C.Scenario.Offline, C.Scenario.Server]:
                return safe_divide(v, avg_power), self.perf_metric
            else:
                return (v / (10 ** 9)) * avg_power, self.perf_metric


HelpInfo.add_configurator_dependency(LoadgenLogReader, LoadgenLogSettings)
