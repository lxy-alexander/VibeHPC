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

from __future__ import annotations
from functools import wraps
import inspect
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Type
import warnings

import matplotlib.pyplot as plt
from tqdm import tqdm

from code.common import logging
from code.common.utils import FastPercentileHeap, ThroughputTracker
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)


class PrefixLogger:
    """
    A wrapper for the logger to add a prefix to log messages.

    Attributes:
        logger (logging.Logger): The original logger.
        prefix (str): The prefix to add to log messages.
    """

    def __init__(self,
                 prefix: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the PrefixLogger.

        Args:
            logger (logging.Logger): The original logger.
            prefix (str): The prefix to add to log messages.
        """
        self.prefix = prefix
        if logger is None:
            # Use a shared logger instance for all PrefixLogger instances
            self.logger = logging.getLogger("__prefix_logger__")

            # Only configure if not already configured
            if not self.logger.handlers:
                # Remove any existing handlers to avoid duplicate logs
                self.logger.propagate = False

                # Create handler with custom formatter
                handler = logging.StreamHandler()
                formatter = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)

                # Set the level based on VERBOSE env var
                level = logging.DEBUG if os.environ.get("VERBOSE", '0') != '0' else logging.INFO
                self.logger.setLevel(level)
        else:
            self.logger = logger

    @staticmethod
    def _get_callsite(n_fbacks: int = 3) -> tuple[str, int, str | None, str]:
        """
        Returns a tuple of (filename, line_no, class_name, func_name) of the caller.
        Args:
            n_fbacks (int): Number of frames to look back in the call stack to find the caller's class.
                           Must be non-negative.

        Returns:
            tuple: A tuple containing the filename, line number, class name (or None if not in a class),
                  and function name of the caller.

        Raises:
            ValueError: If n_fbacks is negative.
        """
        if n_fbacks < 0:
            raise ValueError("n_fbacks must be non-negative")

        frame = inspect.currentframe()
        for _ in range(n_fbacks):
            frame = frame.f_back

        if frame is None:
            raise RuntimeError(f"Cannot get callsite from {n_fbacks} frames back")
        file_path = Path(frame.f_code.co_filename)
        filename = file_path.name

        # If filename is __init__.py, include parent directory
        if filename == "__init__.py" and file_path.parent.name:
            filename = f"{file_path.parent.name}/{filename}"

        line_no = frame.f_lineno
        func_name = frame.f_code.co_name

        # Get the 'self' argument from the frame's locals
        # we supress runtime warnings on inspecting coroutines
        self_arg = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if 'self' in frame.f_locals:
                potential_self = frame.f_locals['self']
                if not inspect.iscoroutine(potential_self):
                    self_arg = potential_self

        if self_arg is not None:
            class_name = self_arg.__class__.__name__
        else:
            class_name = None
        return filename, line_no, class_name, func_name

    def get_prefix(self):
        filename, line_no, class_name, func_name = PrefixLogger._get_callsite()
        if class_name:
            s = f"{filename}:{line_no} {class_name}.{func_name}"
        else:
            s = f"{filename}:{line_no} {func_name}"

        if self.prefix:
            s = f"{self.prefix} {s}"
        return f"[ {s} ] "

    def __getattr__(self, name: str):
        """Wraps logging methods to add a prefix to the log message."""
        attr = getattr(self.logger, name)
        if callable(attr):
            sig = inspect.signature(attr)
            # Kind of hacky but all logging methods have this signature
            if list(sig.parameters.keys()) == ['msg', 'args', 'kwargs']:
                @wraps(attr)
                def wrapper(msg: str, *args, **kwargs):
                    prefix = self.get_prefix()
                    return attr(prefix + msg, *args, **kwargs)
                return wrapper
        return attr


prefix_logger = PrefixLogger()


class LLMServerProgressDisplay:
    """
    A tqdm wrapper to display llm-server progress.

    Methods:
        update(completed=1):
            Updates the progress bar by a specified number of completed units.

        update(total=1):
            Updates the progress bar total

        finish():
            Closes the progress bar.
    """

    def __init__(
        self,
        total: int,
        desc: str = "Processing",
        unit: str = "samples",
        enable_render: bool = True,
        additional_units: Dict[str, str] = {},
        log_dir: os.Pathlike = None
    ):
        """
        Initializes the LLMServerProgressDisplay instance.

        Args:
            total (int): The total number of units to process.
            desc (str): The prefix-description to display with the progress bar.
            unit (str): The unit of measurement for the progress bar.
            enable_render (bool): Flag to enable or disable of progress bar.
            additional_units Dict[str, str]: Additional units to track, name: type (where type=mean|value|99%|throughput_tracker).
            log_dir (Optional[os.Pathlike]): Directory to store log files.
        """
        self.enable_render = enable_render
        self.total = total
        self.desc = desc
        self.unit = unit
        self.log_dir = log_dir
        assert self.log_dir is not None

        self.completed = 0

        self.state_history = {}
        self.iteration_stats = []
        self.progress_bar = None  # lazy init in update_total

        self.additional_units_specs = additional_units
        self.additional_units_values = {}
        for unit, _type in additional_units.items():
            match _type:
                case 'value' | 'mean':
                    self.additional_units_values[unit] = 0
                case '99%':
                    self.additional_units_values[unit] = FastPercentileHeap()
                case 'throughput_tracker':
                    self.additional_units_values[unit] = ThroughputTracker()
                case _:
                    raise ValueError(f"Unknown type({_type}) of additional unit({unit})")

        self.progress_bar_args = {
            'total': self.total,
            'desc': self.desc,
            'unit': self.unit,
            'smoothing': 1,
            'mininterval': 0.34,
            'leave': True,
            'disable': not self.enable_render,
            'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
        }

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.stats_flush_thread = threading.Thread(target=self.stats_periodic_flush)
        self.stats_flush_thread.start()

    def add_additional_unit(self, unit_name: str, unit_type: str):
        """
        Add an additional unit to track after initialization.

        Args:
            unit_name (str): The name of the unit to add.
            unit_type (str): The type of unit (mean|value|99%|throughput_tracker).
        """
        with self.lock:
            if unit_name in self.additional_units_specs:
                logging.debug(f"Unit '{unit_name}' already exists, skipping addition")
                return

            self.additional_units_specs[unit_name] = unit_type

            match unit_type:
                case 'value' | 'mean':
                    self.additional_units_values[unit_name] = 0
                case '99%':
                    self.additional_units_values[unit_name] = FastPercentileHeap()
                case 'throughput_tracker':
                    self.additional_units_values[unit_name] = ThroughputTracker()
                case _:
                    raise ValueError(f"Unknown type({unit_type}) of additional unit({unit_name})")

    def update(self, completed: Optional[int] = None, additional_unit_updates: Dict[str, int] = {}):
        """
        Updates the progress bar by a specified number of completed units.

        Args:
            completed Optional[int]: The number of units completed since the last update. If provided, updates the completed count.
            additional Dict[str, int]: A dictionary of additional units completed since the last update.
        """
        # no updates to render
        if completed and (len(additional_unit_updates) == 0):
            return

        # TODO(vir): ideally we co-schedule this with main thread using priority / move this to another thread
        with self.lock:
            completed = completed or 0
            self.completed += completed
            duration = 1

            # lazy init progress bar
            if self.progress_bar is None:
                if self.enable_render:
                    self.progress_bar = tqdm(**self.progress_bar_args)
                    self.progress_bar.n = self.completed

            else:
                self.progress_bar.n = self.completed
                duration = self.progress_bar.format_dict.get('elapsed')

            for unit, update in additional_unit_updates.items():
                match self.additional_units_specs[unit]:
                    case 'mean':
                        self.additional_units_values[unit] += update
                    case '99%':
                        self.additional_units_values[unit].extend(update)
                    case 'value':
                        self.additional_units_values[unit] = update
                    case 'throughput_tracker':
                        self.additional_units_values[unit].append(update, duration, completed)

            # we try-render update if completed > 0 (not on ONLY extra-stat updates)
            if render_update := completed > 0 and \
                    self.progress_bar is not None and \
                    self.progress_bar.update(n=0):  # we already updated N, try-render now
                instant_rate = self.progress_bar.format_dict.get('rate', 0.0)
                mean_rate = float(self.completed / duration)

                additional_values = {}
                for unit, _type in self.additional_units_specs.items():
                    match _type:
                        case 'mean':
                            additional_values[unit] = float(self.additional_units_values[unit] / duration)
                        case '99%':
                            additional_values[unit] = self.additional_units_values[unit].p()
                        case 'value':
                            additional_values[unit] = float(self.additional_units_values[unit])
                        case 'throughput_tracker':
                            throughput = self.additional_units_values[unit].get_throughput()
                            mean = throughput['mean']
                            margin = throughput['margin']

                            if mean == 0.0:
                                display_str = 'cold-start'
                            else:
                                display_str = f"{mean:.1f}±{margin:.1f}"

                            additional_values[unit] = display_str

                samples_fmt = f'{instant_rate:.2f}{self.unit}/s] Stats=[ {mean_rate:.2f} {self.unit}/s'
                postfix_parts = [f', {value:.2f} {unit}' for unit, value in additional_values.items() if value is not None]
                postfix_fmt = ''.join(postfix_parts)

                self.progress_bar.set_postfix_str(f'{samples_fmt}{postfix_fmt} ', refresh=True)
                self.state_history[self.completed] = additional_values

    def update_total(self, total: int):
        """
        Updates the total number of units to process and refreshes the progress bar.
        Assumes lock is held by the caller.

        Args:
            total (int): The new total number of units to process.

        Returns:
            bool: True if the total was updated, False if it was already set to this value.
        """
        self.total = total
        self.progress_bar_args['total'] = total

        if self.enable_render:
            # very short wait to try update total
            # we re-try later if we fail to acquire the lock
            if self.lock.acquire(timeout=0.0001):
                if self.progress_bar is None:
                    self.progress_bar = tqdm(**self.progress_bar_args)
                    self.progress_bar.n = self.completed
                    self.progress_bar.total = total

                else:
                    self.progress_bar.total = total

                self.lock.release()
                return True
        else:
            if self.total % 10_000 == 0:
                logging.info(f"Status: {self.completed}/{self.total} {self.unit} processed")
            return False

    def record_iteration_stats(self, stats: List[Dict]):
        """
        Records the iteration statistics.
        Saved to self.log_dir on finish.

        Args:
            stats (List[Dict]): A list of dictionaries containing iteration statistics.
        """
        with self.lock:
            self.iteration_stats.extend(stats)

    def finish(self):
        """
        Completes and freezes the progress bar.
        """
        if self.progress_bar is not None and self.enable_render:
            self.progress_bar.close()

        self.stop_event.set()
        self.stats_flush_thread.join()
        self.enable_render = False

        if len(self.state_history) > 0:
            stats_dump = Path(self.log_dir) / "harness_stats.log"
            stats_plot = Path(self.log_dir) / "harness_stats_timeline.png"

            # dump harness stats log
            stats_dump.write_text(json.dumps(self.state_history, separators=(', ', ':')))
            logging.info(f"Harness stats saved to: {stats_dump}")

            completed = list(self.state_history.keys())
            additional_stats_keys = self.additional_units_specs.keys()

            fig, axs = plt.subplots(len(additional_stats_keys), 1, figsize=(10, 5 * len(additional_stats_keys)), squeeze=False)

            for i, key in enumerate(additional_stats_keys):
                type_str = self.additional_units_specs[key]
                values = [self.state_history[n][key] for n in completed]

                if key == 'steady_state_tokens/s':
                    means = [v['mean'] for v in values]
                    margins = [v['margin'] for v in values]
                    axs[i, 0].plot(completed, means, label=f'{key} (mean)')
                    axs[i, 0].fill_between(completed,
                                           [m - margin for m, margin in zip(means, margins)],
                                           [m + margin for m, margin in zip(means, margins)],
                                           alpha=0.2, label=f'{key} (margin)')
                else:
                    axs[i, 0].plot(completed, values, label=key)

                axs[i, 0].set_title(f'[{type_str}] {key}')
                axs[i, 0].set_xlabel('Completed')
                axs[i, 0].set_ylabel(key)
                axs[i, 0].legend()

            # plot and dump harness stats over-time
            plt.tight_layout()
            plt.savefig(stats_plot)
            logging.info(f"Harness timeline plot generated to: {stats_plot}")

            plt.close()

    def stats_periodic_flush(self):
        """
        Periodically flushes accumulated iteration stats to log file.
        """
        stats_file = Path(self.log_dir) / 'harness_iteration_stats.log'
        logging.info(f"Harness iteration stats will be dumped to: {stats_file}")

        with open(stats_file, 'w') as f:
            while not self.stop_event.is_set():
                with self.lock:
                    stats = [json.dumps(stats, separators=(', ', ':')) + '\n' for stats in self.iteration_stats]
                    self.iteration_stats.clear()

                f.writelines(stats)
                f.flush()
                time.sleep(1)

        logging.info(f"Harness iteration stats dumped to: {stats_file}")


class LatencyTracker:
    """
    A thread-safe tracker for latency metrics of loadgen requests.

    This class tracks Input Sequence Length (ISL) and Time To First Token (TTFT)
    metrics for loadgen requests. It provides thread-safe operations for adding
    samples and TTFT measurements, and can export the collected data to CSV format.

    Methods:
        add_sample(loadgen_request_id, ISL):
            Adds a new sample with the given request ID and Input Sequence Length.

        add_TTFT(loadgen_request_id, TTFT):
            Adds Time To First Token measurement for an existing request.

        gen_csv(output_path):
            Generates a CSV file with all tracked latency data.
    """

    def __init__(self):
        self.latency_tracker = {}
        self.lock = threading.Lock()

    def add_sample(self, loadgen_request_id, ISL):
        with self.lock:
            if loadgen_request_id not in self.latency_tracker:
                logging.debug(f"Adding sample for request ID {loadgen_request_id} with ISL {ISL}")
                self.latency_tracker[loadgen_request_id] = {"ISL": ISL, "TTFT": None, }
            else:
                raise ValueError(f"Request ID {loadgen_request_id} already exists in latency tracker, loadgen issued same request ID twice")

    def add_TTFT(self, loadgen_request_id, TTFT):
        # potential perf bottleneck when high token throughput inference is running on backend
        with self.lock:
            try:
                logging.debug(f"Adding TTFT for request ID {loadgen_request_id} with TTFT {TTFT}")
                self.latency_tracker[loadgen_request_id]["TTFT"] = TTFT
            except KeyError:
                raise KeyError(f"Request ID {loadgen_request_id} not found in latency tracker")

    def gen_csv(self, output_path: str):
        with open(output_path, 'w') as f:
            f.write("loadgen_request_id,ISL,TTFT\n")
            for loadgen_request_id, data in self.latency_tracker.items():
                f.write(f"{loadgen_request_id},{data['ISL']},{data['TTFT']}\n")


def get_yaml_string(config_dict, indent=0):
    """
    Convert config dictionary to YAML string manually with type preservation.

    This function recursively converts a configuration dictionary to a YAML-formatted
    string while preserving data types. It automatically removes None values and
    formats different data types appropriately for YAML output.

    Args:
        config_dict (dict): Configuration dictionary to convert
        indent (int): Current indentation level (used internally for recursion)

    Returns:
        str: YAML formatted string

    Examples:
        >>> config = {
        ...     'name': 'test_config',
        ...     'enabled': True,
        ...     'count': 42,
        ...     'ratio': 3.14,
        ...     'tags': ['gpu', 'llm', 123, 4.5, True],
        ...     'settings': {
        ...         'batch_size': 16,
        ...         'temperature': 0.8,
        ...         'enabled_features': [1, 2, 3]
        ...     },
        ...     'disabled_option': None  # This will be removed
        ... }
        >>> print(get_yaml_string(config))
        name: test_config
        enabled: true
        count: 42
        ratio: 3.14
        tags: [gpu, llm, 123, 4.5, true]
        settings:
          batch_size: 16
          temperature: 0.8
          enabled_features: [1, 2, 3]

    Type Handling:
        - bool: Converted to lowercase (true/false)
        - int/float: Preserved as numbers without quotes
        - str: Output as-is
        - list: Formatted as inline arrays with type preservation
        - dict: Recursively formatted with proper indentation
        - None: Automatically removed from output
    """
    def remove_none_values(d):
        # remove None values (recursively)
        # if all values in a dict are None, remove the dict
        match d:
            case dict():
                if all(value is None for value in d.values()):
                    return None
                return {k: remove_none_values(v) for k, v in d.items() if v is not None}
            case list():
                return [remove_none_values(v) for v in d if v is not None]
            case _:
                return d

    def format_yaml_value(item):
        """Format individual values for YAML output, preserving types."""
        match item:
            case bool():
                return str(item).lower()
            case int() | float():
                return str(item)
            case str():
                return item
            case _:
                return str(item)

    config_dict = remove_none_values(config_dict)

    yaml_lines = []
    indent_str = '  ' * indent  # 2 spaces per indent level

    for key, value in config_dict.items():
        match value:
            case bool():
                # Convert True/False to lowercase
                yaml_lines.append(f"{indent_str}{key}: {str(value).lower()}")
            case int() | float():
                # Pass ints and floats as-is
                yaml_lines.append(f"{indent_str}{key}: {value}")
            case str():
                yaml_lines.append(f"{indent_str}{key}: {value}")
            case list():
                # Format lists on single line, preserving item types
                list_items = [format_yaml_value(item) for item in value]
                list_str = '[' + ', '.join(list_items) + ']'
                yaml_lines.append(f"{indent_str}{key}: {list_str}")
            case dict():
                # Handle nested dictionaries
                yaml_lines.append(f"{indent_str}{key}:")
                nested_yaml = get_yaml_string(value, indent + 1)
                # Remove trailing newline and split into individual lines
                nested_lines = nested_yaml.rstrip('\n').split('\n') if nested_yaml.strip() else []
                yaml_lines.extend(nested_lines)
            case None:
                # Skip None values (they should have been filtered out already)
                continue
            case _:
                # Default string representation
                yaml_lines.append(f"{indent_str}{key}: {value}")

    return '\n'.join(yaml_lines) + '\n' if yaml_lines else ''


# Re-export LazyImport from lazy_import.py for backward compatibility
from .lazy_import import LazyImport
