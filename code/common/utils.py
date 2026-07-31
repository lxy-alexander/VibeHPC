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

# Standard library imports
import contextlib
import dataclasses as dcls
import functools
import heapq
import os
import platform
import random
import re
import shutil
import sys
import venv
import importlib
from math import gcd, sqrt
from numbers import Number
from typing import Any, Dict, List, Optional, Union

# Local imports
from code.common import logging, run_command
from code.fields import general as general_fields

# Third-party imports
import numpy as np
from nvmitten.configurator import autoconfigure, bind
import nvtx
from scipy import stats, optimize
import torch


def safe_divide(numerator: Number, denominator: Number) -> Optional[float]:
    """
    Divides 2 numbers, returning None if a DivisionByZero were to occur.

    Args:
        numerator (Number):
            Value for the numerator
        denominator (Number):
            Value for the denominator

    Returns:
        Optional[float]: numerator/denominator. None if denominator is 0.
    """
    if float(denominator) == 0:
        return None
    return float(numerator) / float(denominator)


def get_dyn_ranges(cache_file: str) -> Dict[str, np.uint32]:
    """
    Get dynamic ranges from calibration file for network tensors.

    Args:
        cache_file (str):
            Path to INT8 calibration cache file.

    Returns:
        Dict[str, np.uint32]: Dictionary of tensor name -> dynamic range of tensor
    """
    dyn_ranges = {}
    if not os.path.exists(cache_file):
        raise FileNotFoundError("{} calibration file is not found.".format(cache_file))

    with open(cache_file, "rb") as f:
        lines = f.read().decode('ascii').splitlines()
    for line in lines:
        regex = r"(.+): (\w+)"
        results = re.findall(regex, line)
        # Omit unmatched lines
        if len(results) == 0 or len(results[0]) != 2:
            continue
        results = results[0]
        tensor_name = results[0]
        # Map dynamic range from [0.0 - 1.0] to [0.0 - 127.0]
        dynamic_range = np.uint32(int(results[1], base=16)).view(np.dtype('float32')).item() * 127.0
        dyn_ranges[tensor_name] = dynamic_range
    return dyn_ranges


def load_tensor(fpath: os.PathLike, pin_memory: bool = True):
    assert os.path.exists(fpath), f"Cannot load tensor from {fpath} because it does not exist"

    ret_val = np.load(fpath)
    ret_val = ret_val if not pin_memory else torch.tensor(ret_val).pin_memory()
    logging.debug("Loaded tensor to %s memory: %s (dtype=%s, shape=%s)",
                  "pinned" if pin_memory else "non-pinned",
                  str(fpath),
                  str(ret_val.dtype),
                  str(ret_val.shape))
    return ret_val


def get_e2e_batch_size(batch_size_dict: Dict[str, int]):
    """
    Calculate the end-to-end batch size from a dictionary of batch sizes.

    Args:
        batch_size_dict (Dict[str, int]): Dictionary containing batch sizes for different components.

    Returns:
        int: The least common multiple of all batch sizes.

    Raises:
        ValueError: If the input dictionary is empty or if the calculated LCM is not equal to any of the max engine batch sizes.
    """
    if not batch_size_dict:
        raise ValueError(f"{batch_size_dict} input batch_size_dict is empty.")

    batch_size_list = list(batch_size_dict.values())
    lcm_value = 1
    for val in batch_size_list:
        lcm_value = (lcm_value * val) // gcd(lcm_value, val)

    if lcm_value != max(batch_size_list):
        raise ValueError(f"End-to-end batch size {lcm_value} is not equal to any of the max engine batch size in {batch_size_dict}")

    return lcm_value


def check_eq(val1, val2, error_message="Values are not equal"):
    """
    Check if two values are equal and raise an AssertionError if they are not.

    Args:
        val1: First value to compare
        val2: Second value to compare
        error_message (str, optional): Custom error message to display if values are not equal

    Raises:
        AssertionError: If val1 is not equal to val2
    """
    if val1 != val2:
        raise AssertionError(f"{error_message} ({val1} vs. {val2})")


@autoconfigure
@bind(general_fields.verbose_nvtx)
# pylint: disable=invalid-name
class nvtx_scope:
    def __init__(self,
                 name: str,
                 color: Optional[str] = None,
                 verbose_nvtx: bool = False):
        self.name = name
        # Use hash-based color selection for consistency across runs
        available_colors = ["green", "blue", "yellow", "pink"]
        self.color = color if color else available_colors[hash(name) % len(available_colors)]
        self.enable = verbose_nvtx

        self.markers = None

    def __enter__(self):
        if self.enable:
            assert self.markers is None, "NVTX range already started"
            self.markers = nvtx.start_range(message=self.name, color=self.color)
        return self.markers

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enable:
            assert self.markers is not None, "NVTX range not started"
            nvtx.end_range(self.markers)
            self.markers = None


def with_nvtx_scope(name: str,
                    color: Optional[str] = None):
    """
    Decorator to add NVTX profiling ranges to functions.

    Args:
        name (str): Name of the NVTX range
        color (Optional[str]): Color for the NVTX range. If None, a random color is chosen.

    Returns:
        Callable: Decorated function with NVTX profiling
    """
    def decorator(f, _name, _color):
        @functools.wraps(f)
        def _wrapped_fn(*args, **kwargs):
            with nvtx_scope(_name, _color):
                r = f(*args, **kwargs)
            return r
        return _wrapped_fn
    return functools.partial(decorator, _name=name, _color=color)


class FastPercentileHeap:
    """
    A datastructure to maintain a running percentile of a stream of numbers using two heaps.

    Advantages:
        - fast calculation of current top self.k percentile
        - fast insertion of new values to track

    Attributes:
        percentile (float): The desired percentile (default is 99).
        small (list): Max heap to store the smaller half of the numbers (stored as negative values).
        large (list): Min heap to store the larger half of the numbers.
        n (int): The total number of elements added.
        k (float): The desired percentile as a fraction.
    """

    def __init__(self, percentile=99):
        """
        Initializes the FastPercentileHeap with the given percentile.

        Args:
            percentile (float): The desired percentile (default is 99).
        """
        self.small = []  # max heap (-ve values)
        self.large = []  # min heap

        self.n = 0
        self.k = 1.0 - (percentile / 100.0)

    def extend(self, nums: List[int]):
        """
        Extends the heap with a list of numbers.

        Args:
            nums (List[int]): The list of numbers to be added to the heap.
        """
        for num in nums:
            self.append(num)

    def append(self, num):
        """
        Appends a number to the heap in O(logN) time.

        Args:
            num (float): The number to be added to the heap.
        """
        self.n += 1
        target_large_size = max(1, int(self.n * self.k))

        if len(self.large) < target_large_size:
            heapq.heappush(self.large, num)
        else:
            heapq.heappush(self.small, -num)

        self._rebalance()

    def _rebalance(self):
        """
        Rebalances the heaps in O(1) time when the largest element in self.small is larger than the smallest element in self.large.
        """
        if not self.large or not self.small:
            return

        if -self.small[0] > self.large[0]:
            small_val = -heapq.heappop(self.small)
            large_val = heapq.heappop(self.large)
            heapq.heappush(self.small, -large_val)
            heapq.heappush(self.large, small_val)

    def p(self):
        """
        Calculates the top self.k percentile in O(1) time.

        Returns:
            float: The top self.k percentile value, or None if the heap is empty.
        """
        return self.large[0] if self.large else None


class ThroughputTracker:
    """
    A data structure to maintain throughput measurements in chunks of fixed time duration.
    Helps to estimate the steady-state throughput of a workload.

    Uses curve fitting to detect cold-start vs steady-state based on exponential model:
    y = y_inf * (1 - exp(-x/tau))

    Attributes:
        time_window (float): The duration in seconds for each chunk (default 30 seconds).
        current_chunk (list): Current chunk being filled with (timestamp, token_count) tuples.
        state (dict): Dictionary containing tracking state:
            - start_time: Start time of current chunk
            - total_samples: Total samples processed
            - cold_start: Whether in cold-start phase
            - confirmations: Number of cold-start end confirmations
            - steady_state_start_index: Index where steady-state begins in throughput list
        chunks (dict): Dictionary containing chunk data:
            - throughputs: List of throughput values from all chunks
            - sample_counts: List of sample counts for each chunk
    """

    def __init__(self, time_window=30.0):
        """
        Initializes the ThroughputTracker with the given time window.

        Args:
            time_window (float): Duration in seconds for each chunk (default 30 seconds).
        """
        self.time_window = time_window
        self.current_chunk = []
        self.state = {
            'start_time': None,
            'total_samples': 0,
            'cold_start': True,
            'confirmations': 0,
            'steady_state_start_index': None
        }
        self.chunks = {
            'throughputs': [],
            'sample_counts': []
        }

    def _fit_cold_start_curve(self):
        """
        Fits the exponential curve y = y_inf * (1 - exp(-x/tau)) to cold-start data.
        Returns None if not enough data points for fitting.
        """
        if len(self.chunks['throughputs']) < 4:  # Need at least 4 points for meaningful fit
            return None

        x_data = np.cumsum(self.chunks['sample_counts'])
        y_data = np.array(self.chunks['throughputs'])

        def exp_func(x, y_inf, tau):
            return y_inf * (1 - np.exp(-x / tau))

        try:
            popt, _ = optimize.curve_fit(exp_func, x_data, y_data, p0=[max(y_data), 1.0])
            return popt
        except (RuntimeError, ValueError):
            return None

    def append(self, num_tokens: float, duration: float, num_samples: int = 1):
        """
        Appends a token count to the tracker with specified duration timestamp.

        Args:
            num_tokens (float): The number of tokens to add.
            duration (float): Current elapsed time in seconds since tracking started.
            num_samples (int): The number of samples this update represents (default 1).
        """
        if num_tokens <= 0 or num_samples <= 0:
            return

        if self.state['start_time'] is None:
            self.state['start_time'] = duration

        self.current_chunk.append((duration, num_tokens))
        self.state['total_samples'] += num_samples

        if duration - self.state['start_time'] >= self.time_window:
            self._complete_current_chunk()

    def _complete_current_chunk(self):
        """
        Completes the current chunk and calculates its throughput.
        During cold-start, uses curve fitting to determine when to transition to steady-state.
        """
        if len(self.current_chunk) < 2:
            self.current_chunk = []
            self.state['start_time'] = None
            return

        timestamps = [sample[0] for sample in self.current_chunk]
        token_counts = [sample[1] for sample in self.current_chunk]

        total_tokens = sum(token_counts)
        time_span = timestamps[-1] - timestamps[0]

        if time_span > 0:
            chunk_throughput = total_tokens / time_span
            self.chunks['throughputs'].append(chunk_throughput)
            self.chunks['sample_counts'].append(len(self.current_chunk))

            if self.state['cold_start']:
                fit_params = self._fit_cold_start_curve()
                if fit_params is not None:
                    _, tau = fit_params
                    if self.state['total_samples'] >= 3 * tau:
                        self.state['confirmations'] += 1
                        if self.state['confirmations'] >= 3:
                            self.state['cold_start'] = False
                            self.state['steady_state_start_index'] = len(self.chunks['throughputs']) - 1
                    else:
                        self.state['confirmations'] = 0

        self.current_chunk = []
        self.state['start_time'] = None

    def get_throughput(self, confidence_level=0.95):
        """
        Calculates throughput with confidence interval based on steady-state chunks.

        Args:
            confidence_level (float): Confidence level for the interval (default 0.95).

        Returns:
            dict: Dictionary containing 'mean', 'margin'.
        """
        if self.state['steady_state_start_index'] is None or len(self.chunks['throughputs']) - self.state['steady_state_start_index'] < 2:
            return {
                'mean': 0.0,
                'margin': 0.0,
            }

        steady_state_throughputs = self.chunks['throughputs'][self.state['steady_state_start_index']:]
        mean_throughput = sum(steady_state_throughputs) / len(steady_state_throughputs)
        variance = sum((x - mean_throughput) ** 2 for x in steady_state_throughputs) / (len(steady_state_throughputs) - 1)
        std_dev = sqrt(variance)
        std_error = std_dev / sqrt(len(steady_state_throughputs))

        alpha = 1 - confidence_level
        z_critical = stats.norm.ppf(1 - alpha / 2)
        margin_of_error = z_critical * std_error

        return {
            'mean': mean_throughput,
            'margin': margin_of_error,
        }


class JSONSubsetDataclass(object):
    """
    Provides a from_json method, which allows a dataclass to be constructed from a JSON file which may contain extra
    or missing fields of that dataset.

    This class enables partial loading of JSON data into dataclasses, where missing fields are marked as MISSING
    and extra fields in the JSON are ignored.
    """

    @classmethod
    def from_json(cls, json_dict) -> JSONSubsetDataclass:
        """
        Constructs a dataclass instance from a JSON dictionary, handling missing and extra fields.

        Args:
            json_dict: Dictionary containing JSON data

        Returns:
            JSONSubsetDataclass: Instance of the dataclass with fields populated from the JSON data
        """
        kwargs = dict()
        for f in dcls.fields(cls):
            if f.name not in json_dict:
                kwargs[f.name] = dcls.MISSING
            elif issubclass(f.type, JSONSubsetDataclass):
                kwargs[f.name] = f.type.from_json(json_dict[f.name])
            else:
                kwargs[f.name] = json_dict[f.name]
        return cls(**kwargs)

    def __getattribute__(self, name):
        """
        Overrides attribute access to raise an AttributeError for fields marked as MISSING.

        Args:
            name: Name of the attribute to access

        Returns:
            The value of the attribute if it exists and is not MISSING

        Raises:
            AttributeError: If the attribute was not loaded from JSON and is marked as MISSING
        """
        r = object.__getattribute__(self, name)
        if r is dcls.MISSING:
            raise AttributeError(f"{name} was not loaded from JSON and is removed as an attribute.")
        return r


def parse_kv_string(flags: Optional[Union[str, dict]] = None) -> Dict[str, Any]:
    """
    Parses a string or dictionary into a dictionary with string keys and values of appropriate types.

    Args:
        flags (Optional[Union[str, dict]]): A string in the format 'key1:value1,key2:value2,...' or a dictionary.

    Returns:
        Dict[str, Any]: A dictionary with string keys and values of appropriate types (bool, int, float, or str).
    """
    if flags is None or flags == "":
        return {}

    if isinstance(flags, dict):
        return flags

    flag_dict = {}
    for item in flags.split(','):
        key, value = item.split(':')
        if value.lower() in ['true', 'false']:
            value = value.lower() == 'true'
        else:
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass
        flag_dict[key] = value
    return flag_dict


def dry_runnable(f):
    """Makes a function "dry-runnable". Adds a new keyword argument 'dry_run: bool = False' to the function.

    If dry_run is set to True, a message is printed instead of running the function.
    """
    def _f(*args, dry_run=False, **kwargs):
        if not dry_run:
            return f(*args, **kwargs)
        else:
            param_str = ", ".join([f"{type(arg).__name__}: {arg}" for arg in args] +
                                  [f"{k}: {type(v).__name__} = {v}" for k, v in kwargs.items()])
            print(f"dry run> {f.__name__}({param_str})")
    return _f


@dry_runnable
def safe_copy(input_file, output_file):
    logging.info(f"Copy {input_file} -> {output_file}")
    try:
        shutil.copy(input_file, output_file)
    except Exception as e:
        logging.error(f"Copy failed. Error: {e}")


@dry_runnable
def safe_copytree(src_dir, dst_dir):
    logging.info(f"Copy {src_dir} -> {dst_dir}")
    try:
        shutil.rmtree(dst_dir, ignore_errors=True)
        shutil.copytree(src_dir, dst_dir)
    except Exception as e:
        logging.error(f"Copytree failed. Error: {e}")


@dry_runnable
def safe_rmtree(dir):
    logging.info(f"Remove {dir}")
    try:
        shutil.rmtree(dir)
    except Exception as e:
        logging.error(f"Rmtree failed. Error: {e}")
