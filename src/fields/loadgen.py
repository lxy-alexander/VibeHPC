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
from nvmitten.configurator import Field

import pathlib


__doc__ = """Loadgen control flags

Used to generate Loadgen mlperf.conf and user.conf files.
"""


mlperf_conf_path = Field(
    "mlperf_conf_path",
    description="Path to mlperf.conf",
    from_string=pathlib.Path)

user_conf_path = Field(
    "user_conf_path",
    description="Path to user.conf",
    from_string=pathlib.Path)

test_mode = Field(
    "test_mode",
    description="Testing mode for Loadgen",
    argparse_opts={
        "choices": ["SubmissionRun", "AccuracyOnly", "PerformanceOnly", "FindPeakPerformance"]})

min_duration = Field(
    "min_duration",
    description="Minimum test duration (ms)",
    from_string=int)

max_duration = Field(
    "max_duration",
    description="Maximum test duration (ms)",
    from_string=int)

min_query_count = Field(
    "min_query_count",
    description="Minimum number of queries in test",
    from_string=int)

max_query_count = Field(
    "max_query_count",
    description="Maximum number of queries in test",
    from_string=int)

qsl_rng_seed = Field(
    "qsl_rng_seed",
    description="Seed for RNG that specifies which QSL samples are chosen for performance set and the order in which samples are processed in AccuracyOnly mode",
    from_string=int)

sample_index_rng_seed = Field(
    "sample_index_rng_seed",
    description="Seed for RNG that specifies order in which samples from performance set are included in queries",
    from_string=int)

logfile_suffix = Field(
    "logfile_suffix",
    description="Specify the filename suffix for the LoadGen log files")

logfile_prefix_with_datetime = Field(
    "logfile_prefix_with_datetime",
    description="Prefix filenames for LoadGen log files",
    from_string=bool)

log_copy_detail_to_stdout = Field(
    "log_copy_detail_to_stdout",
    description="Copy LoadGen detailed logging to stdout",
    from_string=bool)

disable_log_copy_summary_to_stdout = Field(
    "disable_log_copy_summary_to_stdout",
    description="Disable copy LoadGen summary logging to stdout",
    from_string=bool)

log_mode = Field(
    "log_mode",
    description="Logging mode for Loadgen",
    argparse_opts={"choices": ["AsyncPoll", "EndOfTestOnly", "Synchronous"]})

log_mode_async_poll_interval_ms = Field(
    "log_mode_async_poll_interval_ms",
    description="Specify the poll interval for asynchrounous logging",
    from_string=int)

log_enable_trace = Field(
    "log_enable_trace",
    description="Enable trace logging",
    from_string=bool)

performance_sample_count = Field(
    "performance_sample_count",
    description="Number of samples to load in performance set. 0=use default",
    from_string=int)

performance_sample_count_override = Field(
    "performance_sample_count_override",
    description="Number of samples to load in performance set; overriding performance_sample_count. 0=don't override",
    from_string=int)

accuracy_sample_count_override = Field(
    "accuracy_sample_count_override",
    description="Number of samples to load in accuracy set; overriding performance_sample_count. 0=don't override",
    from_string=int)

server_target_qps_adj_factor = Field(
    "server_target_qps_adj_factor",
    description="Adjustment factor for target QPS used for server scenario",
    from_string=float)

server_target_qps = Field(
    "server_target_qps",
    description="Target QPS used for server scenario",
    from_string=float)

server_target_latency_ns = Field(
    "server_target_latency_ns",
    description="Desired latency constraint for server scenario",
    from_string=int)

server_target_latency_percentile = Field(
    "server_target_latency_percentile",
    description="Desired latency percentile constraint for server scenario",
    from_string=float)

schedule_rng_seed = Field(
    "schedule_rng_seed",
    description="Seed for RNG that affects the poisson arrival process in server scenario",
    from_string=int)

accuracy_log_rng_seed = Field(
    "accuracy_log_rng_seed",
    description="Affects which samples have their query returns logged to the accuracy log in performance mode.",
    from_string=int)

single_stream_expected_latency_ns = Field(
    "single_stream_expected_latency_ns",
    description="Inverse of desired target QPS",
    from_string=int)

single_stream_target_latency_percentile = Field(
    "single_stream_target_latency_percentile",
    description="Desired latency percentile constraint for single stream scenario",
    from_string=float)

multi_stream_expected_latency_ns = Field(
    "multi_stream_expected_latency_ns",
    description="Expected latency to process a query with multiple Samples, in nanoseconds",
    from_string=int)

multi_stream_target_latency_percentile = Field(
    "multi_stream_target_latency_percentile",
    description="Desired latency percentile to report as a performance metric, for multi stream scenario",
    from_string=float)

multi_stream_samples_per_query = Field(
    "multi_stream_samples_per_query",
    description="Number of samples bundled together as a single query (default: 8)",
    from_string=int)

offline_expected_qps = Field(
    "offline_expected_qps",
    description="Target samples per second rate for the SUT (Offline mode)",
    from_string=float)
