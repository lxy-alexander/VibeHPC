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

"""Qwen3-VL 235B LoadGen fields."""
from datetime import timedelta

from nvmitten.configurator import Field


def _parse_timedelta(value: str):
    """Parse timedelta from seconds (float/int string) or pass through ISO-8601."""
    try:
        return timedelta(seconds=float(value))
    except ValueError:
        return value


test_scenario = Field(
    "q3vl_test_scenario",
    description="The MLPerf inference benchmarking scenario to run.",
    from_string=str,
)

test_mode = Field(
    "q3vl_test_mode",
    description="Whether to run performance measurement or accuracy evaluation.",
    from_string=str,
)

random_seed = Field(
    "q3vl_random_seed",
    description="Random seed for the benchmark run.",
    from_string=int,
)


server_target_latency = Field(
    "q3vl_server_target_latency",
    description="Expected latency constraint for Server scenario.",
    from_string=_parse_timedelta,
)

min_duration = Field(
    "q3vl_min_duration",
    description="Minimum testing duration (seconds or ISO 8601).",
    from_string=_parse_timedelta,
)

max_duration = Field(
    "q3vl_max_duration",
    description="Maximum testing duration (seconds or ISO 8601).",
    from_string=_parse_timedelta,
)

max_concurrency = Field(
    "q3vl_max_concurrency",
    description=(
        "Maximum number of concurrent in-flight requests during benchmarking. "
        "If not set, no concurrency throttling is applied."
    ),
    from_string=lambda x: int(x) if x else None,
)

use_http_client = Field(
    "q3vl_use_http_client",
    description=(
        "Use the aiohttp-based HTTP client with pre-serialized request bodies "
        "instead of the default OpenAI SDK client."
    ),
    from_string=lambda x: x.lower() in ("true", "1", "yes") if x else False,
)





log_output = Field(
    "q3vl_log_output",
    description="Log output settings (serialized or path-based).",
    from_string=str,
)

log_mode = Field(
    "q3vl_log_mode",
    description="How and when logging should be sampled at runtime.",
    from_string=str,
)

enable_trace = Field(
    "q3vl_enable_trace",
    description="Enable trace logging in LoadGen.",
    from_string=lambda x: x.lower() in ("true", "1", "yes"),
)

endpoint_request_timeout = Field(
    "q3vl_endpoint_request_timeout",
    description="Request timeout (seconds or ISO 8601).",
    from_string=_parse_timedelta,
)

endpoint_startup_timeout = Field(
    "q3vl_endpoint_startup_timeout",
    description="Endpoint startup timeout (seconds or ISO 8601).",
    from_string=_parse_timedelta,
)

endpoint_shutdown_timeout = Field(
    "q3vl_endpoint_shutdown_timeout",
    description="Endpoint shutdown timeout (seconds or ISO 8601).",
    from_string=_parse_timedelta,
)

endpoint_poll_interval = Field(
    "q3vl_endpoint_poll_interval",
    description="Endpoint poll interval (seconds or ISO 8601).",
    from_string=_parse_timedelta,
)

endpoint_healthcheck_timeout = Field(
    "q3vl_endpoint_healthcheck_timeout",
    description="Endpoint healthcheck timeout (seconds or ISO 8601).",
    from_string=_parse_timedelta,
)

num_warmup_requests_per_vllm_instance = Field(
    "q3vl_num_warmup_requests_per_vllm_instance",
    description="Warmup requests per vLLM instance.",
    from_string=int,
)

model_repo_id = Field(
    "q3vl_model_repo_id",
    description="HuggingFace repository ID for the model.",
    from_string=str,
)

model_token = Field(
    "q3vl_model_token",
    description="HuggingFace access token for the model repo.",
    from_string=lambda x: x if x else None,
)

model_revision = Field(
    "q3vl_model_revision",
    description="Model revision (commit hash or tag).",
    from_string=str,
)

dataset_repo_id = Field(
    "q3vl_dataset_repo_id",
    description="HuggingFace repository ID for the dataset.",
    from_string=str,
)

dataset_token = Field(
    "q3vl_dataset_token",
    description="HuggingFace access token for the dataset repo.",
    from_string=lambda x: x if x else None,
)

dataset_revision = Field(
    "q3vl_dataset_revision",
    description="Dataset revision (commit hash or tag).",
    from_string=str,
)

dataset_split = Field(
    "q3vl_dataset_split",
    description="Comma-separated dataset splits (e.g., train,test).",
    from_string=lambda x: [item for item in x.split(",") if item] if x else [],
)

etcd_hostname = Field(
    "q3vl_etcd_hostname",
    description="ETCD hostname.",
    from_string=str,
)

etcd_port = Field(
    "q3vl_etcd_port",
    description="ETCD port.",
    from_string=int,
)

etcd_startup_timeout = Field(
    "q3vl_etcd_startup_timeout",
    description="ETCD startup timeout (seconds or ISO 8601).",
    from_string=_parse_timedelta,
)

etcd_healthcheck_timeout = Field(
    "q3vl_etcd_healthcheck_timeout",
    description="ETCD healthcheck timeout (seconds or ISO 8601).",
    from_string=_parse_timedelta,
)

etcd_poll_interval = Field(
    "q3vl_etcd_poll_interval",
    description="ETCD poll interval (seconds or ISO 8601).",
    from_string=_parse_timedelta,
)

nats_hostname = Field(
    "q3vl_nats_hostname",
    description="NATS hostname.",
    from_string=str,
)

nats_port = Field(
    "q3vl_nats_port",
    description="NATS port (client connections).",
    from_string=int,
)

nats_monitoring_port = Field(
    "q3vl_nats_monitoring_port",
    description="NATS monitoring port.",
    from_string=int,
)

nats_startup_timeout = Field(
    "q3vl_nats_startup_timeout",
    description="NATS startup timeout (seconds or ISO 8601).",
    from_string=_parse_timedelta,
)

nats_healthcheck_timeout = Field(
    "q3vl_nats_healthcheck_timeout",
    description="NATS healthcheck timeout (seconds or ISO 8601).",
    from_string=_parse_timedelta,
)

nats_poll_interval = Field(
    "q3vl_nats_poll_interval",
    description="NATS poll interval (seconds or ISO 8601).",
    from_string=_parse_timedelta,
)

vllm_cli = Field(
    "q3vl_vllm_cli",
    description="Comma-separated vLLM CLI flags.",
    from_string=lambda x: [item for item in x.split(",") if item] if x else [],
)

vllm_enable_numa_binding = Field(
    "q3vl_vllm_enable_numa_binding",
    description="Enable NUMA binding for the vLLM backend.",
    from_string=lambda x: x.lower() in ("true", "1", "yes") if x else False,
)

vllm_dyn_log = Field(
    "q3vl_vllm_dyn_log",
    description="Dynamo log level for vLLM/Dynamo processes.",
    from_string=str,
    from_environ="DYN_LOG",
)

vllm_logging_level = Field(
    "q3vl_vllm_logging_level",
    description="Logging level for vLLM.",
    from_string=str,
    from_environ="VLLM_LOGGING_LEVEL",
)

vllm_use_flashinfer_sampler = Field(
    "q3vl_vllm_use_flashinfer_sampler",
    description="Enable FlashInfer sampler in vLLM.",
    from_string=int,
    from_environ="VLLM_USE_FLASHINFER_SAMPLER",
)

vllm_use_flashinfer_moe_fp4 = Field(
    "q3vl_vllm_use_flashinfer_moe_fp4",
    description="Enable FlashInfer MoE FP4 in vLLM.",
    from_string=int,
    from_environ="VLLM_USE_FLASHINFER_MOE_FP4",
)

vllm_use_triton_pos_embed = Field(
    "q3vl_vllm_use_triton_pos_embed",
    description="Enable Triton positional embeddings in vLLM.",
    from_string=int,
    from_environ="VLLM_USE_TRITON_POS_EMBED",
)

vllm_mm_encoder_fp8_attn = Field(
    "q3vl_vllm_mm_encoder_fp8_attn",
    description="Enable FP8 attention in vLLM multimodal encoder.",
    from_string=int,
    from_environ="VLLM_MM_ENCODER_FP8_ATTN",
)

vllm_flashinfer_moe_backend = Field(
    "q3vl_vllm_flashinfer_moe_backend",
    description="FlashInfer MoE backend selection.",
    from_string=str,
    from_environ="VLLM_FLASHINFER_MOE_BACKEND",
)

vllm_flashinfer_workspace_buffer_size = Field(
    "q3vl_vllm_flashinfer_workspace_buffer_size",
    description="FlashInfer workspace buffer size.",
    from_string=int,
    from_environ="VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE",
)

tokio_worker_threads = Field(
    "q3vl_tokio_worker_threads",
    description="Tokio worker thread count for vLLM/Dynamo.",
    from_string=int,
    from_environ="TOKIO_WORKER_THREADS",
)

omp_num_threads = Field(
    "q3vl_omp_num_threads",
    description="OpenMP thread count for vLLM/Dynamo.",
    from_string=int,
    from_environ="OMP_NUM_THREADS",
)

frontend_cli = Field(
    "q3vl_frontend_cli",
    description="Comma-separated Dynamo frontend CLI flags.",
    from_string=lambda x: [item for item in x.split(",") if item] if x else [],
)

frontend_enable_numa_binding = Field(
    "q3vl_frontend_enable_numa_binding",
    description="Enable NUMA binding for the Dynamo frontend.",
    from_string=lambda x: x.lower() in ("true", "1", "yes") if x else False,
)

frontend_startup_timeout = Field(
    "q3vl_frontend_startup_timeout",
    description="Dynamo frontend startup timeout (seconds or ISO 8601).",
    from_string=_parse_timedelta,
)

frontend_healthcheck_timeout = Field(
    "q3vl_frontend_healthcheck_timeout",
    description="Dynamo frontend healthcheck timeout (seconds or ISO 8601).",
    from_string=_parse_timedelta,
)

frontend_poll_interval = Field(
    "q3vl_frontend_poll_interval",
    description="Dynamo frontend poll interval (seconds or ISO 8601).",
    from_string=_parse_timedelta,
)

wandb_entity = Field(
    "q3vl_wandb_entity",
    description="Weights & Biases entity/team name.",
    from_string=str,
    from_environ="WANDB_ENTITY",
)

wandb_project = Field(
    "q3vl_wandb_project",
    description="Weights & Biases project name.",
    from_string=str,
    from_environ="WANDB_PROJECT",
)

wandb_name = Field(
    "q3vl_wandb_name",
    description="Weights & Biases run name (optional).",
    from_string=str,
    from_environ="WANDB_NAME",
)

wandb_api_key = Field(
    "q3vl_wandb_api_key",
    description="Weights & Biases API key.",
    from_string=lambda x: x if x else None,
    from_environ="WANDB_API_KEY",
)
