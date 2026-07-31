"""Utilities for Qwen3-VL 235B vLLM benchmark."""

from __future__ import annotations

from dataclasses import dataclass

from nvmitten.configurator import autoconfigure, bind

import code.fields.loadgen as loadgen_fields

from mlperf_inf_mm_q3vl.schema import (
    Dataset,
    LogSettings,
    Model,
    SamplingParams,
    Settings,
    TestSettings,
)
from mlperf_inf_mm_q3vl_nv.schema import (
    DynamoEndpoint,
    DynamoFrontendLaunchConfig,
    DynamoVllmLaunchConfig,
    EtcdLaunchConfig,
    NatsLaunchConfig,
    Wandb,
)

from . import fields

@autoconfigure
@bind(fields.test_scenario, "scenario")
@bind(fields.test_mode, "mode")
@bind(loadgen_fields.test_mode, "loadgen_test_mode")
@bind(loadgen_fields.offline_expected_qps, "offline_expected_qps")
@bind(loadgen_fields.server_target_qps, "server_target_qps")
@bind(fields.server_target_latency, "server_target_latency")
@bind(fields.min_duration, "min_duration")
@bind(fields.max_duration, "max_duration")
@bind(loadgen_fields.min_query_count, "min_query_count")
@bind(loadgen_fields.max_query_count, "max_query_count")
@bind(loadgen_fields.qsl_rng_seed, "qsl_rng_seed")
@bind(loadgen_fields.sample_index_rng_seed, "sample_index_rng_seed")
@bind(loadgen_fields.schedule_rng_seed, "schedule_rng_seed")
@bind(fields.log_output, "log_output")
@bind(fields.log_mode, "log_mode")
@bind(fields.enable_trace, "enable_trace")
@dataclass
class SettingsBuilder:
    """Build Settings from bound fields in `fields.py`."""

    scenario: str | None = None
    mode: str | None = None
    loadgen_test_mode: str | None = "PerformanceOnly"
    offline_expected_qps: float | None = None
    server_target_qps: float | None = None
    server_target_latency: object | None = None
    min_duration: object | None = None
    max_duration: object | None = None
    min_query_count: int | None = None
    max_query_count: int | None = None
    qsl_rng_seed: int | None = None
    sample_index_rng_seed: int | None = None
    schedule_rng_seed: int | None = None
    log_output: object | None = None
    log_mode: str | None = None
    enable_trace: bool | None = None

    def build(self) -> Settings:
        if self.loadgen_test_mode == "AccuracyOnly":
            self.mode = "accuracy_only"
        elif self.loadgen_test_mode == "PerformanceOnly":
            self.mode = "performance_only"

        test_kwargs = {}
        for attr in [
            "scenario",
            "mode",
            "offline_expected_qps",
            "server_target_qps",
            "server_target_latency",
            "min_duration",
            "max_duration",
            "min_query_count",
            "max_query_count",
            "qsl_rng_seed",
            "sample_index_rng_seed",
            "schedule_rng_seed",
        ]:
            if (value := getattr(self, attr)) is not None:
                test_kwargs[attr] = value

        log_kwargs = {}
        log_output = self.log_output
        if isinstance(log_output, str):
            # LogOutputSettings expects a structured object, not a raw path string.
            log_output = {
                "outdir": log_output,
                "prefix": "mlperf_log_",
                "suffix": "",
            }
        if log_output is not None:
            log_kwargs["log_output"] = log_output
        for attr in ["log_mode", "enable_trace"]:
            if (value := getattr(self, attr)) is not None:
                log_kwargs[attr] = value

        test_settings = TestSettings(**test_kwargs)
        log_settings = LogSettings(**log_kwargs)
        return Settings(test=test_settings, user_conf={}, logging=log_settings)


def build_settings() -> Settings:
    """Convenience wrapper for the configured SettingsBuilder."""
    return SettingsBuilder().build()


@autoconfigure
@bind(fields.vllm_dyn_log, "dyn_log")
@bind(fields.vllm_logging_level, "vllm_logging_level")
@bind(fields.vllm_use_flashinfer_sampler, "vllm_use_flashinfer_sampler")
@bind(fields.vllm_use_flashinfer_moe_fp4, "vllm_use_flashinfer_moe_fp4")
@bind(fields.vllm_use_triton_pos_embed, "vllm_use_triton_pos_embed")
@bind(fields.vllm_mm_encoder_fp8_attn, "vllm_mm_encoder_fp8_attn")
@bind(fields.vllm_flashinfer_moe_backend, "vllm_flashinfer_moe_backend")
@bind(fields.vllm_flashinfer_workspace_buffer_size, "vllm_flashinfer_workspace_buffer_size")
@bind(fields.tokio_worker_threads, "tokio_worker_threads")
@bind(fields.omp_num_threads, "omp_num_threads")
@dataclass
class EnvVarsBuilder:
    """Build vLLM/Dynamo environment overrides from bound fields."""

    dyn_log: str = "debug"
    vllm_logging_level: str = "DEBUG"
    vllm_use_flashinfer_sampler: int = 1
    vllm_use_flashinfer_moe_fp4: int = 1
    vllm_use_triton_pos_embed: int | None = None
    vllm_mm_encoder_fp8_attn: int | None = None
    vllm_flashinfer_moe_backend: str = "latency"
    vllm_flashinfer_workspace_buffer_size: int = 6 * 256 * 1024 * 1024
    tokio_worker_threads: int = 32
    omp_num_threads: int = 64

    def build(self) -> dict[str, str]:
        env_vars: dict[str, str] = {}
        for attr, env_key in [
            ("dyn_log", "DYN_LOG"),
            ("vllm_logging_level", "VLLM_LOGGING_LEVEL"),
            ("vllm_use_flashinfer_sampler", "VLLM_USE_FLASHINFER_SAMPLER"),
            ("vllm_use_flashinfer_moe_fp4", "VLLM_USE_FLASHINFER_MOE_FP4"),
            ("vllm_use_triton_pos_embed", "VLLM_USE_TRITON_POS_EMBED"),
            ("vllm_mm_encoder_fp8_attn", "VLLM_MM_ENCODER_FP8_ATTN"),
            ("vllm_flashinfer_moe_backend", "VLLM_FLASHINFER_MOE_BACKEND"),
            ("vllm_flashinfer_workspace_buffer_size", "VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE"),
            ("tokio_worker_threads", "TOKIO_WORKER_THREADS"),
            ("omp_num_threads", "OMP_NUM_THREADS"),
        ]:
            if (value := getattr(self, attr)) is not None:
                env_vars[env_key] = str(value)
        return env_vars


def build_env_vars() -> dict[str, str]:
    """Convenience wrapper for the configured EnvVarsBuilder."""
    return EnvVarsBuilder().build()


@autoconfigure
@bind(fields.endpoint_request_timeout, "request_timeout")
@bind(fields.endpoint_startup_timeout, "startup_timeout")
@bind(fields.endpoint_shutdown_timeout, "shutdown_timeout")
@bind(fields.endpoint_poll_interval, "poll_interval")
@bind(fields.endpoint_healthcheck_timeout, "healthcheck_timeout")
@bind(fields.num_warmup_requests_per_vllm_instance, "num_warmup_requests_per_vllm_instance")
@bind(fields.model_repo_id, "model_repo_id")
@bind(fields.model_token, "model_token")
@bind(fields.model_revision, "model_revision")
@bind(fields.etcd_hostname, "etcd_hostname")
@bind(fields.etcd_port, "etcd_port")
@bind(fields.etcd_startup_timeout, "etcd_startup_timeout")
@bind(fields.etcd_healthcheck_timeout, "etcd_healthcheck_timeout")
@bind(fields.etcd_poll_interval, "etcd_poll_interval")
@bind(fields.nats_hostname, "nats_hostname")
@bind(fields.nats_port, "nats_port")
@bind(fields.nats_monitoring_port, "nats_monitoring_port")
@bind(fields.nats_startup_timeout, "nats_startup_timeout")
@bind(fields.nats_healthcheck_timeout, "nats_healthcheck_timeout")
@bind(fields.nats_poll_interval, "nats_poll_interval")
@bind(fields.vllm_cli, "vllm_cli")
@bind(fields.vllm_enable_numa_binding, "vllm_enable_numa_binding")
@bind(fields.frontend_cli, "frontend_cli")
@bind(fields.frontend_startup_timeout, "frontend_startup_timeout")
@bind(fields.frontend_healthcheck_timeout, "frontend_healthcheck_timeout")
@bind(fields.frontend_poll_interval, "frontend_poll_interval")
@bind(fields.frontend_enable_numa_binding, "frontend_enable_numa_binding")
@dataclass
class DynamoEndpointBuilder:
    """Build DynamoEndpoint from bound fields in `fields.py`."""

    request_timeout: object | None = None
    startup_timeout: object | None = None
    shutdown_timeout: object | None = None
    poll_interval: object | None = None
    healthcheck_timeout: object | None = None
    num_warmup_requests_per_vllm_instance: int | None = None
    model_repo_id: str | None = "nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4-MLPerf-Inference-Closed-V6.0"
    model_token: str | None = None
    model_revision: str | None = None
    etcd_hostname: str | None = None
    etcd_port: int | None = None
    etcd_startup_timeout: object | None = None
    etcd_healthcheck_timeout: object | None = None
    etcd_poll_interval: object | None = None
    nats_hostname: str | None = None
    nats_port: int | None = None
    nats_monitoring_port: int | None = None
    nats_startup_timeout: object | None = None
    nats_healthcheck_timeout: object | None = None
    nats_poll_interval: object | None = None
    vllm_cli: list[str] | None = None
    vllm_enable_numa_binding: bool | None = None
    frontend_cli: list[str] | None = None
    frontend_startup_timeout: object | None = None
    frontend_healthcheck_timeout: object | None = None
    frontend_poll_interval: object | None = None
    frontend_enable_numa_binding: bool | None = None

    def build(self) -> DynamoEndpoint:
        def _collect_non_none(mapping: list[tuple[str, str]]) -> dict[str, object]:
            kwargs: dict[str, object] = {}
            for attr, key in mapping:
                if (value := getattr(self, attr)) is not None:
                    kwargs[key] = value
            return kwargs

        endpoint_kwargs = _collect_non_none([
            ("request_timeout", "request_timeout"),
            ("startup_timeout", "startup_timeout"),
            ("shutdown_timeout", "shutdown_timeout"),
            ("poll_interval", "poll_interval"),
            ("healthcheck_timeout", "healthcheck_timeout"),
            ("num_warmup_requests_per_vllm_instance", "num_warmup_requests_per_vllm_instance"),
        ])
        model_kwargs = _collect_non_none([
            ("model_repo_id", "repo_id"),
            ("model_token", "token"),
            ("model_revision", "revision"),
        ])
        etcd_kwargs = _collect_non_none([
            ("etcd_hostname", "hostname"),
            ("etcd_port", "port"),
            ("etcd_startup_timeout", "startup_timeout"),
            ("etcd_healthcheck_timeout", "healthcheck_timeout"),
            ("etcd_poll_interval", "poll_interval"),
        ])
        nats_kwargs = _collect_non_none([
            ("nats_hostname", "hostname"),
            ("nats_port", "port"),
            ("nats_monitoring_port", "monitoring_port"),
            ("nats_startup_timeout", "startup_timeout"),
            ("nats_healthcheck_timeout", "healthcheck_timeout"),
            ("nats_poll_interval", "poll_interval"),
        ])
        vllm_kwargs = _collect_non_none([
            ("vllm_cli", "cli"),
            ("vllm_enable_numa_binding", "enable_numa_binding"),
        ])
        frontend_kwargs = _collect_non_none([
            ("frontend_cli", "cli"),
            ("frontend_startup_timeout", "startup_timeout"),
            ("frontend_healthcheck_timeout", "healthcheck_timeout"),
            ("frontend_poll_interval", "poll_interval"),
            ("frontend_enable_numa_binding", "enable_numa_binding"),
        ])

        return DynamoEndpoint(
            **endpoint_kwargs,
            model=Model(**model_kwargs),
            sampling_params=SamplingParams(),
            etcd=EtcdLaunchConfig(**etcd_kwargs),
            nats=NatsLaunchConfig(**nats_kwargs),
            vllm=DynamoVllmLaunchConfig(**vllm_kwargs),
            frontend=DynamoFrontendLaunchConfig(**frontend_kwargs),
        )


def build_dynamo_endpoint() -> DynamoEndpoint:
    """Convenience wrapper for the configured DynamoEndpointBuilder."""
    return DynamoEndpointBuilder().build()


@autoconfigure
@bind(fields.dataset_repo_id, "repo_id")
@bind(fields.dataset_token, "token")
@bind(fields.dataset_revision, "revision")
@bind(fields.dataset_split, "split")
@dataclass
class DatasetBuilder:
    """Build Dataset from bound fields in `fields.py`."""

    repo_id: str = "Shopify/product-catalogue"
    token: str | None = None
    revision: str = "d5c517c509f5aca99053897ef1de797d6d7e5aa5"
    split: list[str] | None = None

    def build(self) -> Dataset:
        kwargs = {
            "repo_id": self.repo_id,
            "token": self.token,
            "revision": self.revision,
            "split": self.split or ["train", "test"],
        }
        return Dataset(**kwargs)


def build_dataset() -> Dataset:
    """Convenience wrapper for the configured DatasetBuilder."""
    return DatasetBuilder().build()


@autoconfigure
@bind(fields.wandb_entity, "entity")
@bind(fields.wandb_project, "project")
@bind(fields.wandb_name, "name")
@bind(fields.wandb_api_key, "api_key")
@dataclass
class WandbBuilder:
    """Build Wandb settings from bound fields in `fields.py`."""

    entity: str | None = None
    project: str | None = None
    name: str | None = None
    api_key: str | None = None

    def build(self) -> Wandb:
        return Wandb(
            entity=self.entity,
            project=self.project,
            name=self.name,
            api_key=self.api_key,
        )


def build_wandb_config() -> Wandb:
    """Convenience wrapper for the configured WandbBuilder."""
    return WandbBuilder().build()
