"""Schema definitions for the NVIDIA-optimized Qwen3-VL (Q3VL) benchmark."""

from __future__ import annotations

from datetime import timedelta
from typing import ClassVar, Self

from mlperf_inf_mm_q3vl.schema import (
    BaseModelWithAttributeDescriptionsFromDocstrings,
    BlacklistedVllmCliFlagError,
    EndpointToDeploy,
    VllmEndpoint,
)
from pydantic import model_validator


class EtcdLaunchConfig(BaseModelWithAttributeDescriptionsFromDocstrings):
    """The launch configuration for the ETCD server."""

    hostname: str = "0.0.0.0"
    """The hostname for the ETCD server."""

    port: int = 2379
    """The port for the ETCD server."""

    startup_timeout: timedelta = timedelta(minutes=10)
    """The startup timeout for the ETCD server."""

    healthcheck_timeout: timedelta = timedelta(seconds=5)
    """The healthcheck timeout for the ETCD server."""

    poll_interval: timedelta = timedelta(seconds=5)
    """The poll interval for the ETCD server."""


class NatsLaunchConfig(BaseModelWithAttributeDescriptionsFromDocstrings):
    """The launch configuration for the NATS server."""

    hostname: str = "0.0.0.0"
    """The hostname for the NATS server."""

    port: int = 4222
    """The port for the NATS server (client connections)."""

    monitoring_port: int = 8222
    """The port for the NATS server monitoring/HTTP endpoint."""

    startup_timeout: timedelta = timedelta(minutes=10)
    """The startup timeout for the NATS server."""

    healthcheck_timeout: timedelta = timedelta(seconds=5)
    """The healthcheck timeout for the NATS server."""

    poll_interval: timedelta = timedelta(seconds=5)
    """The poll interval for the NATS server."""


class DynamoVllmLaunchConfig(BaseModelWithAttributeDescriptionsFromDocstrings):
    """The launch configuration for the vLLM backend of Dynamo."""

    cli: list[str] = []
    """The CLI arguments for the command to launch the vLLM backend of Dynamo."""

    enable_numa_binding: bool = False
    """Whether to bind the vLLM backend to NUMA nodes for memory affinity.
    When enabled, the NUMA topology is auto-detected via NVML."""

    @model_validator(mode="after")
    def validate_cli(self) -> Self:
        """Validate the CLI arguments for the command to launch the vLLM backend of Dynamo."""
        for flag in self.cli:
            if not flag.startswith(("--", "-")):
                raise PositionalCliFlagError(flag)
            if flag in BlacklistedVllmCliFlagError.BLACKLIST:
                raise BlacklistedVllmCliFlagError(flag)
        return self


class PositionalCliFlagError(ValueError):
    """The exception raised when a positional CLI flag is provided."""

    def __init__(self, flag: str) -> None:
        """Initialize the exception."""
        super().__init__(
            f"Positional CLI flag: {flag} is not allowed. Only optional flags are "
            "allowed to be passed in.",
        )


class BlacklistedDynamoFrontendCliFlagError(ValueError):
    """The exception raised when a blacklisted Dynamo frontend CLI flag is provided."""

    BLACKLIST: ClassVar[list[str]] = [
        "--http-host",
        "--http-port",
    ]

    def __init__(self, flag: str) -> None:
        """Initialize the exception."""
        super().__init__(
            f"Blacklisted Dynamo frontend CLI flag: {flag} is not allowed. "
            f"The blacklisted flagsare {self.BLACKLIST}.",
        )


class DynamoFrontendLaunchConfig(BaseModelWithAttributeDescriptionsFromDocstrings):
    """The launch configuration for the Dynamo frontend."""

    cli: list[str] = []
    """The CLI arguments for the command to launch the Dynamo frontend."""

    startup_timeout: timedelta = timedelta(minutes=10)
    """The startup timeout for the Dynamo frontend."""

    healthcheck_timeout: timedelta = timedelta(seconds=5)
    """The healthcheck timeout for the Dynamo frontend."""

    poll_interval: timedelta = timedelta(seconds=5)
    """The poll interval for the Dynamo frontend."""

    enable_numa_binding: bool = False
    """Whether to bind the Dynamo frontend to NUMA node 0 for memory affinity."""

    cpus: str | None = None
    """CPU cores to bind the frontend to as comma-separated string (e.g., "0,1,16,17,32,33,48,49")."""

    @model_validator(mode="after")
    def validate_cli(self) -> Self:
        """Validate the CLI arguments for the command to launch the Dynamo frontend."""
        for flag in self.cli:
            if not flag.startswith(("--", "-")):
                raise PositionalCliFlagError(flag)
            if flag in BlacklistedDynamoFrontendCliFlagError.BLACKLIST:
                raise BlacklistedDynamoFrontendCliFlagError(flag)
        return self


class DynamoEndpoint(EndpointToDeploy):
    """The endpoint to deploy for the NVIDIA-optimized Qwen3-VL (Q3VL) benchmark."""

    etcd: EtcdLaunchConfig
    """The launch configuration for the ETCD server."""

    nats: NatsLaunchConfig
    """The launch configuration for the NATS server."""

    vllm: DynamoVllmLaunchConfig
    """The launch configuration for the vLLM backend of Dynamo."""

    frontend: DynamoFrontendLaunchConfig
    """The launch configuration for the frontend of Dynamo."""

    num_warmup_requests_per_vllm_instance: int = 1000
    """The number of requests per vLLM instance to warm up the endpoint. The total
    number of warmup requests is scaled by the number of vLLM instances.
    """


class Wandb(BaseModelWithAttributeDescriptionsFromDocstrings):
    """The settings for uploading configs and metrics to a Weights & Biases project."""

    entity: str | None = None
    """The wandb team/entity (required when wandb is enabled)."""

    project: str | None = None
    """The wandb project to store the configs/metrics (required when wandb is enabled)."""

    name: str | None = None
    """The name of this run in wandb. If not provided, the name is auto-generated."""

    api_key: str | None = None
    """The wandb API key for the user's account (required when wandb is enabled)."""

    @model_validator(mode="after")
    def validate_required_fields(self) -> Self:
        sets = (v is not None for v in (self.entity, self.project, self.api_key))
        if any(sets) and not all(sets):
            raise ValueError(
                "wandb_config requires entity, project, and api_key when any is set.",
            )
        return self

    def is_configured(self) -> bool:
        return (
            self.entity is not None
            and self.project is not None
            and self.api_key is not None
        )


class VllmProfileEndpoint(VllmEndpoint):
    profile: bool = False
    """Whether to profile the endpoint"""

    payload_timeout: timedelta = timedelta(seconds=10)
    """The timeout for the payload request to the endpoint."""
