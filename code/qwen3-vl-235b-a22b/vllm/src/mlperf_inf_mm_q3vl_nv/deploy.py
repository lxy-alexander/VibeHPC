"""Endpoint deployers for deploying and managing the lifecycles of VLM endpoints."""

from __future__ import annotations

import asyncio
import os
import random
import socket
import subprocess
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self
from urllib.parse import urlparse, urlunparse

import httpx
import requests
from datasets import load_dataset
from huggingface_hub import repo_exists, snapshot_download
from loguru import logger
from mlperf_inf_mm_q3vl.deploy import (
    EndpointDeployer,
    LocalProcessDeadError,
    LocalVllmDeployer,
)
from mlperf_inf_mm_q3vl.log import get_log_file_path
from mlperf_inf_mm_q3vl.task import ShopifyGlobalCatalogue
from mpi4py import MPI
from openai import AsyncOpenAI, DefaultAioHttpClient
from openai.types.chat import ChatCompletion
from pympler import asizeof

from .wandb_utils import log_launch_config_to_wandb

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from datetime import timedelta
    from types import TracebackType

    from mlperf_inf_mm_q3vl.schema import Dataset, LoadedSample, Settings

    from .schema import DynamoEndpoint, DynamoVllmLaunchConfig

HTTP_OK = 200



def _numa_node_has_cpus(node_id: int) -> bool:
    """Check whether a NUMA node has any CPUs via sysfs.

    On Grace Hopper systems, ``nvmlDeviceGetNumaNodeId`` returns the
    GPU-memory NUMA node which typically has no CPUs.  This helper lets
    us detect that case so we can fall through to the CPU-affinity method.
    """
    cpulist_file = Path(f"/sys/devices/system/node/node{node_id}/cpulist")
    try:
        content = cpulist_file.read_text().strip()
        return bool(content) and content != ""
    except (OSError, ValueError):
        # If we can't read sysfs (e.g. inside a container without it
        # mounted), assume the node is valid to avoid breaking the
        # fast path on systems where it works correctly.
        return True


def _get_device_numa_node(device_id: int) -> int:
    """Get the CPU NUMA node closest to a GPU device.

    Strategy: try direct NVML NUMA query, then CPU-affinity fallback,
    then default to node 0.  On Grace Hopper the direct query returns a
    GPU-memory NUMA node with no CPUs, so we validate via sysfs.

    Args:
        device_id: Logical device index.

    Returns:
        The CPU NUMA node ID (``0`` when it cannot be determined).
    """
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

    try:
        numa_id = pynvml.nvmlDeviceGetNumaNodeId(handle)
        if _numa_node_has_cpus(numa_id):
            pynvml.nvmlShutdown()
            return numa_id
    except pynvml.NVMLError:
        pass

    # Fallback: infer from CPU affinity mask
    try:
        cpu_set_size = ((os.cpu_count() or 1) + 63) // 64
        affinity_mask = pynvml.nvmlDeviceGetCpuAffinity(handle, cpu_set_size)
        for i, mask in enumerate(affinity_mask):
            if mask:
                first_cpu = i * 64 + (mask & -mask).bit_length() - 1
                numa_node = _numa_node_for_cpu(first_cpu)
                if numa_node is not None:
                    pynvml.nvmlShutdown()
                    return numa_node
    except pynvml.NVMLError:
        pass

    pynvml.nvmlShutdown()
    logger.warning("Could not determine NUMA node for GPU {}, defaulting to 0", device_id)
    return 0


def _numa_node_for_cpu(cpu_id: int) -> int | None:
    """Determine which NUMA node a CPU belongs to via sysfs.

    Args:
        cpu_id: The logical CPU id.

    Returns:
        The NUMA node id, or ``None`` if it cannot be determined.
    """
    node_path = Path("/sys/devices/system/node")
    if not node_path.exists():
        return None

    for node_dir in node_path.iterdir():
        if not node_dir.name.startswith("node") or not node_dir.name[4:].isdigit():
            continue
        cpulist_file = node_dir / "cpulist"
        if not cpulist_file.exists():
            continue
        try:
            cpulist = cpulist_file.read_text().strip()
            if _cpu_in_cpulist(cpu_id, cpulist):
                return int(node_dir.name[4:])
        except (ValueError, OSError):
            continue
    return None


def _cpu_in_cpulist(cpu_id: int, cpulist: str) -> bool:
    """Check whether *cpu_id* is contained in a Linux cpulist string.

    A cpulist string looks like ``"0-3,8-11"``.
    """
    for part in cpulist.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            if int(start) <= cpu_id <= int(end):
                return True
        elif part.isdigit() and int(part) == cpu_id:
            return True
    return False



def _build_numactl_cmd(
    cmd: list[str],
    numa_node: int | None = None,
    cpu_list: list[int] | None = None,
) -> list[str]:
    """Prepend numactl to a command if NUMA binding is requested."""
    if numa_node is None and not cpu_list:
        return cmd

    numactl_args = ["numactl"]
    if cpu_list:
        cpu_list_str = ",".join(str(c) for c in cpu_list)
        numactl_args.append(f"--physcpubind={cpu_list_str}")
    if numa_node is not None:
        numactl_args.append(f"--cpunodebind={numa_node}")
        numactl_args.append(f"--membind={numa_node}")
    return numactl_args + cmd


class ServiceHealthcheckTimeoutError(RuntimeError):
    """Raised when a service fails to be healthy within the specified timeout."""

    def __init__(self, timeout: timedelta) -> None:
        """Initialize the exception.

        Args:
            timeout: The timeout duration that was exceeded.
        """
        super().__init__(
            f"Service failed to be healthy within the timeout of {timeout}.",
        )


def wait_for_healthy(
    wait_timeout: timedelta,
    poll_interval: timedelta,
    failfast: Callable[[], None],
    healthcheck: Callable[[], bool],
) -> None:
    """Wait for a service to be healthy via healthcheck."""
    start_time = time.time()
    while time.time() - start_time < wait_timeout.total_seconds():
        failfast()
        logger.info(
            "Waiting {:0.2f} seconds for service to be healthy...",
            time.time() - start_time,
        )
        if healthcheck():
            logger.info("Service is now healthy!")
            return

        time.sleep(poll_interval.total_seconds())

    raise ServiceHealthcheckTimeoutError(wait_timeout)


def child_process_failfast(process: subprocess.Popen) -> None:
    """Failfast the child process."""
    returncode = process.poll()
    if returncode is not None:
        raise LocalProcessDeadError(
            returncode=returncode,
            stdout_file_path=process.stdout.name,
            stderr_file_path=process.stderr.name,
        )


def healthcheck_by_status_code(
    healthcheck_url: str,
    healthcheck_timeout: timedelta,
) -> bool:
    """Healthcheck the service by checking the status code."""
    try:
        logger.info("Healthchecking {}...", healthcheck_url)
        response = requests.get(
            healthcheck_url,
            timeout=healthcheck_timeout.total_seconds(),
        )
        if response.status_code == HTTP_OK:
            logger.info(
                "Service with health check URL {} is now healthy!",
                healthcheck_url,
            )
            return True
    except requests.exceptions.RequestException:
        return False


class DynamoFrontendStartupTimeoutError(RuntimeError):
    """Raised when the Dynamo frontend fails to start within the specified timeout."""

    def __init__(self, timeout: timedelta) -> None:
        """Initialize the exception.

        Args:
            timeout: The timeout duration that was exceeded.
        """
        super().__init__(
            f"Dynamo frontend failed to start within the timeout of {timeout}.",
        )


class DynamoFrontendFormatError(RuntimeError):
    """Raised when the Dynamo frontend returns backend info in an unexpected format."""

    def __init__(self, response_text: str) -> None:
        """Initialize the exception.

        Args:
            response_text: The response obtained from the Dynamo frontend.
        """
        super().__init__(
            "Dynamo frontend returned backend information with unexpected "
            f"format: {response_text}",
        )


class DynamoVllmProcessDeadError(RuntimeError):
    """Raised when the Dynamo vLLM process dies."""

    def __init__(self, ranks: Sequence[int]) -> None:
        """Initialize the exception."""
        super().__init__(f"Dynamo vLLM processes on ranks {ranks} have died.")


class UnclearHuggingFaceRepoIdError(RuntimeError):
    """Raised when the Hugging Face repo ID is unclear."""

    def __init__(self, repo_id: str) -> None:
        """Initialize the exception."""
        super().__init__(
            f"The Hugging Face repo ID {repo_id} is unclear. This is because:\n"
            f"- {repo_id} is not a local directory; or\n"
            f"- {repo_id} is not a valid repo ID on Hugging Face Hub; or\n"
            f"- {repo_id} is a valid repo ID on Hugging Face Hub, but this repo is not"
            " accessible by the provided Hugging Face token (if any).",
        )


class MpiDynamoVllmEndpointDeployer(EndpointDeployer):
    """The endpoint deployer for the NVIDIA-optimized Qwen3-VL (Q3VL) benchmark."""

    def __init__(
        self,
        endpoint: DynamoEndpoint,
        settings: Settings,
        warmup_dataset: Dataset,
        base_random_seed: int,
    ) -> None:
        """Initialize the MPI-Dynamo-vLLM endpoint deployer.

        Args:
            endpoint: The configuration for the MPI-Dynamo-vLLM-based endpoint.
            settings: The settings for the benchmark.
            warmup_dataset: The configuration of the warmup dataset.
            base_random_seed: The base random seed for sampling the warmup samples.
        """
        super().__init__(endpoint=endpoint, settings=settings)
        self.endpoint = endpoint
        self.warmup_dataset = warmup_dataset
        self.base_random_seed = base_random_seed

        self.comm_world = MPI.COMM_WORLD
        self.comm_local = self.comm_world.Split_type(MPI.COMM_TYPE_SHARED)
        self.rank = self.comm_world.Get_rank()
        self.local_rank = self.comm_local.Get_rank()
        self.world_size = self.comm_world.Get_size()
        self.local_size = self.comm_local.Get_size()

        self._determine_endpoint_url()

        self._etcd_process: subprocess.Popen | None = None
        self._nats_process: subprocess.Popen | None = None
        self._dynamo_frontend_process: subprocess.Popen | None = None
        self._dynamo_vllm_process: subprocess.Popen | None = None

    def _load_warmup_samples(self) -> tuple[LoadedSample, ...]:
        dataset = load_dataset(
            self.warmup_dataset.repo_id,
            token=self.warmup_dataset.token,
            revision=self.warmup_dataset.revision,
            split="+".join(self.warmup_dataset.split),
        )
        logger.info(
            "Imported {} samples from the warmup dataset splits {}.",
            len(dataset),
            self.warmup_dataset.split,
        )
        random.seed(self.base_random_seed + self.rank)
        sample_indices = random.choices(
            range(len(dataset)),
            k=self.endpoint.num_warmup_requests_per_vllm_instance,
        )
        tic = time.perf_counter()
        loaded_samples = tuple(
            ShopifyGlobalCatalogue.formulate_loaded_sample(
                dataset[index],
                use_guided_decoding=self.endpoint.use_guided_decoding,
            )
            for index in sample_indices
        )
        logger.info(
            "Loaded {} warmup samples to RAM, which took {} seconds and {} GB "
            "in total.",
            len(loaded_samples),
            time.perf_counter() - tic,
            asizeof.asizeof(loaded_samples) / 1024 / 1024 / 1024,
        )
        return loaded_samples

    async def _send_one_warmup_request(
        self,
        openai_api_client: AsyncOpenAI,
        sample: LoadedSample,
    ) -> ChatCompletion:
        """Send one warmup request to the endpoint."""
        return await openai_api_client.chat.completions.create(  # type: ignore[call-overload, misc]
            model=self.endpoint.model.repo_id,
            messages=sample.messages,
            response_format=(
                sample.response_format.model_dump(
                    mode="json",
                    by_alias=True,
                )
                if sample.response_format is not None
                else None
            ),
            frequency_penalty=self.endpoint.sampling_params.frequency_penalty,
            presence_penalty=self.endpoint.sampling_params.presence_penalty,
            temperature=self.endpoint.sampling_params.temperature,
            top_p=self.endpoint.sampling_params.top_p,
            extra_body={
                k: getattr(self.endpoint.sampling_params, k)
                for k in ("top_k", "min_p", "repetition_penalty")
                if getattr(self.endpoint.sampling_params, k) is not None
            },
        )

    async def _send_all_warmup_requests(
        self,
        openai_api_client: AsyncOpenAI,
        loaded_samples: Iterable[LoadedSample],
    ) -> list[ChatCompletion]:
        """Send a bunch of warmup requests to the endpoint."""
        return await asyncio.gather(
            *[
                self._send_one_warmup_request(openai_api_client, sample)
                for sample in loaded_samples
            ],
        )

    def _warmup_endpoint(self) -> None:
        """Warm up the endpoint."""
        loaded_samples = self._load_warmup_samples()
        request_timeout_seconds = self.endpoint.request_timeout.total_seconds()
        openai_api_client = AsyncOpenAI(
            base_url=self.endpoint.url,
            http_client=DefaultAioHttpClient(
                timeout=httpx.Timeout(timeout=request_timeout_seconds, connect=5.0),
            ),
            api_key=self.endpoint.api_key,
            timeout=request_timeout_seconds,
        )

        logger.info("Warming up the endpoint...")
        responses = asyncio.run(
            self._send_all_warmup_requests(openai_api_client, loaded_samples),
        )
        logger.info(
            "Warmed up the endpoint with {} prompt tokens and {} completion tokens!",
            sum(response.usage.prompt_tokens for response in responses),
            sum(response.usage.completion_tokens for response in responses),
        )
        logger.info("Closing the OpenAI client for warmup...")
        try:
            asyncio.run(asyncio.wait_for(openai_api_client.close(), timeout=5.0))
        except TimeoutError:
            logger.error("Failed to close client within 5 seconds")
        except Exception as e:
            logger.exception(f"Error closing client: {e}")
        else:
            logger.info("Closed the OpenAI client for warmup.")

    def _determine_endpoint_url(self) -> str:
        """Determine the endpoint URL.

        Because which node to allocate is decided by the job scheduler, here we need to
        determine the hostname of the head node dynamically.
        """
        head_node_hostname = None
        if self.rank == 0:
            head_node_hostname = socket.gethostbyname(socket.gethostname())
        head_node_hostname = self.comm_world.bcast(head_node_hostname, root=0)
        parsed_url = urlparse(self.endpoint.url)
        # Preserve the port if it exists
        new_netloc = (
            f"{head_node_hostname}:{parsed_url.port}"
            if parsed_url.port
            else head_node_hostname
        )
        self.endpoint.url = urlunparse(
            parsed_url._replace(netloc=new_netloc),
        )
        self.endpoint.etcd.hostname = head_node_hostname
        self.endpoint.nats.hostname = head_node_hostname
        logger.info(
            "After head node discovery, the endpoint settings are: {}",
            self.endpoint,
        )

    def _log_file_path(self, key: str) -> Path:
        """Get the log file path for a given key.

        Args:
            key: The key for the log file.

        Returns:
            The log file path.
        """
        return get_log_file_path(key + f".rank{self.rank}", self.settings)

    def _launch_process(
        self,
        name: str,
        cmd: list[str],
        env: dict[str, str],
        numa_node: int | None = None,
        cpu_list: list[int] | None = None,
    ) -> subprocess.Popen:
        stdout_file_path = self._log_file_path(f"{name}.stdout")
        stderr_file_path = self._log_file_path(f"{name}.stderr")
        cmd = _build_numactl_cmd(cmd, numa_node=numa_node, cpu_list=cpu_list)
        logger.info("Starting {} with command: {}", name, cmd)
        logger.info("Starting {} with environment variables: {}", name, env)
        process = subprocess.Popen(
            cmd,
            stdout=stdout_file_path.open("w"),
            stderr=stderr_file_path.open("w"),
            text=True,
            env=env,
        )
        logger.info("Started {} process with PID: {}", name, process.pid)
        logger.info("{} stdout will be logged to: {}", name, stdout_file_path)
        logger.info("{} stderr will be logged to: {}", name, stderr_file_path)
        return process

    def _launch_process_and_wait_for_healthy(
        self,
        name: str,
        cmd: list[str],
        env: dict[str, str],
        healthcheck_url: str,
        wait_timeout: timedelta,
        healthcheck_timeout: timedelta,
        poll_interval: timedelta,
        numa_node: int | None = None,
        cpu_list: list[int] | None = None,
    ) -> subprocess.Popen:
        process = self._launch_process(name=name, cmd=cmd, env=env, numa_node=numa_node, cpu_list=cpu_list)
        wait_for_healthy(
            wait_timeout=wait_timeout,
            poll_interval=poll_interval,
            failfast=partial(child_process_failfast, process),
            healthcheck=partial(
                healthcheck_by_status_code,
                healthcheck_url,
                healthcheck_timeout,
            ),
        )
        return process

    def _maybe_download(
        self,
        repo_id: str,
        repo_type: Literal["model", "dataset"],
        revision: str,
        token: str,
    ) -> str:
        if Path(repo_id).is_dir():
            logger.info("{} is a local directory and will be used.", repo_id)
            return repo_id
        if repo_exists(repo_id=repo_id, repo_type=repo_type, token=token):
            logger.info(
                "{} exists on Hugging Face Hub and will be attempted to be downloaded.",
                repo_id,
            )
            return snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                token=token,
            )
        raise UnclearHuggingFaceRepoIdError(repo_id)

    def _launch_etcd_and_wait_for_healthy(self) -> None:
        cmd = [
            "etcd",
            "--listen-client-urls",
            f"http://{self.endpoint.etcd.hostname}:{self.endpoint.etcd.port}",
            "--advertise-client-urls",
            f"http://{self.endpoint.etcd.hostname}:{self.endpoint.etcd.port}",
            "--data-dir",
            "/tmp/etcd",
        ]
        self._etcd_process = self._launch_process_and_wait_for_healthy(
            name="etcd",
            cmd=cmd,
            env=os.environ.copy(),
            healthcheck_url=f"http://{self.endpoint.etcd.hostname}:{self.endpoint.etcd.port}/health",
            wait_timeout=self.endpoint.etcd.startup_timeout,
            healthcheck_timeout=self.endpoint.etcd.healthcheck_timeout,
            poll_interval=self.endpoint.etcd.poll_interval,
        )

    def _launch_nats_and_wait_for_healthy(self) -> None:
        cmd = [
            "nats-server",
            "-js",
            "-a",
            self.endpoint.nats.hostname,
            "-p",
            str(self.endpoint.nats.port),
            "-m",
            str(self.endpoint.nats.monitoring_port),
        ]
        self._nats_process = self._launch_process_and_wait_for_healthy(
            name="nats",
            cmd=cmd,
            env=os.environ.copy(),
            healthcheck_url=f"http://{self.endpoint.nats.hostname}:{self.endpoint.nats.monitoring_port}/healthz",
            wait_timeout=self.endpoint.nats.startup_timeout,
            healthcheck_timeout=self.endpoint.nats.healthcheck_timeout,
            poll_interval=self.endpoint.nats.poll_interval,
        )

    def _launch_dynamo_frontend_and_wait_for_healthy(self) -> None:
        parsed_url = urlparse(self.endpoint.url)
        cmd = [
            "--http-host",
            parsed_url.hostname,
            "--http-port",
            str(parsed_url.port),
        ]
        cmd.extend(self.endpoint.frontend.cli)
        env = os.environ.copy()
        env_updates: dict[str, str] = {
            "ETCD_ENDPOINTS": f"http://{self.endpoint.etcd.hostname}:{self.endpoint.etcd.port}",
            "NATS_SERVER": f"nats://{self.endpoint.nats.hostname}:{self.endpoint.nats.port}",
        }
        # Pass HF_TOKEN so frontend can fetch model metadata for gated models
        if self.endpoint.model.token:
            env_updates["HF_TOKEN"] = self.endpoint.model.token
        env.update(env_updates)

        log_launch_config_to_wandb(
            name="dynamo.frontend",
            cmd=cmd,
            env=env,
        )

        # NUMA binding for the frontend
        numa_node = 0 if self.endpoint.frontend.enable_numa_binding else None
        cpu_list = [int(c) for c in self.endpoint.frontend.cpus.split(",")] if self.endpoint.frontend.cpus else None

        self._dynamo_frontend_process = self._launch_process_and_wait_for_healthy(
            name="dynamo.frontend",
            cmd=["python3", "-m", "dynamo.frontend", *cmd],
            env=env,
            healthcheck_url=urlunparse(parsed_url._replace(path="/health")),
            wait_timeout=self.endpoint.frontend.startup_timeout,
            healthcheck_timeout=self.endpoint.frontend.healthcheck_timeout,
            poll_interval=self.endpoint.frontend.poll_interval,
            numa_node=numa_node,
            cpu_list=cpu_list,
        )

    @staticmethod
    def get_num_gpus_from_vllm_endpoint(vllm: DynamoVllmLaunchConfig) -> int:
        """Calculate the number of GPUs required from VllmEndpoint CLI arguments.

        Parses --tensor-parallel-size (or -tp), --pipeline-parallel-size (or -pp),
        and --data-parallel-size (or -dp) from the CLI arguments and returns their
        product.

        Args:
            vllm: The VllmEndpoint configuration.

        Returns:
            The total number of GPUs required (tensor_parallel * pipeline_parallel
            * data_parallel).
        """
        cli = vllm.cli

        # Refer: https://github.com/vllm-project/vllm/blob/83e1c76dbe07e30b7f4e6dbe17ba580f4afc98f0/vllm/model_executor/layers/fused_moe/config.py#L895
        parallel_param_names = ["tensor", "pipeline", "data", "prefill-context"]
        parallel_param_names = [
            param + "-parallel-size" for param in parallel_param_names
        ]
        parallel_param_names.extend(
            [param.replace("-", "_") for param in parallel_param_names],
        )
        parallel_param_names = ["--" + param for param in parallel_param_names]
        parallel_param_names += ["-tp", "-pp", "-dp", "-pcp"]

        parallel_size = 1
        for idx, arg in enumerate(cli):
            if arg in parallel_param_names:
                if idx + 1 < len(cli):
                    parallel_size *= int(cli[idx + 1])
            elif "=" in arg:
                param, value = arg.split("=", 1)
                if param in parallel_param_names:
                    parallel_size *= int(value)

        return parallel_size

    def _launch_dynamo_vllm(self) -> None:
        cmd = [
            "--model",
            self.endpoint.model.repo_id,
            "--revision",
            self.endpoint.model.revision,
        ]
        if self.endpoint.model.token:
            cmd.extend(["--hf-token", self.endpoint.model.token])
        if self.endpoint.api_key:
            cmd.extend(["--api-key", self.endpoint.api_key])
        cmd.extend(self.endpoint.vllm.cli)
        env = os.environ.copy()
        num_gpus = self.get_num_gpus_from_vllm_endpoint(self.endpoint.vllm)
        env_updates: dict[str, str] = {
            "ETCD_ENDPOINTS": f"http://{self.endpoint.etcd.hostname}:{self.endpoint.etcd.port}",
            "NATS_SERVER": f"nats://{self.endpoint.nats.hostname}:{self.endpoint.nats.port}",
            "CUDA_VISIBLE_DEVICES": ",".join(
                str(i)
                for i in range(
                    self.local_rank * num_gpus,
                    (self.local_rank + 1) * num_gpus,
                )
            ),
        }
        # Also set HF_TOKEN env var for vLLM compatibility
        if self.endpoint.model.token:
            env_updates["HF_TOKEN"] = self.endpoint.model.token
        env.update(env_updates)

        # NUMA binding for the vLLM backend — bind to the NUMA node of
        # the first GPU assigned to this rank.
        numa_node = None
        if self.endpoint.vllm.enable_numa_binding:
            first_gpu = self.local_rank * num_gpus
            numa_node = _get_device_numa_node(first_gpu)
            logger.info(
                "Rank {} (GPU {}) -> NUMA {}",
                self.local_rank, first_gpu, numa_node,
            )

        log_launch_config_to_wandb(
            name="dynamo.vllm",
            cmd=cmd,
            env=env,
        )

        self._dynamo_vllm_process = self._launch_process(
            name="dynamo.vllm",
            cmd=["python3", "-m", "dynamo.vllm", *cmd],
            env=env,
            numa_node=numa_node,
        )

    def _startup(self) -> None:
        if self.rank == 0:
            model_repo = self._maybe_download(
                repo_id=self.endpoint.model.repo_id,
                repo_type="model",
                revision=self.endpoint.model.revision,
                token=self.endpoint.model.token,
            )
            dataset_repo = self._maybe_download(
                repo_id=self.warmup_dataset.repo_id,
                repo_type="dataset",
                revision=self.warmup_dataset.revision,
                token=self.warmup_dataset.token,
            )
            self._launch_etcd_and_wait_for_healthy()
            self._launch_nats_and_wait_for_healthy()
            self._launch_dynamo_frontend_and_wait_for_healthy()
        else:
            model_repo = None
            dataset_repo = None
        model_repo = self.comm_world.bcast(model_repo, root=0)
        dataset_repo = self.comm_world.bcast(dataset_repo, root=0)
        if model_repo:
            self.endpoint.model.repo_id = model_repo
        if dataset_repo:
            self.warmup_dataset.repo_id = dataset_repo
        self.comm_world.Barrier()
        self._launch_dynamo_vllm()

    def _check_num_backends(self, healthcheck_timeout: timedelta) -> bool:
        """Check if the number of backends is correct."""
        response = requests.get(
            urlunparse(urlparse(self.endpoint.url)._replace(path="/health")),
            timeout=healthcheck_timeout.total_seconds(),
        )
        response_json = response.json()

        try:
            instances = response_json["instances"]
            num_backends = len(
                [
                    instance
                    for instance in instances
                    if instance["endpoint"] == "generate"
                ],
            )
        except (TypeError, KeyError) as e:
            raise DynamoFrontendFormatError(response.text) from e

        if num_backends < self.world_size:
            logger.info(
                "Waiting for {} backends to be ready...",
                self.world_size - num_backends,
            )
            return False
        return True

    def _failfast(self) -> None:
        fails = self.comm_world.allgather(self._dynamo_vllm_process.poll() is not None)
        if any(fails):
            raise DynamoVllmProcessDeadError(
                ranks=[i for i, fail in enumerate(fails) if fail],
            )

    def _wait_for_ready(self) -> None:
        wait_for_healthy(
            wait_timeout=self.endpoint.startup_timeout,
            poll_interval=self.endpoint.poll_interval,
            failfast=self._failfast,
            healthcheck=partial(
                self._check_num_backends,
                self.endpoint.healthcheck_timeout,
            ),
        )
        self._warmup_endpoint()
        self.comm_world.Barrier()

    def _terminate_or_kill_process(self, proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=self.endpoint.shutdown_timeout.total_seconds())
            logger.info("Process terminated gracefully")
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            logger.info("Process killed")

    def _shutdown(self) -> None:
        for proc in [
            self._dynamo_vllm_process,
            self._dynamo_frontend_process,
            self._etcd_process,
            self._nats_process,
        ]:
            if proc is not None:
                self._terminate_or_kill_process(proc)


class VllmProfileEndpointDeployer(LocalVllmDeployer):
    """Deploy an vLLM endpoint that enables profiling."""

    def __enter__(self) -> Self:
        """Start profiling after the endpoint is deployed."""
        super().__enter__()
        if self.endpoint.profile:
            self._start_profile()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Stop profiling before the endpoint is shut down."""
        if self.endpoint.profile:
            self._stop_profile()
        super().__exit__(exc_type, exc_val, exc_tb)

    def _start_profile(self) -> None:
        profile_url = self.endpoint.url.rstrip("/v1") + "/start_profile"
        try:
            response = requests.post(
                profile_url,
                timeout=self.endpoint.payload_timeout.total_seconds(),
            )
            if response.status_code == HTTP_OK:
                logger.info("Profile started successfully")
                return
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to start profile: {e}")

    def _stop_profile(self) -> None:
        profile_url = self.endpoint.url.rstrip("/v1") + "/stop_profile"
        try:
            response = requests.post(
                profile_url,
                timeout=self.endpoint.payload_timeout.total_seconds(),
            )
            if response.status_code == HTTP_OK:
                logger.info("Profile stopped successfully")
                return
        except requests.exceptions.RequestException:
            pass
