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

import os

import wandb
from code.ops.harness import BenchmarkHarnessOp, Vboost
from loguru import logger

from mlperf_inf_mm_q3vl.benchmark import run_benchmark
from mlperf_inf_mm_q3vl.schema import Verbosity
from mlperf_inf_mm_q3vl_nv.benchmark import run_benchmark_aiohttp
from mlperf_inf_mm_q3vl_nv.deploy import MpiDynamoVllmEndpointDeployer
from mlperf_inf_mm_q3vl_nv.log import setup_loguru_for_benchmark_nv
from mlperf_inf_mm_q3vl_nv.wandb_utils import read_and_log_mlperf_detail_to_wandb
from mpi4py import MPI
from nvmitten.configurator import autoconfigure, bind

from . import fields
from .utils import (
    build_dataset,
    build_dynamo_endpoint,
    build_env_vars,
    build_settings,
    build_wandb_config,
)


@autoconfigure
@bind(fields.random_seed, "random_seed")
@bind(fields.max_concurrency, "max_concurrency")
@bind(fields.use_http_client, "use_http_client")
class Qwen3VL235BHarnessOp(BenchmarkHarnessOp):
    def __init__(
        self,
        *args,
        random_seed: int = 12345,
        max_concurrency: int | None = None,
        use_http_client: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset = build_dataset()
        self.settings = build_settings()
        self.dynamo = build_dynamo_endpoint()
        self.vllm_env_vars = build_env_vars()
        self.wandb_config = build_wandb_config()
        self.random_seed = random_seed
        self.max_concurrency = max_concurrency
        self.use_http_client = use_http_client

    def run(self, scratch_space, dependency_outputs):
        """Run the benchmark using Python code.

        Args:
            scratch_space: The scratch space for temporary files.
            dependency_outputs: Outputs from dependency operations.

        Returns:
            dict: Dictionary containing the result metadata and log directory.
        """
        for key, value in self.vllm_env_vars.items():
            if value is None:
                continue
            os.environ[key] = str(value)
        comm = MPI.COMM_WORLD
        # Ensure log output directory matches Workload log_dir.
        log_outdir = self.wl.log_dir
        if self.settings.logging.log_output is None or not hasattr(
            self.settings.logging.log_output, "outdir"
        ):
            self.settings.logging.log_output = {
                "outdir": log_outdir,
                "prefix": "mlperf_log_",
                "suffix": "",
            }
        else:
            self.settings.logging.log_output.outdir = log_outdir
        setup_loguru_for_benchmark_nv(settings=self.settings, verbosity=Verbosity.INFO)

        # Initialize wandb on rank 0
        if self.wandb_config.is_configured():
            if comm.Get_rank() == 0:
                wandb.login(key=self.wandb_config.api_key)
                dynamo_config = self.dynamo.model_dump(exclude={"model": {"token"}})
                wandb.init(
                    name=self.wandb_config.name,
                    entity=self.wandb_config.entity,
                    project=self.wandb_config.project,
                    config={
                        "random_seed": self.random_seed,
                        "verbosity": Verbosity.INFO.value,
                        "settings": self.settings.model_dump(),
                        "dataset": self.dataset.model_dump(),
                        "dynamo": dynamo_config,
                    },
                )
        with Vboost(), self.power_monitor(), MpiDynamoVllmEndpointDeployer(
            endpoint=self.dynamo,
            settings=self.settings,
            warmup_dataset=self.dataset,
            base_random_seed=self.random_seed,
        ):
            if comm.Get_rank() == 0:
                if self.use_http_client:
                    run_benchmark_aiohttp(
                        settings=self.settings,
                        dataset=self.dataset,
                        endpoint=self.dynamo,
                        random_seed=self.random_seed,
                        max_concurrency=self.max_concurrency,
                    )
                else:
                    if self.max_concurrency is not None:
                        logger.warning(
                            "--max-concurrency is ignored without --use-http-client."
                        )
                    run_benchmark(
                        settings=self.settings,
                        dataset=self.dataset,
                        endpoint=self.dynamo,
                        random_seed=self.random_seed,
                    )
                if self.wandb_config.is_configured():
                    # Also read and log the mlperf_log_detail.txt file
                    read_and_log_mlperf_detail_to_wandb(self.settings)
                    wandb.finish()
            comm.Barrier()
        return self.load_run_results()
