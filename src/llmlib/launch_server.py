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

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import yaml

from code import G_BENCHMARK_MODULES
from code.common import logging
from code.common.constants import Benchmark
from code.common.paths import TRTLLM_DIR, BUILD_DIR
from code.common.systems.system_list import DETECTED_SYSTEM
from code.common.triton.base_config import G_TRITON_BASE_CONFIG
from code.common.workload import EngineIndex, Workload
import code.fields.general as general_fields
from code.fields import harness as harness_fields
from code.fields import loadgen as loadgen_fields
from code.fields.harness import MPIMode
from code.llmlib.builder import TRTLLMBuilderOp, TRTLLMQuantizerOp, HFQuantizerOp
import code.llmlib.fields as llm_fields
from nvmitten.configurator import autoconfigure, bind
from nvmitten.nvidia.accelerator import GPU
from nvmitten.pipeline import Operation

from .config import TritonHarnessConfig, TrtllmEndpointConfig, TrtllmDisaggEndpointConfig, TrtllmExtraYAMLConfig


def _apply_yml_accuracy_override(override_source: Path, dest_yaml_path: Path, test_mode: str) -> None:
    """Apply accuracy YAML overrides when test_mode is AccuracyOnly.

    Convention: for a base YAML at /path/to/config.yml, if an accuracy override
    exists at /path/to/config_accuracy.yml, its contents are deep-merged on top
    of the base YAML. The override file only needs to contain the fields that differ.

    Example base (config.yml):
        max_seq_len: 25776
        max_batch_size: 512

    Example override (config_accuracy.yml):
        max_seq_len: 35840

    Result after merge:
        max_seq_len: 35840
        max_batch_size: 512

    Args:
        override_source: Path to the original trtllm_yml_override file (where _accuracy.yml lives alongside)
        dest_yaml_path: Path to the copied YAML in log_dir (will be modified in-place)
        test_mode: The test mode string (e.g., 'AccuracyOnly')
    """
    if test_mode != 'AccuracyOnly':
        return

    # Derive accuracy override path from source: config.yml -> config_accuracy.yml
    stem = override_source.stem
    suffix = override_source.suffix
    accuracy_path = override_source.parent / f"{stem}_accuracy{suffix}"

    if not accuracy_path.exists():
        logging.info(f"No accuracy override YAML found at {accuracy_path}, using base config as-is")
        return

    logging.info(f"Applying accuracy override YAML: {accuracy_path}")

    # Load base (already copied to dest) and override
    with open(dest_yaml_path, 'r') as f:
        base_config = yaml.safe_load(f) or {}
    with open(accuracy_path, 'r') as f:
        accuracy_overrides = yaml.safe_load(f) or {}

    # Deep merge: accuracy overrides take precedence
    def deep_merge(base: dict, override: dict) -> dict:
        merged = base.copy()
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    merged_config = deep_merge(base_config, accuracy_overrides)

    # Write merged config back to dest
    with open(dest_yaml_path, 'w') as f:
        yaml.dump(merged_config, f, default_flow_style=False, sort_keys=False)

    logging.info(f"Merged accuracy overrides into {dest_yaml_path}")
    for key, value in accuracy_overrides.items():
        logging.info(f"  Override: {key} = {value}")


def setup_tiktoken_for_gpt_oss(benchmark: Benchmark) -> None:
    """
    Temporary workaround, TODO: @shobhitv to remove later
    Download tiktoken encodings for gpt-oss-120b benchmark.

    WAR for openai_harmony trying to download vocab files at runtime from the network.
    Some environments block the download, so we pre-download the tiktoken files.
    """
    if benchmark != Benchmark.GPT_OSS_120B:
        return

    tiktoken_dir = BUILD_DIR / "gpt-oss-tiktoken"
    tiktoken_dir.mkdir(parents=True, exist_ok=True)

    tiktoken_files = {
        "o200k_base.tiktoken": "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
        "cl100k_base.tiktoken": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
    }

    for fname, url in tiktoken_files.items():
        fpath = tiktoken_dir / fname
        if fpath.exists():
            logging.info(f"Tiktoken file already present: {fpath}")
            continue
        logging.info(f"Downloading {fname} to {fpath}...")
        try:
            urllib.request.urlretrieve(url, fpath)
            logging.info(f"Downloaded {fname}")
        except Exception as e:
            logging.warning(f"Failed to download {fname}: {e}")

    # Set env vars so trtllm-serve can find the tiktoken files
    os.environ["TIKTOKEN_CACHE_DIR"] = str(tiktoken_dir)
    os.environ["TIKTOKEN_ENCODINGS_BASE"] = str(tiktoken_dir)
    logging.info(f"Set TIKTOKEN_CACHE_DIR and TIKTOKEN_ENCODINGS_BASE to {tiktoken_dir}")


@autoconfigure
@bind(llm_fields.triton_num_models_per_server, "num_models_per_server")
class GenerateTritonConfigOp(Operation):
    def __init__(self, overwrite=True, num_models_per_server=1):
        """
        Args:
            overwrite (bool): Skip generation if repo exists, else overwrite
            num_models_per_server (int): Number of models to load on each Triton server
        """
        self.system_id = DETECTED_SYSTEM.extras['id']
        self.overwrite = overwrite
        self.workload_name = EngineIndex().wl.benchmark.valstr.lower()
        self.model_version = '1'
        self.num_models_per_server = num_models_per_server
        self.scenario = EngineIndex().wl.scenario.valstr.lower()

        harness_config = TritonHarnessConfig()
        self.trtllm_runtime_flags = harness_config.runtime_flags
        self.tp_size = self.trtllm_runtime_flags['tensor_parallelism']
        self.pp_size = self.trtllm_runtime_flags['pipeline_parallelism']

        self.generation_config = harness_config.gen_config
        self.decoupled = self.generation_config.streaming and self.scenario != 'offline'

        gpus = EngineIndex().wl.system.accelerators[GPU]
        logging.info("Accelerators: ")
        for gpu in gpus:
            logging.info(gpu.pretty_string())
        self.num_gpus = len(gpus)

        self.num_gpus_per_model = self.tp_size * self.pp_size
        num_models = self.num_gpus // self.num_gpus_per_model
        self.num_servers = num_models // self.num_models_per_server

        self.model_store_path_prefix = f"/work/build/triton_model_repos/{self.system_id}/{self.workload_name}/{self.scenario}/repo"

    def run(self, scratch_space, dependency_outputs):
        engine_dir = dependency_outputs[TRTLLMBuilderOp]["engine_dir"]
        assert engine_dir.exists(), f"Engine directory not found at: {engine_dir}"
        assert (engine_dir / "rank0.engine").exists(), "Please specify valid --engine_dir in RUN_ARGS, no engine found at {engine_dir}"

        for repo_idx in range(self.num_servers):
            model_path_str = f"{self.model_store_path_prefix}_{repo_idx}"
            model_repo_path = Path(model_path_str)
            if model_repo_path.exists():
                if not self.overwrite:
                    logging.info(f"Directory {model_path_str} exists, skipping regeneration")
                    continue
                logging.info(f"Directory {model_path_str} already exists, this will be overwritten")
                shutil.rmtree(model_path_str)
            else:
                logging.info(f"Creating {model_path_str}")

            triton_model_name_prefix = "model"
            for m_idx in range(self.num_models_per_server):
                model_idx = m_idx + (repo_idx * self.num_models_per_server)
                triton_model_name = f"{triton_model_name_prefix}-{str(model_idx)}"

                gpu_start_idx = model_idx * self.num_gpus_per_model
                gpu_start_idx %= self.num_gpus // self.num_servers
                gpu_idcs = list(range(gpu_start_idx, gpu_start_idx + self.num_gpus_per_model))
                gpu_idcs = list(map(str, gpu_idcs))
                gpu_idcs = ','.join(gpu_idcs)
                model_dir = model_repo_path.joinpath(triton_model_name, self.model_version)
                model_dir.mkdir(parents=True, exist_ok=False)
                config_file_path = model_repo_path.joinpath(triton_model_name, "config.pbtxt")

                logging.info(f"\tUsing TRTLLM engine at {engine_dir}")

                engine_file_name = str(engine_dir)

                with config_file_path.open(mode='w', encoding='utf-8') as f:
                    f.write(G_TRITON_BASE_CONFIG.format(
                        model_name=triton_model_name,
                        is_decoupled=self.decoupled,
                        beam_width=self.generation_config.runtime_beam_width,
                        engine_path=engine_file_name,
                        gpu_device_idx=gpu_idcs,
                        enable_chunked_context=self.trtllm_runtime_flags['enable_chunked_context'],
                        max_num_tokens=self.trtllm_runtime_flags['max_num_tokens']))
            logging.info(f"Generated triton repository at {model_repo_path}")

        return {"triton_server_repos_path": model_repo_path.parent, "num_gpus_per_model": self.num_gpus_per_model}

    @classmethod
    def output_keys(cls):
        return ["triton_server_repos_path", "num_gpus_per_model"]

    @classmethod
    def immediate_dependencies(cls):
        return {TRTLLMBuilderOp}


@autoconfigure
@bind(general_fields.log_dir)
class RunTritonServerOp(Operation):
    """
        Operation to run tritonserver instance(s)
        - Uses triton's `launch_triton_server.py` script at tensorrtllm_backend/scripts/
        - Will depend on GenerateTritonConfigOp. Will start a separate tritonserver instance for each repo, with CUDA_VISIBLE_DEVICES set accordingly
        - Inside each repo, there may be one or more models - all these are exposed via the same tritonserver instance.
        - Returns the list of tritonserver URLs.
    """

    SCRIPT_PATH = Path("/work/build/triton-inference-server/out/tensorrtllm/scripts/launch_triton_server.py")
    TRITON_SERVER_PATH = Path("/opt/tritonserver/bin/tritonserver")

    def __init__(self, log_dir: Path):
        super().__init__()
        self.server_repos_path = None
        self.num_gpus_per_model = None
        self.log_dir = log_dir
        self.harness_config = TritonHarnessConfig()

    def run(self, scratch_space, dependency_outputs):
        self.server_repos_path = Path(dependency_outputs[GenerateTritonConfigOp]["triton_server_repos_path"])
        self.num_gpus_per_model = dependency_outputs[GenerateTritonConfigOp]["num_gpus_per_model"]

        server_repos = [path for path in self.server_repos_path.iterdir() if path.is_dir()]

        # for each path, run a tritonserver instance like:
        # CUDA_VISIBLE_DEVICES=.. python3 /work/build/triton-inference-server/out/tensorrtllm/scripts/launch_triton_server.py \
        # --tritonserver=/opt/tritonserver/bin/tritonserver \
        # --model_repo=<repo> --tensorrt_llm_model_name=<models> --world_size=<tp*pp>
        # --grpc_port=<> --http_port=<> --metrics_port=<>
        grpc_urls = []
        server_idx = 0
        for server_repo in server_repos:
            model_names = [path.name for path in server_repo.iterdir() if path.is_dir()]
            num_models = len(model_names)
            num_gpus_per_server = num_models * self.num_gpus_per_model
            model_names = ','.join(model_names)

            gpu_ids = list(range(server_idx * num_gpus_per_server, (server_idx + 1) * num_gpus_per_server))
            gpu_ids = ','.join(map(str, gpu_ids))

            # we do 1 core per triton server endpoint, and only support DP
            grpc_url = self.harness_config.get_endpoint_url_for_dp(server_idx)

            # Extract ports from the URL
            grpc_port = grpc_url.split(':')[-1]
            http_port = 8080 + server_idx
            metrics_port = 8888 + server_idx

            cmd = ['python3', str(self.SCRIPT_PATH),
                   '--tritonserver', str(self.TRITON_SERVER_PATH),
                   '--model_repo', str(server_repo),
                   '--tensorrt_llm_model_name', str(model_names),
                   '--world_size', str(self.num_gpus_per_model),
                   '--grpc_port', str(grpc_port),
                   '--http_port', str(http_port),
                   '--metrics_port', str(metrics_port),
                   '--log-file', str(self.log_dir / f'tritonserver_log_{server_idx}.log'),
                   '--log',  # weirdly the script needs this to log into file
                   '--force']
            grpc_urls.append(grpc_url)
            logging.info(f"Starting tritonserver instance {server_idx} with command:\n\tCUDA_VISIBLE_DEVICES={gpu_ids} {' '.join(cmd)}")
            subprocess.Popen(cmd, env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu_ids}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            server_idx += 1

        return {"triton_server_urls": grpc_urls}

    @classmethod
    def output_keys(cls):
        return ["triton_server_urls"]

    @classmethod
    def immediate_dependencies(cls):
        return {GenerateTritonConfigOp}


@autoconfigure
@bind(llm_fields.server_in_foreground)
@bind(llm_fields.nsys_options, "nsys_config_file")
@bind(general_fields.log_dir)
@bind(Workload.FIELD, "workload")
@bind(llm_fields.trtllm_yml_override)
@bind(loadgen_fields.test_mode)
class RunTrtllmServeOp(Operation):
    """ Operation to run trtllm-serve endpoint(s) using trtllm-serve cli """

    def __init__(self,
                 workload: Workload,
                 server_in_foreground: bool = False,
                 nsys_config_file: Path = None,
                 log_dir: Path = None,
                 trtllm_yml_override: Path = None,
                 test_mode: str = None):
        super().__init__()
        self.log_dir = log_dir
        self.wl = workload
        self.blocking = server_in_foreground
        self.nsys_options = None
        self.nsys_cmd_parts = None
        self.trtllm_yml_override = trtllm_yml_override
        self.test_mode = test_mode

        if nsys_config_file is not None:
            with open(nsys_config_file, 'r') as f:
                nsys_options = yaml.safe_load(f)
            self.nsys_cmd_parts = [
                f"{nsys_options['nsys_path']}", "profile",
            ]
            self.nsys_cmd_parts = self.nsys_cmd_parts + nsys_options['extra_flags']
            self.nsys_options = nsys_options

        # Merge user flags with defaults
        self.harness_config = TrtllmEndpointConfig()
        self.trtllm_build_flags = self.harness_config.build_flags
        self.trtllm_runtime_flags = self.harness_config.runtime_flags
        self.trtllm_checkpoint_flags = self.harness_config.checkpoint_flags

        # Only generate YAML if no override provided
        if not trtllm_yml_override:
            self.extra_config_yaml_contents = self.harness_config.extra_config_yaml
        else:
            self.extra_config_yaml_contents = None  # Will read from file

        assert self.harness_config.core_type == harness_fields.CoreType.TRTLLM_ENDPOINT

        if self.harness_config.capture_server_logs_dir is not None:
            self.log_dir = self.harness_config.capture_server_logs_dir
        else:
            self.log_dir = self.harness_config.log_dir

        # Get GPU devices
        gpus = DETECTED_SYSTEM.accelerators[GPU]
        self.devices = [gpu.gpu_index for gpu in gpus]

    def run(self, scratch_space, dependency_outputs):
        # WAR: Setup tiktoken for gpt-oss-120b (openai_harmony needs these files)
        setup_tiktoken_for_gpt_oss(self.wl.benchmark)

        # 1. Get checkpoint / engine path
        if self.harness_config.runtime_flags['trtllm_backend'] == 'pytorch':
            target_path = dependency_outputs[HFQuantizerOp]["quantized_checkpoint_path"]
            assert Path(target_path).exists(), f"Checkpoint path {target_path} does not exist."
        else:
            target_path = dependency_outputs[TRTLLMBuilderOp]["engine_dir"]
            assert Path(target_path).exists(), f"Engine directory {target_path} does not exist."

        # 2. Determine tokenizer path
        if self.harness_config.server_use_hf_tokenizer:
            logging.info("Using HuggingFace tokenizer")
            self.tokenizer_path = None
        elif self.harness_config.runtime_flags['trtllm_backend'] == 'pytorch':
            self.tokenizer_path = target_path
        else:
            quantized_checkpoint_path = dependency_outputs.get(TRTLLMQuantizerOp, {}).get("quantized_checkpoint_path")
            if quantized_checkpoint_path and Path(quantized_checkpoint_path).exists():
                self.tokenizer_path = str(quantized_checkpoint_path)
            else:
                logging.info("Using HuggingFace tokenizer")
                self.tokenizer_path = None

        # 3. Calculate number of trtllm-serve commands to launch on this node
        gpus_per_server = self.harness_config.get_instance_size()
        launch_endpoints = self.harness_config.trtllm_endpoint_urls

        # 4. Determine config YAML path - always in log_dir
        extra_config_path = Path(self.log_dir) / "trtllm_serve_extra_conf.yaml"

        # Use override YAML if provided (copy to log_dir), otherwise generate
        if self.trtllm_yml_override:
            override_source = Path(self.trtllm_yml_override)
            assert override_source.exists(), f"trtllm_yml_override file not found: {override_source}"
            shutil.copy2(override_source, extra_config_path)
            logging.info(f"Copied override YAML: \nfrom: {override_source} \nto: {extra_config_path}")
            # Apply accuracy overrides if test_mode=AccuracyOnly and _accuracy.yml exists
            _apply_yml_accuracy_override(override_source, extra_config_path, self.test_mode)
        else:
            # Create extra args yaml file from generated config
            with extra_config_path.open('w') as f:
                f.write(self.extra_config_yaml_contents)
            logging.info(f"Generated extra config YAML at {extra_config_path}")

        # Read and log YAML contents (works for both override and generated)
        with extra_config_path.open('r') as f:
            yaml_contents = f.read()
        source = f"override from {self.trtllm_yml_override}" if self.trtllm_yml_override else "generated"
        logging.info(f"Extra Config YAML Contents ({source}):\n{yaml_contents}")

        # 4. Launch trtllm-serve processes
        server_processes = []
        mpi_rank = int(os.getenv('SLURM_PROCID', 0))
        for index in range(len(launch_endpoints)):
            nsys_cmd_parts_current = None
            if self.nsys_cmd_parts is not None:
                dp_rank = os.getenv("DP_RANK", index)
                nsys_cmd_parts_current = self.nsys_cmd_parts + [
                    f"--output={self.log_dir}/{self.nsys_options['profile_name']}-dp{dp_rank}-rank{mpi_rank}",
                ]
            endpoint_url = launch_endpoints[index]
            endpoint_port = endpoint_url.split(':')[-1]
            env = os.environ.copy()

            cmd = []
            if self.harness_config.mpi_mode == MPIMode.LEADER:
                # Assert DP=1 in leader mode by checking num_gpus from system name, SLURM world size, and TP*PP are equal
                system_name = DETECTED_SYSTEM.extras["id"]
                num_gpus_from_system = int(system_name.split('x')[-1])
                config_instance_size = self.harness_config.get_instance_size()
                assert num_gpus_from_system == self.harness_config.global_size == config_instance_size, \
                    (
                        f"DP must be 1 in leader mode: num_gpus_from_system={num_gpus_from_system}, "
                        f"global_size={self.harness_config.global_size}, config_instance_size={config_instance_size}.\n"
                        f"System name: {system_name}\n"
                        "Please ensure SYSTEM_NAME matches the total GPU count and SLURM_NTASKS equals config_instance_size."
                    )

                cmd = ['trtllm-llmapi-launch', 'trtllm-serve']
                local_rank_id = os.getenv('SLURM_LOCALID', '0')
                gpu_ids = [local_rank_id]
                # env['CUDA_VISIBLE_DEVICES'] = local_rank_id
                # NOTE: Rank can now see all GPUs that its multi-rank srun step is assigned to via NVIDIA_VISIBLE_DEVICES
                # This is known to cause issue with nsys profiling. Please uncomment line above to enable nsys profiling

            else:
                # Legacy mode: Calculate GPU assignment
                cmd = ['trtllm-serve']
                start_gpu = index * gpus_per_server
                gpu_ids = list(range(start_gpu, start_gpu + gpus_per_server))
                env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))

                # Clean environment: remove SLURM variables to avoid MPI initialization issues in legacy mode
                slurm_vars_removed = []
                for key in list(env.keys()):
                    if key.startswith('SLURM_'):
                        del env[key]
                        slurm_vars_removed.append(key)
                if slurm_vars_removed and index == 0:  # Log only once for the first endpoint
                    logging.info(f"Removed {len(slurm_vars_removed)} SLURM environment variables for running pseudo-MPI programs within single task srun")

            cmd.extend([
                str(target_path),
                '--host', '0.0.0.0',
                '--port', str(endpoint_port),
                '--extra_llm_api_options', str(extra_config_path.absolute())
            ])

            # Add optional arguments only if they are not None
            optional_args = {
                '--num_postprocess_workers': self.trtllm_runtime_flags['num_postprocess_workers'],
                '--tp_size': self.harness_config.tensor_parallelism,
                '--pp_size': self.harness_config.pipeline_parallelism,
                '--ep_size': self.harness_config.moe_expert_parallelism,
                '--max_num_tokens': self.trtllm_runtime_flags['max_num_tokens'],
                '--max_batch_size': self.trtllm_runtime_flags['max_batch_size'],
                '--max_seq_len': self.trtllm_build_flags['max_seq_len'],
                '--max_beam_width': self.trtllm_build_flags['max_beam_width'],
                '--tokenizer': self.tokenizer_path,
            }

            if self.trtllm_runtime_flags['trtllm_backend'] == 'pytorch':
                optional_args |= {'--backend': self.trtllm_runtime_flags['trtllm_backend']}

            for arg_name, arg_value in optional_args.items():
                if arg_value is not None:
                    cmd.extend([arg_name, str(arg_value)])

            if 'disable_gc' in self.trtllm_runtime_flags and self.trtllm_runtime_flags['disable_gc']:
                cmd.append('--disable_gc')

            if self.harness_config.mpi_mode == MPIMode.LEADER:
                # In leader mode, include MPI rank to avoid cluttering a single file
                mpi_rank = int(os.getenv('SLURM_PROCID', 0))
                dp_rank = os.getenv('DP_RANK', None)
                if dp_rank is not None:
                    dp_rank = int(dp_rank)
                    log_file = self.log_dir / f'trtllm_serve_dp{dp_rank}_rank{mpi_rank}.log'
                else:
                    log_file = self.log_dir / f'trtllm_serve_dp{index}_rank{mpi_rank}.log'
            else:
                log_file = self.log_dir / f'trtllm_serve_{index}.log'
            if self.nsys_options and nsys_cmd_parts_current:
                assert "TLLM_PROFILE_START_STOP" in env, "TLLM_PROFILE_START_STOP must be set in the environment when using nsys"
                cmd = nsys_cmd_parts_current + cmd

            with open(log_file, 'w') as f:
                f.write(f"Launch ENV:\n{env}\n\n")
                f.write(f"Launch CMD:\n{' '.join(cmd)}\n\n")
                # Write YAML contents to log (read from file - works for both override and generated)
                with open(extra_config_path, 'r') as yaml_f:
                    yaml_content = yaml_f.read()
                source = f"override from {self.trtllm_yml_override}" if self.trtllm_yml_override else "generated"
                f.write(f"Extra Config ({source}):\n{yaml_content}\n\n")
                server_processes.append(subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT
                ))

            logging.info(f"Launched {endpoint_url}")
            logging.info(f"  CMD: {' '.join(cmd)}")
            logging.info(f"  GPU devices: {gpu_ids}")
            logging.info(f"  Log file: {log_file}")

        if self.blocking:
            for process in server_processes:
                process.wait()

        return {"trtllm_endpoint_urls": launch_endpoints}

    @classmethod
    def output_keys(cls):
        return ["trtllm_endpoint_urls"]

    @classmethod
    def immediate_dependencies(cls):
        if TrtllmEndpointConfig().runtime_flags['trtllm_backend'] == 'pytorch':
            return {HFQuantizerOp}
        else:
            return {TRTLLMBuilderOp}


@bind(general_fields.log_dir)
@autoconfigure
class GenerateTrtllmDisaggConfigOp(Operation):
    def __init__(self, log_dir: Path):
        super().__init__()
        self.log_dir = log_dir

        self.harness_config = TrtllmDisaggEndpointConfig()
        self.trtllm_build_flags = self.harness_config.build_flags
        self.trtllm_runtime_flags = self.harness_config.runtime_flags
        self.trtllm_checkpoint_flags = self.harness_config.checkpoint_flags
        self.extra_config_yaml_contents = self.harness_config.extra_config_yaml
        assert self.harness_config.core_type == harness_fields.CoreType.TRTLLM_DISAGG

    def run(self, scratch_space, dependency_outputs):
        assert self.trtllm_runtime_flags['trtllm_backend'] == 'pytorch', "Can only be used with pytorch backend"
        legacy_mode = (self.harness_config.mpi_mode == MPIMode.LEGACY)

        target_path = dependency_outputs[HFQuantizerOp]["quantized_checkpoint_path"]
        assert Path(target_path).exists(), f"Checkpoint path {target_path} does not exist."

        if self.harness_config.disagg_config_path is not None:
            disagg_config_file_path = Path(self.harness_config.disagg_config_path)
        else:
            disagg_config_file_path = Path(self.log_dir) / "trtllm_serve_disagg_config.yaml"

        if disagg_config_file_path.exists():
            logging.warning(f"Disagg config file {disagg_config_file_path} already exists, overwriting it.")

        gen_config_script_path = TRTLLM_DIR / "docs/source/scripts/disaggregated/gen_yaml.py"
        assert gen_config_script_path.exists(), f"Disagg config script {gen_config_script_path} does not exist"

        logging.info(f"Generating disagg config yaml file at {disagg_config_file_path}")
        gen_script_args = {
            "config": disagg_config_file_path,
            "model": target_path,
            "num_ctx_servers": self.trtllm_runtime_flags['num_ctx_servers'],
            "ctx_tp_size": self.trtllm_runtime_flags['ctx_tp_size'],
            "ctx_batch_size": self.trtllm_runtime_flags['ctx_batch_size'],
            "ctx_max_num_tokens": self.trtllm_runtime_flags['ctx_max_num_tokens'],
            "ctx_enable_attention_dp": self.trtllm_runtime_flags['ctx_enable_attention_dp'],
            "num_gen_servers": self.trtllm_runtime_flags['num_gen_servers'],
            "gen_tp_size": self.trtllm_runtime_flags['gen_tp_size'],
            "gen_batch_size": self.trtllm_runtime_flags['gen_batch_size'],
            "gen_max_num_tokens": self.trtllm_runtime_flags['gen_max_num_tokens'],
            "gen_enable_attention_dp": self.trtllm_runtime_flags['gen_enable_attention_dp'],
            "gen_gpu_memory_fraction": self.trtllm_runtime_flags['gen_gpu_memory_fraction'],
            "worker_start_port": self.trtllm_runtime_flags['worker_start_port'],
            "server_port": self.trtllm_runtime_flags['server_port'],
            "nsys_on": self.trtllm_runtime_flags['nsys_on'],
        }
        if legacy_mode:
            # Get GPU devices to determine num_local_gpus
            gpus = DETECTED_SYSTEM.accelerators[GPU]
            num_local_gpus = len(gpus)

            # Use mpirun when in orchestrator mode
            gen_config_cmd = [
                "mpirun",
                "-n", str(num_local_gpus),
                "python3",
                str(gen_config_script_path),
                *[f"--{key}={value}" for key, value in gen_script_args.items() if value],
            ]
        else:
            # Use direct python3 when in leader mode
            gen_config_cmd = [
                "python3",
                str(gen_config_script_path),
                *[f"--{key}={value}" for key, value in gen_script_args.items() if value],
            ]

        # create custom env
        custom_env = os.environ.copy()
        custom_env['TRTLLM_ENABLE_PDL'] = str(int(self.trtllm_runtime_flags['enable_pdl'] == 1))

        if os.environ.get('SLURM_JOB_NODELIST') is None:
            custom_env['SLURM_JOB_NODELIST'] = 'localhost'
            custom_env['SLURM_TASKS_PER_NODE'] = str(int(self.harness_config.global_size))

        if legacy_mode:
            logging.info(f"Generating disagg config yaml file with mpirun (legacy mode) using {num_local_gpus} GPUs:\n{' '.join(gen_config_cmd)}")
        else:
            logging.info(f"Generating disagg config yaml file with command (MPI mode):\n{' '.join(gen_config_cmd)}")
        subprocess.run(gen_config_cmd, check=True, env=custom_env)

        logging.info(f"Disagg config YAML file generated to: {disagg_config_file_path}")
        return {"disagg_config_file_path": disagg_config_file_path}

    @classmethod
    def output_keys(cls):
        return ["disagg_config_file_path"]

    @classmethod
    def immediate_dependencies(cls):
        return {HFQuantizerOp}


@autoconfigure
@bind(general_fields.log_dir)
class RunTrtllmServeDisaggOp(Operation):
    def __init__(self, log_dir: Path):
        super().__init__()
        self.log_dir = log_dir

        # Merge user flags with defaults
        self.harness_config = TrtllmDisaggEndpointConfig()
        self.trtllm_build_flags = self.harness_config.build_flags
        self.trtllm_runtime_flags = self.harness_config.runtime_flags
        self.trtllm_checkpoint_flags = self.harness_config.checkpoint_flags
        self.extra_config_yaml_contents = self.harness_config.extra_config_yaml
        assert self.harness_config.core_type == harness_fields.CoreType.TRTLLM_DISAGG

        # Get GPU devices
        gpus = DETECTED_SYSTEM.accelerators[GPU]
        self.devices = [gpu.gpu_index for gpu in gpus]

    def run(self, scratch_space, dependency_outputs):
        assert self.trtllm_runtime_flags['trtllm_backend'] == 'pytorch', "Can only be used with pytorch backend"
        legacy_mode = (self.harness_config.mpi_mode == MPIMode.LEGACY)

        # we expect disagg config to be specified in run_llm_server leader mode
        self.disagg_config_file_path = self.harness_config.disagg_config_path
        assert Path(self.disagg_config_file_path).exists(), f"Disagg config file {self.disagg_config_file_path} does not exist"

        if legacy_mode:
            self._launch_legacy_mode()
        else:
            self._launch_leader_mode()

    def _launch_legacy_mode(self):
        raise NotImplementedError("Cannot run disaggregated server in legacy mode. Please use leader mode instead.")

    def _launch_leader_mode(self):
        # TODO(vir): make this a field ?
        launch_type = os.getenv("MLPERF_DISAGG_LAUNCH_TYPE", "worker")
        assert launch_type in ["worker", "leader"], f"Invalid launch type: {launch_type}"

        if launch_type == "leader":
            # launch trtllm-disagg leader process
            leader_log_file_path = self.log_dir / f"leader_log__{os.getenv('HOSTNAME', 'unknown')}_{os.getpid()}.log"

            # Redirect stdout/stderr to log file before exec
            logging.info(f"Starting disagg leader process inline")
            logging.info(f"Log file: {leader_log_file_path}")

            # Open log file and write initial info
            with open(leader_log_file_path, 'a') as log_file:
                log_file.write(f"Launch CMD: trtllm-serve disaggregated -c {self.disagg_config_file_path} -t 1800 -r 1800\n\n")
                log_file.flush()

                # Redirect stdout and stderr to log file
                os.dup2(log_file.fileno(), sys.stdout.fileno())
                os.dup2(log_file.fileno(), sys.stderr.fileno())

            # Replace current process with leader command
            # This preserves the MPI environment
            # TODO(vir): fix hardcoded path
            os.execv("/work/.llm_x86_64/bin/trtllm-serve", [
                "trtllm-serve",
                "disaggregated",
                "--config", str(self.disagg_config_file_path),

                # TODO(vir): change defaults if needed
                "--server_start_timeout", "1800",
                "--request_timeout", "1800"
            ])

        else:  # worker processes
            # launch trtllm-disagg worker processes
            worker_script_path = TRTLLM_DIR / "docs/source/scripts/disaggregated/start_worker.sh"
            worker_log_file_path = self.log_dir / f"worker_log__{os.getenv('HOSTNAME', 'unknown')}_{os.getpid()}.log"

            # Redirect stdout/stderr to log file before exec
            logging.info(f"Starting disagg worker process inline")
            logging.info(f"Log file: {worker_log_file_path}")

            # Open log file and write initial info
            with open(worker_log_file_path, 'a') as log_file:
                log_file.write(f"Launch CMD: bash {worker_script_path} {self.disagg_config_file_path} {self.trtllm_runtime_flags['enable_pdl']}\n\n")
                log_file.flush()

                # Redirect stdout and stderr to log file
                os.dup2(log_file.fileno(), sys.stdout.fileno())
                os.dup2(log_file.fileno(), sys.stderr.fileno())

            # Replace current process with the worker command
            # This preserves the MPI environment
            os.execv("/bin/bash", [
                "bash",
                str(worker_script_path),
                str(self.disagg_config_file_path),
                str(self.trtllm_runtime_flags['enable_pdl'])
            ])

    @classmethod
    def immediate_dependencies(cls):
        return {}


@autoconfigure
@bind(llm_fields.server_in_foreground)
@bind(general_fields.log_dir)
@bind(llm_fields.dynamo_frontend_port)
@bind(llm_fields.dynamo_router_mode)
@bind(llm_fields.dynamo_kv_overlap_weight)
@bind(llm_fields.dynamo_router_replica_sync)
@bind(llm_fields.dynamo_frontend_host)
@bind(llm_fields.dynamo_frontend_secondary)
class RunDisaggFrontendOp(Operation):
    """Operation to launch disaggregated serving frontend (NATS, etcd, router).

    This operation starts the required infrastructure services for disaggregated serving:
    - NATS server (port 4222) for messaging (primary only)
    - etcd (port 2379) for service discovery (primary only)
    - Dynamo frontend (configurable port) for request routing

    For multiple frontends launched via srun --ntasks=N:
    - Primary frontend (rank 0): starts NATS + etcd + router
    - Secondary frontends (rank > 0): only start router, connect to primary's NATS/etcd

    Rank is auto-detected from SLURM_PROCID environment variable.
    Port is computed as: base_port + rank

    Used with: --core_type=disagg_frontend
    """

    # Standard ports
    ETCD_PORT = 2379
    NATS_PORT = 4222
    DEFAULT_FRONTEND_PORT = 8000

    def __init__(self,
                 server_in_foreground: bool = True,
                 log_dir: Path = None,
                 dynamo_frontend_port: int = None,
                 dynamo_router_mode: str = None,
                 dynamo_kv_overlap_weight: float = None,
                 dynamo_router_replica_sync: bool = False,
                 dynamo_frontend_host: str = None,
                 dynamo_frontend_secondary: bool = False):
        super().__init__()
        self.blocking = server_in_foreground
        self.log_dir = log_dir
        self.base_frontend_port = dynamo_frontend_port or self.DEFAULT_FRONTEND_PORT
        self.router_mode = dynamo_router_mode or "round-robin"
        self.kv_overlap_weight = dynamo_kv_overlap_weight
        self.router_replica_sync = dynamo_router_replica_sync
        self.primary_host = dynamo_frontend_host  # For secondary frontends to find primary
        self.is_secondary = dynamo_frontend_secondary  # Explicit secondary frontend flag

    def run(self, scratch_space, dependency_outputs):
        # Detect rank from SLURM environment (must read before cleaning env vars)
        global_rank = int(os.environ.get('SLURM_PROCID', 0))
        local_rank = int(os.environ.get('SLURM_LOCALID', 0))
        num_tasks = int(os.environ.get('SLURM_NTASKS', 1))
        # Primary if global rank 0 AND not explicitly marked as secondary
        # (needed when launching separate srun commands per port)
        is_primary = (global_rank == 0) and not self.is_secondary

        # Compute port from LOCAL rank: base_port + local_rank
        # This ensures each node uses the same port in distributed mode (ntasks-per-node=1)
        # while stacked mode (multiple frontends on same node) gets unique ports
        frontend_port = self.base_frontend_port + local_rank

        # Set up log directory (per-rank if multiple frontends)
        log_dir = Path(self.log_dir)
        if num_tasks > 1:
            log_dir = log_dir / f'fe_{global_rank}'
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"[Rank {global_rank}/{num_tasks}] Frontend starting on port {frontend_port}")

        processes = []

        if is_primary:
            # Primary frontend (rank 0): start NATS and etcd
            logging.info(f"[Rank {global_rank}] Starting PRIMARY frontend (NATS + etcd + router)")

            # Start NATS server
            nats_log = log_dir / "nats_server.log"
            nats_cmd = ["nats-server", "-js"]
            logging.info(f"Starting NATS server on port {self.NATS_PORT}")
            logging.info(f"  CMD: {' '.join(nats_cmd)}")
            with open(nats_log, 'w') as f:
                nats_proc = subprocess.Popen(
                    nats_cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT
                )
                processes.append(nats_proc)
            logging.info(f"NATS server started, PID: {nats_proc.pid}, log: {nats_log}")

            # Start etcd
            etcd_log = log_dir / "etcd.log"
            etcd_data_dir = log_dir / "etcd_data"
            etcd_data_dir.mkdir(parents=True, exist_ok=True)
            etcd_cmd = [
                "etcd",
                "--listen-client-urls", f"http://0.0.0.0:{self.ETCD_PORT}",
                "--advertise-client-urls", f"http://0.0.0.0:{self.ETCD_PORT}",
                "--data-dir", str(etcd_data_dir)
            ]
            logging.info(f"Starting etcd on port {self.ETCD_PORT}")
            logging.info(f"  CMD: {' '.join(etcd_cmd)}")
            with open(etcd_log, 'w') as f:
                etcd_proc = subprocess.Popen(
                    etcd_cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT
                )
                processes.append(etcd_proc)
            logging.info(f"etcd started, PID: {etcd_proc.pid}, log: {etcd_log}")

            # Wait for NATS and etcd to initialize
            time.sleep(3)
        else:
            # Secondary frontend (global rank > 0): connect to primary's NATS/etcd
            assert self.primary_host, \
                "dynamo_frontend_host must be specified for multi-frontend launch"
            logging.info(f"[Rank {global_rank}] Starting SECONDARY frontend (router only)")
            logging.info(f"  Connecting to primary NATS/etcd at {self.primary_host}")
            # Wait a bit for primary to start NATS/etcd
            time.sleep(5)

        # Start Dynamo frontend router
        frontend_log = log_dir / "disagg_frontend.log"
        frontend_cmd = [
            "python3", "-m", "dynamo.frontend",
            "--http-port", str(frontend_port),
            "--router-mode", self.router_mode
        ]

        # Add optional router arguments
        if self.kv_overlap_weight is not None:
            frontend_cmd.extend(["--kv-overlap-score-weight", str(self.kv_overlap_weight)])
        if self.router_replica_sync:
            frontend_cmd.append("--router-replica-sync")

        # Clean environment: remove SLURM variables to avoid MPI initialization issues in dynamo.frontend
        clean_env = os.environ.copy()
        slurm_vars_removed = []
        for key in list(clean_env.keys()):
            if key.startswith('SLURM_'):
                del clean_env[key]
                slurm_vars_removed.append(key)
        if slurm_vars_removed:
            logging.info(f"Removed {len(slurm_vars_removed)} SLURM environment variables for running pseudo-MPI programs within single task srun")

        # For secondary frontends, set environment to connect to primary's NATS/etcd
        if not is_primary:
            clean_env["ETCD_ENDPOINTS"] = f"{self.primary_host}:{self.ETCD_PORT}"
            clean_env["NATS_SERVER"] = f"nats://{self.primary_host}:{self.NATS_PORT}"

        frontend_type = "secondary" if not is_primary else "primary"
        logging.info(f"Starting {frontend_type} frontend on port {frontend_port}")
        logging.info(f"  Router mode: {self.router_mode}")
        if self.kv_overlap_weight is not None:
            logging.info(f"  KV overlap weight: {self.kv_overlap_weight}")
        if self.router_replica_sync:
            logging.info(f"  Router replica sync: enabled")
        logging.info(f"  CMD: {' '.join(frontend_cmd)}")

        with open(frontend_log, 'w') as f:
            frontend_proc = subprocess.Popen(
                frontend_cmd,
                env=clean_env,
                stdout=f,
                stderr=subprocess.STDOUT
            )
            processes.append(frontend_proc)
        logging.info(f"Frontend started, PID: {frontend_proc.pid}, log: {frontend_log}")

        frontend_url = f"localhost:{frontend_port}"
        logging.info(f"Disaggregated serving frontend started. URL: {frontend_url}")
        if is_primary:
            logging.info(f"Workers should connect with: --dynamo_frontend_host=<this_node_hostname>")

        if self.blocking:
            try:
                for proc in processes:
                    proc.wait()
            except KeyboardInterrupt:
                logging.info("Stopping frontend services...")
                for proc in processes:
                    proc.terminate()

        return {"frontend_url": frontend_url}

    @classmethod
    def output_keys(cls):
        return ["frontend_url"]

    @classmethod
    def immediate_dependencies(cls):
        return set()


@autoconfigure
@bind(llm_fields.server_in_foreground)
@bind(general_fields.log_dir)
@bind(Workload.FIELD, "workload")
@bind(llm_fields.dynamo_frontend_host)
@bind(harness_fields.mpi_mode)
@bind(llm_fields.trtllm_yml_override)
@bind(llm_fields.env_yml_override)
@bind(loadgen_fields.test_mode)
class RunDisaggPrefillOp(Operation):
    """Operation to launch disaggregated prefill (context) worker.

    This operation starts a TRT-LLM prefill worker that handles prompt processing
    and registers with the disaggregated serving frontend via NATS/etcd.

    IMPORTANT: Must be launched in MPI leader mode via:
        srun --ntasks=<tensor_parallelism> --nodes=<num_nodes> make run_llm_server ...

    Used with: --core_type=disagg_prefill
    """

    # Standard ports
    ETCD_PORT = 2379
    NATS_PORT = 4222

    def __init__(self,
                 workload: Workload,
                 server_in_foreground: bool = True,
                 log_dir: Path = None,
                 dynamo_frontend_host: str = None,
                 mpi_mode: MPIMode = MPIMode.LEGACY,
                 trtllm_yml_override: Path = None,
                 env_yml_override: Path = None,
                 test_mode: str = None):
        super().__init__()
        self.blocking = server_in_foreground
        self.log_dir = log_dir
        self.wl = workload
        self.dynamo_frontend_host = dynamo_frontend_host
        self.mpi_mode = mpi_mode
        self.trtllm_yml_override = trtllm_yml_override
        self.env_yml_override = env_yml_override
        self.test_mode = test_mode
        # Use TrtllmExtraYAMLConfig for YAML generation (if no override)
        self.harness_config = TrtllmExtraYAMLConfig() if not trtllm_yml_override else None

    @property
    def etcd_endpoints(self) -> str:
        """Get etcd endpoint URL derived from dynamo_frontend_host."""
        return f"{self.dynamo_frontend_host}:{self.ETCD_PORT}"

    @property
    def nats_server(self) -> str:
        """Get NATS server URL derived from dynamo_frontend_host."""
        return f"nats://{self.dynamo_frontend_host}:{self.NATS_PORT}"

    def run(self, scratch_space, dependency_outputs):
        assert self.dynamo_frontend_host, \
            "dynamo_frontend_host must be specified for disaggregated prefill workers"

        # Prefill workers MUST be launched in leader mode with srun
        assert self.mpi_mode == MPIMode.LEADER, (
            "Disaggregated prefill workers must be launched in MPI leader mode.\n"
            "Please launch with: srun --ntasks=<tensor_parallelism> --nodes=<num_nodes> make run_llm_server ..."
        )

        # Get model path from HFQuantizerOp
        model_path = dependency_outputs[HFQuantizerOp]["quantized_checkpoint_path"]
        assert Path(model_path).exists(), f"Model path {model_path} does not exist"

        log_dir = Path(self.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Prefill worker - always uses "prefill" disaggregation mode
        disagg_mode = "prefill"
        worker_name = "prefill"

        # Determine config YAML path - always in log_dir
        config_yaml_path = log_dir / f"disagg_{worker_name}_config.yaml"

        # Use override YAML if provided (copy to log_dir), otherwise generate from config
        if self.trtllm_yml_override:
            override_source = Path(self.trtllm_yml_override)
            assert override_source.exists(), f"trtllm_yml_override file not found: {override_source}"
            shutil.copy2(override_source, config_yaml_path)
            logging.info(f"Copied override YAML: \nfrom: {override_source} \nto: {config_yaml_path}")
            # Apply accuracy overrides if test_mode=AccuracyOnly and _accuracy.yml exists
            _apply_yml_accuracy_override(override_source, config_yaml_path, self.test_mode)
        else:
            # Generate worker config YAML using TrtllmExtraYAMLConfig
            config_yaml_content = self.harness_config.extra_config_yaml
            with open(config_yaml_path, 'w') as f:
                f.write(config_yaml_content)
            logging.info(f"Generated {worker_name} worker config at {config_yaml_path}")

        # Get served model name from benchmark module
        model_repo = G_BENCHMARK_MODULES[self.wl.benchmark].load(("HF_MODEL_REPO",)).HF_MODEL_REPO
        served_model_name, _ = list(model_repo.items())[0]

        # Set environment for worker
        # Reference: https://github.com/ai-dynamo/dynamo/blob/main/examples/basics/multinode/trtllm/start_trtllm_worker.sh
        env = os.environ.copy()
        env.update({
            'ETCD_ENDPOINTS': self.etcd_endpoints,
            'NATS_SERVER': self.nats_server,
            'TRTLLM_SERVER_DISABLE_GC': '1',
            'TRTLLM_WORKER_DISABLE_GC': '1',
            'TLLM_LOG_LEVEL': 'INFO',
        })

        # Load and apply custom env vars from YAML file (if provided)
        custom_env = {}
        if self.env_yml_override:
            env_path = Path(self.env_yml_override)
            if env_path.exists():
                with open(env_path, 'r') as f:
                    custom_env = yaml.safe_load(f) or {}
                if isinstance(custom_env, dict):
                    # Convert all values to strings
                    custom_env = {str(k): str(v) for k, v in custom_env.items()}
                    env.update(custom_env)
                    logging.info(f"Applied env from {env_path}: {custom_env}")
                else:
                    logging.warning(f"env_yml_override must contain a dict, got: {type(custom_env)}")
                    custom_env = {}
            else:
                logging.warning(f"env_yml_override file not found: {env_path}")

        # In leader mode, each MPI rank handles one GPU via SLURM_LOCALID
        mpi_rank = int(os.getenv('SLURM_PROCID', 0))
        worker_log = log_dir / f"disagg_{worker_name}_worker_rank{mpi_rank}.log"

        # Build command using trtllm-llmapi-launch
        cmd = [
            "trtllm-llmapi-launch",
            "python3", "-m", "dynamo.trtllm",
            "--model-path", str(model_path),
            "--served-model-name", served_model_name,
            "--extra-engine-args", str(config_yaml_path),
            "--disaggregation-mode", disagg_mode,
        ]

        logging.info(f"Starting disaggregated prefill worker (MPI rank {mpi_rank})")
        logging.info(f"  CMD: {' '.join(cmd)}")
        logging.info(f"  ETCD_ENDPOINTS: {env['ETCD_ENDPOINTS']}")
        logging.info(f"  NATS_SERVER: {env['NATS_SERVER']}")
        logging.info(f"  MODEL_PATH: {model_path}")
        logging.info(f"  SERVED_MODEL_NAME: {served_model_name}")
        logging.info(f"  CONFIG: {config_yaml_path}")
        logging.info(f"  DISAGGREGATION_MODE: {disagg_mode}")

        # Launch worker via subprocess
        with open(worker_log, 'w') as f:
            f.write(f"MPI Rank: {mpi_rank}\n")
            f.write(f"Launch CMD: {' '.join(cmd)}\n\n")
            f.write(f"Environment:\n")
            for k, v in env.items():
                if k.startswith(('ETCD', 'NATS', 'TRTLLM', 'TLLM', 'SLURM', 'CUDA')):
                    f.write(f"  {k}={v}\n")
            # Log custom env vars from env_yml_override
            if custom_env:
                f.write(f"\nCustom env from {self.env_yml_override}:\n")
                for k, v in custom_env.items():
                    f.write(f"  {k}={v}\n")
            # Write YAML config from file (works for both override and generated)
            with open(config_yaml_path, 'r') as yaml_f:
                yaml_content = yaml_f.read()
            source = f"override from {self.trtllm_yml_override}" if self.trtllm_yml_override else "generated"
            f.write(f"\nConfig YAML ({source}):\n{yaml_content}\n\n")
            f.flush()

            worker_proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT
            )

        logging.info(f"Disaggregated prefill worker started, PID: {worker_proc.pid}, log: {worker_log}")

        if self.blocking:
            try:
                worker_proc.wait()
            except KeyboardInterrupt:
                logging.info("Stopping prefill worker...")
                worker_proc.terminate()

        return {"worker_pid": worker_proc.pid}

    @classmethod
    def output_keys(cls):
        return ["worker_pid"]

    @classmethod
    def immediate_dependencies(cls):
        return {HFQuantizerOp}


@autoconfigure
@bind(llm_fields.server_in_foreground)
@bind(general_fields.log_dir)
@bind(Workload.FIELD, "workload")
@bind(llm_fields.dynamo_frontend_host)
@bind(harness_fields.mpi_mode)
@bind(llm_fields.trtllm_yml_override)
@bind(llm_fields.env_yml_override)
@bind(loadgen_fields.test_mode)
class RunDisaggDecodeOp(Operation):
    """Operation to launch disaggregated decode (generation) worker.

    This operation starts a TRT-LLM decode worker that handles token generation
    and registers with the disaggregated serving frontend via NATS/etcd.

    IMPORTANT: Must be launched in MPI leader mode via:
        srun --ntasks=<tensor_parallelism> --nodes=<num_nodes> make run_llm_server ...

    Used with: --core_type=disagg_decode
    """

    # Standard ports
    ETCD_PORT = 2379
    NATS_PORT = 4222

    def __init__(self,
                 workload: Workload,
                 server_in_foreground: bool = True,
                 log_dir: Path = None,
                 dynamo_frontend_host: str = None,
                 mpi_mode: MPIMode = MPIMode.LEGACY,
                 trtllm_yml_override: Path = None,
                 env_yml_override: Path = None,
                 test_mode: str = None):
        super().__init__()
        self.blocking = server_in_foreground
        self.log_dir = log_dir
        self.wl = workload
        self.dynamo_frontend_host = dynamo_frontend_host
        self.mpi_mode = mpi_mode
        self.trtllm_yml_override = trtllm_yml_override
        self.env_yml_override = env_yml_override
        self.test_mode = test_mode
        # Use TrtllmExtraYAMLConfig for YAML generation (if no override)
        self.harness_config = TrtllmExtraYAMLConfig() if not trtllm_yml_override else None

    @property
    def etcd_endpoints(self) -> str:
        """Get etcd endpoint URL derived from dynamo_frontend_host."""
        return f"{self.dynamo_frontend_host}:{self.ETCD_PORT}"

    @property
    def nats_server(self) -> str:
        """Get NATS server URL derived from dynamo_frontend_host."""
        return f"nats://{self.dynamo_frontend_host}:{self.NATS_PORT}"

    def run(self, scratch_space, dependency_outputs):
        assert self.dynamo_frontend_host, \
            "dynamo_frontend_host must be specified for disaggregated decode workers"

        # Decode workers MUST be launched in leader mode with srun
        assert self.mpi_mode == MPIMode.LEADER, (
            "Disaggregated decode workers must be launched in MPI leader mode.\n"
            "Please launch with: srun --ntasks=<tensor_parallelism> --nodes=<num_nodes> make run_llm_server ..."
        )

        # Get model path from HFQuantizerOp
        model_path = dependency_outputs[HFQuantizerOp]["quantized_checkpoint_path"]
        assert Path(model_path).exists(), f"Model path {model_path} does not exist"

        log_dir = Path(self.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Decode worker - always uses "decode" disaggregation mode
        disagg_mode = "decode"
        worker_name = "decode"

        # Determine config YAML path - always in log_dir
        config_yaml_path = log_dir / f"disagg_{worker_name}_config.yaml"

        # Use override YAML if provided (copy to log_dir), otherwise generate from config
        if self.trtllm_yml_override:
            override_source = Path(self.trtllm_yml_override)
            assert override_source.exists(), f"trtllm_yml_override file not found: {override_source}"
            shutil.copy2(override_source, config_yaml_path)
            logging.info(f"Copied override YAML: \nfrom: {override_source} \nto: {config_yaml_path}")
            # Apply accuracy overrides if test_mode=AccuracyOnly and _accuracy.yml exists
            _apply_yml_accuracy_override(override_source, config_yaml_path, self.test_mode)
        else:
            # Generate worker config YAML using TrtllmExtraYAMLConfig
            config_yaml_content = self.harness_config.extra_config_yaml
            with open(config_yaml_path, 'w') as f:
                f.write(config_yaml_content)
            logging.info(f"Generated {worker_name} worker config at {config_yaml_path}")

        # Get served model name from benchmark module
        model_repo = G_BENCHMARK_MODULES[self.wl.benchmark].load(("HF_MODEL_REPO",)).HF_MODEL_REPO
        served_model_name, _ = list(model_repo.items())[0]

        # Set environment for worker
        # Reference: https://github.com/ai-dynamo/dynamo/blob/main/examples/basics/multinode/trtllm/start_trtllm_worker.sh
        env = os.environ.copy()
        env.update({
            'ETCD_ENDPOINTS': self.etcd_endpoints,
            'NATS_SERVER': self.nats_server,
            'TRTLLM_SERVER_DISABLE_GC': '1',
            'TRTLLM_WORKER_DISABLE_GC': '1',
            'TLLM_LOG_LEVEL': 'INFO',
        })

        # Load and apply custom env vars from YAML file (if provided)
        custom_env = {}
        if self.env_yml_override:
            env_path = Path(self.env_yml_override)
            if env_path.exists():
                with open(env_path, 'r') as f:
                    custom_env = yaml.safe_load(f) or {}
                if isinstance(custom_env, dict):
                    # Convert all values to strings
                    custom_env = {str(k): str(v) for k, v in custom_env.items()}
                    env.update(custom_env)
                    logging.info(f"Applied env from {env_path}: {custom_env}")
                else:
                    logging.warning(f"env_yml_override must contain a dict, got: {type(custom_env)}")
                    custom_env = {}
            else:
                logging.warning(f"env_yml_override file not found: {env_path}")

        # In leader mode, each MPI rank handles one GPU via SLURM_LOCALID
        mpi_rank = int(os.getenv('SLURM_PROCID', 0))
        worker_log = log_dir / f"disagg_{worker_name}_worker_rank{mpi_rank}.log"

        # Build command using trtllm-llmapi-launch
        cmd = [
            "trtllm-llmapi-launch",
            "python3", "-m", "dynamo.trtllm",
            "--model-path", str(model_path),
            "--served-model-name", served_model_name,
            "--extra-engine-args", str(config_yaml_path),
            "--disaggregation-mode", disagg_mode,
        ]

        logging.info(f"Starting disaggregated decode worker (MPI rank {mpi_rank})")
        logging.info(f"  CMD: {' '.join(cmd)}")
        logging.info(f"  ETCD_ENDPOINTS: {env['ETCD_ENDPOINTS']}")
        logging.info(f"  NATS_SERVER: {env['NATS_SERVER']}")
        logging.info(f"  MODEL_PATH: {model_path}")
        logging.info(f"  SERVED_MODEL_NAME: {served_model_name}")
        logging.info(f"  CONFIG: {config_yaml_path}")
        logging.info(f"  DISAGGREGATION_MODE: {disagg_mode}")

        # Launch worker via subprocess
        with open(worker_log, 'w') as f:
            f.write(f"MPI Rank: {mpi_rank}\n")
            f.write(f"Launch CMD: {' '.join(cmd)}\n\n")
            f.write(f"Environment:\n")
            for k, v in env.items():
                if k.startswith(('ETCD', 'NATS', 'TRTLLM', 'TLLM', 'SLURM', 'CUDA')):
                    f.write(f"  {k}={v}\n")
            # Log custom env vars from env_yml_override
            if custom_env:
                f.write(f"\nCustom env from {self.env_yml_override}:\n")
                for k, v in custom_env.items():
                    f.write(f"  {k}={v}\n")
            # Write YAML config from file (works for both override and generated)
            with open(config_yaml_path, 'r') as yaml_f:
                yaml_content = yaml_f.read()
            source = f"override from {self.trtllm_yml_override}" if self.trtllm_yml_override else "generated"
            f.write(f"\nConfig YAML ({source}):\n{yaml_content}\n\n")
            f.flush()

            worker_proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT
            )

        logging.info(f"Disaggregated decode worker started, PID: {worker_proc.pid}, log: {worker_log}")

        if self.blocking:
            try:
                worker_proc.wait()
            except KeyboardInterrupt:
                logging.info("Stopping decode worker...")
                worker_proc.terminate()

        return {"worker_pid": worker_proc.pid}

    @classmethod
    def output_keys(cls):
        return ["worker_pid"]

    @classmethod
    def immediate_dependencies(cls):
        return {HFQuantizerOp}
