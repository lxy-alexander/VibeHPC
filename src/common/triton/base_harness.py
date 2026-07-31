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
from code.common.harness import BaseBenchmarkHarness
from code.common import logging, args_to_string, dict_get
from nvmitten.nvidia.accelerator import GPU
import code.common.arguments as common_args
import os


class TritonLlmHarness(BaseBenchmarkHarness):

    def __init__(self, args, benchmark):
        # trtllm flags
        self.tp_size = args['trtllm_build_flags']['tensor_parallelism']
        self.pp_size = args['trtllm_build_flags']['pipeline_parallelism']
        self.enable_chunked_context = args["trtllm_runtime_flags"]["enable_chunked_context"]
        self.max_num_tokens = args["trtllm_runtime_flags"]["max_num_tokens"]

        self.precision = dict_get(args, "precision", "fp8")
        self.workload_setting = dict_get(args, "workload_setting", None)

        super().__init__(args, benchmark)
        self.model_name = self._get_model_name(args)
        self.model_version = "1"

        custom_args = [
            "num_gpus",
            "use_token_latencies",
            "llm_gen_config_path",
            "gpu_batch_size"
            "trtllm_checkpoint_flags",
            "trtllm_build_flags",
            "trtllm_runtime_flags",
        ]
        self.flag_builder_custom_args = common_args.LOADGEN_ARGS + common_args.SHARED_ARGS + custom_args

        gpus = self.args['system'].accelerators[GPU]
        logging.info("Accelerators: ")
        for gpu in gpus:
            logging.info(gpu.pretty_string())

        self.num_gpus_per_host = len(gpus)
        self.num_gpus_per_model = self.tp_size * self.pp_size
        self.num_clients_per_frontend = args['triton_num_clients_per_frontend']
        self.num_frontends_per_model = args['triton_num_frontends_per_model']
        self.verbose_frontend = dict_get(args, 'triton_verbose_frontend', default=False)
        self.skip_server_spawn = dict_get(args, 'triton_skip_server_spawn', default=False)
        self.num_servers_per_host = dict_get(args, "triton_num_servers", default=1)
        self.num_triton_models = self.num_gpus_per_host // self.num_gpus_per_model

        self.grpc_ports = args["triton_grpc_ports"]

        assert self.num_gpus_per_host > 0
        assert self.num_gpus_per_model > 0
        assert self.num_gpus_per_host % self.num_servers_per_host == 0, f"num_gpus_per_host({self.num_gpus_per_host}) must be a multiple of num_servers_per_host({self.num_servers_per_host})"
        assert self.num_gpus_per_host % self.num_gpus_per_model == 0, f"TP * PP must be a factor of num_gpus_per_host: {self.num_gpus_per_host}"

        self.system_id = args['system_id']

    def _get_engine_fpath(self, device_type, _, batch_size):
        if not self.default_engine_dir:
            tag = ""
        else:
            tag = "{self.name}-{self.scenario.valstr}-{device_type}-b{batch_size}-{self.precision}-tp{self.tp_size}pp{self.pp_size}-{self.workload_setting.shortname()}"
        return f"{self.engine_dir}/{tag}/rank0.engine"

    def _get_model_name(self, config):
        benchmark = config["benchmark"].valstr.lower()
        scenario = config["scenario"].valstr.lower()
        return "{}-{}".format(benchmark, scenario)

    def _get_harness_executable(self):
        return "code/harness/harness_triton_llm/main.py"

    def _construct_terminal_command(self, argstr):
        cmd = f"{self.executable.replace('code/harness/harness_triton_llm/main.py', 'python3 -m code.harness.harness_triton_llm.main')} {argstr}"
        return cmd

    def _get_engine_dirpath(self, device_type, batch_size):
        dirpath = os.path.join(self._get_engine_fpath(device_type, None, batch_size), os.pardir)
        return os.path.abspath(dirpath)

    def _append_config_ver_name(self, system_name):
        system_name += "_Triton"
        return super()._append_config_ver_name(system_name)

    def _build_custom_flags(self, flag_dict):
        def to_cli(value):
            match value:
                case bool() as b: return str(b).lower()
                case _: return str(value)

        # trtllm flags
        flag_dict |= {
            key: ','.join(f"{k}:{to_cli(v)}" for k, v in value.items())
            for key, value in flag_dict.items()
            if key in ['trtllm_checkpoint_flags', 'trtllm_build_flags', 'trtllm_runtime_flags']
        }

        argstr = args_to_string(flag_dict)
        argstr = argstr + " --scenario " + self.scenario.valstr
        argstr = argstr + " --model " + self.name
        argstr = argstr + " --num_gpus_per_host " + str(self.num_gpus_per_host)
        argstr = argstr + " --num_clients_per_frontend " + str(self.num_clients_per_frontend)
        argstr = argstr + " --num_frontends_per_model " + str(self.num_frontends_per_model)
        argstr = argstr + " --num_servers_per_host " + str(self.num_servers_per_host)
        argstr = argstr + " --grpc_ports \"" + str(self.grpc_ports) + "\""
        argstr = argstr + " --system_id " + str(self.system_id)
        if self.verbose_frontend:
            argstr = argstr + " --verbose_frontend"
        if self.skip_server_spawn:
            argstr = argstr + " --skip_server_spawn"
        return argstr

    def run_harness(self, flag_dict=None, skip_generate_measurements=False, use_py_harness=False):
        return super().run_harness(flag_dict, skip_generate_measurements, use_py_harness)
