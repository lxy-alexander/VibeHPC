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
import subprocess
import logging

import socket
from contextlib import closing
import os
from typing import Optional
from pathlib import Path
import psutil

G_TRITON_SERVER_PATH = "/work/build/triton-inference-server"
G_TRTLLM_BACKEND_PATH = G_TRITON_SERVER_PATH + "/out/tensorrtllm"

G_LAUNCH_SCRIPT_LOCATION = Path(G_TRTLLM_BACKEND_PATH + "/scripts")

# from https://stackoverflow.com/a/45690594/5076583


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class TritonSutBackend:
    """
    The part of the triton SUT that manages spawning and termination of the tritonserver process.
    Example launch command:
        python3 launch_triton_server.py \
            --tritonserver /opt/tritonserver/bin/tritonserver \
            --model_repo build/triton_model_repo_0/ \
            --tensorrt_llm_model_name <model_names> \
            --grpc_port <port> \
            --http_port <port> \
            --metrics_port <port> \
            --world_size <tp*pp> \
            --force
    """

    def __init__(self,
                 binary_exec: str = "/opt/tritonserver/bin/tritonserver",
                 model_repo: str = "/work/build/triton_model_repo",
                 cuda_visible_devices: Optional[str] = None,
                 oversubscribe: bool = False,
                 world_size: int = 1):

        cmd = [
            "python3", f"{G_LAUNCH_SCRIPT_LOCATION}/launch_triton_server.py",
            f"--tritonserver={binary_exec}", f"--model_repo={model_repo}"
        ]

        model_names = os.listdir(model_repo)
        num_models = len(model_names)
        model_names = ','.join(model_names)
        cmd.append(f"--tensorrt_llm_model_name={model_names}")

        if num_models > 1:
            cmd.append("--multi-model")
            world_size = 1  # processes will be spawned automatically

        self.grpc_port = find_free_port()
        self.http_port = find_free_port()
        self.metrics_port = find_free_port()

        cmd.extend([
            f"--grpc_port={self.grpc_port}",
            f"--http_port={self.http_port}",
            f"--metrics_port={self.metrics_port}"
        ])

        cmd.append("--force")
        cmd.append(f"--world_size={world_size}")
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

        logging.info(f"[TritonSutBackend] Executing command: {' '.join(cmd)}")
        subprocess.Popen(cmd, env=env)

    def get_grpc_port(self):
        return self.grpc_port

    def get_http_port(self):
        return self.http_port

    def get_metrics_port(self):
        return self.metrics_port

    def is_ready(self):
        """Checks if the GRPC port is active"""
        for conn in psutil.net_connections(kind='tcp'):
            if conn.laddr.port == self.grpc_port:
                return True
        return False
