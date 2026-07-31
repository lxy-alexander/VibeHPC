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

from code.harness.harness_triton_llm.dataset import LlmDataset
from code.harness.harness_triton_llm.frontend import TritonSutGrpcFrontend, TritonSutGrpcStreamFrontend
from code.harness.harness_triton_llm.backend import TritonSutBackend
from code.harness.harness_triton_llm.utils import LlmConfig, get_llm_gen_config, split_string_into_chunks, LoadingBarManager

from code.common.utils import parse_cli_flags
from code.common import dict_get

import argparse
from pathlib import Path
import mlperf_loadgen as lg
import threading
import logging
import multiprocessing as mp
import multiprocessing.connection as mp_conn
from functools import partial
import signal
import os
import psutil
import time

G_DEFAULT_PORTS = {'http': 8000, 'grpc': 8001, 'metrics': 8002}

scenario_map = {
    "Offline": lg.TestScenario.Offline,
    "SingleStream": lg.TestScenario.SingleStream,
    "Server": lg.TestScenario.Server,
}

test_mode_map = {
    "PerformanceOnly": lg.TestMode.PerformanceOnly,
    "AccuracyOnly": lg.TestMode.AccuracyOnly,
    "SubmissionRun": lg.TestMode.SubmissionRun,
}

log_mode_map = {
    "AsyncPoll": lg.LoggingMode.AsyncPoll,
    "EndOfTestOnly": lg.LoggingMode.EndOfTestOnly,
    "Synchronous": lg.LoggingMode.Synchronous,
}


def dummy_flush():
    pass


"""
    Called by LoadGen's SuT when issuing queries
"""


def dispatch_queries_from_loadgen(dataset, frontend_input_queues, load_balancing_counters, query_samples):
    for query_sample in query_samples:
        # get idx of frontend with minimum cumulative ISL
        next_frontend_idx = load_balancing_counters.index(min(load_balancing_counters))

        sample_id = query_sample.id
        sample_input_ids, sample_input_lens, sample_stop_ids = dataset.get_input(query_sample.index)
        frontend_input_queues[next_frontend_idx].send([sample_id, sample_input_ids, sample_input_lens, sample_stop_ids])

        # Update work-load counters
        load_balancing_counters[next_frontend_idx] += sample_input_lens[0, 0]


def frontend_process(frontend_class: type,
                     lg2fe_recv_conn: mp_conn.Connection,
                     triton_model_name: str,
                     frontend_name: str,
                     grpc_url: str,
                     llm_config: LlmConfig,
                     args: argparse.Namespace,
                     ready_signal: mp_conn.Connection,
                     fe2lg_send_conn: mp_conn.Connection,
                     master_proc_pid: int):
    try:
        if args.verbose_frontend:
            logging.info(f"Initializing frontend {frontend_name}, PID: {os.getpid()}")
        frontend = frontend_class(llm_config=llm_config,
                                  verbose=args.verbose_frontend,
                                  triton_model_name=triton_model_name,
                                  frontend_name=frontend_name,
                                  llm_batch_size=args.gpu_batch_size,
                                  num_frontends_per_model=args.num_frontends_per_model,
                                  url=grpc_url,
                                  num_clients=args.num_clients_per_frontend,
                                  report_loadgen_conn=fe2lg_send_conn)
        ready_signal.send(None)

        while True:
            query_sample = lg2fe_recv_conn.recv()
            if query_sample is None:
                break
            frontend.dispatch_query_samples(query_sample)
        frontend.notify_dispatch_done()
    except KeyboardInterrupt as e:
        logging.info(f"Frontend process {frontend_name} caught KeyboardInterrupt")
        if master_proc_pid != -1:
            os.kill(master_proc_pid, signal.SIGINT)
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            child.kill()
            logging.info('Killing child pid {}'.format(child.pid))

        current_process.kill()


def send_results_to_loadgen(output_conn: mp_conn.Connection, bar_manager: LoadingBarManager, bar_idx: int):
    num_first_toks_recvd = 0
    num_samples_completed = 0
    while True:
        result, total_samples, completed_samples = output_conn.recv()
        if result is None:
            break
        is_first, sample_id, output_ids = result
        if is_first:
            qsr = lg.QuerySampleResponse(sample_id, output_ids.ctypes.data, 4, 1)
            lg.FirstTokenComplete([qsr])
            num_first_toks_recvd += 1

        else:
            seq_len = output_ids.shape[-1]
            qsr = lg.QuerySampleResponse(sample_id, output_ids.ctypes.data, 4 * seq_len, seq_len)
            lg.QuerySamplesComplete([qsr])
            num_samples_completed += 1

        bar_manager.update(bar_idx, completed_samples, total_samples)


def parse_args():
    parser = argparse.ArgumentParser()

    # Test args
    parser.add_argument("--scenario", choices=["Offline", "Server"], default="Offline")
    parser.add_argument("--test_mode", choices=["PerformanceOnly", "AccuracyOnly", "SubmissionRun"], default="PerformanceOnly")
    parser.add_argument("--model", choices=["gptj", "llama2-70b", "mixtral-8x7b", "llama3_1-405b"], default="gptj")
    parser.add_argument("--llm_gen_config_path", help="Path to generation_config.json")
    parser.add_argument("--num_gpus_per_host", help="Number of GPUs to perform benchmarking on. See --num_servers_per_host", default=0, type=int)
    parser.add_argument("--gpu_batch_size", help="BS of TRTLLM engine", type=int)

    # QSL args
    parser.add_argument("--tensor_path", type=str, help="path to the directory that contains \"input_ids_padded.npy\", \"input_lens.npy\" \"stop_ids_padded.npy\" ")
    parser.add_argument("--dispatcher_type", type=str, choices=["sequential", "mlperf"], default="mlperf", help="The dispatching behavior of queries.")

    # Config args
    parser.add_argument("--performance_sample_count", type=int, default=5000, help="Number of samples to run benchmark on")
    parser.add_argument("--mlperf_conf_path", help="Path to mlperf.conf", default="build/loadgen-configs/DGX-H100_H100-SXM-80GBx1_TRT/gptj-99/Offline/mlperf.conf")
    parser.add_argument("--user_conf_path", help="Path to user.conf", default="build/loadgen-configs/DGX-H100_H100-SXM-80GBx1_TRT/gptj-99/Offline/user.conf")

    # Log args
    parser.add_argument("--log_mode", type=str, default="AsyncPoll", help="Logging mode for Loadgen")
    parser.add_argument("--log_mode_async_poll_interval_ms", type=int, default=1000, help="Specify the poll interval for asynchrounous logging")
    parser.add_argument("--logfile_outdir", type=str, default='/work/build/logs/triton', help="Specify the existing output directory for the LoadGen logs")
    parser.add_argument("--logfile_prefix", type=str, default='triton-grpc', help="Specify the filename prefix for the LoadGen log files")
    parser.add_argument("--logfile_suffix", type=str, default='', help="Specify the filename suffix for the LoadGen log files")

    # Trtllm flags
    parser.add_argument("--trtllm_runtime_flags", type=parse_cli_flags, default={}, help="Dictionary of runtime flags for TRTLLM.")
    parser.add_argument("--trtllm_build_flags", type=parse_cli_flags, default={}, help="Dictionary of build flags for TRTLLM.")

    # Triton control knobs
    parser.add_argument("--skip_server_spawn", action="store_true", help="Skip starting a tritonserver process")
    parser.add_argument("--verbose_frontend", action="store_true", help="Make triton frontend verbose, this enables logging of stats")
    parser.add_argument("--protocol", type=str, default="grpc", help="Protocol to use for triton client-server communication")
    parser.add_argument("--num_clients_per_frontend", type=int, default=1,
                        help="Number of triton clients per frontend. Total number of gRPC clients are (num_models * num_frontends_per_model * num_clients_per_frontend)")
    parser.add_argument("--num_frontends_per_model", type=int, default=1, help="Number of frontend processes per GPU")
    parser.add_argument("--use_token_latencies", type=bool, help="Ask loadgen to record token latencies or not")
    parser.add_argument("--server_mpi_oversubscribe", action="store_true", help="Use `--oversubscribe` flag for mpirun to launch tritonserver")

    parser.add_argument("--num_servers_per_host", type=int, default=1,
                        help="Number of tritonserver instances to spawn in each host. Recommended: 1 for Offline, and <num_models_per_host> for Server (1 model perf tritonserver instance). Must be a multiple of num_gpus_per_host. Ignored if --skip_server_spawn is used")
    parser.add_argument("--grpc_ports", type=str,
                        help=f"GRPC port for the tritonserver process to listen at, ignored if --skip_server_spawn is not used. Usage: --grpc_ports=<hostname>:<port_0>,<port_1>,<port_2>|<hostname>:<port_0>,<port_1>,<port_2>. Default: localhost:{G_DEFAULT_PORTS['grpc']}", default=f"localhost:G_DEFAULT_PORTS['grpc']")

    parser.add_argument("--system_id", type=str, required=True)

    args, _ = parser.parse_known_args()
    assert args.num_gpus_per_host > 0, "num GPUs must be a positive integer"
    return args


def sigint_callback_main(children_procs, sig, frame):
    logging.info("SIGINT received, terminating.")
    if children_procs is not None:
        for child in children_procs:
            child.terminate()
    signal_name = signal.Signals(sig).name
    psutil.Process().kill()


if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = parse_args()
    llm_config = get_llm_gen_config(args.model, args.scenario, args.llm_gen_config_path)

    # Initialize settings
    test_settings = lg.TestSettings()
    test_settings.scenario = scenario_map[args.scenario]
    test_settings.mode = test_mode_map[args.test_mode]

    # Load config
    test_settings.FromConfig(args.mlperf_conf_path, args.model, args.scenario, 2)
    test_settings.FromConfig(args.user_conf_path, args.model, args.scenario, 1)
    test_settings.server_coalesce_queries = True
    test_settings.use_token_latencies = args.use_token_latencies

    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.logfile_outdir
    log_output_settings.prefix = args.logfile_prefix
    log_output_settings.suffix = args.logfile_suffix

    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.log_mode = log_mode_map[args.log_mode]
    log_settings.log_mode_async_poll_interval_ms = args.log_mode_async_poll_interval_ms

    # Initialize QSL
    files = ["input_ids_padded.npy", "input_lens.npy", "stop_ids_padded.npy"]
    input_ids, input_lens, stop_ids = [Path(args.tensor_path) / file for file in files]
    assert input_ids.exists(), f"Path {input_ids} does not exist"
    assert input_lens.exists(), f"Path {input_lens} does not exist"
    assert stop_ids.exists() or (not llm_config.use_stop_tokens), f"Path {stop_ids} does not exist"
    stop_ids = None if not llm_config.use_stop_tokens else stop_ids

    dataset = LlmDataset(path_input_ids=input_ids, path_input_lens=input_lens, path_stop_ids=stop_ids)
    qsl = lg.ConstructQSL(dataset.get_size(), args.performance_sample_count, dataset.load_samples_to_ram, dataset.unload_samples_from_ram)

    frontend_class = TritonSutGrpcFrontend
    if llm_config.streaming:
        frontend_class = TritonSutGrpcStreamFrontend

    def process_host_data(host_data):
        host_name, port_list = host_data.split(':')
        port_list = port_list.split(',')
        return host_name, port_list

    hosts = args.grpc_ports.split('|')
    hosts = [process_host_data(host) for host in hosts]
    host_port_mapping = {
        host_name: port_list
        for host_name, port_list in hosts
    }

    num_hosts = len(host_port_mapping)

    tp_size = dict_get(args.trtllm_build_flags, "tensor_parallelism", 1)
    pp_size = dict_get(args.trtllm_build_flags, "pipeline_parallelism", 1)
    num_gpus_per_model = tp_size * pp_size
    num_triton_models = args.num_gpus_per_host // num_gpus_per_model
    num_triton_models_per_server = num_triton_models // args.num_servers_per_host

    num_triton_frontends = num_triton_models * args.num_frontends_per_model
    assert args.num_gpus_per_host % num_gpus_per_model == 0, "TP * PP is not a factor of num_gpus_per_host"

    model_name_prefix = f"{llm_config.model_name.lower()}-{args.scenario.lower()}"

    visible_devices_list = ','.join(map(str, range(args.num_gpus_per_host)))
    num_devices_per_server = args.num_gpus_per_host // args.num_servers_per_host
    visible_devices_by_server = split_string_into_chunks(visible_devices_list, num_devices_per_server)

    backends = []
    if not args.skip_server_spawn:
        grpc_ports_by_server = []
        assert num_hosts == 1, "Can not spawn tritonserver instances on multiple hosts, please spawn independantly and use --skip_server_spawn --grpc_ports=<host_0>:<port_0>,<port_0>-<host_1>:<port_0>"
        assert "localhost" in host_port_mapping, "If --skip_server_spawn not used, tritonserver may only reside on localhost"

        base_repo_path = f"/work/build/triton_model_repos/{args.system_id}/{llm_config.model_name.lower()}/{args.scenario.lower()}"
        for server_idx in range(args.num_servers_per_host):
            model_repo = f"{base_repo_path}/repo_{server_idx}"
            backend = TritonSutBackend(oversubscribe=args.server_mpi_oversubscribe, model_repo=model_repo,
                                       cuda_visible_devices=visible_devices_by_server[server_idx], world_size=num_gpus_per_model)
            grpc_port = backend.get_grpc_port()
            grpc_ports_by_server.append(grpc_port)
            backends.append(backend)
        host_port_mapping["localhost"] = grpc_ports_by_server
        logging.info(f"Spawned tritonserver processes on ports {grpc_ports_by_server}")

        for backend in backends:
            while not backend.is_ready():
                time.sleep(5)
                continue
    # Register SIGINT callback to cleanup tritonserver proc
    signal.signal(signal.SIGINT, partial(sigint_callback_main, None))

    children_processes = []
    children_input_connections = []
    children_ready_signals = []
    children_output_connections = []

    bar_manager = LoadingBarManager()
    for host, ports in host_port_mapping.items():
        logging.info(f"Initializing frontends for host {host}")
        for idx in range(num_triton_frontends):
            # spawn a child process that will hold a frontend.
            # This frontend will be responsible for sending queries to GPU #{gpu_idx}
            model_idx = idx // args.num_frontends_per_model
            server_idx = model_idx // num_triton_models_per_server
            frontend_idx = idx % args.num_frontends_per_model
            triton_model_name = f"model-{model_idx}"
            frontend_name = f"{host}::{model_name_prefix}-{model_idx}-{frontend_idx}"
            lg2fe_recv_conn, lg2fe_send_conn = mp.Pipe(duplex=False)  # inputs to frontend
            fe2lg_recv_conn, fe2lg_send_conn = mp.Pipe(duplex=False)  # outputs from frontend
            lg2fe_ready_signal, fe2lg_ready_signal = mp.Pipe(duplex=False)

            grpc_port = ports[server_idx]
            grpc_url = f"{host}:{grpc_port}"

            bar_idx = bar_manager.add_loading_bar(name=f"{grpc_url}-frontend_{frontend_idx} progress: ")

            child = mp.Process(target=frontend_process, args=(frontend_class, lg2fe_recv_conn,
                                                              triton_model_name, frontend_name, grpc_url,
                                                              llm_config, args, fe2lg_ready_signal,
                                                              fe2lg_send_conn, os.getpid()))
            child.daemon = True
            children_processes.append(child)
            children_input_connections.append(lg2fe_send_conn)
            children_output_connections.append(fe2lg_recv_conn)
            children_ready_signals.append(lg2fe_ready_signal)
            child.start()

    signal.signal(signal.SIGINT, partial(sigint_callback_main, children_processes))

    qsr_consumers = []
    for output_conn_idx in range(len(children_output_connections)):
        output_conn = children_output_connections[output_conn_idx]
        qsr_consumer = threading.Thread(target=send_results_to_loadgen, args=(output_conn, bar_manager, output_conn_idx))
        qsr_consumer.start()
        qsr_consumers.append(qsr_consumer)

    for ready_signal in children_ready_signals:
        _ = ready_signal.recv()

    load_balancing_counters = [0] * len(children_input_connections)
    sut = lg.ConstructSUT(partial(dispatch_queries_from_loadgen, dataset, children_input_connections, load_balancing_counters),
                          dummy_flush)
    logging.info("Initialized TritonSUT. Starting benchmark run")
    bar_manager.start()

    # Start test
    lg.StartTestWithLogSettings(sut, qsl, test_settings, log_settings)
    logging.info("Benchmark run complete")

    bar_manager.stop()

    for input_conn in children_input_connections:
        input_conn.send(None)

    for child in children_processes:
        child.join()

    for qsr_consumer in qsr_consumers:
        qsr_consumer.join()

    # Destroying SuT, Qsl
    lg.DestroySUT(sut)
    logging.info("Destroyed SUT")

    lg.DestroyQSL(qsl)
    logging.info("Destroyed QSL")

    if not args.skip_server_spawn:
        os.system('pkill -2 tritonserver')
