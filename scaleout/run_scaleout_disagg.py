#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Scaleout script for disaggregated serving
# This script launches servers (CTX/GEN workers and optionally master servers), then harness

import argparse
import os
import sys
import subprocess
import re
import datetime
import getpass
from pathlib import Path

import yaml

def run_srun(args, env=None, dry_run=False, background=False, log_file=None):
    cmd = ["srun"] + args
    sys.stderr.write("================================================\n")
    sys.stderr.write("Executing srun command:\n")
    sys.stderr.write(f"{' '.join(cmd)}\n")
    sys.stderr.write("================================================\n")
    
    if log_file:
        try:
            with open(log_file, "a") as f:
                f.write(' '.join(cmd) + '\n')
        except Exception as e:
            sys.stderr.write(f"WARNING: Failed to write to log file {log_file}: {e}\n")

    if not dry_run:
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
            
        if background:
            # Return the Popen object for background processes
            return subprocess.Popen(cmd, env=run_env)
        else:
            # Wait for foreground processes
            subprocess.check_call(cmd, env=run_env)
            return None
    return None

def get_gpu_count_from_system(system_name):
    # Parse GPUs per server from atomic system names (e.g., "...x8" -> 8)
    match = re.search(r'x(\d+)$', system_name)
    if match:
        return int(match.group(1))
    return None

def main():
    parser = argparse.ArgumentParser(description="Scaleout script for disaggregated serving")
    parser.add_argument("--stage", default="all", help="Stage to run (server|harness|all)")
    parser.add_argument("--launch-master", default="true", choices=["true", "false"], help="Launch master servers in server stage (default: true)")
    parser.add_argument("--dry-run", action="store_true", help="Print srun commands without executing")
    parser.add_argument("--harness-system", help="Aggregate system config for harness (default: computed from ctx/gen systems)")
    parser.add_argument("--gpus-per-node", type=int, required=True, help="GPUs per node")
    parser.add_argument("--num-ctx-servers", type=int, required=True, help="Number of context servers")
    parser.add_argument("--num-gen-servers", type=int, required=True, help="Number of generation servers")
    parser.add_argument("--ctx-atomic-system", required=True, help="Atomic system for CTX workers")
    parser.add_argument("--gen-atomic-system", required=True, help="Atomic system for GEN workers")
    parser.add_argument("--ctx-run-args", required=True, help="Benchmark arguments for CTX workers")
    parser.add_argument("--gen-run-args", required=True, help="Benchmark arguments for GEN workers")
    parser.add_argument("--harness-run-args", required=True, help="Benchmark arguments for Harness")
    parser.add_argument("--container-image", help="Container image")
    parser.add_argument("--mlperf-scratch-path", default="/lustre/share/coreai_mlperf_inference/mlperf_inference_storage_clone", help="Scratch path")
    parser.add_argument("--extra-srun-flags", default="", help="Additional srun flags")
    parser.add_argument("--base-port", type=int, default=30000, help="Base port for all servers (default: 30000)")
    parser.add_argument("--audit", action="store_true", help="Run audit harness (run_audit_harness) instead of regular harness")
    parser.add_argument("--num-master-servers", type=int, default=1, help="Number of master/frontend servers to launch (default: 1)")
    parser.add_argument("--log-dir", help="Log directory (default: auto-generated with timestamp)")

    args = parser.parse_args()
    
    stage = args.stage
    launch_master = args.launch_master
    dry_run = args.dry_run
    harness_target = "run_audit_harness" if args.audit else "run_harness"

    # Parse GPU counts from atomic systems early (needed for harness system computation)
    num_ctx_gpus_per_server = get_gpu_count_from_system(args.ctx_atomic_system)
    num_gen_gpus_per_server = get_gpu_count_from_system(args.gen_atomic_system)

    if num_ctx_gpus_per_server is None:
        sys.stderr.write(f"ERROR: Failed to parse GPU count from ctx_atomic_system: {args.ctx_atomic_system}\n")
        sys.exit(1)
    if num_gen_gpus_per_server is None:
        sys.stderr.write(f"ERROR: Failed to parse GPU count from gen_atomic_system: {args.gen_atomic_system}\n")
        sys.exit(1)

    num_ctx_gpus = args.num_ctx_servers * num_ctx_gpus_per_server
    num_gen_gpus = args.num_gen_servers * num_gen_gpus_per_server
    total_gpus = num_ctx_gpus + num_gen_gpus

    # Compute harness system if not provided
    if args.harness_system:
        harness_system = args.harness_system
    else:
        base_system = re.sub(r'x\d+$', '', args.ctx_atomic_system)
        harness_system = f"{base_system}x{total_gpus}"

    script_path = Path(__file__).resolve()
    # script_dir is .../scaleout
    # host_vol is .../NVIDIA (parent of scaleout)
    host_vol = str(script_path.parent.parent)
    container_vol = "/work"
    
    # Determine log directory: use --log-dir if provided, else auto-generate
    # --log-dir expects a host path (like run_scaleout.sh)
    if args.log_dir:
        # User provided host path
        host_log_dir = args.log_dir
        # Convert host path to container path (replace host_vol prefix with container_vol)
        log_dir = args.log_dir.replace(host_vol, container_vol, 1)
    else:
        # Auto-generate with timestamp (container path)
        timestamp = datetime.datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
        slurm_jobid = os.environ.get("SLURM_JOBID", "unknown")
        log_dir = f"/work/build/logs/scaleout_disagg_{harness_system}_slurm-{slurm_jobid}_{timestamp}"
        # Convert container path to host path (replace container_vol prefix with host_vol)
        host_log_dir = log_dir.replace(container_vol, host_vol, 1)
            
    if not dry_run and not os.path.exists(host_log_dir):
        os.makedirs(host_log_dir, exist_ok=True)
        
    srun_log_file = os.path.join(host_log_dir, "srun_commands.log")
    
    container_image = args.container_image
    if not container_image:
        docker_tag = f"{getpass.getuser()}-aarch64"
        container_image = os.path.join(host_vol, "build", "sqsh_images", f"mlperf-inference-{docker_tag}-release.sqsh")
        
    if not os.environ.get("SLURM_JOB_NODELIST") or not os.environ.get("SLURM_JOBID"):
        sys.stderr.write("ERROR: Not running in a SLURM allocation\n")
        sys.exit(1)
        
    # Modify args
    ctx_run_args = args.ctx_run_args
    gen_run_args = args.gen_run_args
    harness_run_args = args.harness_run_args
    
    if stage in ["server", "all"]:
        if "--mpi_mode=leader" not in ctx_run_args: ctx_run_args += " --mpi_mode=leader"
        if "--server_in_foreground" not in ctx_run_args: ctx_run_args += " --server_in_foreground"
        if "--mpi_mode=leader" not in gen_run_args: gen_run_args += " --mpi_mode=leader"
        if "--server_in_foreground" not in gen_run_args: gen_run_args += " --server_in_foreground"
        
    if stage in ["harness", "all"]:
        if "--mpi_mode=leader" not in harness_run_args: harness_run_args += " --mpi_mode=leader"
    
    srun_flags_list = args.extra_srun_flags.split() if args.extra_srun_flags else []

    # Print whether harness system was provided or computed
    if args.harness_system:
        print(f"Using user-provided harness system: {harness_system}")
    else:
        print(f"Computed harness system: {harness_system} ({args.num_ctx_servers} CTX × {num_ctx_gpus_per_server} + {args.num_gen_servers} GEN × {num_gen_gpus_per_server} = {total_gpus} GPUs)")

    # Determine deployment mode
    if num_ctx_gpus_per_server <= args.gpus_per_node:
        ctx_deployment_mode = "intra_node"
    else:
        ctx_deployment_mode = "cross_node"
        if num_ctx_gpus_per_server % args.gpus_per_node != 0:
            sys.stderr.write(f"ERROR: CTX cross-node deployment requires num_ctx_gpus_per_server ({num_ctx_gpus_per_server}) to be a multiple of gpus_per_node ({args.gpus_per_node})\n")
            sys.exit(1)
            
    if num_gen_gpus_per_server <= args.gpus_per_node:
        gen_deployment_mode = "intra_node"
    else:
        gen_deployment_mode = "cross_node"
        if num_gen_gpus_per_server % args.gpus_per_node != 0:
            sys.stderr.write(f"ERROR: GEN cross-node deployment requires num_gen_gpus_per_server ({num_gen_gpus_per_server}) to be a multiple of gpus_per_node ({args.gpus_per_node})\n")
            sys.exit(1)
            
    total_nodes_allocated = (total_gpus + args.gpus_per_node - 1) // args.gpus_per_node # round up to the nearest integer
    total_gpus_allocated = total_nodes_allocated * args.gpus_per_node
    total_gpus_wasted = total_gpus_allocated - total_gpus
    
    # Get Slurm nodes
    try:
        slurm_output = subprocess.check_output(["scontrol", "show", "hostname", os.environ["SLURM_JOB_NODELIST"]], text=True)
        node_array = slurm_output.strip().splitlines()
        allocated_node_count = len(node_array)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"ERROR: Failed to get node list: {e}\n")
        sys.exit(1)
        
    slurm_gpus_allocated = allocated_node_count * args.gpus_per_node
    
    if slurm_gpus_allocated != total_gpus_allocated:
        sys.stderr.write("ERROR: GPU count mismatch\n")
        sys.stderr.write(f"  SLURM allocated: {slurm_gpus_allocated} ({allocated_node_count} nodes x {args.gpus_per_node} GPUs/node)\n")
        sys.stderr.write(f"  Required: {total_gpus_allocated} GPUs\n")
        if total_gpus_wasted > 0:
            sys.stderr.write(f"  Note: {total_gpus_wasted} GPUs will be wasted (intra-node leftover)\n")
        sys.exit(1)
        
    print("============================================")
    print(f"MLPerf Disaggregated Scaleout - {stage}")
    print("============================================")
    print(f"Harness system: {harness_system}")
    print(f"GPUs per node: {args.gpus_per_node}")
    print(f"Allocated nodes: {allocated_node_count}")
    print("")
    print(f"GEN: {args.num_gen_servers} servers x {num_gen_gpus_per_server} GPUs = {num_gen_gpus} GPUs [{args.gen_atomic_system}]")
    print(f"     Mode: {gen_deployment_mode}")
    print("")
    print(f"CTX: {args.num_ctx_servers} servers x {num_ctx_gpus_per_server} GPUs = {num_ctx_gpus} GPUs [{args.ctx_atomic_system}]")
    print(f"     Mode: {ctx_deployment_mode}")
    print("")
    if launch_master == "true":
        print(f"Master servers: {args.num_master_servers}")
        print("")
    print(f"Total: {total_nodes_allocated} nodes, {total_gpus_allocated} GPUs allocated, {total_gpus} GPUs used")
    if total_gpus_wasted > 0:
        print(f"GPU waste: {total_gpus_wasted} GPUs (leftover in last node)")
    print(f"Log directory: {log_dir}")
    print(f"Harness target: {harness_target}")
    print(f"Base port: {args.base_port}")
    print("============================================")
    
    master_urls = []
    gen_worker_urls = []
    ctx_worker_urls = []
    
    # Track running background processes to prevent premature exit if needed?
    # The bash script just backgrounds them and then runs harness. 
    # If harness finishes, the script exits, and bash kills background jobs?
    # Standard shell behavior: background jobs are children. If script exits, they might be SIGHUPed or orphaned.
    # In SLURM, srun jobs are separate steps. They should persist until cancelled or job ends.
    
    # Common srun arguments
    common_srun_args = [
        "--overlap",
        "--export=ALL",
        "--mpi=pmix",
        f"--container-image={container_image}",
        f"--container-mounts={host_vol}:{container_vol},{args.mlperf_scratch_path}:/home/mlperf_inference_storage",
        f"--container-workdir={container_vol}",
        "--container-remap-root",
    ]
    
    def launch_worker(worker_type, worker_idx, gpus_per_server, deployment_mode, atomic_system):
        if worker_type == "GEN":
            start_gpu = worker_idx * num_gen_gpus_per_server
            port_offset = worker_idx
            current_run_args = gen_run_args
        else:
            start_gpu = num_gen_gpus + worker_idx * num_ctx_gpus_per_server
            port_offset = args.num_gen_servers + worker_idx
            current_run_args = ctx_run_args
            
        end_gpu = start_gpu + gpus_per_server - 1
        start_node = start_gpu // args.gpus_per_node
        end_node = end_gpu // args.gpus_per_node
        
        server_nodes = node_array[start_node : end_node + 1]
        server_nodes_str = ",".join(server_nodes)
        
        cuda_devices = []
        if deployment_mode == "intra_node":
            gpu_offset = start_gpu % args.gpus_per_node
            for g in range(gpus_per_server):
                cuda_devices.append(str(gpu_offset + g))
        else:
            for g in range(args.gpus_per_node):
                cuda_devices.append(str(g))
        cuda_devices_str = ",".join(cuda_devices)
        
        first_node = server_nodes[0]
        worker_url = f"{first_node}:{args.base_port + port_offset}"
        
        worker_args = f"{current_run_args} --trtllm_server_urls={worker_url}"
        worker_log_dir = f"{log_dir}/{worker_type.lower()}/{worker_type.lower()}{worker_idx}"
        
        num_nodes = len(server_nodes)
        num_tasks_per_node = len(cuda_devices)
        ipc_port = 10012 + port_offset
        ipc_addr = f"tcp://127.0.0.1:{ipc_port}"
        
        print(f"  Launching {worker_type} worker {worker_idx} on {server_nodes_str} (GPUs: {cuda_devices_str})")
        
        env_vars = os.environ.copy()
        env_vars.update({
            "LOG_DIR": worker_log_dir,
            "NVIDIA_VISIBLE_DEVICES": cuda_devices_str,
            "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR": ipc_addr
        })
        
        srun_cmd = common_srun_args + [
            f"--nodelist={server_nodes_str}",
            f"--ntasks-per-node={num_tasks_per_node}",
            f"--nodes={num_nodes}",
        ] + srun_flags_list + [
            "make", "run_llm_server",
            f"RUN_ARGS={worker_args}",
            f"SYSTEM_NAME={atomic_system}"
        ]
        
        run_srun(srun_cmd, env=env_vars, dry_run=dry_run, background=True, log_file=srun_log_file)
        return worker_url

    if stage in ["server", "all"]:
        print("============================================")
        print("Stage 1: Launching Servers")
        print("============================================")
        
        total_workers = args.num_gen_servers + args.num_ctx_servers
        print("--------------------------------------------")
        
        print(f"Launching {args.num_gen_servers} GEN worker(s)...")
        for idx in range(args.num_gen_servers):
            url = launch_worker("GEN", idx, num_gen_gpus_per_server, gen_deployment_mode, args.gen_atomic_system)
            gen_worker_urls.append(url)
            
        print(f"Launching {args.num_ctx_servers} CTX worker(s)...")
        for idx in range(args.num_ctx_servers):
            url = launch_worker("CTX", idx, num_ctx_gpus_per_server, ctx_deployment_mode, args.ctx_atomic_system)
            ctx_worker_urls.append(url)
            
        print("============================================")
        print("All workers launched")
        print("============================================")
        
        if launch_master == "true":
            print("============================================")
            print(f"Launching {args.num_master_servers} Master Server(s)")
            print("============================================")

            for master_idx in range(args.num_master_servers):
                # Assign to nodes in round-robin fashion
                master_node_idx = master_idx % len(node_array)
                master_node = node_array[master_node_idx]
                master_port = args.base_port + total_workers + master_idx
                master_url_single = f"{master_node}:{master_port}"
                master_urls.append(master_url_single)

                print(f"Master {master_idx}: {master_url_single}")

                # Build config dictionary
                config = {
                    "hostname": master_node,
                    "port": master_port,
                    "backend": "pytorch",
                    "context_servers": {
                        "num_instances": args.num_ctx_servers,
                        "urls": ctx_worker_urls,  # All masters share same workers
                    },
                    "generation_servers": {
                        "num_instances": args.num_gen_servers,
                        "urls": gen_worker_urls,  # All masters share same workers
                    },
                }

                # Write config file
                host_server_config_file = os.path.join(host_log_dir, f"master_server_config_{master_idx}.yaml")
                container_config_file = f"{log_dir}/master_server_config_{master_idx}.yaml"

                if not dry_run:
                    try:
                        with open(host_server_config_file, "w") as f:
                            yaml.dump(config, f, default_flow_style=False)
                        print(f"  Config created: {host_server_config_file}")
                    except Exception as e:
                        sys.stderr.write(f"ERROR: Failed to write server config file {host_server_config_file}: {e}\n")
                        sys.exit(1)
                else:
                    print(f"  Dry run: Would create config at {host_server_config_file}")
                    if master_idx == 0:  # Only print one example config
                        print(yaml.dump(config, default_flow_style=False))

                # Launch master
                master_server_log = f"{log_dir}/master_server_{master_idx}.log"
                master_cmd = f"trtllm-serve disaggregated --config_file {container_config_file} --server_start_timeout 7200 --request_timeout 7200 > {master_server_log} 2>&1"

                master_srun_cmd = common_srun_args + [
                    f"--nodelist={master_node}",
                    "--ntasks=1",
                    "--nodes=1",
                ] + srun_flags_list + [
                    "bash", "-c", master_cmd
                ]

                run_srun(master_srun_cmd, env={"LOG_DIR": log_dir}, dry_run=dry_run, background=True, log_file=srun_log_file)

            print("============================================")
            print(f"All {args.num_master_servers} master server(s) launched")
            print(f"Master URLs: {','.join(master_urls)}")
            print("============================================")

    if stage in ["harness", "all"]:
        print("============================================")
        print("Stage 2: Launching MLPerf Harness")
        print("============================================")

        if not master_urls:
            # Reconstruct master URLs if running harness-only stage
            total_workers = args.num_gen_servers + args.num_ctx_servers
            for master_idx in range(args.num_master_servers):
                master_node_idx = master_idx % len(node_array)
                master_node = node_array[master_node_idx]
                master_port = args.base_port + total_workers + master_idx
                master_urls.append(f"{master_node}:{master_port}")

        # Create comma-separated URL string for harness
        master_urls_str = ",".join(master_urls)

        if "--trtllm_server_urls=" not in harness_run_args:
            harness_run_args += f" --trtllm_server_urls={master_urls_str}"

        print(f"Master URLs: {master_urls_str}")
        print(f"Harness target: {harness_target}")
        print("--------------------------------------------")

        harness_srun_cmd = common_srun_args + [
            "--nodes=1",
        ] + srun_flags_list + [
            "make", harness_target,
            f"RUN_ARGS={harness_run_args}",
            f"SYSTEM_NAME={harness_system}"
        ]
        
        run_srun(harness_srun_cmd, env={"LOG_DIR": log_dir}, dry_run=dry_run, background=False, log_file=srun_log_file)

if __name__ == "__main__":
    main()
