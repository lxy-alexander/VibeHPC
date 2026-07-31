"""Frontend server management for disaggregated serving.

This module handles launching and managing frontend servers (NATS, etcd, router)
for the disaggregated serving architecture.
"""

import os
import time
from collections import defaultdict
from pathlib import Path

from base import BaseSrun


class FrontendSrun(BaseSrun):
    """Disaggregated serving frontend (NATS, etcd, router).

    Launches all frontend replicas with a single srun --ntasks=N command.
    Rank detection (via SLURM_PROCID) determines primary vs secondary:
    - Primary frontend (rank 0): starts NATS + etcd + router
    - Secondary frontends (rank > 0): only start router, connect to primary's NATS/etcd
    """

    # Base port for frontends (each frontend uses base_port + rank)
    FRONTEND_BASE_PORT = 8000

    def __init__(self, config: dict, global_config: dict, allocated_nodes: dict, log_dir: Path,
                 dry_run: bool = False, num_frontends: int = 1):
        super().__init__(config, global_config, allocated_nodes, log_dir, dry_run)
        self.system = config['system']
        self.num_frontends = num_frontends
        self.base_port = self.FRONTEND_BASE_PORT

        # Frontend node(s) - can be:
        # - single node string (stacked)
        # - list of node strings (distributed, one port)
        # - list of (node, port_offset) tuples (distributed, multi-port)
        frontend_nodes = allocated_nodes['frontend']
        if isinstance(frontend_nodes, list):
            if frontend_nodes and isinstance(frontend_nodes[0], tuple):
                # Multi-port mode: list of (node, port_offset) tuples
                self.frontend_assignments = frontend_nodes  # [(node, port_offset), ...]
                self.nodes = list(dict.fromkeys([node for node, _ in frontend_nodes]))  # unique nodes
                self.primary_node = frontend_nodes[0][0]
                self.distribute_frontends = True
                self.multi_port_mode = True
            else:
                # Single-port distributed mode: list of node strings
                self.frontend_assignments = [(node, 0) for node in frontend_nodes]
                self.nodes = frontend_nodes
                self.primary_node = frontend_nodes[0]
                self.distribute_frontends = True
                self.multi_port_mode = False
        else:
            # Stacked mode: single node string
            self.frontend_assignments = [(frontend_nodes, i) for i in range(num_frontends)]
            self.nodes = [frontend_nodes]
            self.primary_node = frontend_nodes
            self.distribute_frontends = False
            self.multi_port_mode = num_frontends > 1

        # For backwards compatibility
        self.node = self.primary_node

        # Frontend-specific flags
        self.router_mode = config.get('router_mode', None)  # e.g., 'round-robin', 'kv', 'random'
        self.kv_overlap_weight = config.get('kv_overlap_weight', None)  # For KV router mode

        # Default kv_overlap_weight to 1 if router_mode is 'kv' and not explicitly set
        if self.router_mode == 'kv' and self.kv_overlap_weight is None:
            self.kv_overlap_weight = 1.0

        # Router replica sync for multiple frontends
        self.router_replica_sync = num_frontends > 1

        self.extra_frontend_flags = {k: v for k, v in config.items()
                                     if k not in ['num_prefill_workers', 'num_decode_workers', 'system',
                                                  'router_mode', 'kv_overlap_weight', 'num_frontends',
                                                  'distribute_frontends']}

    def launch(self):
        """Launch all frontends.

        Rank 0 (primary): starts NATS + etcd + router
        Rank > 0 (secondary): only starts router, connects to primary's NATS/etcd

        Supports three modes:
        - Stacked (default): All frontends on single node (--nodes=1 --ntasks=N)
        - Distributed single-port: One frontend per node (--nodes=N --ntasks=N --ntasks-per-node=1)
        - Distributed multi-port: Multiple ports per node when frontends > nodes
        """
        workspace = self.get_workspace()
        benchmark = self.global_config['benchmark']
        scenario = self.global_config['scenario']

        # Pass NATS/etcd endpoints to all frontends so they can connect to the primary
        export_env = f'ALL,ETCD_ENDPOINTS={self.primary_node}:2379,NATS_SERVER=nats://{self.primary_node}:4222'

        # Log directory for frontends (per-rank subdirs created by launch_server.py)
        frontend_log_dir = self.log_dir / 'frontend'
        frontend_log_dir.mkdir(parents=True, exist_ok=True)

        # Convert host log dir to container path
        container_log_dir = str(frontend_log_dir).replace(str(workspace), '/work')

        # Group frontend assignments by port offset for multi-port mode
        # This allows us to launch each port level with a separate srun
        port_groups = defaultdict(list)
        for idx, (node, port_offset) in enumerate(self.frontend_assignments):
            port_groups[port_offset].append((idx, node))

        # Track global frontend rank for proper primary detection
        global_rank = 0

        for port_offset in sorted(port_groups.keys()):
            assignments = port_groups[port_offset]
            nodes_for_port = [node for _, node in assignments]
            num_tasks = len(assignments)
            actual_port = self.base_port + port_offset

            # Create per-port log directory for separate srun commands
            # This ensures each frontend gets its own log files (disagg_frontend.log, etc.)
            port_log_dir = frontend_log_dir / f'fe_{port_offset}'
            port_log_dir.mkdir(parents=True, exist_ok=True)
            port_container_log_dir = str(port_log_dir).replace(str(workspace), '/work')

            # Build RUN_ARGS for make run_llm_server
            run_args = [
                f"--benchmarks={benchmark}",
                f"--scenarios={scenario}",
                "--core_type=disagg_frontend",
                f"--dynamo_frontend_port={actual_port}",
                f"--dynamo_frontend_host={self.primary_node}",  # All frontends need to know primary's host
            ]

            # Add router configuration
            if self.router_mode:
                run_args.append(f"--dynamo_router_mode={self.router_mode}")
            if self.kv_overlap_weight is not None:
                run_args.append(f"--dynamo_kv_overlap_weight={self.kv_overlap_weight}")
            if self.router_replica_sync:
                run_args.append("--dynamo_router_replica_sync")

            # For secondary port groups, tell them they're not primary
            if port_offset > 0:
                run_args.append("--dynamo_frontend_secondary")

            # Append --test_mode=AccuracyOnly when --accuracy flag is set
            if self.global_config.get('accuracy'):
                run_args.append("--test_mode=AccuracyOnly")

            run_args_str = " ".join(run_args)

            # Determine launch mode
            unique_nodes = list(dict.fromkeys(nodes_for_port))
            # Container name for pyxis caching (matches sbatch pattern)
            container_name = f'dynamo_frontend_port{actual_port}'

            if len(unique_nodes) == 1:
                # All tasks on single node
                cmd = [
                    'srun',
                    '--overlap',
                    '--nodes=1',
                    f'--ntasks={num_tasks}',
                    f'--nodelist={unique_nodes[0]}',
                    f'--output={port_log_dir}/frontend_%t.stdout',
                    f'--error={port_log_dir}/frontend_%t.stderr',
                    f'--export={export_env}',
                    f'--container-image={self.get_container_image()}',
                    f'--container-name={container_name}',
                    f'--container-mounts={workspace}:/work,{self.get_storage_path()}:/home/mlperf_inference_storage',
                    '--container-workdir=/work',
                    '--container-remap-root',
                    '--mpi=pmix',
                    'make', 'run_llm_server',
                    f'RUN_ARGS={run_args_str}',
                    f'LOG_DIR={port_container_log_dir}',
                    f'SYSTEM_NAME={self.system}'
                ]
                node_desc = f"on {unique_nodes[0]}"
            else:
                # Distributed across multiple nodes
                nodelist = ','.join(unique_nodes)
                cmd = [
                    'srun',
                    '--overlap',
                    f'--nodes={len(unique_nodes)}',
                    f'--ntasks={num_tasks}',
                    '--ntasks-per-node=1',
                    f'--nodelist={nodelist}',
                    f'--output={port_log_dir}/frontend_%t.stdout',
                    f'--error={port_log_dir}/frontend_%t.stderr',
                    f'--export={export_env}',
                    f'--container-image={self.get_container_image()}',
                    f'--container-name={container_name}',
                    f'--container-mounts={workspace}:/work,{self.get_storage_path()}:/home/mlperf_inference_storage',
                    '--container-workdir=/work',
                    '--container-remap-root',
                    '--mpi=pmix',
                    'make', 'run_llm_server',
                    f'RUN_ARGS={run_args_str}',
                    f'LOG_DIR={port_container_log_dir}',
                    f'SYSTEM_NAME={self.system}'
                ]
                node_desc = f"across {len(unique_nodes)} nodes"

            desc = f"Launch {num_tasks} Frontend(s) {node_desc} (port {actual_port})"
            self.run_command(cmd, desc, background=True)
            global_rank += num_tasks

        # Wait for frontends to initialize
        if not self.dry_run:
            delay = int(os.environ.get('FRONTEND_INIT_DELAY', '15'))
            if delay > 0:
                print(f"Waiting {delay}s for frontend(s) to initialize...")
                time.sleep(delay)

    def get_url(self, rank: int = 0) -> str:
        """Get frontend URL for a specific rank."""
        if rank < len(self.frontend_assignments):
            node, port_offset = self.frontend_assignments[rank]
            return f"http://{node}:{self.base_port + port_offset}"
        else:
            # Fallback to last assignment
            node, port_offset = self.frontend_assignments[-1]
            return f"http://{node}:{self.base_port + port_offset}"

    def get_all_urls(self) -> list:
        """Get all frontend URLs."""
        return [f"http://{node}:{self.base_port + port_offset}"
                for node, port_offset in self.frontend_assignments]
