#!/usr/bin/env python3
"""
Launch script for disaggregated serving deployment.
Supports reading configuration from YAML files or MLPerf system configs.

Usage:
    # From YAML config file:
    python3 scripts/slurm_llm/dynamo_disagg/launch_disagg_cluster.py \\
        --config scripts/slurm_llm/dynamo_disagg/config/sample/disagg_deployment_minimal.yaml \\
        --container-image /path/to/image.sqsh

    # From MLPerf system config:
    python3 scripts/slurm_llm/dynamo_disagg/launch_disagg_cluster.py \\
        --system GB200-NVL72_GB200-186GB_aarch64x20 \\
        --benchmark deepseek-r1 --scenario Interactive \\
        --container-image /path/to/image.sqsh

    # With dry-run and test nodelist:
    python3 scripts/slurm_llm/dynamo_disagg/launch_disagg_cluster.py \\
        --config scripts/slurm_llm/dynamo_disagg/config/test_res_alloc/test_shared_5p_3d.yaml \\
        --container-image /test/image.sqsh --nodelist "node0" --dry-run --verbose

    # Run harness after server launch with custom RUN_ARGS:
    # NOTE: Use --flag="value" form (with =) for args whose values start with "--"
    python3 scripts/slurm_llm/dynamo_disagg/launch_disagg_cluster.py \\
        --system GB200-NVL72_GB200-186GB_aarch64x20 \\
        --benchmark deepseek-r1 --scenario Interactive \\
        --container-image /path/to/image.sqsh \\
        --run-harness-args="--benchmarks=deepseek-r1 --scenarios=Interactive --test_mode=PerformanceOnly" \\
        --run-harness-nodeidx 0

    # Accuracy run (servers use _accuracy.yml overrides automatically):
    python3 scripts/slurm_llm/dynamo_disagg/launch_disagg_cluster.py \\
        --system GB200-NVL72_GB200-186GB_aarch64x20 \\
        --benchmark deepseek-r1 --scenario Interactive \\
        --container-image /path/to/image.sqsh \\
        --accuracy

Note: Run inside salloc/sbatch with a valid SLURM job allocation (or use --nodelist for testing).
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Any
import yaml

# Import base and frontend classes from separate modules
# Support both direct execution and package import
from base import BaseSrun
from frontend import FrontendSrun


@contextmanager
def scoped_sys_path(paths: List[str]):
    """Temporarily add paths to sys.path."""
    original_path = sys.path.copy()
    sys.path = paths + sys.path
    try:
        yield
    finally:
        sys.path = original_path


def get_field_value(config_dict: dict, field_name: str) -> Any:
    """Extract a value from config dict where keys are Field objects.

    Args:
        config_dict: Config dict with Field objects as keys
        field_name: The field name to look for (e.g., 'dynamo_cluster')

    Returns:
        The value associated with the field, or None if not found
    """
    for key, value in config_dict.items():
        # Field objects have a 'name' attribute
        if hasattr(key, 'name') and key.name == field_name:
            return value
    return None


def load_config_from_mlperf_system(system_name: str, benchmark: str, scenario: str,
                                   config_id: str = 'dynamo_cluster', config_dir: Path = None) -> dict:
    """Load disaggregated serving config from MLPerf Python config files.

    Imports the Python config module and extracts the dynamo_cluster field.
    Uses lazy imports in code.llmlib to avoid loading heavy dependencies.

    Args:
        system_name: MLPerf system name (e.g., 'GB200-NVL72_GB200-186GB_aarch64x20')
        benchmark: Benchmark name (e.g., 'deepseek-r1')
        scenario: Scenario name (e.g., 'Interactive')
        config_id: Config ID to load from ATOMIC_EXPORTS (must be 'dynamo_cluster')
        config_dir: Path to configs directory (default: auto-detect)

    Returns:
        dict: Configuration dict compatible with DisaggCluster YAML format
    """
    # Validate config_id - should be 'dynamo_cluster' for dynamo disaggregated serving
    if config_id == 'default':
        print("\n" + "=" * 80)
        print("WARNING: Using config_id='default' for dynamo disaggregated serving.")
        print("This will load harness config from EXPORTS instead of ATOMIC_EXPORTS['dynamo_cluster'].")
        print("The loadgen settings (server_target_qps, min_query_count, etc.) may not match")
        print("the dynamo cluster configuration. Consider using --config-id=dynamo_cluster.")
        print("=" * 80 + "\n")
    elif config_id != 'dynamo_cluster':
        raise ValueError(
            f"config_id must be 'dynamo_cluster' (or 'default' with warning) for dynamo disaggregated serving. "
            f"Got: '{config_id}'. "
            f"Please use --config-id=dynamo_cluster or ensure your config file has a 'dynamo_cluster' entry in ATOMIC_EXPORTS."
        )

    # Auto-detect workspace and config directory
    # Find git root and append closed/NVIDIA
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).parent
        )
        git_root = Path(result.stdout.strip())
        workspace_dir = git_root / 'closed' / 'NVIDIA'
    except subprocess.CalledProcessError:
        # Fallback: assume script is at scripts/slurm_llm/dynamo_disagg/
        workspace_dir = Path(__file__).parent.parent.parent.parent

    if config_dir is None:
        config_dir = workspace_dir / 'configs'

    config_path = config_dir / system_name / scenario / f'{benchmark}.py'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Loading config from: {config_path}")
    print(f"  Config ID: {config_id}")

    # Import the config module with proper Python path setup
    # Need workspace in path so 'code.*' imports work
    system_config_dir = config_dir / system_name
    with scoped_sys_path([str(workspace_dir), str(system_config_dir)]):
        # Import as module: {scenario}.{benchmark}
        module_name = f"{scenario}.{benchmark}"
        module = importlib.import_module(module_name)

        # Get ATOMIC_EXPORTS
        if not hasattr(module, 'ATOMIC_EXPORTS'):
            raise ValueError(f"ATOMIC_EXPORTS not found in {config_path}")

        atomic_exports = module.ATOMIC_EXPORTS

        # Find the workload setting (typically there's only one)
        if len(atomic_exports) == 0:
            raise ValueError(f"ATOMIC_EXPORTS is empty in {config_path}")

        # Use first workload setting
        workload_setting = next(iter(atomic_exports.keys()))
        atomic_configs = atomic_exports[workload_setting]

        # Get config by config_id
        if config_id not in atomic_configs:
            available_ids = list(atomic_configs.keys())
            raise ValueError(f"config_id '{config_id}' not found. Available: {available_ids}")

        config_dict = atomic_configs[config_id]

    # Extract dynamo_cluster field
    dynamo_cluster = get_field_value(config_dict, 'dynamo_cluster')
    if dynamo_cluster is None:
        raise ValueError(f"llm_fields.dynamo_cluster not found in config '{config_id}' in {config_path}")

    # Extract values from dynamo_cluster dict
    num_prefill_workers = dynamo_cluster.get('num_prefill_workers')
    num_decode_workers = dynamo_cluster.get('num_decode_workers')
    gpus_per_node = dynamo_cluster.get('gpus_per_node', 4)

    prefill_dict = dynamo_cluster.get('prefill', {})
    decode_dict = dynamo_cluster.get('decode', {})

    prefill_system = prefill_dict.get('system')
    decode_system = decode_dict.get('system')

    # Worker config: trtllm_yml_override (YAML path) is REQUIRED
    # NOTE: config_id (Python config loading) is NOT supported - it's broken for ctx/gen worker configs
    prefill_yml_override = prefill_dict.get('trtllm_yml_override')
    decode_yml_override = decode_dict.get('trtllm_yml_override')

    # Assert that trtllm_yml_override is specified (not config_id)
    if prefill_dict.get('config_id'):
        raise ValueError(
            f"prefill.config_id is not supported for dynamo workers. "
            f"Loading server configs from Python is broken. "
            f"Please use 'trtllm_yml_override' with a YAML config path instead."
        )
    if decode_dict.get('config_id'):
        raise ValueError(
            f"decode.config_id is not supported for dynamo workers. "
            f"Loading server configs from Python is broken. "
            f"Please use 'trtllm_yml_override' with a YAML config path instead."
        )
    if not prefill_yml_override:
        raise ValueError(
            f"prefill.trtllm_yml_override is required in dynamo_cluster config. "
            f"Please specify the YAML config path for prefill workers."
        )
    if not decode_yml_override:
        raise ValueError(
            f"decode.trtllm_yml_override is required in dynamo_cluster config. "
            f"Please specify the YAML config path for decode workers."
        )

    # Validate required fields
    if num_prefill_workers is None:
        raise ValueError(f"'num_prefill_workers' not found in dynamo_cluster")
    if num_decode_workers is None:
        raise ValueError(f"'num_decode_workers' not found in dynamo_cluster")
    if prefill_system is None:
        raise ValueError(f"'prefill.system' not found in dynamo_cluster")
    if decode_system is None:
        raise ValueError(f"'decode.system' not found in dynamo_cluster")

    # System for frontend and harness is the system from the config path (the --system argument)
    cluster_system = system_name

    # Extract GPU counts from worker system names (e.g., "GB300...x2" -> 2)
    def get_gpus_from_system(sys_name: str) -> int:
        try:
            return int(sys_name.rsplit('x', 1)[-1])
        except (ValueError, IndexError):
            raise ValueError(f"Cannot parse GPU count from system name: {sys_name}")

    prefill_gpus = get_gpus_from_system(prefill_system)
    decode_gpus = get_gpus_from_system(decode_system)

    # Extract optional frontend settings
    num_frontends = dynamo_cluster.get('num_frontends', 1)
    distribute_frontends = dynamo_cluster.get('distribute_frontends', False)
    frontend_dict = dynamo_cluster.get('frontend', {})

    # Build config dict in YAML format
    config = {
        'benchmark': benchmark,
        'scenario': scenario,
        'disagg_cluster': {
            'num_prefill_workers': num_prefill_workers,
            'num_decode_workers': num_decode_workers,
            'num_frontends': num_frontends,
            'distribute_frontends': distribute_frontends,
            'system': cluster_system,
        },
        'prefill': {
            'system': prefill_system,
            'gpus_per_worker': prefill_gpus,
            'gpus_per_node': gpus_per_node,
        },
        'decode': {
            'system': decode_system,
            'gpus_per_worker': decode_gpus,
            'gpus_per_node': gpus_per_node,
        },
    }

    # Add frontend settings if present
    if frontend_dict:
        config['disagg_cluster']['frontend'] = frontend_dict

    # Add worker config source (trtllm_yml_override only - config_id is not supported)
    config['prefill']['config'] = prefill_yml_override
    config['decode']['config'] = decode_yml_override

    # Pass through env_vars if specified
    if prefill_dict.get('env_vars'):
        config['prefill']['env_vars'] = prefill_dict['env_vars']
    if decode_dict.get('env_vars'):
        config['decode']['env_vars'] = decode_dict['env_vars']

    print(f"  Prefill: {num_prefill_workers} worker(s) x {prefill_gpus} GPUs = {prefill_system}")
    print(f"  Decode: {num_decode_workers} worker(s) x {decode_gpus} GPUs = {decode_system}")

    return config


class WorkerSrun(BaseSrun):
    """Base class for worker srun steps (prefill/decode)."""

    # Base ports for different worker types (to avoid conflicts on shared nodes)
    WORKER_BASE_PORTS = {
        'prefill': 30000,
        'decode': 31000
    }

    def __init__(self, config: dict, global_config: dict, worker_type: str, allocated_nodes: dict, log_dir: Path, dry_run: bool = False, gpu_offset: int = 0):
        super().__init__(config, global_config, allocated_nodes, log_dir, dry_run)
        self.worker_type = worker_type
        self.nodes = allocated_nodes[worker_type]
        self.system = config['system']
        self.gpus_per_node = config['gpus_per_node']
        self.gpus_per_worker = config['gpus_per_worker']
        # YAML config path (trtllm_yml_override) - REQUIRED
        # NOTE: config_id (Python config loading) is NOT supported - it's broken
        self.config_path = config.get('config')
        if not self.config_path:
            raise ValueError(
                f"{worker_type}: 'config' (trtllm_yml_override YAML path) is required. "
                f"config_id is not supported for dynamo workers."
            )
        # Extract primary frontend hostname (where NATS/etcd runs)
        # allocated_nodes['frontend'] can be: str, list of str, or list of (node, port) tuples
        frontend_alloc = allocated_nodes['frontend']
        if isinstance(frontend_alloc, list):
            first_item = frontend_alloc[0]
            if isinstance(first_item, tuple):
                # Multi-port mode: list of (node, port_offset) tuples
                self.frontend_node = first_item[0]
            else:
                # Distributed single-port mode: list of node strings
                self.frontend_node = first_item
        else:
            # Stacked mode: single node string
            self.frontend_node = frontend_alloc
        self.num_workers = global_config['disagg_cluster'][f'num_{worker_type}_workers']
        # GPU offset for intra-node mixed deployments (prefill+decode on same node)
        self.gpu_offset = gpu_offset
        # Base port for this worker type (different for prefill/decode to avoid conflicts)
        self.base_port = self.WORKER_BASE_PORTS.get(worker_type, 30000)
        # Optional environment variables from YAML file
        self.env_vars = self._load_env_vars(config.get('env_vars'))

    def _load_env_vars(self, env_vars_path: str) -> dict:
        """Load environment variables from YAML file if provided."""
        if not env_vars_path:
            return {}
        # Handle relative paths (relative to workspace)
        env_path = Path(env_vars_path)
        if not env_path.is_absolute():
            workspace = self.get_workspace()
            env_path = workspace / env_vars_path
        if not env_path.exists():
            print(f"WARNING: env_vars file not found: {env_path}")
            return {}
        with open(env_path, 'r') as f:
            env_data = yaml.safe_load(f)
        if not isinstance(env_data, dict):
            print(f"WARNING: env_vars file must contain a dict, got: {type(env_data)}")
            return {}
        # Convert all values to strings
        return {str(k): str(v) for k, v in env_data.items()}

    def get_nodelist(self) -> str:
        """Convert nodes list to comma-separated nodelist."""
        return ','.join(self.nodes)

    def build_run_args(self) -> str:
        """Build RUN_ARGS for worker."""
        base_args = (
            f"--benchmarks={self.global_config['benchmark']} "
            f"--scenarios={self.global_config['scenario']} "
            f"--core_type=disagg_{self.worker_type} "
            f"--dynamo_frontend_host={self.frontend_node} "
            f"--mpi_mode=leader"
        )
        # Use trtllm_yml_override (YAML path) - config_id is not supported
        args = f"{base_args} --trtllm_yml_override={self.config_path}"

        # Append --test_mode=AccuracyOnly when --accuracy flag is set
        if self.global_config.get('accuracy'):
            args = f"{args} --test_mode=AccuracyOnly"

        return args

    def launch(self):
        """Launch workers using run_scaleout.sh."""
        workspace = self.get_workspace()
        jobid = self.get_slurm_jobid()
        nodelist = self.get_nodelist()
        worker_log_dir = self.log_dir / self.worker_type
        worker_log_dir.mkdir(parents=True, exist_ok=True)

        # Use local run_scaleout.sh from dynamo_disagg directory
        script_dir = Path(__file__).parent
        cmd = [
            str(script_dir / 'run_scaleout.sh'),
            f'--workspace={workspace}',  # Explicit workspace to override SLURM_SUBMIT_DIR
            f'--jobid={jobid}',
            f'--nodelist={nodelist}',
            '--stage=server',
            f'--atomic-system={self.system}',
            f'--dp-multiplicity={self.num_workers}',
            f'--gpus-per-node={self.gpus_per_node}',
            f'--gpu-offset={self.gpu_offset}',
            f'--base-port={self.base_port}',
            f'--container-image={self.get_container_image()}',
            f'--log-dir={worker_log_dir}',
            f'--run-args={self.build_run_args()}'
        ]

        # Redirect run_scaleout.sh's own output to log files
        log_prefix = str(worker_log_dir / 'run_scaleout')
        # Pass custom env vars to subprocess (propagated via srun --export=ALL)
        self.run_command(cmd, f"Launch {self.worker_type.title()} Workers on {nodelist}",
                         background=True, log_prefix=log_prefix, env=self.env_vars if self.env_vars else None)


class PrefillWorkerSrun(WorkerSrun):
    """Prefill (context) worker srun step."""

    def __init__(self, config: dict, global_config: dict, allocated_nodes: dict, log_dir: Path, dry_run: bool = False, gpu_offset: int = 0):
        super().__init__(config, global_config, 'prefill', allocated_nodes, log_dir, dry_run, gpu_offset)


class DecodeWorkerSrun(WorkerSrun):
    """Decode (generation) worker srun step."""

    def __init__(self, config: dict, global_config: dict, allocated_nodes: dict, log_dir: Path, dry_run: bool = False, gpu_offset: int = 0):
        super().__init__(config, global_config, 'decode', allocated_nodes, log_dir, dry_run, gpu_offset)


class DisaggCluster:
    """Orchestrator for disaggregated serving cluster deployment."""

    def __init__(self, config_path: Path = None, config_dict: dict = None,
                 container_image: str = None, storage_path: str = None,
                 accuracy: bool = False, config_id: str = 'dynamo_cluster',
                 dry_run: bool = False, verbose: bool = False,
                 nodelist: List[str] = None):
        """Initialize DisaggCluster.

        Args:
            config_path: Path to YAML config file (mutually exclusive with config_dict)
            config_dict: Pre-loaded config dict from MLPerf system config (mutually exclusive with config_path)
            container_image: Container image path (required, always from CLI)
            storage_path: Path to shared storage containing models/data (optional, from CLI)
            accuracy: If True, pass --test_mode=AccuracyOnly to server workers (enables _accuracy.yml overrides)
            config_id: Config ID for harness (default: 'dynamo_cluster')
            dry_run: Print commands without executing
            verbose: Enable verbose output
            nodelist: Optional nodelist for testing (bypasses SLURM)
        """
        if config_path is None and config_dict is None:
            raise ValueError("Either config_path or config_dict must be provided")
        if config_path is not None and config_dict is not None:
            raise ValueError("Only one of config_path or config_dict can be provided")
        if container_image is None:
            raise ValueError("container_image is required")

        self.config_path = Path(config_path) if config_path else None
        self.storage_path = storage_path  # CLI override for storage path
        self.config_id = config_id  # Config ID for harness
        self.dry_run = dry_run
        self.verbose = verbose
        self._override_nodelist = nodelist  # For testing without SLURM

        if config_dict is not None:
            self.config = config_dict
        else:
            self.config = self._load_config()

        # Always set container_image from CLI
        self.config['container_image'] = container_image

        # Set storage_path if provided via CLI
        if storage_path:
            self.config['storage_path'] = storage_path

        # Store accuracy flag for server workers
        self.config['accuracy'] = accuracy

        self._validate_config()

        # Infer num_nodes from gpus_per_worker
        self._infer_num_nodes()

        # Get nodelist (from CLI override or SLURM)
        self.nodelist = self._fetch_nodelist()

        # Infer number of clusters
        self._infer_num_clusters()

        # Validate nodelist can be evenly divided
        self._validate_nodelist_divisibility()

        # Allocate nodes to all clusters
        self.cluster_allocations = self._allocate_all_clusters()

        # Create unified log directory
        self.log_dir = self._create_log_directory()

        # Initialize srun steps for all clusters
        self.clusters = []
        for i, allocation in enumerate(self.cluster_allocations):
            cluster_log_dir = self.log_dir / f'cluster_{i}'
            frontend_config = self._build_frontend_config()
            num_frontends = frontend_config.get('num_frontends', 1)

            # Create single FrontendSrun that launches all frontends via srun --ntasks=N
            frontend = FrontendSrun(
                frontend_config,
                self.config,
                allocation,
                cluster_log_dir,
                dry_run,
                num_frontends=num_frontends
            )

            cluster = {
                'id': i,
                'allocated_nodes': allocation,
                'log_dir': cluster_log_dir,
                'frontend': frontend,  # Single FrontendSrun (launches all N frontends)
                'num_frontends': num_frontends,  # For reference
                'prefill_workers': PrefillWorkerSrun(
                    self.config['prefill'],
                    self.config,
                    allocation,
                    cluster_log_dir,
                    dry_run,
                    gpu_offset=allocation.get('prefill_gpu_offset', 0)
                ),
                'decode_workers': DecodeWorkerSrun(
                    self.config['decode'],
                    self.config,
                    allocation,
                    cluster_log_dir,
                    dry_run,
                    gpu_offset=allocation.get('decode_gpu_offset', 0)
                )
            }
            self.clusters.append(cluster)

        # Generate worker mapping file
        self.generate_worker_mapping()

    def _load_config(self) -> dict:
        """Load and parse YAML configuration."""
        if self.config_path is None:
            raise RuntimeError("_load_config called but config_path is None")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def _require_fields(self, config: dict, fields: List[str], prefix: str = ""):
        """Validate that all required fields exist in config."""
        for field in fields:
            if field not in config:
                field_path = f"{prefix}.{field}" if prefix else field
                raise ValueError(f"{field_path} is required")

    def _validate_config(self):
        """Validate required fields in configuration."""
        # Validate top-level fields (container_image is set from CLI in __init__)
        self._require_fields(self.config,
                             ['benchmark', 'scenario', 'disagg_cluster', 'prefill', 'decode'])

        # Validate disagg_cluster
        self._require_fields(self.config['disagg_cluster'],
                             ['num_prefill_workers', 'num_decode_workers'],
                             prefix='disagg_cluster')

        # Default system to 'minimal' if not specified (used for frontend and harness)
        if 'system' not in self.config['disagg_cluster']:
            self.config['disagg_cluster']['system'] = 'minimal'

        # Validate workers
        for worker_type in ['prefill', 'decode']:
            worker_config = self.config[worker_type]

            # Required fields
            self._require_fields(worker_config,
                                 ['gpus_per_worker', 'gpus_per_node', 'system'],
                                 prefix=worker_type)

            # config (YAML path via trtllm_yml_override) is REQUIRED
            # NOTE: config_id (Python config loading) is NOT supported - it's broken for server configs
            if 'config_id' in worker_config:
                raise ValueError(
                    f"{worker_type}: 'config_id' is not supported for dynamo workers. "
                    f"Loading server configs from Python is broken. "
                    f"Please use 'config' (trtllm_yml_override) with a YAML config path instead."
                )
            if 'config' not in worker_config:
                raise ValueError(
                    f"{worker_type}: 'config' (trtllm_yml_override YAML path) is required. "
                    f"Please specify the YAML config path for {worker_type} workers."
                )

            # Validate total GPU requirements
            gpus_per_worker = worker_config['gpus_per_worker']
            gpus_per_node = worker_config['gpus_per_node']
            num_workers = self.config['disagg_cluster'][f'num_{worker_type}_workers']
            total_gpus = gpus_per_worker * num_workers

            # Validate gpus_per_worker fits evenly in gpus_per_node (for intra-node)
            # or gpus_per_node fits evenly in gpus_per_worker (for cross-node)
            if gpus_per_worker < gpus_per_node:
                # Intra-node: multiple workers per node
                if gpus_per_node % gpus_per_worker != 0:
                    raise ValueError(
                        f"{worker_type}: gpus_per_node ({gpus_per_node}) must be evenly divisible by "
                        f"gpus_per_worker ({gpus_per_worker}) for intra-node deployment"
                    )
            else:
                # Cross-node: worker spans multiple nodes
                if gpus_per_worker % gpus_per_node != 0:
                    raise ValueError(
                        f"{worker_type}: gpus_per_worker ({gpus_per_worker}) must be evenly divisible by "
                        f"gpus_per_node ({gpus_per_node}) for cross-node deployment"
                    )

        # Validate total GPUs matches system name suffix
        prefill_total = self.config['prefill']['gpus_per_worker'] * self.config['disagg_cluster']['num_prefill_workers']
        decode_total = self.config['decode']['gpus_per_worker'] * self.config['disagg_cluster']['num_decode_workers']
        total_gpus = prefill_total + decode_total

        # Parse expected total from system name (e.g., "aarch64x72" -> 72)
        system_name = self.config['disagg_cluster'].get('system', '')
        try:
            expected_gpus = int(system_name.rsplit('x', 1)[-1])
            if total_gpus != expected_gpus:
                print(f"WARNING: Total GPUs ({total_gpus}) doesn't match system name suffix (x{expected_gpus})")
                print(f"  Prefill: {self.config['disagg_cluster']['num_prefill_workers']} workers × {self.config['prefill']['gpus_per_worker']} GPUs = {prefill_total}")
                print(f"  Decode: {self.config['disagg_cluster']['num_decode_workers']} workers × {self.config['decode']['gpus_per_worker']} GPUs = {decode_total}")
                print(f"  Total: {total_gpus} GPUs")
        except (ValueError, IndexError):
            pass  # Can't parse system name, skip validation

        # Validate uniform gpus_per_node across prefill and decode
        prefill_gpus_per_node = self.config['prefill']['gpus_per_node']
        decode_gpus_per_node = self.config['decode']['gpus_per_node']
        if prefill_gpus_per_node != decode_gpus_per_node:
            raise ValueError(
                f"gpus_per_node must be the same for prefill and decode. "
                f"Got prefill.gpus_per_node={prefill_gpus_per_node}, decode.gpus_per_node={decode_gpus_per_node}"
            )

    def _infer_num_nodes(self):
        """Infer num_nodes for workers based on total GPU requirements.

        Uses ceiling division to support partial node allocation (intra-node workers).
        Also calculates total GPUs and checks if prefill+decode can share nodes.
        """
        import math
        for worker_type in ['prefill', 'decode']:
            worker_config = self.config[worker_type]

            gpus_per_worker = worker_config['gpus_per_worker']
            gpus_per_node = worker_config['gpus_per_node']
            num_workers = self.config['disagg_cluster'][f'num_{worker_type}_workers']

            # Calculate total GPUs needed
            total_gpus = gpus_per_worker * num_workers
            worker_config['total_gpus'] = total_gpus

            # Calculate num_nodes using ceiling division (supports partial nodes)
            num_nodes = math.ceil(total_gpus / gpus_per_node)

            # Set the inferred value
            worker_config['num_nodes'] = num_nodes

            # Calculate workers per node for verbose output
            workers_per_node = gpus_per_node // gpus_per_worker if gpus_per_worker <= gpus_per_node else 1

            if self.verbose:
                if gpus_per_worker < gpus_per_node:
                    print(f"Inferred {worker_type}.num_nodes = {num_nodes} "
                          f"(intra-node: {num_workers} workers × {gpus_per_worker} GPUs = {total_gpus} total, "
                          f"up to {workers_per_node} workers/node)")
                else:
                    nodes_per_worker = gpus_per_worker // gpus_per_node
                    print(f"Inferred {worker_type}.num_nodes = {num_nodes} "
                          f"(cross-node: {num_workers} workers × {nodes_per_worker} nodes/worker)")

        # Determine allocation mode based on worker sizes
        # Two modes:
        # 1. SHARED: Both intra-node (< gpus_per_node) AND combined GPUs fit in one node
        # 2. SEPARATE: All other cases (cross-node, mixed, or intra-node needing multiple nodes)
        #
        # Validation (already done in _validate_config):
        # - Cross-node (>= gpus_per_node): gpus_per_worker % gpus_per_node == 0
        # - Intra-node (< gpus_per_node): gpus_per_node % gpus_per_worker == 0
        prefill_gpus = self.config['prefill']['total_gpus']
        decode_gpus = self.config['decode']['total_gpus']
        gpus_per_node = self.config['prefill']['gpus_per_node']  # Same for both
        combined_gpus = prefill_gpus + decode_gpus

        prefill_gpus_per_worker = self.config['prefill']['gpus_per_worker']
        decode_gpus_per_worker = self.config['decode']['gpus_per_worker']
        prefill_intra = prefill_gpus_per_worker < gpus_per_node
        decode_intra = decode_gpus_per_worker < gpus_per_node

        # Only enable shared allocation when BOTH are intra-node AND combined fits in one node
        if prefill_intra and decode_intra and combined_gpus <= gpus_per_node:
            self.config['_allocation_mode'] = 'shared'
            self.config['_can_share_nodes'] = True
            self.config['_shared_node_count'] = 1
            if self.verbose:
                print(f"\nAllocation Mode: SHARED (single node)")
                print(f"  Combined GPUs: {prefill_gpus} (prefill) + {decode_gpus} (decode) = {combined_gpus} <= {gpus_per_node}")
        else:
            # Separate allocation for all other cases
            self.config['_allocation_mode'] = 'separate'
            self.config['_can_share_nodes'] = False
            self.config['_shared_node_count'] = None
            if self.verbose:
                print(f"\nAllocation Mode: SEPARATE")
                print(f"  Prefill: {'intra-node' if prefill_intra else 'cross-node'} ({prefill_gpus_per_worker} GPUs/worker)")
                print(f"  Decode: {'intra-node' if decode_intra else 'cross-node'} ({decode_gpus_per_worker} GPUs/worker)")

    def _get_slurm_jobid(self) -> str:
        """Get SLURM job ID from config or environment."""
        if 'slurm_jobid' in self.config and self.config['slurm_jobid']:
            return str(self.config['slurm_jobid'])

        jobid = os.environ.get('SLURM_JOBID')
        if not jobid:
            if self.dry_run and self._override_nodelist:
                # Use a fake job ID for dry-run testing
                return 'DRYRUN'
            raise RuntimeError("SLURM_JOBID not found. Run inside salloc or specify in config.")
        return jobid

    def _create_log_directory(self) -> Path:
        """Create unified log directory for all components."""
        timestamp = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
        jobid = self._get_slurm_jobid()
        benchmark = self.config['benchmark']

        # Get workspace path
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--show-toplevel'],
                capture_output=True,
                text=True,
                check=True
            )
            git_root = Path(result.stdout.strip())
            workspace = git_root / 'closed' / 'NVIDIA'
        except subprocess.CalledProcessError:
            workspace = Path.cwd()

        log_dir = workspace / 'build' / 'logs' / f'disagg_{benchmark}_slurm-{jobid}_{timestamp}'
        log_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"Log directory: {log_dir}")

        return log_dir

    def _fetch_nodelist(self) -> List[str]:
        """Fetch nodelist from CLI override or SLURM."""
        if self._override_nodelist:
            if self.verbose:
                print(f"Using CLI nodelist: {', '.join(self._override_nodelist)}")
            return self._override_nodelist

        return self._fetch_nodelist_from_slurm()

    def _fetch_nodelist_from_slurm(self) -> List[str]:
        """Fetch nodelist from SLURM job ID."""
        jobid = self._get_slurm_jobid()

        if self.verbose:
            print(f"Fetching nodelist for SLURM job {jobid}...")

        try:
            # Get node range from squeue
            cmd = ['squeue', '-j', jobid, '-h', '-o', '%N']
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            node_range = result.stdout.strip()

            if not node_range:
                raise RuntimeError(f"Could not get nodelist for job {jobid}")

            # Expand node range using scontrol
            cmd = ['scontrol', 'show', 'hostnames', node_range]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            nodes = result.stdout.strip().split('\n')

            if self.verbose:
                print(f"Found {len(nodes)} nodes: {', '.join(nodes)}")

            return nodes
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to fetch nodelist from SLURM: {e}")

    def _infer_num_clusters(self):
        """Infer or use explicit number of clusters.

        If `num_clusters` is specified in disagg_cluster config, use that value.
        Otherwise, infer from nodelist size.
        """
        prefill_nodes = self.config['prefill']['num_nodes']
        decode_nodes = self.config['decode']['num_nodes']
        allocation_mode = self.config.get('_allocation_mode', 'separate')

        if allocation_mode == 'shared':
            self.nodes_per_cluster = 1
            self.use_shared_allocation = True
        else:
            # Separate allocation: distinct node sets for prefill and decode
            self.nodes_per_cluster = prefill_nodes + decode_nodes
            self.use_shared_allocation = False

        total_nodes = len(self.nodelist)

        # Check for explicit num_clusters in config
        explicit_num_clusters = self.config['disagg_cluster'].get('num_clusters', None)

        if explicit_num_clusters is not None:
            # User explicitly specified number of clusters
            self.num_clusters = explicit_num_clusters
            required_nodes = self.num_clusters * self.nodes_per_cluster
            if required_nodes > total_nodes:
                raise ValueError(
                    f"Explicit num_clusters={self.num_clusters} requires {required_nodes} nodes "
                    f"({self.num_clusters} clusters × {self.nodes_per_cluster} nodes/cluster), "
                    f"but only {total_nodes} nodes available."
                )
            if self.verbose and required_nodes < total_nodes:
                print(f"\nNote: Using {required_nodes} of {total_nodes} available nodes "
                      f"(num_clusters={self.num_clusters} explicitly set)")
        else:
            # Infer from nodelist size
            self.num_clusters = total_nodes // self.nodes_per_cluster

        if self.verbose:
            print(f"\nCluster Sizing:")
            if allocation_mode == 'shared':
                print(f"  Nodes per cluster: {self.nodes_per_cluster} (shared: prefill+decode on same node)")
            else:
                print(f"  Nodes per cluster: {self.nodes_per_cluster} ({prefill_nodes} prefill + {decode_nodes} decode)")
            print(f"  Total nodes available: {total_nodes}")
            print(f"  Number of clusters: {self.num_clusters}")
            if explicit_num_clusters is not None:
                print(f"  (explicit: num_clusters={explicit_num_clusters} in config)")

    def _validate_nodelist_divisibility(self):
        """Validate that nodelist can be evenly divided into clusters."""
        total_nodes = len(self.nodelist)
        if total_nodes % self.nodes_per_cluster != 0:
            raise ValueError(
                f"Nodelist size ({total_nodes}) must be evenly divisible by nodes per cluster ({self.nodes_per_cluster}). "
                f"Got remainder: {total_nodes % self.nodes_per_cluster}"
            )

    def _build_frontend_config(self) -> dict:
        """Build frontend config from disagg_cluster settings."""
        frontend_config = {
            'num_prefill_workers': self.config['disagg_cluster']['num_prefill_workers'],
            'num_decode_workers': self.config['disagg_cluster']['num_decode_workers'],
            'system': self.config['disagg_cluster']['system'],
            'num_frontends': self.config['disagg_cluster'].get('num_frontends', 1),
            'distribute_frontends': self.config['disagg_cluster'].get('distribute_frontends', False)
        }

        # Add frontend-specific flags if provided
        if 'frontend' in self.config['disagg_cluster']:
            frontend_flags = self.config['disagg_cluster']['frontend']
            if 'router_mode' in frontend_flags:
                frontend_config['router_mode'] = frontend_flags['router_mode']
            if 'kv_overlap_weight' in frontend_flags:
                frontend_config['kv_overlap_weight'] = frontend_flags['kv_overlap_weight']
            # Override num_frontends if specified in frontend section
            if 'num_frontends' in frontend_flags:
                frontend_config['num_frontends'] = frontend_flags['num_frontends']
            # Override distribute_frontends if specified in frontend section
            if 'distribute_frontends' in frontend_flags:
                frontend_config['distribute_frontends'] = frontend_flags['distribute_frontends']
            # Add any additional frontend flags
            for key, value in frontend_flags.items():
                if key not in frontend_config:
                    frontend_config[key] = value

        return frontend_config

    def _allocate_all_clusters(self) -> List[dict]:
        """Allocate nodes to all clusters.

        Two allocation modes:
        1. shared: Both intra-node and fit within 1 node - GPU-level allocation with offsets
        2. separate: All other cases - node-level allocation with distinct node sets
        """
        allocations = []
        prefill_num = self.config['prefill']['num_nodes']
        decode_num = self.config['decode']['num_nodes']
        allocation_mode = self.config.get('_allocation_mode', 'separate')

        # Check if frontends should be distributed across nodes
        distribute_frontends = self.config['disagg_cluster'].get('distribute_frontends', False)
        if 'frontend' in self.config['disagg_cluster']:
            distribute_frontends = self.config['disagg_cluster']['frontend'].get(
                'distribute_frontends', distribute_frontends)
        num_frontends = self.config['disagg_cluster'].get('num_frontends', 1)

        for i in range(self.num_clusters):
            start_idx = i * self.nodes_per_cluster
            end_idx = start_idx + self.nodes_per_cluster
            cluster_nodes = self.nodelist[start_idx:end_idx]

            if allocation_mode == 'shared':
                # Shared allocation: both worker types use the same node
                # Prefill starts at GPU 0, decode starts after prefill's GPUs
                prefill_gpus = self.config['prefill']['total_gpus']
                decode_gpus = self.config['decode']['total_gpus']

                # Determine frontend node allocation for shared mode
                if distribute_frontends and num_frontends > 1:
                    # Distributed: use first N nodes for frontends
                    frontend_allocation = cluster_nodes[:num_frontends]
                else:
                    # Stacked: all frontends on single node
                    frontend_allocation = cluster_nodes[0]

                allocation = {
                    'frontend': frontend_allocation,
                    'prefill': cluster_nodes,  # Single shared node
                    'decode': cluster_nodes,   # Same node
                    'prefill_gpu_offset': 0,
                    'decode_gpu_offset': prefill_gpus
                }

                if self.verbose:
                    print(f"\nCluster {i} Shared Allocation:")
                    print(f"  Node: {cluster_nodes[0]}")
                    fe_str = ', '.join(frontend_allocation) if isinstance(frontend_allocation, list) else frontend_allocation
                    print(f"  Frontend: {fe_str}" + (" (distributed)" if distribute_frontends else " (stacked)"))
                    print(f"  Prefill: GPUs 0-{prefill_gpus - 1} (offset: 0)")
                    print(f"  Decode:  GPUs {prefill_gpus}-{prefill_gpus + decode_gpus - 1} (offset: {prefill_gpus})")

            else:
                # Separate allocation: distinct node sets for prefill and decode
                prefill_nodes = cluster_nodes[:prefill_num]
                decode_nodes = cluster_nodes[prefill_num:prefill_num + decode_num]

                # Determine frontend node allocation - prefer prefill (CTX) nodes, overflow to decode (GEN)
                if distribute_frontends and num_frontends > 1:
                    # Distributed: place frontends on worker nodes, overflow round-robin
                    # Strategy:
                    # 1. Distribute one per node: CTX nodes first, then GEN nodes (port 8000)
                    # 2. Overflow: round-robin across CTX then GEN nodes (port 8001, 8002, ...)
                    all_worker_nodes = prefill_nodes + decode_nodes
                    total_worker_nodes = len(all_worker_nodes)

                    if num_frontends <= total_worker_nodes:
                        # All frontends fit on one port per node
                        # Prefer CTX nodes first, then GEN nodes
                        if num_frontends <= len(prefill_nodes):
                            frontend_allocation = prefill_nodes[:num_frontends]
                        else:
                            overflow_count = num_frontends - len(prefill_nodes)
                            frontend_allocation = prefill_nodes + decode_nodes[:overflow_count]
                    else:
                        # More frontends than nodes - round-robin across all nodes
                        # CTX nodes port 0, GEN nodes port 0, CTX nodes port 1, GEN nodes port 1, ...
                        frontend_allocation = []
                        remaining = num_frontends
                        port_offset = 0
                        while remaining > 0:
                            # Add CTX nodes at current port
                            for node in prefill_nodes:
                                if remaining <= 0:
                                    break
                                frontend_allocation.append((node, port_offset))
                                remaining -= 1
                            # Add GEN nodes at current port
                            for node in decode_nodes:
                                if remaining <= 0:
                                    break
                                frontend_allocation.append((node, port_offset))
                                remaining -= 1
                            port_offset += 1
                else:
                    # Stacked: all frontends on first prefill node
                    frontend_allocation = prefill_nodes[0]

                allocation = {
                    'frontend': frontend_allocation,
                    'prefill': prefill_nodes,
                    'decode': decode_nodes,
                    'prefill_gpu_offset': 0,
                    'decode_gpu_offset': 0
                }

                if self.verbose:
                    print(f"\nCluster {i} Separate Allocation:")
                    if distribute_frontends and isinstance(frontend_allocation, list):
                        if frontend_allocation and isinstance(frontend_allocation[0], tuple):
                            # Tuple allocation: (node, port_offset)
                            fe_str = ', '.join([f"{node}:{8000+port}" for node, port in frontend_allocation])
                            num_port0 = sum(1 for _, p in frontend_allocation if p == 0)
                            num_overflow = sum(1 for _, p in frontend_allocation if p > 0)
                            max_port = max(p for _, p in frontend_allocation)
                            if max_port > 0:
                                fe_mode = f"(round-robin across {max_port + 1} port(s))"
                            else:
                                fe_mode = f"(distributed across {num_port0} nodes)"
                        else:
                            fe_str = ', '.join(frontend_allocation)
                            if num_frontends <= len(prefill_nodes):
                                fe_mode = f"(distributed: {num_frontends} on CTX)"
                            else:
                                fe_mode = f"(distributed: {len(prefill_nodes)} on CTX, {num_frontends - len(prefill_nodes)} on GEN)"
                    else:
                        fe_str = frontend_allocation if isinstance(frontend_allocation, str) else frontend_allocation
                        fe_mode = "(stacked)"
                    print(f"  Frontend: {fe_str} {fe_mode}")
                    print(f"  Prefill:  {', '.join(allocation['prefill'])} ({len(allocation['prefill'])} nodes)")
                    print(f"  Decode:   {', '.join(allocation['decode'])} ({len(allocation['decode'])} nodes)")

            allocations.append(allocation)

        return allocations

    def print_summary(self):
        """Print deployment summary."""
        allocation_mode = self.config.get('_allocation_mode', 'separate')

        print(f"\n{'='*80}")
        print("Disaggregated Serving Deployment")
        print(f"{'='*80}")
        print(f"Config: {self.config_path}")
        print(f"Benchmark: {self.config['benchmark']}")
        print(f"Scenario: {self.config['scenario']}")
        print(f"Total nodes: {len(self.nodelist)}")
        print(f"Number of clusters: {self.num_clusters}")
        print(f"Nodes per cluster: {self.nodes_per_cluster}")
        print(f"Allocation mode: {allocation_mode.upper()}")
        print(f"Log directory: {self.log_dir}")
        print(f"Worker mapping: {self.log_dir / 'worker_mapping.txt'}")

        for i, cluster in enumerate(self.clusters):
            print(f"\n--- Cluster {i} ---")
            frontend = cluster['frontend']
            num_frontends = cluster['num_frontends']
            print(f"  Frontends: {num_frontends} on {frontend.node}:{frontend.base_port}-{frontend.base_port + num_frontends - 1}")
            if num_frontends > 1:
                print(f"  Frontend URLs: {', '.join(frontend.get_all_urls())}")
            prefill = cluster['prefill_workers']
            decode = cluster['decode_workers']
            if allocation_mode == 'shared':
                print(f"  Nodes: {prefill.get_nodelist()} (shared)")
                print(f"  Prefill: {prefill.num_workers} workers × {prefill.gpus_per_worker} GPUs, offset={prefill.gpu_offset}, port={prefill.base_port}")
                print(f"  Decode:  {decode.num_workers} workers × {decode.gpus_per_worker} GPUs, offset={decode.gpu_offset}, port={decode.base_port}")
            else:
                print(f"  Prefill: {prefill.get_nodelist()} ({len(prefill.nodes)} nodes, {prefill.num_workers} workers × {prefill.gpus_per_worker} GPUs)")
                print(f"  Decode:  {decode.get_nodelist()} ({len(decode.nodes)} nodes, {decode.num_workers} workers × {decode.gpus_per_worker} GPUs)")

        if self.dry_run:
            print("\n*** DRY RUN MODE - No commands will be executed ***")
        print(f"{'='*80}\n")

    def print_completion_info(self):
        """Print deployment completion information."""
        print(f"\n{'='*80}")
        print("Deployment Complete!")
        print(f"{'='*80}")
        print(f"Number of clusters: {self.num_clusters}")
        print(f"Log directory: {self.log_dir}")

        for i, cluster in enumerate(self.clusters):
            print(f"\n--- Cluster {i} ---")
            frontend = cluster['frontend']
            print(f"  Frontend URL(s): {', '.join(frontend.get_all_urls())}")
            print(f"  Test: curl {frontend.get_url()}/v1/models")

        print("\nTo check logs:")
        for i, cluster in enumerate(self.clusters):
            cluster_log_dir = cluster['log_dir']
            num_frontends = cluster['num_frontends']
            print(f"  Cluster {i}:")
            print(f"    tail -f {cluster_log_dir}/frontend/fe_*/disagg_frontend.log")
            print(f"    tail -f {cluster_log_dir}/prefill/*.log")
            print(f"    tail -f {cluster_log_dir}/decode/*.log")

        print(f"\nWorker mapping: {self.log_dir / 'worker_mapping.txt'}")
        print(f"{'='*80}\n")

    def generate_worker_mapping(self) -> str:
        """Generate a mapping file that visualizes worker <-> node:gpu_id relationships.

        Returns the path to the generated mapping file.
        """
        gpus_per_node = self.config['prefill']['gpus_per_node']
        allocation_mode = self.config.get('_allocation_mode', 'separate')

        lines = []
        lines.append("=" * 80)
        lines.append("DISAGGREGATED SERVING - WORKER MAPPING")
        lines.append("=" * 80)
        lines.append(f"Allocation Mode: {allocation_mode.upper()}")
        lines.append(f"GPUs per Node: {gpus_per_node}")
        lines.append("")

        for cluster_idx, cluster in enumerate(self.clusters):
            lines.append(f"{'='*80}")
            lines.append(f"CLUSTER {cluster_idx}")
            lines.append(f"{'='*80}")

            allocation = cluster['allocated_nodes']
            prefill_workers = cluster['prefill_workers']
            decode_workers = cluster['decode_workers']

            # Frontends
            frontend = cluster['frontend']
            num_frontends = cluster['num_frontends']
            mode_str = "distributed" if frontend.distribute_frontends else "stacked"
            lines.append(f"\nFRONTENDS ({num_frontends} instances, {mode_str}):")
            for rank in range(num_frontends):
                primary_marker = " (primary - NATS/etcd)" if rank == 0 else ""
                if rank < len(frontend.frontend_assignments):
                    node, port_offset = frontend.frontend_assignments[rank]
                    port = frontend.base_port + port_offset
                else:
                    node, port = frontend.node, frontend.base_port + rank
                lines.append(f"  [rank {rank}] Node: {node}, Port: {port}{primary_marker}")
            lines.append(f"  URLs: {', '.join(frontend.get_all_urls())}")

            # Generate prefill mapping
            lines.append(f"\nPREFILL WORKERS ({prefill_workers.num_workers} workers × {prefill_workers.gpus_per_worker} GPUs):")
            lines.append(f"  Nodes: {prefill_workers.get_nodelist()}")
            lines.append(f"  GPU Offset: {prefill_workers.gpu_offset}")
            lines.append(f"  Base Port: {prefill_workers.base_port}")
            lines.append("")

            prefill_mapping = self._compute_worker_gpu_mapping(
                worker_type='prefill',
                nodes=prefill_workers.nodes,
                num_workers=prefill_workers.num_workers,
                gpus_per_worker=prefill_workers.gpus_per_worker,
                gpus_per_node=gpus_per_node,
                gpu_offset=prefill_workers.gpu_offset,
                base_port=prefill_workers.base_port
            )
            for worker_id, mapping in prefill_mapping.items():
                lines.append(f"  {worker_id}:")
                lines.append(f"    URL: {mapping['url']}")
                lines.append(f"    GPUs: {mapping['gpu_list']}")

            # Generate decode mapping
            lines.append(f"\nDECODE WORKERS ({decode_workers.num_workers} workers × {decode_workers.gpus_per_worker} GPUs):")
            lines.append(f"  Nodes: {decode_workers.get_nodelist()}")
            lines.append(f"  GPU Offset: {decode_workers.gpu_offset}")
            lines.append(f"  Base Port: {decode_workers.base_port}")
            lines.append("")

            decode_mapping = self._compute_worker_gpu_mapping(
                worker_type='decode',
                nodes=decode_workers.nodes,
                num_workers=decode_workers.num_workers,
                gpus_per_worker=decode_workers.gpus_per_worker,
                gpus_per_node=gpus_per_node,
                gpu_offset=decode_workers.gpu_offset,
                base_port=decode_workers.base_port
            )
            for worker_id, mapping in decode_mapping.items():
                lines.append(f"  {worker_id}:")
                lines.append(f"    URL: {mapping['url']}")
                lines.append(f"    GPUs: {mapping['gpu_list']}")

            # Visual node diagram
            lines.append(f"\nNODE DIAGRAM:")
            lines.append("-" * 60)
            node_diagram = self._generate_node_diagram(
                cluster_idx, prefill_mapping, decode_mapping, gpus_per_node
            )
            lines.extend(node_diagram)
            lines.append("")

        # Write to file
        mapping_file = self.log_dir / 'worker_mapping.txt'
        with open(mapping_file, 'w') as f:
            f.write('\n'.join(lines))

        return str(mapping_file)

    def _compute_worker_gpu_mapping(self, worker_type: str, nodes: List[str],
                                    num_workers: int, gpus_per_worker: int,
                                    gpus_per_node: int, gpu_offset: int,
                                    base_port: int) -> dict:
        """Compute mapping from worker ID to node:gpu_ids.

        Handles both cross-node (gpus_per_worker >= gpus_per_node) and
        intra-node (gpus_per_worker < gpus_per_node) cases.
        """
        mapping = {}

        if gpus_per_worker >= gpus_per_node:
            # Cross-node: worker spans multiple nodes
            nodes_per_worker = gpus_per_worker // gpus_per_node
            for worker_idx in range(num_workers):
                start_node = worker_idx * nodes_per_worker
                worker_nodes = nodes[start_node:start_node + nodes_per_worker]
                gpu_list = []
                for node in worker_nodes:
                    gpu_list.append(f"{node}:0-{gpus_per_node - 1}")
                mapping[f"{worker_type}_{worker_idx}"] = {
                    'url': f"{worker_nodes[0]}:{base_port}",
                    'gpu_list': ', '.join(gpu_list)
                }
        else:
            # Intra-node: multiple workers per node
            workers_per_node = gpus_per_node // gpus_per_worker
            worker_idx = 0
            for node_idx, node in enumerate(nodes):
                for slot in range(workers_per_node):
                    if worker_idx >= num_workers:
                        break
                    start_gpu = gpu_offset + slot * gpus_per_worker
                    end_gpu = start_gpu + gpus_per_worker - 1
                    port = base_port + slot
                    mapping[f"{worker_type}_{worker_idx}"] = {
                        'url': f"{node}:{port}",
                        'gpu_list': f"{node}:{start_gpu}-{end_gpu}"
                    }
                    worker_idx += 1

        return mapping

    def _generate_node_diagram(self, cluster_idx: int, prefill_mapping: dict,
                               decode_mapping: dict, gpus_per_node: int) -> List[str]:
        """Generate ASCII diagram showing GPU allocation per node."""
        lines = []

        # Collect all nodes and their GPU assignments
        node_gpus = {}  # node -> {gpu_id: worker_type}

        for worker_id, mapping in prefill_mapping.items():
            for gpu_spec in mapping['gpu_list'].split(', '):
                if ':' in gpu_spec:
                    node, gpu_range = gpu_spec.rsplit(':', 1)
                    if node not in node_gpus:
                        node_gpus[node] = {}
                    if '-' in gpu_range:
                        start, end = map(int, gpu_range.split('-'))
                        for g in range(start, end + 1):
                            node_gpus[node][g] = 'P'  # P for Prefill
                    else:
                        node_gpus[node][int(gpu_range)] = 'P'

        for worker_id, mapping in decode_mapping.items():
            for gpu_spec in mapping['gpu_list'].split(', '):
                if ':' in gpu_spec:
                    node, gpu_range = gpu_spec.rsplit(':', 1)
                    if node not in node_gpus:
                        node_gpus[node] = {}
                    if '-' in gpu_range:
                        start, end = map(int, gpu_range.split('-'))
                        for g in range(start, end + 1):
                            node_gpus[node][g] = 'D'  # D for Decode
                    else:
                        node_gpus[node][int(gpu_range)] = 'D'

        # Generate diagram
        lines.append(f"Legend: P=Prefill, D=Decode, .=Unused")
        lines.append("")

        for node in sorted(node_gpus.keys()):
            gpu_chars = []
            for g in range(gpus_per_node):
                gpu_chars.append(node_gpus[node].get(g, '.'))
            gpu_str = ' '.join(gpu_chars)
            lines.append(f"  {node}: [{gpu_str}]")
            lines.append(f"  {'':>{len(node)}}   " + ' '.join(str(g) for g in range(gpus_per_node)))

        return lines

    def _cancel_job_steps(self):
        """Cancel all job steps for this SLURM job using scancel."""
        jobid = self._get_slurm_jobid()
        if not jobid:
            return

        print(f"Cancelling all job steps for job {jobid}...")
        try:
            # Get list of running steps
            result = subprocess.run(
                ['squeue', '--steps', '-j', jobid, '-h', '-o', '%i'],
                capture_output=True, text=True, timeout=10
            )
            steps = [s.strip() for s in result.stdout.strip().split('\n') if s.strip()]

            if steps:
                print(f"  Found {len(steps)} running step(s): {', '.join(steps)}")
                # Cancel all steps
                subprocess.run(
                    ['scancel', '--signal=TERM'] + steps,
                    capture_output=True, timeout=30
                )
                # Give time for graceful shutdown
                time.sleep(3)
                # Force cancel if still running
                subprocess.run(
                    ['scancel'] + steps,
                    capture_output=True, timeout=30
                )
                print("  All steps cancelled.")
            else:
                print("  No running steps found.")
        except Exception as e:
            print(f"  Warning: Failed to cancel steps: {e}")

    def wait_for_processes(self):
        """Wait for all background processes to complete.

        This is essential for sbatch jobs - if the main script exits,
        all background srun processes would be terminated.
        """
        processes = []
        for cluster in self.clusters:
            # Add frontend process (single srun launches all frontends)
            frontend = cluster['frontend']
            if frontend.process:
                processes.append(('frontend', frontend.process))
            if cluster['prefill_workers'].process:
                processes.append(('prefill', cluster['prefill_workers'].process))
            if cluster['decode_workers'].process:
                processes.append(('decode', cluster['decode_workers'].process))

        if not processes:
            return

        print(f"Waiting for {len(processes)} background process(es)...")
        print("Press Ctrl+C to terminate all processes.\n")

        try:
            # Wait for any process to exit
            while processes:
                for name, proc in processes[:]:
                    ret = proc.poll()
                    if ret is not None:
                        print(f"Process '{name}' exited with code {ret}")
                        processes.remove((name, proc))
                if processes:
                    time.sleep(1)
        except KeyboardInterrupt:
            print("\nReceived interrupt, cancelling SLURM job steps...")
            # Use scancel to reliably cancel all job steps
            self._cancel_job_steps()

            # Also terminate local processes
            for name, proc in processes:
                if proc.poll() is None:
                    proc.terminate()
            # Give processes time to terminate
            for name, proc in processes:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

        print("All processes have exited.")

    def run_harness(self, run_args: str, harness_node_idx: int = 0, cluster_idx: int = None, audit: bool = False):
        """Run harness against the deployed cluster(s).

        Args:
            run_args: RUN_ARGS string for make run_harness
            harness_node_idx: Index into nodelist for which node runs the harness (default: 0)
            cluster_idx: Index of specific cluster to target (default: None = all clusters)
            audit: If True, run audit harness (run_audit_harness) for compliance testing
        """
        # Build frontend URL(s) - includes ALL frontends from selected cluster(s)
        # Use frontend.get_all_urls() to get the correct distributed URLs (without http:// prefix)
        frontend_urls = []
        if cluster_idx is not None:
            # Target specific cluster (all its frontends)
            if cluster_idx < 0 or cluster_idx >= len(self.clusters):
                raise ValueError(f"cluster_idx ({cluster_idx}) out of range for {len(self.clusters)} clusters")
            cluster = self.clusters[cluster_idx]
            frontend = cluster['frontend']
            # get_all_urls returns http://node:port, strip the http:// prefix
            frontend_urls.extend([url.replace("http://", "") for url in frontend.get_all_urls()])
        else:
            # Target all clusters (all frontends from all clusters)
            for c in self.clusters:
                frontend = c['frontend']
                # get_all_urls returns http://node:port, strip the http:// prefix
                frontend_urls.extend([url.replace("http://", "") for url in frontend.get_all_urls()])

        frontend_url = ",".join(frontend_urls)
        harness_system = self.config['disagg_cluster']['system']  # Use disagg_cluster.system for harness

        # Select harness node from nodelist
        if harness_node_idx < 0 or harness_node_idx >= len(self.nodelist):
            raise ValueError(f"harness_node_idx ({harness_node_idx}) out of range for nodelist size ({len(self.nodelist)})")
        harness_node = self.nodelist[harness_node_idx]

        first_cluster = self.clusters[0]
        workspace = first_cluster['frontend'].get_workspace()
        harness_log_dir = self.log_dir / 'harness'
        harness_log_dir.mkdir(parents=True, exist_ok=True)

        # Add --trtllm_server_urls to connect to the frontend if not already specified
        if '--trtllm_server_urls' not in run_args:
            run_args = f"{run_args} --trtllm_server_urls={frontend_url}"

        # Add --config_id to use ATOMIC_EXPORTS config for harness (if dynamo_cluster)
        # This ensures harness uses the same loadgen settings (QPS, etc.) as the dynamo cluster
        # NOTE: For audit runs of gpt-oss-120b, don't pass config_id - COMPLIANCE_OVERRIDES
        # only work with EXPORTS (default config), not ATOMIC_EXPORTS (code/main.py doesn't
        # apply COMPLIANCE_OVERRIDES in the atomic config path). The audit harness needs
        # COMPLIANCE_OVERRIDES to apply tensor_path and min_query_count overrides for TEST07/TEST09.
        if '--config_id' not in run_args and not (audit and 'gpt-oss' in self.config.get('benchmark', '')):
            run_args = f"{run_args} --config_id={self.config_id}"

        # Pass RUN_ARGS and SYSTEM_NAME as Make variables (not env vars)
        # This is the same pattern used by run_scaleout.sh
        harness_target = 'run_audit_harness' if audit else 'run_harness'
        cmd = [
            'srun',
            '--overlap',
            f'--jobid={self._get_slurm_jobid()}',
            '--nodes=1',
            '--ntasks=1',
            f'--nodelist={harness_node}',
            f'--output={harness_log_dir}/harness.stdout',
            f'--error={harness_log_dir}/harness.stderr',
            f'--container-image={self.config["container_image"]}',
            f'--container-mounts={workspace}:/work,{first_cluster["frontend"].get_storage_path()}:/home/mlperf_inference_storage',
            '--container-workdir=/work',
            '--container-remap-root',
            '--export=MLPINF_HTTP_USE_COMPLETIONS=1,MLPINF_USE_DYNAMO=1',
            'make', harness_target,
            f'RUN_ARGS={run_args}',
            f'SYSTEM_NAME={harness_system}',
            'NO_DISPLAY_RESULTS=0'  # Ensure display_results runs
        ]

        print(f"\n{'='*80}")
        print(f"Running {'Audit ' if audit else ''}Harness")
        print(f"{'='*80}")
        print(f"Harness node: {harness_node} (nodelist index: {harness_node_idx})")
        print(f"Frontend URL(s): {frontend_url}")
        total_frontends = sum(c['num_frontends'] for c in self.clusters)
        print(f"Targeting: {total_frontends} frontend(s) across {'all ' + str(len(self.clusters)) + ' clusters' if cluster_idx is None else f'cluster {cluster_idx}'}")
        print(f"System: {harness_system}")
        print(f"RUN_ARGS: {run_args}")
        print(f"Log directory: {harness_log_dir}")

        # Run harness in foreground (not background)
        first_cluster['frontend'].run_command(cmd, "Run Harness", background=False)

    def wait_for_servers_ready(self, timeout: int = 300, poll_interval: int = 10) -> bool:
        """Wait for frontend servers and workers to become ready.

        Performs a comprehensive health check:
        1. Check /v1/models to verify workers have registered
        2. Send a test inference request to verify both prefill and decode workers are operational

        Args:
            timeout: Maximum seconds to wait (default: 300)
            poll_interval: Seconds between health check attempts (default: 10)

        Returns:
            True if servers are ready, False if timeout reached
        """
        if self.dry_run:
            print(f"[DRY-RUN] Would wait for servers to be ready (timeout: {timeout}s)")
            return True

        # Build list of frontend URLs to check (handles both stacked and distributed modes)
        frontend_urls = []
        for cluster in self.clusters:
            frontend = cluster['frontend']
            frontend_urls.extend(frontend.get_all_urls())

        print(f"\nWaiting for {len(frontend_urls)} frontend(s) and workers to be ready (timeout: {timeout}s)...")
        for url in frontend_urls:
            print(f"  - {url}")

        start_time = time.time()
        ready_frontends = set()

        def check_frontend_ready(url: str) -> tuple:
            """Check if frontend and workers are ready.

            Returns:
                (is_ready, model_id, status_msg)
            """
            try:
                # Step 1: Check /v1/models to see if workers have registered
                models_url = f"{url}/v1/models"
                req = urllib.request.Request(models_url, method='GET')
                with urllib.request.urlopen(req, timeout=5) as response:
                    if response.status != 200:
                        return (False, None, "models endpoint not ready")
                    data = json.loads(response.read().decode('utf-8'))
                    if not data.get('data') or len(data['data']) == 0:
                        return (False, None, "no models registered")
                    model_id = data['data'][0].get('id', 'unknown')

                # Step 2: Send a test inference request to verify end-to-end
                # This confirms both prefill (context) and decode (generation) workers work
                test_url = f"{url}/v1/chat/completions"
                test_payload = json.dumps({
                    "model": model_id,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                    "stream": False
                }).encode('utf-8')
                test_req = urllib.request.Request(
                    test_url,
                    data=test_payload,
                    headers={'Content-Type': 'application/json'},
                    method='POST'
                )
                with urllib.request.urlopen(test_req, timeout=30) as test_response:
                    if test_response.status == 200:
                        return (True, model_id, "test request succeeded")
                    else:
                        return (False, model_id, f"test request failed with status {test_response.status}")

            except urllib.error.HTTPError as e:
                return (False, None, f"HTTP error {e.code}")
            except urllib.error.URLError as e:
                return (False, None, f"connection failed: {e.reason}")
            except (TimeoutError, ConnectionRefusedError):
                return (False, None, "connection timeout/refused")
            except json.JSONDecodeError:
                return (False, None, "invalid JSON response")
            except OSError as e:
                return (False, None, f"OS error: {e}")

        while time.time() - start_time < timeout:
            elapsed = int(time.time() - start_time)

            for url in frontend_urls:
                if url in ready_frontends:
                    continue

                is_ready, model_id, status_msg = check_frontend_ready(url)
                if is_ready:
                    ready_frontends.add(url)
                    print(f"  [{elapsed}s] {url} is ready! Model: {model_id} ({status_msg})")

            if len(ready_frontends) == len(frontend_urls):
                print(f"\nAll frontends and workers ready after {elapsed} seconds!")
                return True

            # Progress update every poll_interval
            print(f"  [{elapsed}s] {len(ready_frontends)}/{len(frontend_urls)} frontends ready, waiting...")
            time.sleep(poll_interval)

        # Timeout reached
        elapsed = int(time.time() - start_time)
        not_ready = [url for url in frontend_urls if url not in ready_frontends]
        print(f"\nWarning: Timeout after {elapsed}s. {len(not_ready)} frontend(s) not ready:")
        for url in not_ready:
            is_ready, model_id, status_msg = check_frontend_ready(url)
            print(f"  - {url}: {status_msg}")
        return False

    def deploy(self, run_harness_args: str = None, run_harness_nodeidx: int = 0, server_init_delay: int = 600, audit: bool = False):
        """Deploy all clusters.

        Args:
            run_harness_args: Custom RUN_ARGS for make run_harness (triggers harness run if provided)
            run_harness_nodeidx: Index into nodelist for harness node (default: 0)
            server_init_delay: Max seconds to wait for servers before running harness (default: 600).
                              Uses health check polling - proceeds early if servers are ready.
            audit: If True, run audit harness (run_audit_harness) for compliance testing.
        """
        self.print_summary()

        # Launch all clusters
        for i, cluster in enumerate(self.clusters):
            print(f"\n{'='*80}")
            print(f"Launching Cluster {i}")
            print(f"{'='*80}\n")

            # Launch all frontends with single srun --ntasks=N
            cluster['frontend'].launch()

            cluster['prefill_workers'].launch()
            cluster['decode_workers'].launch()

        self.print_completion_info()

        if run_harness_args is not None:
            # Wait for servers to be ready using health check polling
            servers_ready = self.wait_for_servers_ready(timeout=server_init_delay)
            if not servers_ready:
                print("\nERROR: Servers not ready after timeout. Aborting harness.")
                print("Check worker logs for initialization issues.")
                self._cancel_job_steps()
                raise RuntimeError("Servers failed to initialize within timeout")
            self.run_harness(
                run_args=run_harness_args,
                harness_node_idx=run_harness_nodeidx,
                audit=audit
            )
            # After harness completes, terminate servers
            print("\nHarness complete. Terminating servers...")
            self._cancel_job_steps()
        else:
            # Wait for all background processes (essential for sbatch)
            if not self.dry_run:
                self.wait_for_processes()


def main():
    parser = argparse.ArgumentParser(
        description='Launch disaggregated serving cluster from YAML or MLPerf system config',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Two configuration modes:

1. YAML configuration file (--config):
   python3 scripts/slurm_llm/dynamo_disagg/launch_disagg_cluster.py \\
       --config scripts/slurm_llm/dynamo_disagg/config/sample/disagg_deployment_minimal.yaml \\
       --container-image /path/to/image.sqsh

2. MLPerf system configuration (--system):
   python3 scripts/slurm_llm/dynamo_disagg/launch_disagg_cluster.py \\
       --system GB200-NVL72_GB200-186GB_aarch64x20 \\
       --benchmark deepseek-r1 --scenario Interactive \\
       --container-image /path/to/image.sqsh

Example YAML configuration:
  benchmark: deepseek-r1
  scenario: Interactive

  disagg_cluster:
    num_prefill_workers: 1
    num_decode_workers: 1

  prefill:
    system: GB200-NVL72_GB200-186GB_aarch64x4
    gpus_per_worker: 4
    gpus_per_node: 4
    config: /work/scripts/.../prefill_config.yaml  # YAML path required

  decode:
    system: GB200-NVL72_GB200-186GB_aarch64x16
    gpus_per_worker: 16
    gpus_per_node: 4
    config: /work/scripts/.../decode_config.yaml  # YAML path required

MLPerf system config requirements (in configs/{system}/{scenario}/{benchmark}.py):
  # Frontend system is inferred from the --system argument (config path)
  llm_fields.dynamo_cluster: {
      'num_prefill_workers': 1,               # required
      'num_decode_workers': 1,                # required
      'gpus_per_node': 4,                     # optional, default shown
      'prefill': {
          'system': '...x4',                  # required
          'trtllm_yml_override': '/work/path/to/prefill.yaml',  # required
      },
      'decode': {
          'system': '...x16',                 # required
          'trtllm_yml_override': '/work/path/to/decode.yaml',   # required
      },
  }

NOTE: 'config_id' (Python config loading) is NOT supported for workers.
      Loading server configs from Python is broken. Use 'trtllm_yml_override' only.

Notes:
  - --container-image is REQUIRED and must be provided via CLI
  - Nodelist is ALWAYS fetched from $SLURM_JOBID
  - Number of clusters = len(nodelist) / (prefill_nodes + decode_nodes)
        """
    )

    # Config source: either YAML file or MLPerf system config
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        '--config', '-c',
        help='Path to YAML configuration file'
    )
    config_group.add_argument(
        '--system', '-s',
        help='MLPerf system name (e.g., GB200-NVL72_GB200-186GB_aarch64x20)'
    )

    # Arguments for MLPerf system config mode
    parser.add_argument(
        '--benchmark', '-b',
        help='Benchmark name (required with --system)'
    )
    parser.add_argument(
        '--scenario',
        default='Interactive',
        help='Scenario name (default: Interactive)'
    )
    parser.add_argument(
        '--config-id',
        default='dynamo_cluster',
        help='Config ID from ATOMIC_EXPORTS. Must be "dynamo_cluster" for dynamo disaggregated serving.'
    )

    # CLI overrides for worker configs
    parser.add_argument(
        '--prefill-yml',
        help='Override prefill worker trtllm_yml_override path'
    )
    parser.add_argument(
        '--decode-yml',
        help='Override decode worker trtllm_yml_override path'
    )
    parser.add_argument(
        '--container-image',
        required=True,
        help='Container image path (required)'
    )
    parser.add_argument(
        '--storage-path',
        help='Path to shared storage containing models/data (default: /home/mlperf_inference_storage). '
             'Mounted to both /home/mlperf_inference_storage and /work/build inside container.'
    )

    # Harness arguments
    parser.add_argument(
        '--run-harness-args',
        type=str,
        default=None,
        help='Custom RUN_ARGS for make run_harness. If provided, runs harness then terminates servers. '
             'If not provided, skips harness and waits for servers to terminate. '
             'IMPORTANT: Use = form (--run-harness-args="...") since values starting with "--" '
             'confuse argparse when space-separated.'
    )
    parser.add_argument(
        '--run-harness-nodeidx',
        type=int,
        default=0,
        help='Index into nodelist for which node runs the harness (default: 0). '
             'Uses disagg_cluster.system as the system ID.'
    )
    parser.add_argument(
        '--server-init-delay',
        type=int,
        default=600,
        help='Max seconds to wait for servers to initialize (default: 600). '
             'Uses health check polling - proceeds early if servers are ready. '
             'Large models like DeepSeek-R1 typically need 300-600 seconds.'
    )
    parser.add_argument(
        '--accuracy',
        action='store_true',
        help='Enable accuracy mode. Server workers automatically apply _accuracy.yml overrides '
             'via the trtllm_yml_override convention. When used with --run-harness-args, '
             'appends --test_mode=AccuracyOnly to harness args.'
    )
    parser.add_argument(
        '--audit',
        action='store_true',
        help='Run audit harness (run_audit_harness) instead of regular harness for compliance testing'
    )

    # Common arguments
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Print commands without executing them'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--nodelist',
        help='Comma-separated nodelist for testing (bypasses SLURM, e.g., "node0,node1,node2")'
    )

    args = parser.parse_args()

    # Validate MLPerf system config arguments
    if args.system and not args.benchmark:
        parser.error("--benchmark is required when using --system")

    # Handle --accuracy: append --test_mode=AccuracyOnly to harness args if provided
    if args.accuracy:
        if args.run_harness_args is not None:
            args.run_harness_args = f"{args.run_harness_args} --test_mode=AccuracyOnly"

    # Store accuracy flag for propagation to server workers
    accuracy = args.accuracy

    # Parse nodelist if provided
    nodelist = None
    if args.nodelist:
        nodelist = [n.strip() for n in args.nodelist.split(',')]

    try:
        if args.config:
            # Load from YAML file
            cluster = DisaggCluster(
                config_path=args.config,
                container_image=args.container_image,
                storage_path=args.storage_path,
                accuracy=accuracy,
                config_id=args.config_id,
                dry_run=args.dry_run,
                verbose=args.verbose,
                nodelist=nodelist
            )
        else:
            # Load from MLPerf system config
            config_dict = load_config_from_mlperf_system(
                system_name=args.system,
                benchmark=args.benchmark,
                scenario=args.scenario,
                config_id=args.config_id
            )

            # Apply CLI overrides
            if args.prefill_yml:
                config_dict['prefill']['config'] = args.prefill_yml
                # Clear config_id if yml override is specified
                config_dict['prefill'].pop('config_id', None)
                if args.verbose:
                    print(f"Override: prefill config = {args.prefill_yml}")
            if args.decode_yml:
                config_dict['decode']['config'] = args.decode_yml
                config_dict['decode'].pop('config_id', None)
                if args.verbose:
                    print(f"Override: decode config = {args.decode_yml}")

            cluster = DisaggCluster(
                config_dict=config_dict,
                container_image=args.container_image,
                storage_path=args.storage_path,
                accuracy=accuracy,
                config_id=args.config_id,
                dry_run=args.dry_run,
                verbose=args.verbose,
                nodelist=nodelist
            )

        cluster.deploy(
            run_harness_args=args.run_harness_args,
            run_harness_nodeidx=args.run_harness_nodeidx,
            server_init_delay=args.server_init_delay,
            audit=args.audit
        )
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
