# Dynamo + TensorRT-LLM Disaggregated Serving

## Build the container image
```bash
docker build -f Dockerfile.dynamo -t mlperf-dynamo .
```

For use with enroot, please import the docker image into a SquashFS file
```bash
enroot import dockerd://mlperf-dynamo:latest
```

## Prerequisites
- SLURM allocation with `enroot` and `pyxis` support
- Container image with Dynamo + TRT-LLM
  - TODO: Add container building steps
- Prefill config YAML (`prefill_config.yaml`)
- Decode config YAML (`decode_config.yaml`)
- To benchmark via `make run_harness`, please set `MLPINF_HTTP_USE_COMPLETIONS=1` and `MLPINF_USE_DYNAMO=1` in your env

## Terminology
### 1. Prefill/Decode Worker
A worker instance launched by Dynamo to run compute-bound prefill or memory-bound decode operations.
### 2. Frontend server
A combination of NATS server, ETCD server and Dynamo router that manages routing of requests between workers.

### 3. Disagg Cluster
`M` prefill workers + `N` decode workers + `K` frontend servers sharing one NATS/etcd instance.
- All frontends in a cluster share the same NATS and etcd for worker coordination
- Each frontend has its own HTTP port (8000, 8001, etc.)
- Harness can target multiple frontend URLs for load distribution

## TLDR - Launch a generic disagg cluster with `N` prefill, `M` decode workers
Starting with a trtllm config yaml file for each the prefill and decode workers, you may stand up a disagg cluster using the following steps.

### 1. Write the disagg cluster yml file

Create a `disagg_cluster.yml` file as follows. Here, we are launching 1 prefill server, and 4 decode servers across 72 GPUs.
```yaml
benchmark: deepseek-r1
scenario: Interactive

disagg_cluster:
  num_prefill_workers: 1
  num_decode_workers: 2
  system: GB200-NVL72_GB200-186GB_aarch64x36

  # Optional: Multiple frontends (default: 1)
  # num_frontends: 2

  # Optional: Frontend-specific configuration
  # frontend:
  #   router_mode: round-robin  # Options: round-robin, kv, random
  #   kv_overlap_weight: 0.5    # For KV router mode (0.0-1.0)

prefill:
  system: GB200-NVL72_GB200-186GB_aarch64x4
  gpus_per_worker: 4
  gpus_per_node: 4
  config: /work/prefill_config.yaml

decode:
  system: GB200-NVL72_GB200-186GB_aarch64x16
  gpus_per_worker: 16
  gpus_per_node: 4
  config: /work/decode_config.yaml
```

### Frontend Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `num_frontends` | Number of frontend replicas (sharing same NATS/etcd) | `1` |
| `router_mode` | Routing strategy: `round-robin`, `kv`, `random` | `round-robin` |
| `kv_overlap_weight` | Weight for KV router mode (0.0-1.0) | `1.0` (when `kv` mode) |

### Multiple Frontends

To run multiple frontends within a single cluster (sharing one NATS/etcd instance):

```yaml
disagg_cluster:
  num_prefill_workers: 2
  num_decode_workers: 8
  num_frontends: 2  # Multiple frontends sharing same NATS/etcd
  system: GB200-NVL72_GB200-186GB_aarch64x4
  frontend:
    router_mode: kv
    kv_overlap_weight: 0.0
```

Multiple frontends provide:
- Load distribution across frontend routers
- All frontends share the same NATS and etcd (primary frontend hosts them)
- Each frontend listens on a unique port (8000, 8001, etc.)
- Harness automatically targets all frontend URLs

The harness will automatically target all frontend URLs when running benchmarks.

### 2. Run the script `launch_disagg_cluster.sh`
```bash
# 1. Install dependencies (handy to use a venv for this)
pip install -r scripts/slurm_llm/dynamo_disagg/requirements.txt

# 2. Allocate nodes and launch
salloc --nodes=9 --time=2:00:00
python3 scripts/slurm_llm/dynamo_disagg/launch_disagg_cluster.py \
    --config scripts/slurm_llm/dynamo_disagg/config/sample/disagg_deployment_minimal.yaml \
    --container-image /path/to/image.sqsh
```

### 3. Optionally, `sbatch` your run
```bash
# Use the provided sbatch script:
sbatch --nodes=9 --time 2:0:0 -p backfill scripts/slurm_llm/dynamo_disagg/example_batch_script.sh
```

Or create your own:
```bash
#!/bin/bash

# Set up venv with pip
python3 -m venv --clear /tmp/disagg_venv
source /tmp/disagg_venv/bin/activate
python3 -m ensurepip --upgrade
pip install --upgrade pip
pip install -r scripts/slurm_llm/dynamo_disagg/requirements.txt

# Use -u for unbuffered output
python3 -u scripts/slurm_llm/dynamo_disagg/launch_disagg_cluster.py \
    --config scripts/slurm_llm/dynamo_disagg/config/sample/disagg_deployment_minimal.yaml \
    --container-image /path/to/image.sqsh
```

### 4. Check status
You should see three running job steps - one each for prefill workers, decode workers and frontend server.
```bash
$ sacct -j $SLURM_JOBID
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode
------------ ---------- ---------- ---------- ---------- ---------- --------
...
1522062.0          make            acct              144    RUNNING      0:0
1522062.1          make            acct              144    RUNNING      0:0
1522062.2          make            acct              576    RUNNING      0:0
```

## Resource Allocation

`launch_disagg_cluster.py` automatically allocates nodes from your SLURM allocation to prefill and decode workers.

### Allocation Formula

Assuming `gpus_per_worker` is a multiple of `gpus_per_node`:
  - TODO: Only inter-node for now. Intra-node is a WIP

```
nodes_per_prefill_worker = gpus_per_worker / gpus_per_node
nodes_per_decode_worker  = gpus_per_worker / gpus_per_node

prefill_nodes = num_prefill_workers × nodes_per_prefill_worker
decode_nodes  = num_decode_workers × nodes_per_decode_worker
total_nodes   = prefill_nodes + decode_nodes
```

The **frontend runs on the first node** of the cluster allocation. Multiple frontends share the same node.

### Node Assignment

Nodes from `SLURM_JOB_NODELIST` are assigned sequentially:

```
Prefill nodes:  [node0, node1, ...]        ← Frontend runs on node0
Decode nodes:   [nodeN, nodeN+1, ...]
```

### Prefill + Decode Isolation

Prefill and decode workers are allocated to **separate nodes** (no node overlap between prefill and decode).
  - This will change with intra-node


## Manual invocation: Standing up prefill/decode workers

### 1a. Launch Frontend (Single)

For a single frontend with default settings (port 8000, round-robin router):

```bash
srun --overlap --nodes=1 --ntasks=1 -w $FRONTEND_NODE \
    --container-image=$CONTAINER_IMAGE \
    --container-mounts="$(pwd):/work" \
    --container-workdir=/work --container-remap-root \
    make run_llm_server RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=Interactive \
        --core_type=disagg_frontend" &
```

### 1a-alt. Launch Multiple Frontends

Use `make run_llm_server --core_type=disagg_frontend` with the following arguments:

```bash
# Primary frontend (port 8000) - starts NATS + etcd + router
srun --overlap --nodes=1 --ntasks=1 -w $FRONTEND_NODE \
    --container-image=$CONTAINER_IMAGE \
    --container-mounts="$(pwd):/work" \
    --container-workdir=/work --container-remap-root \
    make run_llm_server RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=Interactive \
        --core_type=disagg_frontend \
        --dynamo_frontend_port=8000 \
        --dynamo_router_mode=kv \
        --dynamo_router_replica_sync=true" &

# Secondary frontend (port 8001) - router only, connects to primary's NATS/etcd
srun --overlap --nodes=1 --ntasks=1 -w $FRONTEND_NODE \
    --container-image=$CONTAINER_IMAGE \
    --container-mounts="$(pwd):/work" \
    --container-workdir=/work --container-remap-root \
    make run_llm_server RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=Interactive \
        --core_type=disagg_frontend \
        --dynamo_frontend_port=8001 \
        --dynamo_router_mode=kv \
        --dynamo_router_replica_sync=true \
        --dynamo_is_secondary_frontend=true \
        --dynamo_frontend_host=$FRONTEND_NODE" &
```

| Argument | Description |
|----------|-------------|
| `--dynamo_frontend_port` | HTTP port for this frontend (default: 8000) |
| `--dynamo_router_mode` | Router mode: `round-robin`, `kv`, `random` (default: round-robin) |
| `--dynamo_kv_overlap_weight` | KV overlap weight 0.0-1.0 (default: 1.0) |
| `--dynamo_router_replica_sync` | Enable sync for multiple frontends |
| `--dynamo_is_secondary_frontend` | Skip NATS/etcd startup (for secondary frontends) |
| `--dynamo_frontend_host` | Primary frontend host (required for secondary) |

**Recommended**: Use `launch_disagg_cluster.py` with `num_frontends: 2` in your config instead of manual setup.

### 1b. Launch Prefill Worker

```bash
srun --overlap --nodes=1 --ntasks=4 -w $PREFILL_NODE \
    --container-image=$CONTAINER_IMAGE \
    --container-mounts="$(pwd):/work" \
    --container-workdir=/work --container-remap-root \
    --mpi=pmix \
    make run_llm_server RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=Interactive \
        --core_type=disagg_prefill \
        --dynamo_frontend_host=$FRONTEND_NODE \
        --trtllm_yml_override=/work/prefill_config.yaml \
        --mpi_mode=leader" &
```

### 1c. Launch Decode Worker

```bash
srun --overlap --nodes=4 --ntasks=16 -w $DECODE_NODES \
    --container-image=$CONTAINER_IMAGE \
    --container-mounts="$(pwd):/work" \
    --container-workdir=/work --container-remap-root \
    --mpi=pmix \
    make run_llm_server RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=Interactive \
        --core_type=disagg_decode \
        --dynamo_frontend_host=$FRONTEND_NODE \
        --trtllm_yml_override=/work/decode_config.yaml \
        --mpi_mode=leader" &
```

### 1d. Run Benchmark

Wait for workers to register, then:

```bash
srun --overlap --nodes=1 --ntasks=1 -w $FRONTEND_NODE \
    --container-image=$CONTAINER_IMAGE \
    --container-mounts="$(pwd):/work" \
    --container-workdir=/work --container-remap-root \
    --export=MLPINF_HTTP_USE_COMPLETIONS=1,MLPINF_USE_DYNAMO=1 \
    make run_harness RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=Interactive \
        --core_type=dynamo_endpoint \
        --trtllm_server_urls=$FRONTEND_NODE:8000 \
        --test_mode=AccuracyOnly \
        --server_target_qps=10"
```


---

## Checking in your configuration to `configs`

Internalize YAML configs into the MLPerf config system for reproducible deployments and sharing performant configs.

### Move YAML Configs

```bash
mkdir -p configs/GB200-NVL72_GB200-186GB_aarch64x4/Interactive/disagg_prefill
mkdir -p configs/GB200-NVL72_GB200-186GB_aarch64x16/Interactive/disagg_decode

mv prefill_config.yaml configs/GB200-NVL72_GB200-186GB_aarch64x4/Interactive/disagg_prefill/deepseek-r1.yml
mv decode_config.yaml configs/GB200-NVL72_GB200-186GB_aarch64x16/Interactive/disagg_decode/deepseek-r1.yml
```

### 3b. Create System Config

Create `configs/GB200-NVL72_GB200-186GB_aarch64x36/Interactive/deepseek-r1.py`:

```python
import code.llmlib.fields as llm_fields

dynamo_endpoint = {
    llm_fields.dynamo_cluster: {
        'num_prefill_workers': 1,
        'num_decode_workers': 2,
        'gpus_per_node': 4,
        'prefill': {
            'system': 'GB200-NVL72_GB200-186GB_aarch64x4',
            'trtllm_yml_override': '/work/configs/GB200-NVL72_GB200-186GB_aarch64x4/Interactive/disagg_prefill/deepseek-r1.yml',
        },
        'decode': {
            'system': 'GB200-NVL72_GB200-186GB_aarch64x16',
            'trtllm_yml_override': '/work/configs/GB200-NVL72_GB200-186GB_aarch64x16/Interactive/disagg_decode/deepseek-r1.yml',
        },
    },
    # ... other fields
}
```

### 3c. Launch via System Name
With this, the checked in configuration under `configs` will be picked up

```bash
# Deploy servers only (no harness)
python3 scripts/slurm_llm/dynamo_disagg/launch_disagg_cluster.py \
    --system GB200-NVL72_GB200-186GB_aarch64x12 \
    --benchmark deepseek-r1 \
    --scenario Interactive \
    --container-image /path/to/image.sqsh

# Deploy servers AND run harness
python3 scripts/slurm_llm/dynamo_disagg/launch_disagg_cluster.py \
    --system GB200-NVL72_GB200-186GB_aarch64x12 \
    --benchmark deepseek-r1 \
    --scenario Interactive \
    --container-image /path/to/image.sqsh \
    --run-harness-args "--benchmarks=deepseek-r1 --scenarios=Interactive --test_mode=PerformanceOnly --server_target_qps=5"
```

---

## CLI Reference

### launch_disagg_cluster.py

#### Configuration Options
| Argument | Description |
|----------|-------------|
| `--config` | YAML cluster config file path |
| `--system` | MLPerf system name (e.g., `GB200-NVL72_GB200-186GB_aarch64x12`) |
| `--benchmark` | Benchmark name (e.g., `deepseek-r1`) |
| `--scenario` | Scenario (default: `Interactive`) |
| `--config-id` | Config ID to load from ATOMIC_EXPORTS (default: `default`) |
| `--container-image` | Container image path (required) |
| `--storage-path` | Path to shared storage for models/data. Mounted to `/home/mlperf_inference_storage` and `/work/build/models` inside container. Default: `/lustre/share/coreai_mlperf_inference/mlperf_inference_storage_clone` |
| `--prefill-yml` | Override prefill YAML path |
| `--decode-yml` | Override decode YAML path |

#### Harness Options
| Argument | Description |
|----------|-------------|
| `--run-harness-args` | Full RUN_ARGS string for `make run_harness`. If provided, runs harness after servers are ready, then terminates servers. If not provided, only deploys servers. |
| `--run-harness-nodeidx` | Node index for harness execution (default: 0) |
| `--server-init-delay` | Max seconds to wait for servers with health check polling (default: 600). For large models like DeepSeek-R1, use 900+. |

#### Other Options
| Argument | Description |
|----------|-------------|
| `--nodelist` | Override SLURM nodelist (for testing) |
| `--dry-run` | Print commands without executing |
| `--verbose` | Enable verbose output |

#### Harness Args Examples
```bash
# Performance run
--run-harness-args "--benchmarks=deepseek-r1 --scenarios=Interactive --test_mode=PerformanceOnly --server_target_qps=5"

# Accuracy run
--run-harness-args "--benchmarks=deepseek-r1 --scenarios=Interactive --test_mode=AccuracyOnly"

# With custom query count
--run-harness-args "--benchmarks=deepseek-r1 --scenarios=Interactive --test_mode=PerformanceOnly --server_target_qps=40 --min_query_count=144000"
```

### Core Types for `make run_llm_serve`

| Core Type | Description |
|-----------|-------------|
| `disagg_frontend` | Frontend (NATS + etcd + router) |
| `disagg_prefill` | Prefill worker |
| `disagg_decode` | Decode worker |

### Core Types for `make run_harness`
| `dynamo_endpoint` | mimics `trtllm_endpoint` and skips config gen+logging |

---

## Logs

```
build/logs/disagg_{benchmark}_slurm-{jobid}_{timestamp}/
├── cluster_0/
│   ├── frontend/
│   │   ├── fe_0/disagg_frontend.log   # Primary frontend (NATS/etcd)
│   │   └── fe_1/disagg_frontend.log   # Secondary frontend (if num_frontends > 1)
│   ├── prefill/*.stdout
│   └── decode/*.stdout
├── harness/*.stdout  (if --run-harness)
└── worker_mapping.txt
```
