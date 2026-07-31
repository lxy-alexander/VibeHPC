# MLPerf Inference Scaleout Orchestrator

Automates running MLPerf LLM workloads across multiple SLURM nodes. This script will launch multiple TensorRT-LLM servers followed by the MLPerf harness. We are launching the servers in leader mode i.e. MPI world is managed by SLURM as opposed to legacy workflows where the MLPerf harness would create an MPI world within a node. This design allows us to break the single node barrier and allows us to run cross-node servers.

## Usage

```bash
# 1. Allocate nodes
salloc --nodes=N --partition=PARTITION --time=HH:MM:SS

# 2. Launch servers
./scaleout/run_scaleout.sh \
    --stage all \
    --atomic-system <system> \
    --gpus-per-node <N> \
    --dp-multiplicity <N> \
    --run-args "--benchmarks=... --scenarios=... --core_type=trtllm_endpoint"
```

Or, use `sbatch` to queue your job

```bash
# 1. Allocate nodes
sbatch --nodes=N --partition=PARTITION --time=HH:MM:SS \
  ./scaleout/run_scaleout.sh \
    --stage all \
    --atomic-system <system> \
    --gpus-per-node <N> \
    --dp-multiplicity <N> \
    --run-args "--benchmarks=... --scenarios=... --core_type=trtllm_endpoint"
```

Note: Specify `--stage server/harness` to run each phase separately.

**Config folder resolution**: The `--atomic-system` determines the config folder used when launching servers (via `run_llm_server`), while the `--harness-system` (computed as `atomic-system × dp-multiplicity` by default) determines the config folder used when running the harness (via `run_harness`).

## Examples

### Cross-Node (18 nodes, 4 GPUs/node, 9 DP ranks)
DeepSeek-R1 on 18 nodes with 4 GPUs per node. Each DP rank uses 8 GPUs spanning 2 nodes.

```bash
salloc --nodes=18 --partition=gb200

./scaleout/run_scaleout.sh \
    --stage all \
    --atomic-system GB200-NVL72_GB200-186GB_aarch64x8 \
    --gpus-per-node 4 \
    --dp-multiplicity 9 \
    --run-args "--benchmarks=deepseek-r1 --scenarios=Offline --core_type=trtllm_endpoint"
```

### Single-Node (1 node, 8 GPUs/node, 8 DP ranks)
Llama3.1-8b on 1 nodes with 8 GPUs per node. Each DP rank uses 1 GPU.

```bash
salloc --nodes=1 --partition=b200

./scaleout/run_scaleout.sh \
    --stage all \
    --atomic-system B200-SXM-180GBx1 \
    --gpus-per-node 8 \
    --dp-multiplicity 8 \
    --run-args "--benchmarks=llama3_1-8b --scenarios=Offline --core_type=trtllm_endpoint"
```

### Two-Stage Launch (Server and Harness Separately)
Run server and harness as separate stages. This is useful for debugging or managing long-running servers.

```bash
salloc --nodes=18 --partition=gb200

# Stage 1: Launch servers (runs in background)
./scaleout/run_scaleout.sh \
    --stage server \
    --atomic-system GB200-NVL72_GB200-186GB_aarch64x8 \
    --gpus-per-node 4 \
    --dp-multiplicity 9 \
    --run-args "--benchmarks=deepseek-r1 --scenarios=Offline --core_type=trtllm_endpoint"

# Stage 2: Launch harness (after servers are ready)
./scaleout/run_scaleout.sh \
    --stage harness \
    --atomic-system GB200-NVL72_GB200-186GB_aarch64x8 \
    --gpus-per-node 4 \
    --dp-multiplicity 9 \
    --run-args "--benchmarks=deepseek-r1 --scenarios=Offline --core_type=trtllm_endpoint"
```

### Running Compliance Tests (Audit)
Run audit harness to execute compliance tests. This is required for MLPerf submissions.

```bash
salloc --nodes=18 --partition=gb200

./scaleout/run_scaleout.sh \
    --stage all \
    --atomic-system GB200-NVL72_GB200-186GB_aarch64x8 \
    --gpus-per-node 4 \
    --dp-multiplicity 9 \
    --run-args "--benchmarks=deepseek-r1 --scenarios=Offline --core_type=trtllm_endpoint" \
    --audit
```

## Command Line Arguments

### Required Arguments

- `--stage <server|harness|all>` - Execution stage:
  - `server`: Launch TensorRT-LLM servers only
  - `harness`: Launch MLPerf harness only (assumes servers are already running)
  - `all`: Launch servers, then launch harness sequentially

- `--atomic-system <name>` - Atomic system configuration name (e.g., `GB200-NVL72_GB200-186GB_aarch64x8`)
  - This should correspond to 1 DP rank for the benchmark
  - Format should end with `x<N>` where N is the number of GPUs per DP rank

- `--gpus-per-node <N>` - Number of GPUs available per node in the allocation

- `--dp-multiplicity <N>` - Number of Data Parallel (DP) ranks to use
  - Total GPUs required = DP multiplicity × GPUs per DP rank (from atomic-system)

- `--run-args "<args>"` - Benchmark run arguments passed to the underlying MLPerf make target
  - Example: `"--benchmarks=deepseek-r1 --scenarios=Offline --core_type=trtllm_endpoint"`

### Optional Arguments

- `--container-image <path>` - Path to container image
  - Default: `build/sqsh_images/mlperf-inference-$USER-aarch64-release.sqsh`

- `--mlperf-scratch-path <path>` - Path to MLPerf scratch directory
  - Default: `/lustre/share/coreai_mlperf_inference/mlperf_inference_storage_clone`

- `--extra-srun-flags "<flags>"` - Additional flags to pass to srun commands
  - Example: `"--exclusive --cpu-bind=none"`

- `--base-port <N>` - Base port number for server URLs
  - Default: `30000`
  - For intra-node deployments, ports are incremented from this base

- `--harness-system <name>` - Override harness system (default: calculated from atomic-system × dp-multiplicity)
  - Example: `GB200-NVL72_GB200-186GB_aarch64x72`

- `--audit` - Run audit harness (run_audit_harness) instead of regular harness
  - Executes all compliance tests for the workload

- `--dry-run` - Print srun commands without executing them
  - Useful for verifying the command structure and environment variables before actual execution


## Disaggregated Serving

Disaggregated serving separates context (prefill) and generation (decode) workloads across different GPU pools. This allows independent optimization of time-to-first-token (TTFT) and time-per-output-token (TPOT), and eliminates interference between the two phases.

The disaggregated workflow uses:
- **CTX workers**: Handle context/prefill phase only
- **GEN workers**: Handle generation/decode phase only  
- **Master server**: Coordinates requests between CTX and GEN workers

### Usage

```bash
# 1. Allocate nodes
salloc --nodes=N --partition=PARTITION --time=HH:MM:SS

# 2. Launch disaggregated servers
./scaleout/run_scaleout_disagg.py \
    --stage all \
    --gpus-per-node <N> \
    --num-ctx-servers <N> \
    --num-gen-servers <N> \
    --ctx-atomic-system <system> \
    --gen-atomic-system <system> \
    --ctx-run-args "--benchmarks=... --scenarios=... --core_type=trtllm_endpoint" \
    --gen-run-args "--benchmarks=... --scenarios=... --core_type=trtllm_endpoint" \
    --harness-run-args "--benchmarks=... --scenarios=... --core_type=trtllm_endpoint"
```

Note: Specify `--stage server/harness` to run each phase separately.

### Examples

#### Cross-Node DeepSeek-R1

DeepSeek-R1 with cross-node deployment. CTX servers use 4 GPUs each (intra-node), GEN servers use 8 GPUs each (cross-node, spanning 2 nodes).

```bash
salloc --nodes=3 --partition=gb200

./scaleout/run_scaleout_disagg.py \
    --stage all \
    --gpus-per-node 4 \
    --num-ctx-servers 1 \
    --num-gen-servers 1 \
    --ctx-atomic-system GB200-NVL72_GB200-186GB_aarch64x4 \
    --gen-atomic-system GB200-NVL72_GB200-186GB_aarch64x8 \
    --ctx-run-args "--benchmarks=deepseek-r1 --scenarios=Interactive --core_type=trtllm_endpoint" \
    --gen-run-args "--benchmarks=deepseek-r1 --scenarios=Interactive --core_type=trtllm_endpoint" \
    --harness-run-args "--benchmarks=deepseek-r1 --scenarios=Interactive --core_type=trtllm_endpoint"
```

**Resource breakdown:**
- CTX: 1 server × 4 GPUs = 4 GPUs (1 node)
- GEN: 1 server × 8 GPUs = 8 GPUs (2 nodes)
- Total: 3 nodes, 12 GPUs allocated, 12 GPUs used
- Harness system (computed): `GB200-NVL72_GB200-186GB_aarch64x12`

#### Intra-Node Llama2-70B

Llama2-70B with intra-node deployment. Both CTX and GEN servers fit within single nodes.

```bash
salloc --nodes=1 --partition=b200

./scaleout/run_scaleout_disagg.py \
    --stage all \
    --gpus-per-node 8 \
    --num-ctx-servers 3 \
    --num-gen-servers 5 \
    --ctx-atomic-system B200-SXM-180GBx1 \
    --gen-atomic-system B200-SXM-180GBx1 \
    --ctx-run-args "--benchmarks=llama2-70b --scenarios=Interactive --core_type=trtllm_endpoint" \
    --gen-run-args "--benchmarks=llama2-70b --scenarios=Interactive --core_type=trtllm_endpoint" \
    --harness-run-args "--benchmarks=llama2-70b --scenarios=Interactive --core_type=trtllm_endpoint"
```

**Resource breakdown:**
- CTX: 3 servers × 1 GPU = 3 GPUs
- GEN: 5 servers × 1 GPU = 5 GPUs
- Total: 1 node, 8 GPUs allocated, 8 GPUs used
- Harness system (computed): `B200-SXM-180GBx8`

#### Running Compliance Tests (Audit)
Run audit harness for disaggregated serving to execute compliance tests.

```bash
salloc --nodes=3 --partition=gb200

./scaleout/run_scaleout_disagg.py \
    --stage all \
    --gpus-per-node 4 \
    --num-ctx-servers 1 \
    --num-gen-servers 1 \
    --ctx-atomic-system GB200-NVL72_GB200-186GB_aarch64x4 \
    --gen-atomic-system GB200-NVL72_GB200-186GB_aarch64x8 \
    --ctx-run-args "--benchmarks=deepseek-r1 --scenarios=Interactive --core_type=trtllm_endpoint" \
    --gen-run-args "--benchmarks=deepseek-r1 --scenarios=Interactive --core_type=trtllm_endpoint" \
    --harness-run-args "--benchmarks=deepseek-r1 --scenarios=Interactive --core_type=trtllm_endpoint" \
    --audit
```

### Command Line Arguments

#### Required Arguments

- `--stage <server|harness|all>` - Execution stage:
  - `server`: Launch CTX/GEN workers and optionally master servers
  - `harness`: Launch MLPerf harness only (assumes servers are already running)
  - `all`: Launch servers, then launch harness sequentially

- `--gpus-per-node <N>` - Number of GPUs available per node in the allocation

- `--num-ctx-servers <N>` - Number of context/prefill servers to launch

- `--num-gen-servers <N>` - Number of generation/decode servers to launch

- `--ctx-atomic-system <name>` - Atomic system for CTX workers
  - Format should end with `x<N>` where N is GPUs per CTX server
  - Example: `GB200-NVL72_GB200-186GB_aarch64x4`

- `--gen-atomic-system <name>` - Atomic system for GEN workers
  - Format should end with `x<N>` where N is GPUs per GEN server
  - Example: `GB200-NVL72_GB200-186GB_aarch64x8`

- `--ctx-run-args "<args>"` - Benchmark arguments for CTX workers
  - Example: `"--benchmarks=deepseek-r1 --scenarios=Interactive --core_type=trtllm_endpoint --config_id=ctx"`

- `--gen-run-args "<args>"` - Benchmark arguments for GEN workers
  - Example: `"--benchmarks=deepseek-r1 --scenarios=Interactive --core_type=trtllm_endpoint --config_id=gen"`

- `--harness-run-args "<args>"` - Benchmark arguments for Harness
  - Example: `"--benchmarks=deepseek-r1 --scenarios=Interactive --core_type=trtllm_endpoint"`

#### Optional Arguments

- `--harness-system <name>` - Override harness system (default: computed from ctx/gen systems)
  - Computed as: `base_system` + `x` + `(num_ctx_servers × ctx_gpus_per_server + num_gen_servers × gen_gpus_per_server)`
  - Base system is derived from `--ctx-atomic-system`
  - Example: `GB200-NVL72_GB200-186GB_aarch64x12`

- `--launch-master <true|false>` - Launch master server in server stage (default: `true`)
  - Set to `false` to launch only workers without master server

- `--container-image <path>` - Path to container image
  - Default: `build/sqsh_images/mlperf-inference-$USER-aarch64-release.sqsh`

- `--mlperf-scratch-path <path>` - Path to MLPerf scratch directory
  - Default: `/lustre/share/coreai_mlperf_inference/mlperf_inference_storage_clone`

- `--extra-srun-flags "<flags>"` - Additional flags to pass to srun commands
  - Example: `"--exclusive --cpu-bind=none"`

- `--base-port <N>` - Base port number for server URLs
  - Default: `30000`
  - Ports are incremented sequentially: GEN workers (0..N-1), CTX workers (N..M-1), master (M)

- `--audit` - Run audit harness (run_audit_harness) instead of regular harness
  - Executes all compliance tests required for the workload

- `--dry-run` - Print srun commands without executing

## Environment Variables

- `SERVER_SPAWN_TIME` - Sleep time in seconds between server launch waves (default: `60`)
  - Controls the delay between launching successive batches of servers on the same nodes
  - Launching multiple servers simultaneously on same node results in TCP port contention
  - Example: `SERVER_SPAWN_TIME=120 ./scaleout/run_scaleout.sh ...`

- `LOG_DIR` - Custom directory for logs
  - Default (regular scaleout): `/work/build/logs/scaleout_<atomic-system>x<dp-multiplicity>_<timestamp>`
  - Default (disaggregated): `/work/build/logs/scaleout_disagg_<harness-system>_<timestamp>`
  - Example: `export LOG_DIR=/custom/log/path` before running the script
