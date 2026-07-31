# Llama2-70b Disaggregated Inference Scripts

Scripts for running llama2-70b benchmarks with disaggregated inference architecture on SLURM clusters.

## What is Disaggregated Inference?

Disaggregated inference separates the model serving into two specialized components:
- **CTX servers** (context/prefill): Process input prompts in parallel - optimized for high throughput
- **GEN servers** (generation/decode): Generate output tokens sequentially - optimized for low latency

This separation allows independent scaling and better resource utilization.

## Files Overview

### Main Scripts

```
llama2_70b_disagg.sh          # Main entry point - submits SLURM job
benchmark_llama2_70b.sh        # Core benchmark orchestration logic
common.sh                      # Shared utilities and helper functions
```

### Helper Scripts (`scripts/` directory)

```
gen_worker_config.py           # Generates YAML configs for CTX/GEN workers
gen_server_config.py           # Generates YAML config for coordinator server
start_worker.sh                # Launches CTX/GEN worker processes
start_server.sh                # Launches coordinator server process
bench_e2e.sh                   # Runs MLPerf harness client and collects results
```

### Documentation

```
NODE_ALLOCATION_SIMULATION.txt # Node allocation calculation examples
```

## How It Works

1. **llama2_70b_disagg.sh** - Entry point that:
   - Parses command-line parameters (QPS, batch sizes, GPU counts, etc.)
   - Calculates required nodes based on GPU configuration
   - Submits SLURM batch job calling `benchmark_llama2_70b.sh`

2. **benchmark_llama2_70b.sh** - Orchestrator that:
   - Sets up the container environment
   - Generates worker and server configuration files
   - Launches CTX workers, GEN workers, and coordinator server
   - Calls `bench_e2e.sh` to run the benchmark
   - Collects and organizes results

3. **Worker/Server Scripts** - Process managers that:
   - `gen_worker_config.py` / `gen_server_config.py`: Create YAML configs with model paths, batch sizes, memory settings
   - `start_worker.sh` / `start_server.sh`: Launch TRT-LLM processes with proper GPU assignments
   - `bench_e2e.sh`: Waits for server health, runs MLPerf harness, captures metrics

## Usage

The main script is called from the repository root with scenario-specific parameters. See `closed/NVIDIA/scaleout/REPRODUCE.md` for complete usage examples.

**Basic pattern:**
```bash
cd closed/NVIDIA/scripts/slurm_llm
./disagg_trtllm_llama2_70b/llama2_70b_disagg.sh \
    --scenario [Server|Interactive] \
    --ctx-num-servers N \
    --gen-num-servers M \
    --target-qps Q \
    --gpu [GB200|GB300] \
    [additional parameters...]
```

**Key Parameters:**
- `--scenario`: MLPerf scenario (Server or Interactive for disaggregated)
- `--ctx-num-servers`: Number of context/prefill servers
- `--gen-num-servers`: Number of generation/decode servers
- `--num-servers`: Number of coordinator master servers
- `--target-qps`: Target queries per second
- `--ctx-mbs` / `--gen-mbs`: Max batch size for CTX/GEN
- `--ctx-mnt` / `--gen-mnt`: Max num tokens for CTX/GEN
- `--gpu`: GPU type (GB200, GB300)
- `--partition`: SLURM partition
- `--segment`: SLURM segment for node allocation
- `--exp-tag`: Experiment identifier for logs
- `--test-mode`: PerformanceOnly or AccuracyOnly

## Output Structure

Logs are organized by scenario and configuration:

```
build/logs/
├── Server/
│   └── {exp_tag}/
│       ├── ctx_config.yaml          # CTX worker configuration
│       ├── gen_config.yaml          # GEN worker configuration
│       ├── output_ctx_0.log         # CTX worker outputs
│       ├── output_gen_*.log         # GEN worker outputs
│       ├── coordinator_*.log        # Coordinator logs
│       ├── harness.log              # MLPerf benchmark results
│       └── hostnames/               # Worker endpoint information
└── Interactive/
    └── {exp_tag}/
        └── (same structure as Server)
```

## Why Each Component is Needed

| Component | Purpose |
|-----------|---------|
| **llama2_70b_disagg.sh** | CLI interface for users - handles all parameter validation and SLURM submission |
| **benchmark_llama2_70b.sh** | Orchestrates the entire benchmark lifecycle - setup, launch, teardown |
| **common.sh** | Shared logging and utility functions to avoid code duplication |
| **gen_worker_config.py** | Generates precise YAML configs for TRT-LLM workers based on parameters |
| **gen_server_config.py** | Generates coordinator config that routes requests to workers |
| **start_worker.sh** | Manages worker process lifecycle with proper GPU assignment |
| **start_server.sh** | Manages coordinator server with health checks |
| **bench_e2e.sh** | Runs MLPerf LoadGen and handles client-side benchmarking |

## Requirements

- SLURM cluster with GPU nodes
- Container image with TRT-LLM and MLPerf harness
- Llama2-70b model weights (FP4 quantized recommended)
- Access to appropriate SLURM partitions and accounts

## References

For complete reproduction commands and configuration examples, see:
- `closed/NVIDIA/scaleout/REPRODUCE.md` - Full benchmark reproduction guide
- `NODE_ALLOCATION_SIMULATION.txt` - Node calculation examples
