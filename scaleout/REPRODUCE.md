# MLPerf Inference Scaleout - Reproduction Commands

All commands below should be run from `closed/NVIDIA/`.

## DeepSeek-R1

### GB200x72

```bash
# Offline
salloc --nodes=18 --partition=gb200
./scaleout/run_scaleout.sh --stage all --atomic-system GB200-NVL72_GB200-186GB_aarch64x8 --gpus-per-node 4 --dp-multiplicity 9 --run-args "--benchmarks=deepseek-r1 --scenarios=Offline --core_type=trtllm_endpoint"

# Server
salloc --nodes=18 --partition=gb200
./scaleout/run_scaleout.sh --stage all --atomic-system GB200-NVL72_GB200-186GB_aarch64x8 --gpus-per-node 4 --dp-multiplicity 9 --run-args "--benchmarks=deepseek-r1 --scenarios=Server --core_type=trtllm_endpoint"
```

### GB300x72

```bash
# Offline
salloc --nodes=18 --partition=gb300
./scaleout/run_scaleout.sh --stage all --atomic-system GB300-NVL72_GB300-288GB_aarch64x8 --gpus-per-node 4 --dp-multiplicity 9 --run-args "--benchmarks=deepseek-r1 --scenarios=Offline --core_type=trtllm_endpoint"

# Server
salloc --nodes=18 --partition=gb300
./scaleout/run_scaleout.sh --stage all --atomic-system GB300-NVL72_GB300-288GB_aarch64x8 --gpus-per-node 4 --dp-multiplicity 9 --run-args "--benchmarks=deepseek-r1 --scenarios=Server --core_type=trtllm_endpoint"
```

### GB300x288
```bash
# Offline
sbatch --job-name deepseek_r1.GB200x288_Offline \
    --nodes 72 \
    --partition gb300 \
    ./scaleout/run_scaleout.sh \
    --stage all  --atomic-system GB300-NVL72_GB300-288GB_aarch64x8 \
    --gpus-per-node 4 --dp-multiplicity 36 \
    --run-args "--benchmarks=deepseek-r1 --scenarios=Offline --core_type=trtllm_endpoint" \
    --harness-system GB300-NVL72_GB300-288GB_aarch64x288

# Server
sbatch --job-name deepseek_r1.GB200x288_Server \
    --nodes 72 \
    --partition gb300 \
    ./scaleout/run_scaleout.sh \
    --stage all  --atomic-system GB300-NVL72_GB300-288GB_aarch64x8 \
    --gpus-per-node 4 --dp-multiplicity 36 \
    --run-args "--benchmarks=deepseek-r1 --scenarios=Server --core_type=trtllm_endpoint" \
    --harness-system GB300-NVL72_GB300-288GB_aarch64x288

# Accuracy test: Add `--test_mode=AccuracyOnly` to --run-args
```


## Llama3.1-405b

### B300x8

```bash
# Offline
salloc --nodes=1 --partition=b300
./scaleout/run_scaleout.sh \
    --stage all \
    --atomic-system B300-SXM-270GBx2 \
    --gpus-per-node 8 \
    --dp-multiplicity 4 \
    --run-args "--benchmarks=llama3.1-405b --scenarios=Offline --core_type=trtllm_endpoint"
```

### GB200x72

```bash
# Offline
salloc --nodes=18 --partition=gb200
./scaleout/run_scaleout.sh --stage all --atomic-system GB200-NVL72_GB200-186GB_aarch64x4 --gpus-per-node 4 --dp-multiplicity 18 --run-args "--benchmarks=llama3_1-405b --scenarios=Offline --core_type=trtllm_endpoint"

# Server (disaggregated)
salloc --nodes=18 --partition=gb200
./scaleout/run_scaleout_disagg.py \
    --stage all \
    --gpus-per-node 4 \
    --num-ctx-servers 14 \
    --num-gen-servers 2 \
    --ctx-atomic-system GB200-NVL72_GB200-186GB_aarch64x4 \
    --gen-atomic-system GB200-NVL72_GB200-186GB_aarch64x8 \
    --harness-system GB200-NVL72_GB200-186GB_aarch64x72 \
    --ctx-run-args "--benchmarks=llama3_1-405b --scenarios=Server --core_type=trtllm_endpoint --config_id=ctx" \
    --gen-run-args "--benchmarks=llama3_1-405b --scenarios=Server --core_type=trtllm_endpoint" \
    --harness-run-args "--benchmarks=llama3_1-405b --scenarios=Server --core_type=trtllm_endpoint --config_id=disagg"

# Interactive (disaggregated)
salloc --nodes=18 --partition=gb200
./scaleout/run_scaleout_disagg.py \
    --stage all \
    --gpus-per-node 4 \
    --num-ctx-servers 14 \
    --num-gen-servers 2 \
    --ctx-atomic-system GB200-NVL72_GB200-186GB_aarch64x4 \
    --gen-atomic-system GB200-NVL72_GB200-186GB_aarch64x8 \
    --harness-system GB200-NVL72_GB200-186GB_aarch64x72 \
    --ctx-run-args "--benchmarks=llama3_1-405b --scenarios=Interactive --core_type=trtllm_endpoint" \
    --gen-run-args "--benchmarks=llama3_1-405b --scenarios=Interactive --core_type=trtllm_endpoint" \
    --harness-run-args "--benchmarks=llama3_1-405b --scenarios=Interactive --core_type=trtllm_endpoint"
```

### GB300x72

```bash
# Offline
salloc --nodes=18 --partition=gb300
./scaleout/run_scaleout.sh --stage all --atomic-system GB300-NVL72_GB300-288GB_aarch64x2 --gpus-per-node 4 --dp-multiplicity 36 --run-args "--benchmarks=llama3_1-405b --scenarios=Offline --core_type=trtllm_endpoint"

# Server
salloc --nodes=18 --partition=gb300
./scaleout/run_scaleout.sh --stage all --atomic-system GB300-NVL72_GB300-288GB_aarch64x2 --gpus-per-node 4 --dp-multiplicity 36 --run-args "--benchmarks=llama3_1-405b --scenarios=Server --core_type=trtllm_endpoint"

# Interactive (disaggregated)
salloc --nodes=18 --partition=gb300
./scaleout/run_scaleout_disagg.py \
    --stage all \
    --gpus-per-node 4 \
    --num-ctx-servers 26 \
    --num-gen-servers 5 \
    --ctx-atomic-system GB300-NVL72_GB300-288GB_aarch64x2 \
    --gen-atomic-system GB300-NVL72_GB300-288GB_aarch64x4 \
    --harness-system GB300-NVL72_GB300-288GB_aarch64x72 \
    --ctx-run-args "--benchmarks=llama3_1-405b --scenarios=Interactive --core_type=trtllm_endpoint" \
    --gen-run-args "--benchmarks=llama3_1-405b --scenarios=Interactive --core_type=trtllm_endpoint" \
    --harness-run-args "--benchmarks=llama3_1-405b --scenarios=Interactive --core_type=trtllm_endpoint"
```

## Llama2-70b

### B300x8

```bash
# Offline
salloc --nodes=1 --partition=b300
./scaleout/run_scaleout.sh \
    --stage all \
    --atomic-system B300-SXM-270GBx1 \
    --gpus-per-node 8 \
    --dp-multiplicity 8 \
    --run-args "--benchmarks=llama2-70b --scenarios=Offline --core_type=trtllm_endpoint"

# Server
salloc --nodes=1 --partition=b300
./scaleout/run_scaleout.sh \
    --stage all \
    --atomic-system B300-SXM-270GBx1 \
    --gpus-per-node 8 \
    --dp-multiplicity 8 \
    --run-args "--benchmarks=llama2-70b --scenarios=Server --core_type=trtllm_endpoint --server_target_qps=400"
```

### GB200x72

```bash
# Offline
salloc --nodes=18 --partition=gb200
./scaleout/run_scaleout.sh \
    --stage all \
    --atomic-system GB200-NVL72_GB200-186GB_aarch64x1 \
    --gpus-per-node 4 \
    --dp-multiplicity 72 \
    --run-args "--benchmarks=llama2-70b --scenarios=Offline --core_type=trtllm_endpoint"

# Server (disaggregated, run from closed/NVIDIA/scripts/slurm_llm):
cd closed/NVIDIA/scripts/slurm_llm
./disagg_trtllm_llama2_70b/llama2_70b_disagg.sh \
    --scenario Server \
    --ctx-num-servers 28 --gen-num-servers 44 --num-servers 9 \
    --target-qps 3000 \
    --ctx-mbs 4096 --ctx-mnt 4352 \
    --gen-mbs 1024 --gen-mnt 1024 \
    --ctx-stream 30 --gen-stream 150 \
    --cg-sizes "1,2,4,8,16,32,64,128,256,512,640,768,1024"
# Add --test-mode AccuracyOnly for accuracy runs
# Add --audit for audit runs

# Interactive (disaggregated, run from closed/NVIDIA/scripts/slurm_llm):
cd closed/NVIDIA/scripts/slurm_llm
./disagg_trtllm_llama2_70b/llama2_70b_disagg.sh \
    --ctx-num-servers 28 --gen-num-servers 44 --num-servers 9 \
    --target-qps 2790 \
    --ctx-mbs 4096 --ctx-mnt 4352 \
    --gen-mbs 512 --gen-mnt 512 \
    --ctx-stream 30 --gen-stream 150 \
    --cg-sizes "1,2,4,8,16,32,64,128,256,512"
# Add --test-mode AccuracyOnly for accuracy runs
# Add --audit for audit runs
```

### GB300x72

```bash
# Offline
salloc --nodes=18 --partition=gb300
./scaleout/run_scaleout.sh \
    --stage all \
    --atomic-system GB300-NVL72_GB300-288GB_aarch64x1 \
    --gpus-per-node 4 \
    --dp-multiplicity 72 \
    --run-args "--benchmarks=llama2-70b --scenarios=Offline --core_type=trtllm_endpoint"

# Server (disaggregated, run from closed/NVIDIA/scripts/slurm_llm):
cd closed/NVIDIA/scripts/slurm_llm
./disagg_trtllm_llama2_70b/llama2_70b_disagg.sh \
    --scenario Server \
    --ctx-num-servers 28 --gen-num-servers 44 --num-servers 9 \
    --target-qps 3200 \
    --ctx-mbs 4608 --ctx-mnt 4608 \
    --gen-mbs 2048 --gen-mnt 2048 \
    --ctx-stream 30 --gen-stream 100 \
    --cg-sizes "1,2,4,8,16,32,64,128,256,512,640,768,1024,1280,1536,1792,2048"
# Add --test-mode AccuracyOnly for accuracy runs
# Add --audit for audit runs

# Interactive (disaggregated, run from closed/NVIDIA/scripts/slurm_llm):
cd closed/NVIDIA/scripts/slurm_llm
./disagg_trtllm_llama2_70b/llama2_70b_disagg.sh \
    --ctx-num-servers 28 --gen-num-servers 44 --num-servers 9 \
    --target-qps 3000 \
    --ctx-mbs 4608 --ctx-mnt 4608 \
    --gen-mbs 1536 --gen-mnt 1536 \
    --ctx-stream 30 --gen-stream 100 \
    --cg-sizes "1,2,4,8,16,32,64,128,256,512,640,768,1024,1280,1536"
# Add --test-mode AccuracyOnly for accuracy runs
# Add --audit for audit runs
```

## GPT-OSS-120b

### GB200x72

```bash
# Offline
salloc --nodes=18 --partition=gb200
./scaleout/run_scaleout.sh --stage all --atomic-system GB200-NVL72_GB200-186GB_aarch64x1 --gpus-per-node 4 --dp-multiplicity 72 --run-args "--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint"

# Server
salloc --nodes=18 --partition=gb200
./scaleout/run_scaleout.sh --stage all --atomic-system GB200-NVL72_GB200-186GB_aarch64x1 --gpus-per-node 4 --dp-multiplicity 72 --run-args "--benchmarks=gpt-oss-120b --scenarios=Server --core_type=trtllm_endpoint"

# Interactive (disaggregated)
salloc --nodes=18 --partition=gb200
./scaleout/run_scaleout_disagg.py \
    --stage all \
    --gpus-per-node 4 \
    --num-ctx-servers 24 \
    --num-gen-servers 12 \
    --num-master-servers 12 \
    --ctx-atomic-system GB200-NVL72_GB200-186GB_aarch64x1 \
    --gen-atomic-system GB200-NVL72_GB200-186GB_aarch64x4 \
    --ctx-run-args "--benchmarks=gpt-oss-120b --scenarios=Interactive --core_type=trtllm_endpoint" \
    --gen-run-args "--benchmarks=gpt-oss-120b --scenarios=Interactive --core_type=trtllm_endpoint" \
    --harness-run-args "--benchmarks=gpt-oss-120b --scenarios=Interactive --core_type=trtllm_endpoint"
```

### GB300x72

```bash
# Offline
salloc --nodes=18 --partition=gb300
./scaleout/run_scaleout.sh --stage all --atomic-system GB300-NVL72_GB300-288GB_aarch64x1 --gpus-per-node 4 --dp-multiplicity 72 --run-args "--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint"

# Server
salloc --nodes=18 --partition=gb300
./scaleout/run_scaleout.sh --stage all --atomic-system GB300-NVL72_GB300-288GB_aarch64x1 --gpus-per-node 4 --dp-multiplicity 72 --run-args "--benchmarks=gpt-oss-120b --scenarios=Server --core_type=trtllm_endpoint"

# Interactive (disaggregated)
salloc --nodes=18 --partition=gb300
./scaleout/run_scaleout_disagg.py \
    --stage all \
    --gpus-per-node 4 \
    --num-ctx-servers 20 \
    --num-gen-servers 13 \
    --num-master-servers 12 \
    --ctx-atomic-system GB300-NVL72_GB300-288GB_aarch64x1 \
    --gen-atomic-system GB300-NVL72_GB300-288GB_aarch64x4 \
    --ctx-run-args "--benchmarks=gpt-oss-120b --scenarios=Interactive --core_type=trtllm_endpoint" \
    --gen-run-args "--benchmarks=gpt-oss-120b --scenarios=Interactive --core_type=trtllm_endpoint" \
    --harness-run-args "--benchmarks=gpt-oss-120b --scenarios=Interactive --core_type=trtllm_endpoint"
```
