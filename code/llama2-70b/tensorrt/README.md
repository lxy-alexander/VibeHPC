# NVIDIA B300 x1 Llama2-70B Test Execution Runbook


## 1. Model Download

Llama2-70B can be quantized from the original Hugging Face checkpoint, or you can directly use an already converted TensorRT-LLM NVFP4 checkpoint.

Recommended checkpoint directory:

```text
$MLPERF_SCRATCH_PATH/models/Llama2/fp4-quantized-modelopt/llama2-70b-chat-hf-tp1pp1-fp4
```

If you already have the converted checkpoint, sync it directly to the target directory:

```bash
set -euxo pipefail

export MLPERF_SCRATCH_PATH=/path/to

SRC_CKPT=/path/to/llama2-70b-chat-hf-tp1pp1-fp4
TARGET_CKPT="$MLPERF_SCRATCH_PATH/models/Llama2/fp4-quantized-modelopt/llama2-70b-chat-hf-tp1pp1-fp4"

mkdir -p "$(dirname "$TARGET_CKPT")"
rsync -avh --progress "$SRC_CKPT"/ "$TARGET_CKPT"/

test -s "$TARGET_CKPT/config.json"
test -s "$TARGET_CKPT/rank0.safetensors"
du -sh "$TARGET_CKPT"
ls -lh "$TARGET_CKPT"
```

Expected checkpoint characteristics:

```text
Architecture: LlamaForCausalLM
Tensor parallelism: 1
Pipeline parallelism: 1
Quantization: NVFP4
KV cache dtype: FP8
```

## 2. Download Docker (Use NVIDIA Docker)

Use the NVIDIA x86 MLPerf TensorRT-LLM image on B300 x86 machines.

Pull the NVIDIA Docker image:

```bash
docker pull nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_x86
```

Clone the VibeHPC project and prepare third-party repositories:

```bash
export VibeHPC_REPO=/path/to/VibeHPC/inference_results_v6.0.git
export VibeHPC_ROOT=/path/to/inference_results_v6.0

git clone --progress "$VibeHPC_REPO" "$VibeHPC_ROOT"

cd "$VibeHPC_ROOT/closed/NVIDIA"
export VibeHPC_PATH="$(pwd)"

mkdir -p 3rdparty

git clone --depth 1 --progress https://github.com/NVIDIA/TensorRT-LLM.git 3rdparty/trtllm
git clone --depth 1 --progress https://github.com/mlcommons/inference.git 3rdparty/mlc-inference

cd 3rdparty/mlc-inference
git submodule update --init --recursive --jobs 8

cd ../trtllm

git config --global --add safe.directory '*'
git fetch --all --tags

git checkout v1.3.0rc0

git rev-parse HEAD
git describe --tags --always
```

Enter the container:

```bash
export VibeHPC_PATH=/path/to/inference_results_v6.0/closed/NVIDIA
export MLPERF_SCRATCH_PATH=/path/to

docker run --rm -it \
  --gpus all \
  --net host \
  --ipc host \
  --shm-size=64gb \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $VibeHPC_PATH:/work \
  -v $MLPERF_SCRATCH_PATH:$MLPERF_SCRATCH_PATH \
  -e MLPERF_SCRATCH_PATH=$MLPERF_SCRATCH_PATH \
  -w /work \
  nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_x86
```

Inside the container, run the following first:

```bash
cd /work
git config --global --add safe.directory '*'
git config --global --add safe.directory /work/3rdparty/trtllm
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

make link_dirs
mkdir -p "$MLPERF_SCRATCH_PATH/logs"
ln -sfn "$MLPERF_SCRATCH_PATH/logs" /work/build/logs

ls -l /work/build
```

## 3. Data Download

Inside the container, use the NVIDIA harness to download data:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

make link_dirs

BENCHMARKS="llama2-70b" make download_data
```

Check download results:

```bash
find "$MLPERF_SCRATCH_PATH/data" -maxdepth 3 -iname "*llama2*" -o -iname "*cnn*"
```

## 4. Data Processing

Run the commands in this section inside the NVIDIA Docker container. If you have not entered the container yet, complete Section 2 to pull the Docker image and start the container.

Set the MLPerf Inference v6.1 RNG seeds:

```bash
cd /work

LOADGEN=/work/code/common/mlcommons/loadgen.py
cp -a "$LOADGEN" "${LOADGEN}.before-v61-seeds"

python3 - <<'PY'
from pathlib import Path

p = Path("/work/code/common/mlcommons/loadgen.py")
s = p.read_text()

old = """\
        ts.FromConfig(str(self.user_conf_path),
                      benchmark_name,
                      server_name,
                      1)
        return ts
"""

new = """\
        ts.FromConfig(str(self.user_conf_path),
                      benchmark_name,
                      server_name,
                      1)

        # MLPerf Inference v6.1 official RNG seeds.
        ts.qsl_rng_seed = 2085463073848966840
        ts.sample_index_rng_seed = 2747215439041700203
        ts.schedule_rng_seed = 16159082839903944936

        return ts
"""

if old in s:
    p.write_text(s.replace(old, new, 1))
    print("patched v6.1 RNG seeds")
else:
    print("seed patch block not found, maybe already patched")
PY

python3 -m py_compile "$LOADGEN"
```

Preprocess data:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

make link_dirs

BENCHMARKS="llama2-70b" make preprocess_data

find /work/build/preprocessed_data/llama2-70b \
  -maxdepth 3 -type f | sort | head -n 100
```

Check the B300 x1 configuration:

```bash
cd /work

CFG_70B=/work/configs/B300-SXM-270GBx1/Offline/llama2-70b.py

python3 -m py_compile "$CFG_70B"

grep -nE "min_duration|performance_sample_count|offline_expected_qps|gpu_batch_size|max_num_tokens|max_concurrency|num_postprocess_workers|stream_interval|kvcache_free_gpu_mem_frac|enable_max_num_tokens_tuning|enable_batch_size_tuning|use_graphs" \
  "$CFG_70B"
```

Generate or confirm the engine:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

make generate_engines \
  RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline" \
  SYSTEM_NAME="$SYSTEM_NAME"
```

Check the engine:

```bash
ENGINE_70B=/work/build/engines/B300-SXM-270GBx1/Offline/llama2-70b/gpu-fp4-b3072-tp1-pp1.cp990

test -s "$ENGINE_70B/rank0.engine"
test -s "$ENGINE_70B/config.json"
du -sh "$ENGINE_70B"
ls -lh "$ENGINE_70B"

python3 - <<'PY'
import json
from pathlib import Path

p = Path("/work/build/engines/B300-SXM-270GBx1/Offline/llama2-70b/gpu-fp4-b3072-tp1-pp1.cp990/config.json")
cfg = json.loads(p.read_text())
build = cfg.get("build_config", {})

for key in ("max_batch_size", "max_num_tokens", "max_input_len", "max_seq_len"):
    print(f"{key}: {build.get(key)}")
PY
```

## 5. Run Tests

### 1. Run PerformanceOnly

Use the following for Llama2-70B on B300 x1:

```text
SYSTEM_NAME=B300-SXM-270GBx1
```

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1
export RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline"

rm -rf \
  /work/build/loadgen-configs/B300-SXM-270GBx1_TRT/llama2-70b*/Offline

LOG="$MLPERF_SCRATCH_PATH/logs/b300-1gpu-llama70b/v61_performance.log"
mkdir -p "$(dirname "$LOG")"

make run_harness \
  RUN_ARGS="$RUN_ARGS --test_mode=PerformanceOnly" \
  SYSTEM_NAME="$SYSTEM_NAME" \
  2>&1 | tee "$LOG"
```

Check:

```bash
grep -E "qsl_rng_seed|sample_index_rng_seed|schedule_rng_seed|Result is|Min duration satisfied|Min queries satisfied|result_tokens_per_second|INVALID|VALID" \
  "$LOG" | tail -n 80
```

### 2. Run AccuracyOnly

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1
export RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline"

LOG="$MLPERF_SCRATCH_PATH/logs/b300-1gpu-llama70b/v61_accuracy.log"
mkdir -p "$(dirname "$LOG")"

make run_harness \
  RUN_ARGS="$RUN_ARGS --test_mode=AccuracyOnly" \
  SYSTEM_NAME="$SYSTEM_NAME" \
  2>&1 | tee "$LOG"
```

Check:

```bash
grep -E "qsl_rng_seed|sample_index_rng_seed|schedule_rng_seed" "$LOG" | tail -n 10
grep -Ei "accuracy|rouge|tokens per sample|gen_len|pass|fail|result" "$LOG" | tail -n 120
```
