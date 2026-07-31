# NVIDIA B300 x1 Llama2-70B  测试执行文档


## 1. 模型下载

Llama2-70B 可从原始 Hugging Face checkpoint 量化，也可直接使用已经转换完成的 TensorRT-LLM NVFP4 checkpoint。

推荐 checkpoint 目录：

```text
$MLPERF_SCRATCH_PATH/models/Llama2/fp4-quantized-modelopt/llama2-70b-chat-hf-tp1pp1-fp4
```

如果已有转换后的 checkpoint，直接同步到目标目录：

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

期望 checkpoint 特征：

```text
Architecture: LlamaForCausalLM
Tensor parallelism: 1
Pipeline parallelism: 1
Quantization: NVFP4
KV cache dtype: FP8
```

## 2. 下载 Docker（使用 NVIDIA Docker）

B300 x86 机器使用 NVIDIA x86 MLPerf TensorRT-LLM 镜像。

拉取 NVIDIA Docker 镜像：

```bash
docker pull nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_x86
```

进入容器：

```bash
export VibeHPC_PATH=/path/to
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

容器内建议先执行：

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

准备第三方库：

```bash
cd /work
mkdir -p 3rdparty

if [ ! -d 3rdparty/trtllm ]; then
  git clone --recurse-submodules https://github.com/NVIDIA/TensorRT-LLM.git 3rdparty/trtllm
fi

if [ ! -d 3rdparty/mlc-inference ]; then
  git clone --recurse-submodules https://github.com/mlcommons/inference.git 3rdparty/mlc-inference
fi

if [ ! -d 3rdparty/mitten ]; then
  git clone --recurse-submodules https://github.com/NVIDIA/mitten.git 3rdparty/mitten
fi

git config --global --add safe.directory /work/3rdparty/trtllm
```

## 3. 下载数据

在容器内使用 NVIDIA harness 下载数据：

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

make link_dirs

BENCHMARKS="llama2-70b" make download_data
```

检查下载结果：

```bash
find "$MLPERF_SCRATCH_PATH/data" -maxdepth 3 -iname "*llama2*" -o -iname "*cnn*"
```

## 4. 数据处理

本节命令需要在 NVIDIA Docker 容器内执行。若还没有进入容器，先完成第 2 节的 Docker 镜像下载和容器启动。

设置 MLPerf Inference v6.1 RNG seeds：

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

预处理数据：

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

make link_dirs

BENCHMARKS="llama2-70b" make preprocess_data

find /work/build/preprocessed_data/llama2-70b \
  -maxdepth 3 -type f | sort | head -n 100
```

检查 B300 x1 配置：

```bash
cd /work

CFG_70B=/work/configs/B300-SXM-270GBx1/Offline/llama2-70b.py

python3 -m py_compile "$CFG_70B"

grep -nE "min_duration|performance_sample_count|offline_expected_qps|gpu_batch_size|max_num_tokens|max_concurrency|num_postprocess_workers|stream_interval|kvcache_free_gpu_mem_frac|enable_max_num_tokens_tuning|enable_batch_size_tuning|use_graphs" \
  "$CFG_70B"
```

生成或确认 engine：

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

make generate_engines \
  RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline" \
  SYSTEM_NAME="$SYSTEM_NAME"
```

检查 engine：

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

## 5. 跑测试

### 1）. 跑 PerformanceOnly

Llama2-70B B300 x1 使用：

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

检查：

```bash
grep -E "qsl_rng_seed|sample_index_rng_seed|schedule_rng_seed|Result is|Min duration satisfied|Min queries satisfied|result_tokens_per_second|INVALID|VALID" \
  "$LOG" | tail -n 80
```

### 2）. 跑 AccuracyOnly

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

检查：

```bash
grep -E "qsl_rng_seed|sample_index_rng_seed|schedule_rng_seed" "$LOG" | tail -n 10
grep -Ei "accuracy|rouge|tokens per sample|gen_len|pass|fail|result" "$LOG" | tail -n 120
```
