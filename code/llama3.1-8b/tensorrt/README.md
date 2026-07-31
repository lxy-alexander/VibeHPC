# NVIDIA B300 x1 Llama3.1-8B  测试执行文档


## 1. 模型下载

Llama3.1-8B 使用 Meta Llama 3.1 8B Instruct 模型，需要提前获得 Meta Llama 模型访问权限。

目标目录：

```text
$MLPERF_SCRATCH_PATH/models/Llama-3.1-8B-Instruct
```

```bash
set -euxo pipefail

export MLPERF_SCRATCH_PATH=/path/to

python3 -m pip install -U "huggingface_hub[cli]"

# 如需认证，先执行：
# huggingface-cli login

huggingface-cli download \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --local-dir "$MLPERF_SCRATCH_PATH/models/Llama-3.1-8B-Instruct" \
  --local-dir-use-symlinks False

MODEL_8B="$MLPERF_SCRATCH_PATH/models/Llama-3.1-8B-Instruct"

test -s "$MODEL_8B/config.json"
test -s "$MODEL_8B/model.safetensors.index.json"
find "$MODEL_8B" -maxdepth 1 -name "model-*.safetensors" | sort
du -sh "$MODEL_8B"
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

BENCHMARKS="llama3_1-8b" make download_data
```

检查下载结果：

```bash
find "$MLPERF_SCRATCH_PATH/data" -maxdepth 3 -iname "*llama*" -o -iname "*cnn*"
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

准备 B300 x1 配置：

```bash
cd /work

if [ ! -f configs/B300-SXM-270GBx1/Offline/llama3_1-8b.py ]; then
  mkdir -p configs/B300-SXM-270GBx1/Offline
  cp configs/B200-SXM-180GBx1/Offline/llama3_1-8b.py \
    configs/B300-SXM-270GBx1/Offline/llama3_1-8b.py
fi

python3 -m py_compile configs/B300-SXM-270GBx1/Offline/llama3_1-8b.py
grep -nE "min_duration|performance_sample_count|offline_expected_qps|gpu_batch_size|max_num_tokens|max_concurrency|tensor_parallelism|pipeline_parallelism" \
  configs/B300-SXM-270GBx1/Offline/llama3_1-8b.py
```

预处理数据：

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

make link_dirs

BENCHMARKS="llama3_1-8b" make preprocess_data

find /work/build/preprocessed_data \
  -maxdepth 3 -type f \
  \( -path "*llama3.1-8b*" -o -path "*llama3_1-8b*" \) \
  | sort | head -n 50
```

生成 NVFP4 checkpoint 和 engine：

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

make generate_engines \
  RUN_ARGS="--benchmarks=llama3_1-8b --scenarios=Offline --model_path=$MLPERF_SCRATCH_PATH/models/Llama-3.1-8B-Instruct --llm_quantizer_outdir=$MLPERF_SCRATCH_PATH/models/Llama3.1-8B"
```

检查构建结果：

```bash
find /work/build/models/Llama3.1-8B \
  -maxdepth 4 -type f \
  \( -name "config.json" -o -name "rank*.safetensors" \) \
  -ls

find /work/build/engines/B300-SXM-270GBx1/Offline \
  -path "*llama3*" -type f \
  \( -name "config.json" -o -name "rank*.engine" \) \
  -ls
```

## 5. 跑测试

### 1）. 跑 PerformanceOnly

Llama3.1-8B B300 x1 使用：

```text
SYSTEM_NAME=B300-SXM-270GBx1
```

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1
export RUN_ARGS="--benchmarks=llama3_1-8b --scenarios=Offline"

rm -rf \
  /work/build/loadgen-configs/B300-SXM-270GBx1_TRT/llama3_1-8b*/Offline \
  /work/build/loadgen-configs/B300-SXM-270GBx1_TRT/llama3.1-8b*/Offline

LOG="$MLPERF_SCRATCH_PATH/logs/b300-1gpu-llama8b/v61_performance.log"
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
export RUN_ARGS="--benchmarks=llama3_1-8b --scenarios=Offline"

LOG="$MLPERF_SCRATCH_PATH/logs/b300-1gpu-llama8b/v61_accuracy.log"
mkdir -p "$(dirname "$LOG")"

make run_harness \
  RUN_ARGS="$RUN_ARGS --test_mode=AccuracyOnly" \
  SYSTEM_NAME="$SYSTEM_NAME" \
  2>&1 | tee "$LOG"
```

检查：

```bash
grep -E "qsl_rng_seed|sample_index_rng_seed|schedule_rng_seed" "$LOG" | tail -n 10
grep -Ei "accuracy|rouge|tokens per sample|pass|fail|result" "$LOG" | tail -n 100
```
