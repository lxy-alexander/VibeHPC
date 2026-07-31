# NVIDIA B300 x1 GPT-OSS-120B  测试执行文档


## 1. 模型下载

GPT-OSS-120B B300 x1 使用 MLCommons storage 中的模型文件。

目标目录：

```text
$MLPERF_SCRATCH_PATH/models/gpt-oss-model/gpt-oss-120b
```

期望路径：

```text
$MLPERF_SCRATCH_PATH/models/gpt-oss/gpt-oss-120b
```

```bash
set -euxo pipefail

sudo apt update
sudo apt install -y aria2 curl python3

export MLPERF_SCRATCH_PATH=/path/to

SRC_BASE="https://inference.mlcommons-storage.org/gpt-oss_model/gpt-oss-120b"
TARGET_DIR="$MLPERF_SCRATCH_PATH/models/gpt-oss-model/gpt-oss-120b"
URL_FILE="$MLPERF_SCRATCH_PATH/cache/gpt-oss-120b-model-urls.txt"

mkdir -p "$TARGET_DIR" "$(dirname "$URL_FILE")"
: > "$URL_FILE"

echo "===== Prepare model file list ====="

for i in $(seq 0 14); do
  name=$(printf "model-%05d-of-00014.safetensors" "$i")

  if [ -s "$TARGET_DIR/$name" ]; then
    echo "skip existing $name"
    continue
  fi

  echo "$SRC_BASE/$name" >> "$URL_FILE"
done

for f in \
  config.json \
  generation_config.json \
  tokenizer.json \
  tokenizer_config.json \
  special_tokens_map.json \
  model.safetensors.index.json
do
  if [ -s "$TARGET_DIR/$f" ]; then
    echo "skip existing $f"
    continue
  fi

  echo "$SRC_BASE/$f" >> "$URL_FILE"
done

if [ -s "$URL_FILE" ]; then
  echo "===== Download model files directly to TARGET_DIR ====="
  aria2c \
    --console-log-level=notice \
    --summary-interval=5 \
    -c \
    -x 16 \
    -s 16 \
    -j 4 \
    -k 64M \
    --file-allocation=none \
    --retry-wait=5 \
    --max-tries=0 \
    --timeout=60 \
    --connect-timeout=30 \
    --allow-overwrite=false \
    --auto-file-renaming=false \
    --dir="$TARGET_DIR" \
    -i "$URL_FILE"
else
  echo "all model files already exist"
fi

echo "===== Create expected model path ====="
mkdir -p "$MLPERF_SCRATCH_PATH/models/gpt-oss"
ln -sfn "$TARGET_DIR" "$MLPERF_SCRATCH_PATH/models/gpt-oss/gpt-oss-120b"

echo "===== Final check ====="
find "$TARGET_DIR" -maxdepth 1 -name "model-*-of-00014.safetensors" | wc -l
du -sh "$TARGET_DIR"
ls -lh "$TARGET_DIR"/config.json "$TARGET_DIR"/tokenizer.json "$TARGET_DIR"/model.safetensors.index.json
ls -l "$MLPERF_SCRATCH_PATH/models/gpt-oss/gpt-oss-120b"
```

期望结果：

```text
safetensors count: 15
```

## 2. 下载 Docker（使用 NVIDIA Docker）

B300 x86 机器使用 NVIDIA x86 MLPerf TensorRT-LLM 镜像。

拉取 NVIDIA Docker 镜像：

```bash
docker pull nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_x86
```

进入容器

```bash
export VibeHPC_PATH=/path/to
export MLPERF_SCRATCH_PATH=/path/to

docker run --rm -it \
  --gpus all \
  --net host \
  --ipc host \
  --shm-size=32gb \
  --ulimit memlock=-1 \
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
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

make link_dirs
mkdir -p $MLPERF_SCRATCH_PATH/logs
ln -sfn $MLPERF_SCRATCH_PATH/logs build/logs
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

创建 venv，并使用 `mlc-scripts` 下载 GPT-OSS-120B 数据。

```bash
set -euxo pipefail

export MLPERF_SCRATCH_PATH=/path/to

sudo apt update
sudo apt install -y python3-venv python3-full

mkdir -p "$MLPERF_SCRATCH_PATH"/{data,models,preprocessed_data,logs,venvs,cache}
python3 -m venv "$MLPERF_SCRATCH_PATH/venvs/mlperf"
source "$MLPERF_SCRATCH_PATH/venvs/mlperf/bin/activate"

python -m pip install -U pip
python -m pip install mlc-scripts

mlcr get-dataset-mlperf-inference-gpt-oss,_mlc,_r2-downloader \
  --outdirname="$MLPERF_SCRATCH_PATH/data/gpt-oss-120b-raw" \
  -j
```

整理为 NVIDIA harness 期望的数据目录：

```bash
export MLPERF_SCRATCH_PATH=/path/to

mkdir -p "$MLPERF_SCRATCH_PATH/data/gpt-oss/v4/acc"
mkdir -p "$MLPERF_SCRATCH_PATH/data/gpt-oss/v4/perf"
mkdir -p "$MLPERF_SCRATCH_PATH/data/gpt-oss/v4/compliance/test07"

ln -sfn "$MLPERF_SCRATCH_PATH/data/gpt-oss-120b-raw/gpt-oss-dataset/acc/acc_eval_ref.parquet" \
  "$MLPERF_SCRATCH_PATH/data/gpt-oss/v4/acc/acc_eval_ref.parquet"

ln -sfn "$MLPERF_SCRATCH_PATH/data/gpt-oss-120b-raw/gpt-oss-dataset/acc/acc_eval_compliance_gpqa.parquet" \
  "$MLPERF_SCRATCH_PATH/data/gpt-oss/v4/acc/acc_eval_compliance_gpqa.parquet"

ln -sfn "$MLPERF_SCRATCH_PATH/data/gpt-oss-120b-raw/gpt-oss-dataset/perf/perf_eval_ref.parquet" \
  "$MLPERF_SCRATCH_PATH/data/gpt-oss/v4/perf/perf_eval_ref.parquet"

ls -lh "$MLPERF_SCRATCH_PATH/data/gpt-oss/v4/acc"
ls -lh "$MLPERF_SCRATCH_PATH/data/gpt-oss/v4/perf"
```

## 4. 数据处理

本节命令需要在 NVIDIA Docker 容器内执行。若还没有进入容器，先完成第 2 节的 Docker 镜像下载和容器启动，再回到本节运行预处理。

进入容器后初始化目录链接：

```bash
cd /work
git config --global --add safe.directory '*'
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

make link_dirs

mkdir -p $MLPERF_SCRATCH_PATH/logs
ln -sfn $MLPERF_SCRATCH_PATH/logs build/logs

ls -l build
```

期望链接：

```text
build/data -> $MLPERF_SCRATCH_PATH/data
build/models -> $MLPERF_SCRATCH_PATH/models
build/preprocessed_data -> $MLPERF_SCRATCH_PATH/preprocessed_data
build/logs -> $MLPERF_SCRATCH_PATH/logs
```

生成 `input_ids_padded.npy` 和 `input_lens.npy`：

```bash
set -euxo pipefail

cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

source "$MLPERF_SCRATCH_PATH/venvs/mlperf/bin/activate"

python -m ensurepip --upgrade || true
python -m pip install --upgrade pip setuptools wheel
python -m pip install pandas pyarrow numpy transformers

python code/gpt-oss-120b/tensorrt/preprocess_compliance_data.py \
  --input-file "$MLPERF_SCRATCH_PATH/data/gpt-oss/v4/acc/acc_eval_ref.parquet" \
  --output-dir "$MLPERF_SCRATCH_PATH/data/gpt-oss/v4/acc" \
  --tokenizer "$MLPERF_SCRATCH_PATH/models/gpt-oss-model/gpt-oss-120b"

python code/gpt-oss-120b/tensorrt/preprocess_compliance_data.py \
  --input-file "$MLPERF_SCRATCH_PATH/data/gpt-oss/v4/perf/perf_eval_ref.parquet" \
  --output-dir "$MLPERF_SCRATCH_PATH/data/gpt-oss/v4/perf" \
  --tokenizer "$MLPERF_SCRATCH_PATH/models/gpt-oss-model/gpt-oss-120b"

python code/gpt-oss-120b/tensorrt/preprocess_compliance_data.py \
  --input-file "$MLPERF_SCRATCH_PATH/data/gpt-oss/v4/acc/acc_eval_compliance_gpqa.parquet" \
  --output-dir "$MLPERF_SCRATCH_PATH/data/gpt-oss/v4/compliance/test07" \
  --tokenizer "$MLPERF_SCRATCH_PATH/models/gpt-oss-model/gpt-oss-120b"
```

检查预处理产物：

```bash
export MLPERF_SCRATCH_PATH=/path/to

python - <<'PY'
import os
from pathlib import Path
import numpy as np

scratch = Path(os.environ["MLPERF_SCRATCH_PATH"])

for rel, n in [
    ("data/gpt-oss/v4/acc", 4395),
    ("data/gpt-oss/v4/perf", 6396),
    ("data/gpt-oss/v4/compliance/test07", 990),
]:
    p = scratch / rel
    ids = np.load(p / "input_ids_padded.npy", mmap_mode="r")
    lens = np.load(p / "input_lens.npy", mmap_mode="r")
    print(p, ids.shape, lens.shape)
    assert ids.shape[0] == n
    assert lens.shape[0] == n
PY
```

确认模型兼容路径：

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to

mkdir -p "$MLPERF_SCRATCH_PATH/models/gpt-oss"
ln -sfn "$MLPERF_SCRATCH_PATH/models/gpt-oss-model/gpt-oss-120b" \
  "$MLPERF_SCRATCH_PATH/models/gpt-oss/gpt-oss-120b"

ls -l build/models/gpt-oss/gpt-oss-120b
ls -lh build/models/gpt-oss/gpt-oss-120b/config.json
```

## 5. 跑测试

### 1）. 跑 PerformanceOnly

GPT-OSS-120B B300 x1 使用：

```text
SYSTEM_NAME=B300-SXM-270GBx1
```

启动服务前建议清理残留进程：

```bash
echo "== before =="
nvidia-smi

echo "== kill possible leftovers =="
pkill -9 -f trtllm-serve || true
pkill -9 -f tensorrt_llm || true
pkill -9 -f llmapi || true
pkill -9 -f 'python.*code.main' || true
pkill -9 -f '/usr/bin/python' || true
pkill -9 -f '/usr/local/bin/python' || true

sleep 5

echo "== users of nvidia devices =="
fuser -v /dev/nvidia* || true

echo "== after kill =="
nvidia-smi
```

启动 endpoint。此终端保持运行，不要关闭：

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=PerformanceOnly"

make run_llm_server
```

服务启动后，在另一个终端进入同一个容器，或重新启动一个挂载相同目录的容器，然后运行 harness：

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=PerformanceOnly"

make run_harness
```

### 2）. 跑 AccuracyOnly

AccuracyOnly 建议重新启动 fresh endpoint。先停止 PerformanceOnly 的 server，再在 server 终端执行：

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=AccuracyOnly"

make run_llm_server
```

等 server ready 后，在另一个终端运行 accuracy harness：

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=AccuracyOnly"

make run_harness
```

### 3）. 跑涉及的 Audit

GPT-OSS-120B 涉及 TEST07 和 TEST09。

TEST07 建议重新启动 fresh endpoint：

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=PerformanceOnly"

make run_llm_server
```

等 server ready 后，在另一个终端运行 TEST07：

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=PerformanceOnly --trtllm_server_urls=127.0.0.1:30000 --performance_sample_count=990 --performance_sample_count_override=990"

make run_audit_test07
```

TEST09 建议重新启动 fresh endpoint：

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=AccuracyOnly"

make run_llm_server
```

等 server ready 后，在另一个终端运行 TEST09：

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=AccuracyOnly"

make run_audit_test09
```
