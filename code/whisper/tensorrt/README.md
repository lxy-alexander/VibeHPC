# NVIDIA B300 x1 Whisper  测试执行文档


## 1. 模型下载

Whisper 使用 `whisper-large-v3` 模型。

目标目录：

```text
$MLPERF_SCRATCH_PATH/models/whisper-large-v3
```

```bash
set -euxo pipefail

export MLPERF_SCRATCH_PATH=/path/to

TARGET_DIR="$MLPERF_SCRATCH_PATH/models/whisper-large-v3"
mkdir -p "$TARGET_DIR"

curl -L -f --retry 10 --retry-delay 5 \
  -o "$TARGET_DIR/multilingual.tiktoken" \
  https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken

curl -L -f --retry 10 --retry-delay 5 \
  -o "$TARGET_DIR/mel_filters.npz" \
  https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz

curl -L -f --retry 10 --retry-delay 5 \
  -o "$TARGET_DIR/large-v3.pt" \
  https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt

md5sum "$TARGET_DIR/multilingual.tiktoken" | grep "da95f6601b7c4327d4464d081b9dcf09"
md5sum "$TARGET_DIR/mel_filters.npz" | grep "61b070b259b27b8f8550e632d5300c8b"
md5sum "$TARGET_DIR/large-v3.pt" | grep "017baacdaada84d0d5cb030140875b65"

ls -lh "$TARGET_DIR"
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

ls -l build
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

Whisper 使用 LibriSpeech `dev-clean` 和 `dev-other` 数据。

在容器内准备下载清单并下载数据：

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to

make link_dirs

cp 3rdparty/mlc-inference/speech2text/utils/inference_librispeech.csv \
  /work/code/whisper/tensorrt/utils/inference_librispeech.csv

mkdir -p build/data/whisper-large-v3/LibriSpeech

python /work/code/whisper/tensorrt/utils/download_librispeech.py \
  /work/code/whisper/tensorrt/utils/inference_librispeech.csv \
  /work/build/data/whisper-large-v3/LibriSpeech \
  -e /work/build/data/whisper-large-v3

ls -lh /work/build/data/whisper-large-v3
```

## 4. 数据处理

本节命令需要在 NVIDIA Docker 容器内执行。若还没有进入容器，先完成第 2 节的 Docker 镜像下载和容器启动。

如果当前代码库没有 B300 x1 Whisper 配置，先从相近配置复制一个起点：

```bash
cd /work

if [ ! -f configs/B300-SXM-270GBx1/Offline/whisper.py ]; then
  mkdir -p configs/B300-SXM-270GBx1/Offline
  cp configs/B200-SXM-180GBx8/Offline/whisper.py \
    configs/B300-SXM-270GBx1/Offline/whisper.py

  python3 - <<'PY'
from pathlib import Path

p = Path("configs/B300-SXM-270GBx1/Offline/whisper.py")
s = p.read_text()

s = s.replace("loadgen_fields.offline_expected_qps: 770", "loadgen_fields.offline_expected_qps: 120")
s = s.replace("'encoder': 320", "'encoder': 768")
s = s.replace("'decoder': 320", "'decoder': 768")
s = s.replace("'max_batch_size': 320", "'max_batch_size': 768")

p.write_text(s)
print(f"created {p}")
PY
fi
```

执行预处理：

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

make link_dirs

cp 3rdparty/mlc-inference/speech2text/utils/inference_librispeech.csv \
  /work/code/whisper/tensorrt/utils/inference_librispeech.csv

BENCHMARKS="whisper" make preprocess_data
```

检查预处理产物：

```bash
ls -lh /work/build/preprocessed_data/whisper-large-v3
ls -lh /work/build/preprocessed_data/whisper-large-v3/dev-all-repack.json
```

可选生成 checkpoint 和 engines：

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

make generate_engines RUN_ARGS="--benchmarks=whisper --scenarios=Offline"
```

## 5. 跑测试

### 1）. 跑 PerformanceOnly

Whisper B300 x1 使用：

```text
SYSTEM_NAME=B300-SXM-270GBx1
```

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1
export WHISPER_MULTIPROCESS=1
git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=whisper --scenarios=Offline --test_mode=PerformanceOnly"

make run_harness RUN_ARGS="$RUN_ARGS"
```

### 2）. 跑 AccuracyOnly

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1
export WHISPER_MULTIPROCESS=1
git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=whisper --scenarios=Offline --test_mode=AccuracyOnly"

make run_harness RUN_ARGS="$RUN_ARGS"
```

### 3）. 跑涉及的 Audit

Whisper 涉及 TEST01，通过 `run_audit_harness` 运行。

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1
export WHISPER_MULTIPROCESS=1
git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=whisper --scenarios=Offline"

make run_audit_harness RUN_ARGS="$RUN_ARGS"
```
