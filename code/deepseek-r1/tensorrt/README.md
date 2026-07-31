# NVIDIA B300 x8 DeepSeek-R1  测试执行文档


## 1. 模型下载

DeepSeek-R1 B300 x8 使用 NVIDIA 的 FP4/NVFP4 checkpoint。


目标目录：

```text
$MLPERF_SCRATCH_PATH/models/deepseek-r1/fp4-quantized-modelopt/deepseek_r1-torch-fp4
```

```bash
set -euxo pipefail

sudo apt update
sudo apt install -y aria2 curl python3

REPO="nvidia/DeepSeek-R1-NVFP4-v2"
REVISION="main"

export MLPERF_SCRATCH_PATH=/path/to

TARGET_DIR="$MLPERF_SCRATCH_PATH/models/deepseek-r1/fp4-quantized-modelopt/deepseek_r1-torch-fp4"
URL_FILE="$MLPERF_SCRATCH_PATH/cache/deepseek-r1-nvfp4-v2-urls.txt"

mkdir -p "$TARGET_DIR" "$(dirname "$URL_FILE")"
: > "$URL_FILE"

echo "===== Prepare safetensor file list ====="

for i in $(seq 1 163); do
  name=$(printf "model-%05d-of-000163.safetensors" "$i")

  if [ -s "$TARGET_DIR/$name" ]; then
    echo "skip existing $name"
    continue
  fi

  echo "https://huggingface.co/${REPO}/resolve/${REVISION}/${name}" >> "$URL_FILE"
  echo "  out=${name}" >> "$URL_FILE"
done

if [ -s "$URL_FILE" ]; then
  echo "===== Download safetensors directly to TARGET_DIR ====="
  aria2c \
    --console-log-level=notice \
    --summary-interval=5 \
    -c \
    -x 1 \
    -s 1 \
    -j 10 \
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
  echo "all safetensor files already exist"
fi

echo "===== Verify target count ====="
find "$TARGET_DIR" -maxdepth 1 -name "model-*-of-000163.safetensors" | wc -l
du -sh "$TARGET_DIR"

echo "===== Download small config/tokenizer files ====="
cd "$TARGET_DIR"

for f in \
  LICENSE \
  README.md \
  config.json \
  configuration_deepseek.py \
  generation_config.json \
  hf_quant_config.json \
  model.safetensors.index.json \
  modeling_deepseek.py \
  tokenizer.json \
  tokenizer_config.json
do
  if [ ! -s "$TARGET_DIR/$f" ]; then
    curl -L -f --retry 10 --retry-delay 5 \
      -o "$TARGET_DIR/$f" \
      "https://huggingface.co/${REPO}/resolve/${REVISION}/${f}"
  fi
done

echo "===== Final check ====="
find "$TARGET_DIR" -maxdepth 1 -name "model-*-of-000163.safetensors" | wc -l
du -sh "$TARGET_DIR"
ls -lh "$TARGET_DIR"/config.json "$TARGET_DIR"/tokenizer.json "$TARGET_DIR"/model.safetensors.index.json
```

期望结果：

```text
safetensors count: 163
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
export SYSTEM_NAME=B300-SXM-270GBx8

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

创建数据目录，并使用 MLCommons R2 downloader 下载 DeepSeek-R1 评测数据：

```bash
export MLPERF_SCRATCH_PATH=/path/to

mkdir -p $MLPERF_SCRATCH_PATH/data/deepseek-r1

bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d $MLPERF_SCRATCH_PATH/data/deepseek-r1 \
  https://inference.mlcommons-storage.org/metadata/deepseek-r1-datasets-fp8-eval.uri

ls -lh $MLPERF_SCRATCH_PATH/data/deepseek-r1
```

DeepSeek-R1 需要以下两个 pkl 文件：

```text
mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl
mlperf_deepseek_r1_calibration_dataset_500_fp8_eval.pkl
```

## 4. 数据处理

本节命令需要在 NVIDIA Docker 容器内执行。若还没有进入容器，先完成第 2 节的 Docker 镜像下载和容器启动，再回到本节运行预处理。

进入容器后初始化目录链接：

```bash
cd /work
git config --global --add safe.directory '*'
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx8

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

创建预处理虚拟环境并执行 DeepSeek-R1 数据预处理：

```bash
export MLPERF_SCRATCH_PATH=/path/to

python3 -m venv $MLPERF_SCRATCH_PATH/venvs/deepseek-preprocess
source $MLPERF_SCRATCH_PATH/venvs/deepseek-preprocess/bin/activate

pip install -U pip
pip install "numpy==2.3.0" "torch==2.7.0" pandas pyarrow datasets

cd /work
export SYSTEM_NAME=B300-SXM-270GBx8

make link_dirs
mkdir -p $MLPERF_SCRATCH_PATH/preprocessed_data/deepseek-r1

python3 code/deepseek-r1/tensorrt/preprocess_data.py \
  --data_dir $MLPERF_SCRATCH_PATH/data \
  --preprocessed_data_dir $MLPERF_SCRATCH_PATH/preprocessed_data
```

检查预处理产物：

```bash
ls -lh build/preprocessed_data/deepseek-r1/input_lens.npy
ls -lh build/preprocessed_data/deepseek-r1/input_ids_padded.npy
ls -lh build/preprocessed_data/deepseek-r1/mlperf_deepseek_r1_calibration_dataset_500_fp8_calibration/data.parquet

find build/models/deepseek-r1/fp4-quantized-modelopt/deepseek_r1-torch-fp4 \
  -maxdepth 1 -name 'model-*-of-000163.safetensors' | wc -l
du -sh build/models/deepseek-r1/fp4-quantized-modelopt/deepseek_r1-torch-fp4
```

如果校准数据目录名没有对齐，创建兼容链接：

```bash
ln -sfn \
  mlperf_deepseek_r1_calibration_dataset_500_fp8_eval.pkl \
  build/preprocessed_data/deepseek-r1/mlperf_deepseek_r1_calibration_dataset_500_fp8_calibration

ls -lh build/preprocessed_data/deepseek-r1/mlperf_deepseek_r1_calibration_dataset_500_fp8_calibration/data.parquet

make link_dirs
deactivate
```

为 DeepSeek-R1 建立模型兼容路径：

```bash
cd /work
mkdir -p /work/build/models/deepseek-r1

ln -sfn \
  /work/build/models/deepseek-r1/fp4-quantized-modelopt/deepseek_r1-torch-fp4 \
  /work/build/models/deepseek-r1/deepseek-r1

ls -l /work/build/models/deepseek-r1/deepseek-r1
ls -lh /work/build/models/deepseek-r1/deepseek-r1/config.json
```

可选预构建：

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to

make prebuild ENV=release BENCHMARK=deepseek
```

如果 `make prebuild` 报缺少原始模型路径：

```text
build/models/deepseek-r1/deepseek-r1
```

先确认上面的兼容链接是否已经创建。正常流程优先只使用 NVFP4 checkpoint。

## 5. 跑测试

### 1）. 跑 PerformanceOnly

DeepSeek-R1 B300 x8 使用：

```text
SYSTEM_NAME=B300-SXM-270GBx8
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
export SYSTEM_NAME=B300-SXM-270GBx8
git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=Offline --core_type=trtllm_endpoint --test_mode=PerformanceOnly"

make run_llm_server
```


服务启动后，在另一个终端进入同一个容器，或重新启动一个挂载相同目录的容器，然后运行 harness：

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx8
git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=Offline --core_type=trtllm_endpoint --test_mode=PerformanceOnly"

make run_harness
```

### 2）. 跑 AccuracyOnly

AccuracyOnly 建议重新启动 fresh endpoint。先停止 PerformanceOnly 的 server，再在 server 终端执行：

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx8
git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=Offline --core_type=trtllm_endpoint --test_mode=AccuracyOnly"

make run_llm_server SYSTEM_NAME=B300-SXM-270GBx8
```

等 server ready 后，在另一个终端运行 accuracy harness：

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx8
git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=Offline --core_type=trtllm_endpoint --test_mode=AccuracyOnly"

make run_harness SYSTEM_NAME=B300-SXM-270GBx8
```

### 3）. 跑涉及的 Audit

DeepSeek-R1 本文保留原始记录中的 TEST06 Audit。建议重新启动 fresh endpoint：

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx8
git config --global --add safe.directory /work/3rdparty/trtllm

make run_llm_server \
  RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=Offline --core_type=trtllm_endpoint" \
  SYSTEM_NAME=B300-SXM-270GBx8
```

等 server ready 后，在另一个终端运行 TEST06：

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx8
git config --global --add safe.directory /work/3rdparty/trtllm

make run_audit_test06 \
  RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=Offline --core_type=trtllm_endpoint" \
  SYSTEM_NAME=B300-SXM-270GBx8
```
