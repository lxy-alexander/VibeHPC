# NVIDIA B300 x1 GPT-OSS-120B Test Execution Runbook


## 1. Model Download

GPT-OSS-120B on B300 x1 uses model files from MLCommons storage.

Target directory:

```text
$MLPERF_SCRATCH_PATH/models/gpt-oss-model/gpt-oss-120b
```

Expected path:

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

Expected result:

```text
safetensors count: 15
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
  --shm-size=32gb \
  --ulimit memlock=-1 \
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
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

make link_dirs
mkdir -p $MLPERF_SCRATCH_PATH/logs
ln -sfn $MLPERF_SCRATCH_PATH/logs build/logs
```

## 3. Data Download

Create a venv and use `mlc-scripts` to download GPT-OSS-120B data.

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

Arrange the files into the data directory expected by the NVIDIA harness:

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

## 4. Data Processing

Run the commands in this section inside the NVIDIA Docker container. If you have not entered the container yet, complete Section 2 to pull the Docker image and start the container, then return here for preprocessing.

After entering the container, initialize directory links:

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

Expected links:

```text
build/data -> $MLPERF_SCRATCH_PATH/data
build/models -> $MLPERF_SCRATCH_PATH/models
build/preprocessed_data -> $MLPERF_SCRATCH_PATH/preprocessed_data
build/logs -> $MLPERF_SCRATCH_PATH/logs
```

Generate `input_ids_padded.npy` and `input_lens.npy`:

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

Check preprocessing outputs:

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

Confirm the model compatibility path:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to

mkdir -p "$MLPERF_SCRATCH_PATH/models/gpt-oss"
ln -sfn "$MLPERF_SCRATCH_PATH/models/gpt-oss-model/gpt-oss-120b" \
  "$MLPERF_SCRATCH_PATH/models/gpt-oss/gpt-oss-120b"

ls -l build/models/gpt-oss/gpt-oss-120b
ls -lh build/models/gpt-oss/gpt-oss-120b/config.json
```

## 5. Run Tests

### 1. Run PerformanceOnly

Use the following for GPT-OSS-120B on B300 x1:

```text
SYSTEM_NAME=B300-SXM-270GBx1
```

Before starting the service, clean up any leftover processes:

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

Start the endpoint. Keep this terminal running:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=PerformanceOnly"

make run_llm_server
```

After the service starts, enter the same container from another terminal or start another container with the same mounts, then run the harness:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=PerformanceOnly"

make run_harness
```

### 2. Run AccuracyOnly

For AccuracyOnly, start a fresh endpoint. Stop the PerformanceOnly server first, then run the following in the server terminal:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=AccuracyOnly"

make run_llm_server
```

After the server is ready, run the accuracy harness in another terminal:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=AccuracyOnly"

make run_harness
```

### 3. Run Required Audit Tests

GPT-OSS-120B requires TEST07 and TEST09.

For TEST07, start a fresh endpoint:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=PerformanceOnly"

make run_llm_server
```

After the server is ready, run TEST07 in another terminal:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=PerformanceOnly --trtllm_server_urls=127.0.0.1:30000 --performance_sample_count=990 --performance_sample_count_override=990"

make run_audit_test07
```

For TEST09, start a fresh endpoint:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=AccuracyOnly"

make run_llm_server
```

After the server is ready, run TEST09 in another terminal:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=AccuracyOnly"

make run_audit_test09
```
