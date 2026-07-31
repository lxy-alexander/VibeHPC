# NVIDIA B300 x8 DeepSeek-R1 Test Execution Runbook


## 1. Model Download

DeepSeek-R1 on B300 x8 uses NVIDIA FP4/NVFP4 checkpoints.


Target directory:

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

Expected result:

```text
safetensors count: 163
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
export SYSTEM_NAME=B300-SXM-270GBx8

make link_dirs
mkdir -p $MLPERF_SCRATCH_PATH/logs
ln -sfn $MLPERF_SCRATCH_PATH/logs build/logs
```

## 3. Data Download

Create the data directory and use the MLCommons R2 downloader to download DeepSeek-R1 evaluation data:

```bash
export MLPERF_SCRATCH_PATH=/path/to

mkdir -p $MLPERF_SCRATCH_PATH/data/deepseek-r1

bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d $MLPERF_SCRATCH_PATH/data/deepseek-r1 \
  https://inference.mlcommons-storage.org/metadata/deepseek-r1-datasets-fp8-eval.uri

ls -lh $MLPERF_SCRATCH_PATH/data/deepseek-r1
```

DeepSeek-R1 requires the following two pkl files:

```text
mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl
mlperf_deepseek_r1_calibration_dataset_500_fp8_eval.pkl
```

## 4. Data Processing

Run the commands in this section inside the NVIDIA Docker container. If you have not entered the container yet, complete Section 2 to pull the Docker image and start the container, then return here for preprocessing.

After entering the container, initialize directory links:

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

Expected links:

```text
build/data -> $MLPERF_SCRATCH_PATH/data
build/models -> $MLPERF_SCRATCH_PATH/models
build/preprocessed_data -> $MLPERF_SCRATCH_PATH/preprocessed_data
build/logs -> $MLPERF_SCRATCH_PATH/logs
```

Create a preprocessing virtual environment and run DeepSeek-R1 data preprocessing:

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

Check preprocessing outputs:

```bash
ls -lh build/preprocessed_data/deepseek-r1/input_lens.npy
ls -lh build/preprocessed_data/deepseek-r1/input_ids_padded.npy
ls -lh build/preprocessed_data/deepseek-r1/mlperf_deepseek_r1_calibration_dataset_500_fp8_calibration/data.parquet

find build/models/deepseek-r1/fp4-quantized-modelopt/deepseek_r1-torch-fp4 \
  -maxdepth 1 -name 'model-*-of-000163.safetensors' | wc -l
du -sh build/models/deepseek-r1/fp4-quantized-modelopt/deepseek_r1-torch-fp4
```

If the calibration data directory name does not match, create a compatibility link:

```bash
ln -sfn \
  mlperf_deepseek_r1_calibration_dataset_500_fp8_eval.pkl \
  build/preprocessed_data/deepseek-r1/mlperf_deepseek_r1_calibration_dataset_500_fp8_calibration

ls -lh build/preprocessed_data/deepseek-r1/mlperf_deepseek_r1_calibration_dataset_500_fp8_calibration/data.parquet

make link_dirs
deactivate
```

Create a model compatibility path for DeepSeek-R1:

```bash
cd /work
mkdir -p /work/build/models/deepseek-r1

ln -sfn \
  /work/build/models/deepseek-r1/fp4-quantized-modelopt/deepseek_r1-torch-fp4 \
  /work/build/models/deepseek-r1/deepseek-r1

ls -l /work/build/models/deepseek-r1/deepseek-r1
ls -lh /work/build/models/deepseek-r1/deepseek-r1/config.json
```

Optional prebuild:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to

make prebuild ENV=release BENCHMARK=deepseek
```

If `make prebuild` reports a missing original model path:

```text
build/models/deepseek-r1/deepseek-r1
```

First confirm that the compatibility link above has been created. The normal flow should use only the NVFP4 checkpoint.

## 5. Run Tests

### 1. Run PerformanceOnly

Use the following for DeepSeek-R1 on B300 x8:

```text
SYSTEM_NAME=B300-SXM-270GBx8
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
export SYSTEM_NAME=B300-SXM-270GBx8
git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=Offline --core_type=trtllm_endpoint --test_mode=PerformanceOnly"

make run_llm_server
```


After the service starts, enter the same container from another terminal or start another container with the same mounts, then run the harness:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx8
git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=Offline --core_type=trtllm_endpoint --test_mode=PerformanceOnly"

make run_harness
```

### 2. Run AccuracyOnly

For AccuracyOnly, start a fresh endpoint. Stop the PerformanceOnly server first, then run the following in the server terminal:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx8
git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=Offline --core_type=trtllm_endpoint --test_mode=AccuracyOnly"

make run_llm_server SYSTEM_NAME=B300-SXM-270GBx8
```

After the server is ready, run the accuracy harness in another terminal:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx8
git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=Offline --core_type=trtllm_endpoint --test_mode=AccuracyOnly"

make run_harness SYSTEM_NAME=B300-SXM-270GBx8
```

### 3. Run Required Audit Tests

For DeepSeek-R1, this runbook keeps the TEST06 audit from the source notes. Start a fresh endpoint:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx8
git config --global --add safe.directory /work/3rdparty/trtllm

make run_llm_server \
  RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=Offline --core_type=trtllm_endpoint" \
  SYSTEM_NAME=B300-SXM-270GBx8
```

After the server is ready, run TEST06 in another terminal:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx8
git config --global --add safe.directory /work/3rdparty/trtllm

make run_audit_test06 \
  RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=Offline --core_type=trtllm_endpoint" \
  SYSTEM_NAME=B300-SXM-270GBx8
```
