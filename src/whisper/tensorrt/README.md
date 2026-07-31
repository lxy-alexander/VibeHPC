# NVIDIA B300 x1 Whisper Test Execution Runbook


## 1. Model Download

Whisper uses the `whisper-large-v3` model.

Target directory:

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

ls -l build
```

## 3. Data Download

Whisper uses the LibriSpeech `dev-clean` and `dev-other` datasets.

Inside the container, prepare the download manifest and download the data:

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

## 4. Data Processing

Run the commands in this section inside the NVIDIA Docker container. If you have not entered the container yet, complete Section 2 to pull the Docker image and start the container.

If the current codebase does not have a B300 x1 Whisper configuration, first copy a nearby configuration as a starting point:

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

Run preprocessing:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

make link_dirs

cp 3rdparty/mlc-inference/speech2text/utils/inference_librispeech.csv \
  /work/code/whisper/tensorrt/utils/inference_librispeech.csv

BENCHMARKS="whisper" make preprocess_data
```

Check preprocessing outputs:

```bash
ls -lh /work/build/preprocessed_data/whisper-large-v3
ls -lh /work/build/preprocessed_data/whisper-large-v3/dev-all-repack.json
```

Optionally generate the checkpoint and engines:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1

make generate_engines RUN_ARGS="--benchmarks=whisper --scenarios=Offline"
```

## 5. Run Tests

### 1. Run PerformanceOnly

Use the following for Whisper on B300 x1:

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

### 2. Run AccuracyOnly

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1
export WHISPER_MULTIPROCESS=1
git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=whisper --scenarios=Offline --test_mode=AccuracyOnly"

make run_harness RUN_ARGS="$RUN_ARGS"
```

### 3. Run Required Audit Tests

Whisper requires TEST01, run through `run_audit_harness`.

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1
export WHISPER_MULTIPROCESS=1
git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=whisper --scenarios=Offline"

make run_audit_harness RUN_ARGS="$RUN_ARGS"
```
