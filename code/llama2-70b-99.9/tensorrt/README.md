# NVIDIA B300 x1 Llama2-70B-99.9 Test Execution Runbook


## 1. Model Download

Llama2-70B-99.9 uses the public FP4 checkpoint below:

```text
centml/llama2-70b-chat-hf-torch-fp4_mlperf-inf-v6.0
```

Download target:

```text
$MLPERF_SCRATCH_PATH/models/llama2-70b-chat-hf-torch-fp4-v6.0
```

NVIDIA compatibility paths:

```text
$MLPERF_SCRATCH_PATH/models/Llama2/fp4-quantized-modelopt/llama2-70b-chat-hf-torch-fp4
$MLPERF_SCRATCH_PATH/models/Llama2/fp4-quantized-modelopt/llama2-70b-chat-hf-tp1pp1-fp4
```

```bash
set -euxo pipefail

export MLPERF_SCRATCH_PATH=/path/to

python3 -m pip install -U "huggingface_hub[cli]"

# If authentication is required, run:
# huggingface-cli login

LLAMA2_70B_MODEL_PATH="$MLPERF_SCRATCH_PATH/models/llama2-70b-chat-hf-torch-fp4-v6.0"

huggingface-cli download \
  centml/llama2-70b-chat-hf-torch-fp4_mlperf-inf-v6.0 \
  --local-dir "$LLAMA2_70B_MODEL_PATH" \
  --local-dir-use-symlinks False

DEFAULT_HF_CKPT="$MLPERF_SCRATCH_PATH/models/Llama2/fp4-quantized-modelopt/llama2-70b-chat-hf-torch-fp4"
DEFAULT_TRTLLM_CKPT="$MLPERF_SCRATCH_PATH/models/Llama2/fp4-quantized-modelopt/llama2-70b-chat-hf-tp1pp1-fp4"

mkdir -p "$(dirname "$DEFAULT_HF_CKPT")"

if [ ! -e "$DEFAULT_HF_CKPT" ]; then
  ln -s "$LLAMA2_70B_MODEL_PATH" "$DEFAULT_HF_CKPT"
fi

if [ ! -e "$DEFAULT_TRTLLM_CKPT" ]; then
  ln -s "$LLAMA2_70B_MODEL_PATH" "$DEFAULT_TRTLLM_CKPT"
fi

find "$LLAMA2_70B_MODEL_PATH" -maxdepth 2 -type f | sort | head -n 100
du -sh "$LLAMA2_70B_MODEL_PATH"
ls -l "$DEFAULT_HF_CKPT" "$DEFAULT_TRTLLM_CKPT"
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
  --name mlperf-b300-llama70b-999 \
  --gpus all \
  --net host \
  --ipc host \
  --shm-size=128gb \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v $VibeHPC_PATH:/work \
  -v $MLPERF_SCRATCH_PATH:$MLPERF_SCRATCH_PATH \
  -e MLPERF_SCRATCH_PATH=$MLPERF_SCRATCH_PATH \
  -w /work \
  nvcr.io/nvidia/mlperf/mlperf-inference:tensorrt_llm_release-feat-1.2-mlpinf-b5ddff4_mlperf-main-f538816_jan28_x86
```

For another terminal connected to the same running container:

```bash
docker exec -it mlperf-b300-llama70b-999 bash
```

Inside the container, run the following first:

```bash
cd /work
git config --global --add safe.directory '*'
git config --global --add safe.directory /work
git config --global --add safe.directory /work/3rdparty/trtllm

export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1
export CUDA_VISIBLE_DEVICES=0

nvidia-smi -L

python3 - <<'PY'
import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

make link_dirs
mkdir -p "$MLPERF_SCRATCH_PATH/logs"
rm -rf /work/build/logs
ln -s "$MLPERF_SCRATCH_PATH/logs" /work/build/logs

ls -lah /work/build
```

Expected GPU count:

```text
GPU count: 1
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
find "$MLPERF_SCRATCH_PATH/data" -maxdepth 4 -iname "*llama2*" -o -iname "*open_orca*"
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

Add the B300 x1 99.9 accuracy target to the Offline and Server configs:

```bash
cd /work

python3 - <<'PY'
from pathlib import Path

configs = [
    Path("configs/B300-SXM-270GBx1/Offline/llama2-70b.py"),
    Path("configs/B300-SXM-270GBx1/Server/llama2-70b.py"),
]

export_99 = "C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): ifb_config,"
export_999 = "C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.999), C.PowerSetting.MaxP): ifb_config,"

atomic_99 = '''C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        "default": ifb_config,
    },'''
atomic_999 = '''C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.999), C.PowerSetting.MaxP): {
        "default": ifb_config,
    },'''

for cfg in configs:
    s = cfg.read_text()
    if "AccuracyTarget(0.999)" not in s:
        s = s.replace(export_99, export_99 + "\n    " + export_999)
        s = s.replace(atomic_99, atomic_99 + "\n    " + atomic_999)
        cfg.write_text(s)
        print(f"patched {cfg}")
    else:
        print(f"already patched {cfg}")
PY

python3 -m py_compile \
  configs/B300-SXM-270GBx1/Offline/llama2-70b.py \
  configs/B300-SXM-270GBx1/Server/llama2-70b.py

grep -n "AccuracyTarget(0.999)" \
  configs/B300-SXM-270GBx1/Offline/llama2-70b.py \
  configs/B300-SXM-270GBx1/Server/llama2-70b.py
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

Check the model compatibility links:

```bash
export MLPERF_SCRATCH_PATH=/path/to

LLAMA2_70B_MODEL_PATH="$MLPERF_SCRATCH_PATH/models/llama2-70b-chat-hf-torch-fp4-v6.0"
DEFAULT_HF_CKPT="$MLPERF_SCRATCH_PATH/models/Llama2/fp4-quantized-modelopt/llama2-70b-chat-hf-torch-fp4"
DEFAULT_TRTLLM_CKPT="$MLPERF_SCRATCH_PATH/models/Llama2/fp4-quantized-modelopt/llama2-70b-chat-hf-tp1pp1-fp4"

test -e "$LLAMA2_70B_MODEL_PATH"
test -e "$DEFAULT_HF_CKPT"
test -e "$DEFAULT_TRTLLM_CKPT"
ls -l "$DEFAULT_HF_CKPT" "$DEFAULT_TRTLLM_CKPT"
du -sh "$LLAMA2_70B_MODEL_PATH"
```

## 5. Run Tests

### 1. Run PerformanceOnly

Use the following for Llama2-70B-99.9 on B300 x1:

```text
SYSTEM_NAME=B300-SXM-270GBx1
```

Before starting the service, clean up any leftover processes:

```bash
echo "== before =="
nvidia-smi

echo "== kill possible leftovers =="
pkill -9 -f run_llm_server || true
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

Start the Offline endpoint. Keep this terminal running:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1
export CUDA_VISIBLE_DEVICES=0
export LLAMA2_70B_MODEL_PATH="$MLPERF_SCRATCH_PATH/models/llama2-70b-chat-hf-torch-fp4-v6.0"

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --core_type=trtllm_endpoint --accuracy_target=.999 --test_mode=PerformanceOnly --model_path=$LLAMA2_70B_MODEL_PATH"

make run_llm_server RUN_ARGS="$RUN_ARGS" SYSTEM_NAME="$SYSTEM_NAME"
```

After the Offline endpoint is ready, run the Offline harness in another terminal:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1
export CUDA_VISIBLE_DEVICES=0
export LLAMA2_70B_MODEL_PATH="$MLPERF_SCRATCH_PATH/models/llama2-70b-chat-hf-torch-fp4-v6.0"

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --core_type=trtllm_endpoint --accuracy_target=.999 --test_mode=PerformanceOnly --model_path=$LLAMA2_70B_MODEL_PATH"

LOG="$MLPERF_SCRATCH_PATH/logs/b300-1gpu-llama70b-999/offline_performance.log"
mkdir -p "$(dirname "$LOG")"

make run_harness RUN_ARGS="$RUN_ARGS" SYSTEM_NAME="$SYSTEM_NAME" 2>&1 | tee "$LOG"
```

Check:

```bash
grep -E "qsl_rng_seed|sample_index_rng_seed|schedule_rng_seed|Result is|Min duration satisfied|Min queries satisfied|result_tokens_per_second|INVALID|VALID" \
  "$LOG" | tail -n 80
```

To run Server PerformanceOnly, stop the Offline endpoint first, then start a fresh Server endpoint:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1
export CUDA_VISIBLE_DEVICES=0
export LLAMA2_70B_MODEL_PATH="$MLPERF_SCRATCH_PATH/models/llama2-70b-chat-hf-torch-fp4-v6.0"

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=llama2-70b --scenarios=Server --core_type=trtllm_endpoint --accuracy_target=.999 --test_mode=PerformanceOnly --model_path=$LLAMA2_70B_MODEL_PATH"

make run_llm_server RUN_ARGS="$RUN_ARGS" SYSTEM_NAME="$SYSTEM_NAME"
```

After the Server endpoint is ready, run the Server harness in another terminal:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1
export CUDA_VISIBLE_DEVICES=0
export LLAMA2_70B_MODEL_PATH="$MLPERF_SCRATCH_PATH/models/llama2-70b-chat-hf-torch-fp4-v6.0"

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=llama2-70b --scenarios=Server --core_type=trtllm_endpoint --accuracy_target=.999 --test_mode=PerformanceOnly --model_path=$LLAMA2_70B_MODEL_PATH"

LOG="$MLPERF_SCRATCH_PATH/logs/b300-1gpu-llama70b-999/server_performance.log"
mkdir -p "$(dirname "$LOG")"

make run_harness RUN_ARGS="$RUN_ARGS" SYSTEM_NAME="$SYSTEM_NAME" 2>&1 | tee "$LOG"
```

### 2. Run AccuracyOnly

For AccuracyOnly, stop the previous endpoint first and start a fresh Offline endpoint:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1
export CUDA_VISIBLE_DEVICES=0
export LLAMA2_70B_MODEL_PATH="$MLPERF_SCRATCH_PATH/models/llama2-70b-chat-hf-torch-fp4-v6.0"

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --core_type=trtllm_endpoint --accuracy_target=.999 --test_mode=AccuracyOnly --model_path=$LLAMA2_70B_MODEL_PATH"

make run_llm_server RUN_ARGS="$RUN_ARGS" SYSTEM_NAME="$SYSTEM_NAME"
```

After the Offline endpoint is ready, run the Offline accuracy harness in another terminal:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1
export CUDA_VISIBLE_DEVICES=0
export LLAMA2_70B_MODEL_PATH="$MLPERF_SCRATCH_PATH/models/llama2-70b-chat-hf-torch-fp4-v6.0"

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --core_type=trtllm_endpoint --accuracy_target=.999 --test_mode=AccuracyOnly --model_path=$LLAMA2_70B_MODEL_PATH"

LOG="$MLPERF_SCRATCH_PATH/logs/b300-1gpu-llama70b-999/offline_accuracy.log"
mkdir -p "$(dirname "$LOG")"

make run_harness RUN_ARGS="$RUN_ARGS" SYSTEM_NAME="$SYSTEM_NAME" 2>&1 | tee "$LOG"
```

Check:

```bash
grep -E "qsl_rng_seed|sample_index_rng_seed|schedule_rng_seed" "$LOG" | tail -n 10
grep -Ei "accuracy|rouge|tokens per sample|gen_len|pass|fail|result" "$LOG" | tail -n 120
```

To run Server AccuracyOnly, stop the Offline endpoint first and start a fresh Server endpoint:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1
export CUDA_VISIBLE_DEVICES=0
export LLAMA2_70B_MODEL_PATH="$MLPERF_SCRATCH_PATH/models/llama2-70b-chat-hf-torch-fp4-v6.0"

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=llama2-70b --scenarios=Server --core_type=trtllm_endpoint --accuracy_target=.999 --test_mode=AccuracyOnly --model_path=$LLAMA2_70B_MODEL_PATH"

make run_llm_server RUN_ARGS="$RUN_ARGS" SYSTEM_NAME="$SYSTEM_NAME"
```

After the Server endpoint is ready, run the Server accuracy harness in another terminal:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1
export CUDA_VISIBLE_DEVICES=0
export LLAMA2_70B_MODEL_PATH="$MLPERF_SCRATCH_PATH/models/llama2-70b-chat-hf-torch-fp4-v6.0"

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=llama2-70b --scenarios=Server --core_type=trtllm_endpoint --accuracy_target=.999 --test_mode=AccuracyOnly --model_path=$LLAMA2_70B_MODEL_PATH"

LOG="$MLPERF_SCRATCH_PATH/logs/b300-1gpu-llama70b-999/server_accuracy.log"
mkdir -p "$(dirname "$LOG")"

make run_harness RUN_ARGS="$RUN_ARGS" SYSTEM_NAME="$SYSTEM_NAME" 2>&1 | tee "$LOG"
```

### 3. Run Required Audit Tests

Llama2-70B-99.9 requires TEST06. Run TEST06 with the same `RUN_ARGS` used for result logs, but without `--test_mode=AccuracyOnly`.

Start a fresh Offline endpoint:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1
export CUDA_VISIBLE_DEVICES=0
export LLAMA2_70B_MODEL_PATH="$MLPERF_SCRATCH_PATH/models/llama2-70b-chat-hf-torch-fp4-v6.0"

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --core_type=trtllm_endpoint --accuracy_target=.999 --model_path=$LLAMA2_70B_MODEL_PATH"

make run_llm_server RUN_ARGS="$RUN_ARGS" SYSTEM_NAME="$SYSTEM_NAME"
```

After the Offline endpoint is ready, run TEST06 in another terminal:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1
export CUDA_VISIBLE_DEVICES=0
export LLAMA2_70B_MODEL_PATH="$MLPERF_SCRATCH_PATH/models/llama2-70b-chat-hf-torch-fp4-v6.0"

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --core_type=trtllm_endpoint --accuracy_target=.999 --model_path=$LLAMA2_70B_MODEL_PATH"

make run_audit_test06 RUN_ARGS="$RUN_ARGS" SYSTEM_NAME="$SYSTEM_NAME"
```

For Server TEST06, stop the Offline endpoint first and start a fresh Server endpoint:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1
export CUDA_VISIBLE_DEVICES=0
export LLAMA2_70B_MODEL_PATH="$MLPERF_SCRATCH_PATH/models/llama2-70b-chat-hf-torch-fp4-v6.0"

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=llama2-70b --scenarios=Server --core_type=trtllm_endpoint --accuracy_target=.999 --model_path=$LLAMA2_70B_MODEL_PATH"

make run_llm_server RUN_ARGS="$RUN_ARGS" SYSTEM_NAME="$SYSTEM_NAME"
```

After the Server endpoint is ready, run TEST06 in another terminal:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx1
export CUDA_VISIBLE_DEVICES=0
export LLAMA2_70B_MODEL_PATH="$MLPERF_SCRATCH_PATH/models/llama2-70b-chat-hf-torch-fp4-v6.0"

git config --global --add safe.directory /work/3rdparty/trtllm

export RUN_ARGS="--benchmarks=llama2-70b --scenarios=Server --core_type=trtllm_endpoint --accuracy_target=.999 --model_path=$LLAMA2_70B_MODEL_PATH"

make run_audit_test06 RUN_ARGS="$RUN_ARGS" SYSTEM_NAME="$SYSTEM_NAME"
```
