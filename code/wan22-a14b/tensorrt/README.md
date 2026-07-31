# NVIDIA B300 x8 WAN2.2-A14B Test Execution Runbook


## 1. Model Download

WAN2.2-A14B uses the Hugging Face model:

```text
Wan-AI/Wan2.2-T2V-A14B-Diffusers
```

Recommended download path:

```text
$MLPERF_SCRATCH_PATH/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers
```

```bash
set -euxo pipefail

export MLPERF_SCRATCH_PATH=/path/to

python3 -m pip install -U "huggingface_hub[cli]"

# If authentication is required, run:
# huggingface-cli login

huggingface-cli download Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --local-dir "$MLPERF_SCRATCH_PATH/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers" \
  --local-dir-use-symlinks False

du -sh "$MLPERF_SCRATCH_PATH/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers"
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
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx8

make link_dirs
mkdir -p $MLPERF_SCRATCH_PATH/logs
ln -sfn $MLPERF_SCRATCH_PATH/logs build/logs
```

## 3. Data Download

WAN2.2-A14B prompts and fixed latent come from the `mlc-inference` sub-repository:

```text
3rdparty/mlc-inference/text_to_video/wan-2.2-t2v-a14b/data/vbench_prompts.txt
3rdparty/mlc-inference/text_to_video/wan-2.2-t2v-a14b/data/fixed_latent.pt
```

After Section 2 is complete, these two files should already be at the paths above.

## 4. Data Processing

Run the commands in this section inside the NVIDIA Docker container. If you have not entered the container yet, complete Section 2 to pull the Docker image and start the container.

Run the preprocessing script:

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx8

make link_dirs

sh /work/code/wan22-a14b/tensorrt/preprocess_data.sh

ls -lh "$MLPERF_SCRATCH_PATH/preprocessed_data/wan22-a14b/prompts.txt"
ls -lh "$MLPERF_SCRATCH_PATH/preprocessed_data/wan22-a14b/fixed_latent.pt"
```

Install `visual_gen`:

```bash
cd /work/3rdparty/trtllm/tensorrt_llm/visual_gen
pip install -e . --no-build-isolation

export PYTHONPATH=/work/3rdparty/trtllm/tensorrt_llm/visual_gen:$PYTHONPATH
export PIP_NO_CACHE_DIR=1

cd /work
```

## 5. Run Tests

### 1. Run PerformanceOnly

Use the following for WAN2.2-A14B on B300 x8:

```text
SYSTEM_NAME=B300-SXM-270GBx8
```

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx8
export PIP_NO_CACHE_DIR=1
export PYTHONPATH=/work/3rdparty/trtllm/tensorrt_llm/visual_gen:$PYTHONPATH

export RUN_ARGS="--benchmarks=wan22-a14b --scenarios=Offline --test_mode=PerformanceOnly --wan22_model_path=$MLPERF_SCRATCH_PATH/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers"

make run_harness RUN_ARGS="$RUN_ARGS"
```

### 2. Run AccuracyOnly

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx8
export PIP_NO_CACHE_DIR=1
export PYTHONPATH=/work/3rdparty/trtllm/tensorrt_llm/visual_gen:$PYTHONPATH

export RUN_ARGS="--benchmarks=wan22-a14b --scenarios=Offline --test_mode=AccuracyOnly --wan22_model_path=$MLPERF_SCRATCH_PATH/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers"

make run_harness RUN_ARGS="$RUN_ARGS"
```

### 3. Run Required Audit Tests

WAN2.2-A14B requires TEST01, run through `run_audit_harness`.

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx8
export PIP_NO_CACHE_DIR=1
export PYTHONPATH=/work/3rdparty/trtllm/tensorrt_llm/visual_gen:$PYTHONPATH

export RUN_ARGS="--benchmarks=wan22-a14b --scenarios=Offline --wan22_model_path=$MLPERF_SCRATCH_PATH/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers"

make run_audit_harness RUN_ARGS="$RUN_ARGS"
```

Optional smoke test:

```bash
make run_harness RUN_ARGS="--benchmarks=wan22-a14b --scenarios=Offline --test_mode=AccuracyOnly --wan22_total_sample_count=20 --accuracy_sample_count_override=20 --wan22_model_path=$MLPERF_SCRATCH_PATH/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers"

make run_harness RUN_ARGS="--benchmarks=wan22-a14b --scenarios=Offline --test_mode=PerformanceOnly --wan22_total_sample_count=5 --performance_sample_count_override=5 --min_query_count=5 --max_query_count=5 --min_duration=0 --verbose --wan22_model_path=$MLPERF_SCRATCH_PATH/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers"
```
