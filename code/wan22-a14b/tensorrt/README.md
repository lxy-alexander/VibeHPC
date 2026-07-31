# NVIDIA B300 x8 WAN2.2-A14B  测试执行文档


## 1. 模型下载

WAN2.2-A14B 使用 Hugging Face 模型：

```text
Wan-AI/Wan2.2-T2V-A14B-Diffusers
```

建议下载到：

```text
$MLPERF_SCRATCH_PATH/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers
```

```bash
set -euxo pipefail

export MLPERF_SCRATCH_PATH=/path/to

python3 -m pip install -U "huggingface_hub[cli]"

# 如需认证，先执行：
# huggingface-cli login

huggingface-cli download Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --local-dir "$MLPERF_SCRATCH_PATH/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers" \
  --local-dir-use-symlinks False

du -sh "$MLPERF_SCRATCH_PATH/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers"
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
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx8

make link_dirs
mkdir -p $MLPERF_SCRATCH_PATH/logs
ln -sfn $MLPERF_SCRATCH_PATH/logs build/logs
```

## 3. 下载数据

WAN2.2-A14B 的 prompt 和 fixed latent 来自 `mlc-inference` 子仓库：

```text
3rdparty/mlc-inference/text_to_video/wan-2.2-t2v-a14b/data/vbench_prompts.txt
3rdparty/mlc-inference/text_to_video/wan-2.2-t2v-a14b/data/fixed_latent.pt
```

准备第三方库：

```bash
cd /work
mkdir -p 3rdparty

if [ ! -d 3rdparty/mlc-inference ]; then
  git clone --recurse-submodules https://github.com/mlcommons/inference.git 3rdparty/mlc-inference
fi

if [ ! -d 3rdparty/trtllm ]; then
  git clone --recurse-submodules https://github.com/NVIDIA/TensorRT-LLM.git 3rdparty/trtllm
fi

if [ ! -d 3rdparty/mitten ]; then
  git clone --recurse-submodules https://github.com/NVIDIA/mitten.git 3rdparty/mitten
fi

cd /work/3rdparty/trtllm
git checkout feat/visual_gen
```

## 4. 数据处理

本节命令需要在 NVIDIA Docker 容器内执行。若还没有进入容器，先完成第 2 节的 Docker 镜像下载和容器启动。

执行预处理脚本：

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx8

make link_dirs

sh /work/code/wan22-a14b/tensorrt/preprocess_data.sh

ls -lh "$MLPERF_SCRATCH_PATH/preprocessed_data/wan22-a14b/prompts.txt"
ls -lh "$MLPERF_SCRATCH_PATH/preprocessed_data/wan22-a14b/fixed_latent.pt"
```

安装 visual_gen：

```bash
cd /work/3rdparty/trtllm/tensorrt_llm/visual_gen
pip install -e . --no-build-isolation

export PYTHONPATH=/work/3rdparty/trtllm/tensorrt_llm/visual_gen:$PYTHONPATH
export PIP_NO_CACHE_DIR=1

cd /work
```

## 5. 跑测试

### 1）. 跑 PerformanceOnly

WAN2.2-A14B B300 x8 使用：

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

### 2）. 跑 AccuracyOnly

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx8
export PIP_NO_CACHE_DIR=1
export PYTHONPATH=/work/3rdparty/trtllm/tensorrt_llm/visual_gen:$PYTHONPATH

export RUN_ARGS="--benchmarks=wan22-a14b --scenarios=Offline --test_mode=AccuracyOnly --wan22_model_path=$MLPERF_SCRATCH_PATH/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers"

make run_harness RUN_ARGS="$RUN_ARGS"
```

### 3）. 跑涉及的 Audit

WAN2.2-A14B 涉及 TEST01，通过 `run_audit_harness` 运行。

```bash
cd /work
export MLPERF_SCRATCH_PATH=/path/to
export SYSTEM_NAME=B300-SXM-270GBx8
export PIP_NO_CACHE_DIR=1
export PYTHONPATH=/work/3rdparty/trtllm/tensorrt_llm/visual_gen:$PYTHONPATH

export RUN_ARGS="--benchmarks=wan22-a14b --scenarios=Offline --wan22_model_path=$MLPERF_SCRATCH_PATH/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers"

make run_audit_harness RUN_ARGS="$RUN_ARGS"
```

可选 smoke test：

```bash
make run_harness RUN_ARGS="--benchmarks=wan22-a14b --scenarios=Offline --test_mode=AccuracyOnly --wan22_total_sample_count=20 --accuracy_sample_count_override=20 --wan22_model_path=$MLPERF_SCRATCH_PATH/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers"

make run_harness RUN_ARGS="--benchmarks=wan22-a14b --scenarios=Offline --test_mode=PerformanceOnly --wan22_total_sample_count=5 --performance_sample_count_override=5 --min_query_count=5 --max_query_count=5 --min_duration=0 --verbose --wan22_model_path=$MLPERF_SCRATCH_PATH/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers"
```
