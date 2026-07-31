# Qwen3-VL-235B-A22B

This directory contains the source code for NVIDIA's submission towards the
[vision-language model (VLM) benchmark](https://github.com/mlcommons/inference/tree/master/multimodal/qwen3-vl)
in the MLPerf Inference Benchmark Suite, starting from the v6.0 round.

## Steps to prepare a submission

This repo integrates a highly automated workflow to get benchmarking results through harness (i.e., [Mitten](https://github.com/NVIDIA/mitten)).

### 1.Docker environment



Building the image with all dependencies for benchmarking and submission can be done *** **Via docker only** *** in this version.

**Before** running the build command or launching from a prebuilt image, make sure all submodules are initialized by `git submodule update --init --recursive`.

#### - Build the image from scratch

```bash
cd /path/to/mlperf-inference/closed/NVIDIA \
    && BUILDX_BUILDER=default make -f Makefile.docker prebuild_q3vl
```
This command builds a vLLM-based container image and all mlperf-inference related tools, and then launches the result image with the correct mounting for you.

#### - Use pre-built docker image

We have pushed fully functional images to <br>
arm64 based image: registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/v6.03-rc2-feb10-q3vl-aarch64 <br>
amd64 based iamge: registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/v6.03-rc2-feb10-q3vl-amd64 <br>

At launch time, make sure to mount `/path/to/mlperf-inference/closed/NVIDIA` to `/work` for the following steps to run correctly. 

### 2. Collect results through harness

#### - Running with slurm system

```bash
srun -N1 --mpi=pmix  --ntasks-per-node=${NUM_GPUS}  --container-image=registry.gitlab.com/nvidia/mlperf-inference-partner/nv-mlpinf-partner/v6.03-rc2-feb10-q3vl-aarch64 \
--container-mounts mlperf-inference/closed/NVIDIA:/work \
--pty bash -c 'make run_harness RUN_ARGS="--benchmarks=qwen3-vl-235b-a22b \
            --scenarios=${SCENARIO} \
            --test_mode=${MODE}"'
```
***Note***: 

You should set `${NUM_GPUS}` based on what system you are benchmarking:
- GB(2|3)00: `export NUM_GPUS=4`

You should set `${SCENARIO}` based on whether you want to run the "Offline" or the
"Server" scenario:
- Offline: `export SCENARIO=Offline`
- Server: `export SCENARIO=Server`

You should set `${MODE}` based on whether you want to run the "Performance Only" or the
"Accuracy Only" mode:
- Performance Only: `export MODE=PerformanceOnly`
- Accuracy Only: `export MODE=AccuracyOnly`


#### - Running with interactive docker

Simply do the following inside your docker container, and multi-processing will be handled by MPI automatically
```bash
make run_harness RUN_ARGS="--benchmarks=qwen3-vl-235b-a22b \
            --scenarios=${SCENARIO} \
            --test_mode=${MODE}"
```

For each system, PerformanceOnly scenario will collect the performance results and AccuracyOnly will provide the accuracy check for corresponding system and mode.  


### 3. Stage your results for a submission

Running with harness will generate all logs and results for a submission. Before proceeding, make sure you only have the results
that you want to submit in `/work/build/logs`, then follow the steps below to prepare your results in the submission ready format:

```bash
make stage_results # This will copy your logs from closed/nvidia/build/logs/xxx to a staging path
make stage_compliance # Stage compliance results
make truncate_results # It truncates the result logs 
make check_submission_staging # Make sure the results collected are all valid and each performance result should have an accuracy log under the same system & scnario
make purge_artifacts_repo && make pull_artifacts_repo # Clean the existing results repo and pull the latest main branch
make push_submission_staging # Push results to a remote branch
```



## Developer guide

The following sections are for developer experiments, the image will not include MLPerf submission related requirements, and scripts below will not stage your results in submission-ready format. 

### Key software versions

- CUDA: 13.0.1
- vLLM: https://github.com/CentML/vllm/tree/mlperf-inf-mm-q3vl-v6.0
- Dynamo: https://github.com/CentML/dynamo/tree/mlperf-inf-mm-q3vl-v6.0
- FlashInfer: https://github.com/CentML/flashinfer/tree/mlperf-inf-mm-q3vl-v6.0

### Build the container image

You can leverage [scripts/build_image.sh](scripts/build_image.sh) to build a container
image end-to-end for running this benchmark. At the
[closed/NVIDIA/code/qwen3-vl-235b-a22b/vllm](closed/NVIDIA/code/qwen3-vl-235b-a22b/vllm)
directory (i.e., where this `README.md` is), run the following command:

```bash
bash scripts/build_image.sh
```

If you would like to run the benchmark on a `amd64` (i.e., x86) system, you would need
to build the image on an `amd64` (i.e., x86) system. Conversely, you would need to build
the image on an `arm64` (i.e., `aarch64`) system for running the benchmark on a `arm64`
(i.e., `aarch64`) system.

Building the vLLM base image can be intensive on CPU and host memory resources. We
recommend to build the image on a machine with at least 72 CPU threads and 574 GB of
host memory.

### Build the container image for the latest release

The current latest release is version RC2. To build the container image for RC2, please
run the following command:

```bash
bash scripts/build_image.sh --vllm-revision mlperf-inf-mm-q3vl-v6.0-rc2 --vllm-force-rebuild --dynamo-revision mlperf-inf-mm-q3vl-v6.0-rc2 --result-force-rebuild
```

### NVFP4 Quantization with LLM-COMPRESSOR

> [!NOTE]
> We calibrated an NVFP4 model checkpoint using the following steps and uploaded it to 
> [nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4-MLPerf-Inference-Closed-V6.0](https://huggingface.co/nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4-MLPerf-Inference-Closed-V6.0).
> Please use this provided checkpoint if you don't have a specific need to calibrate
> a checkpoint on your own. 
> All benchmarking scripts use 
> [nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4-MLPerf-Inference-Closed-V6.0](https://huggingface.co/nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4-MLPerf-Inference-Closed-V6.0)
> by default.

[LLM-Compressor](https://github.com/vllm-project/llm-compressor) provides an [example script](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w4a4_fp4/qwen3_vl_moe_w4a4_fp4.py) for quantization of Qwen3-VL-235B-A22B-Instruct.
We replace the default calibration dataset with Shopify dataset by the following steps to get the checkpoint we use for the submission results. 

1. Follow the readme to install the stable version of llmcompressor `pip install llmcompressor` (and possibly other dependencies like torchvision, openai etc.) in a pytorch environment, note that the 
quantization script **Does Not** require to run in the vllm base or submission container.

2. Implement a mapping function to process the input dataset to a valid format of vllm message. (For shopify dataset, you can find the reference example [here](https://github.com/mlcommons/inference/blob/master/multimodal/qwen3-vl/src/mlperf_inf_mm_q3vl/task.py))

3. Apply the preprocess function in original script with your customized transformation onto your dataset.

Please refer to [qwen3_vl_moe_w4a4_fp4_with_customized_dataset.py](scripts/quantization/qwen3_vl_moe_w4a4_fp4_with_customized_dataset.py) as an example quantization script.

**Tip** 
Use following envs to prevent the script from hanging on some circumstances.
```bash
export OPENBLAS_NUM_THREADS=16
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
```


### Launch a benchmarking run

#### Run the benchmark locally

Start the container in the interactive mode via:

```bash
docker run --runtime=nvidia --gpus all --rm -it --ipc=host --cap-add=SYS_NICE \
  -v $(pwd -P):/mlperf-inf-mm-q3vl-nv \
  -v ${HF_HOME}:/root/.cache/huggingface -w /mlperf-inf-mm-q3vl-nv \
  gitlab-master.nvidia.com:5005/mlpinf/mlperf-inference/mlperf-inf-mm-q3vl-nv:${ARCH}64_cuda13.0.1_CentML_dynamo-mlperf-inf-mm-q3vl-v6.0-rc2_CentML_vllm-mlperf-inf-mm-q3vl-v6.0-rc2
```

You should set `${ARCH}` based on what system you are benchmarking:
- GB(2|3)00: `export ARCH=arm`
- H200: `export ARCH=amd`

You should set `${HF_HOME}` to where you want to keep the model checkpoint and the
dataset, for example, `export HF_HOME=/tmp/.cache/huggingface`.

Inside the container, you can download the model checkpoint and the dataset via:

```bash
hf download nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4-MLPerf-Inference-Closed-V6.0
hf download Shopify/product-catalogue --repo-type dataset
```

Please consult HuggingFace's documentations on how to use the `hf` CLI to
[download the model](https://huggingface.co/docs/hub/en/models-downloading) and
[download the dataset](https://huggingface.co/docs/hub/en/datasets-downloading).

Then, inside the container, you can launch the benchmark via:

```bash
bash scripts/benchmarks/local/${SYSTEM}/${SCENARIO}.sh --mode ${MODE}
```

You should set `${SYSTEM}` based on what system you are benchmarking:
- GB300: `export SYSTEM=gb300-nvl4` 
- GB200: `export SYSTEM=gb200-nvl4`
- H200: `export SYSTEM=8xh200`

You should set `${SCENARIO}` based on whether you want to run the "Offline" or the
"Server" scenario:
- Offline: `export SCENARIO=offline`
- Server: `export SCENARIO=server`

You should set `${MODE}` based on whether you want to run the "Performance Only" or the
"Accuracy Only" mode:
- Performance Only: `export MODE=performance_only`
- Accuracy Only: `export MODE=accuracy_only`

If you run the "Accuracy Only" mode, accuracy evaluation will take place automatically.
Once finished, you will see a `accuracy.txt` file in your log directory (by default, 
`./output/`) which contains the accuracy score that's expected by the submission
checker. 

> [!NOTE]
> I'm explaining this as if you know nothing about Bash or shell scripting. If you are
> fluent in Bash, by this point you should be able to tell that you can just find the
> container image and the script corresponding to your use case and launch it directly.
> You can pass in `--help` to see what CLI flags each script can takes.

#### Run the benchmark in a Slurm cluster

[scripts/benchmarks/slurm](scripts/benchmarks/slurm) contains examples of how you would
launch the benchmark (per each type of system) in a Slurm cluster:

```bash
cd scripts/benchmarks/slurm/${SYSTEM}/
sbatch ${SCENARIO}.sh
```

You should set `${SYSTEM}` based on what system you are benchmarking:
- GB300: `export SYSTEM=gb300-nvl4` 
- GB200: `export SYSTEM=gb200-nvl4`
- H200: `export SYSTEM=1x8xh200`

You should set `${SCENARIO}` based on whether you want to run the "Offline" or the
"Server" scenario:
- Offline: `export SCENARIO=offline`
- Server: `export SCENARIO=server`

You should set `${MODE}` based on whether you want to run the "Performance Only" or the
"Accuracy Only" mode:
- Performance Only: `export MODE=performance_only`
- Accuracy Only: `export MODE=accuracy_only`

Different organization usually has significantly different Slurm setup. You should
contact and work with your cluster admin on how to adapt those examples for the Slurm
cluster you have access to.

### Profile with Nsight Systems

To collect Nsight Systems traces on vLLM for this benchmark, you can leverage the
`mlperf-inf-mm-q3vl benchmark nv vllm-profiler` command:

1. Wrap the command with `nsys profile --options`.
2. Set `--vllm.profile True` as a CLI flag for `mlperf-inf-mm-q3vl benchmark nv vllm-profiler`
   to ask vLLM to start/stop the profiler.
3. Pass vLLM supported profiling flags to control the capture range. Check
   [vLLM Profile Documentation](https://docs.vllm.ai/en/stable/contributing/profiling/#openai-server) for more information.

#### Example

If you want to profile a benchmark run where:
- It runs the performance only mode in the server scenario where the target QPS is 10
  requests per second.
- The maximum number of batched tokens is 32768 (therefore, you want to capture a CUDA
  graph that can support up to 32768 tokens).
- All 4 GPUs on a single node are configured in a fully tensor parallel fashion.
- Prefix caching across requests are disabled (this is required by the
  [MLPerf Inference rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#94-llm-benchmarks)).
- The `nsys` trace is captured starting from the 1000-th to the 1100-th `EngineCore`
  iterations.
The commands (inside the container) would look like the following:

```bash
# Set `HF_TOKEN` to the HuggingFace access token that you would like to use to access
# your model checkpoint (in this case, `nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4-MLPerf-Inference-Closed-V6.0`).
export HF_TOKEN=...

export VLLM_NVTX_SCOPES_FOR_PROFILING=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_USE_FLASHINFER_MOE_FP4=1
export VLLM_FLASHINFER_MOE_BACKEND=latency
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=$((6 * 256 * 1024 * 1024))

nsys profile \
    --wait primary \
    -f true \
    -o test_output \
    --gpu-metrics-devices=all \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    --capture-range-end repeat \
    --trace cuda,nvtx \
    mlperf-inf-mm-q3vl benchmark nv vllm-profiler \
    --settings.test.scenario server \
    --settings.test.mode performance_only \
    --settings.test.server_target_qps 10 \
    --vllm.profile True \
    --vllm.model.repo_id=nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4-MLPerf-Inference-Closed-V6.0 \
    --vllm.model.revision=main \
    --vllm.model.token ${HF_TOKEN} \
    --vllm.cli=--async-scheduling \
    --vllm.cli=--max-model-len=32768 \
    --vllm.cli=--max-num-seqs=1024 \
    --vllm.cli=--max-num-batched-tokens=32768 \
    --vllm.cli=--compilation-config='{
        "max_cudagraph_capture_size": 32768,
        "cudagraph_capture_sizes": [
            1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
            136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248,
            256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480,
            496, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 16384, 32768
        ]
    }' \
    --vllm.cli=--limit-mm-per-prompt.video=0 \
    --vllm.cli=--tensor-parallel-size=4 \
    --vllm.cli=--no-enable-prefix-caching \
    --vllm.cli=--enable-layerwise-nvtx-tracing \
    --vllm.cli=--enable-logging-iteration-details \
    --vllm.cli=--profiler-config.profiler=cuda \
    --vllm.cli="--profiler-config.delay_iterations=1000" \
    --vllm.cli="--profiler-config.max_iterations=100"
```

