#!/bin/bash
#SBATCH -A coreai_mlperf_inference
#SBATCH --output=outputs/%j-stdout.txt
#SBATCH --error=outputs/%j-stderr.txt
#SBATCH --partition=36x2-a01r
#SBATCH --time=1:30:00
#SBATCH -N1
#SBATCH --ntasks-per-node=4

CONTAINER_IMAGE=${CONTAINER_IMAGE:-"gitlab-master.nvidia.com:5005/mlpinf/mlperf-inference/mlperf-inf-mm-q3vl-nv:arm64_cuda13.0.1_CentML_dynamo-mlperf-inf-mm-q3vl-v6.0_CentML_vllm-mlperf-inf-mm-q3vl-v6.0"}
HF_CACHE_HOST_DIR=${CACHE_HOST_DIR:-"/lustre/fsw/coreai_mlperf_inference/${USER}/.cache/huggingface"}
OUTPUT_HOST_DIR=${OUTPUT_HOST_DIR:-"$(pwd -P)/outputs/${SLURM_JOB_ID}/"}
MODE=${MODE:-"performance_only"}
TARGET_QPS=${TARGET_QPS:-38}
HF_TOKEN=${HF_TOKEN:-""}
WANDB_ENTITY=${WANDB_ENTITY:-"nvidia"}
WANDB_PROJECT=${WANDB_PROJECT:-"mlperf-inf-mm-q3vl-nv-v6.0"}
WANDB_NAME=${WANDB_NAME:-"ptyche-${SLURM_JOB_ID}"}
WANDB_API_KEY=${WANDB_API_KEY:-""}

mkdir -p "${OUTPUT_HOST_DIR}"

mounts="${HF_CACHE_HOST_DIR}:/root/.cache/huggingface,${OUTPUT_HOST_DIR}:/output/"

if [ -n "${HF_TOKEN}" ]; then
    hf_token_flag="--dynamo.model.token=${HF_TOKEN}"
else
    hf_token_flag=""
fi

if [ -n "${WANDB_ENTITY}" ] && [ -n "${WANDB_PROJECT}" ] && [ -n "${WANDB_API_KEY}" ]; then
    wandb_flags="--wandb_config.entity=${WANDB_ENTITY} --wandb_config.project=${WANDB_PROJECT} --wandb_config.api_key=${WANDB_API_KEY}"
    if [ -n "${WANDB_NAME}" ]; then
        wandb_flags="${wandb_flags} --wandb_config.name=${WANDB_NAME}"
    fi
else
    wandb_flags=""
fi

export DYN_LOG=debug  # Change from 'info' to 'debug'
export VLLM_LOGGING_LEVEL=DEBUG

export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_USE_FLASHINFER_MOE_FP4=1
export VLLM_FLASHINFER_MOE_BACKEND=latency
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=$((6 * 256 * 1024 * 1024)) 
export TOKIO_WORKER_THREADS=32
export OMP_NUM_THREADS=64

export VLLM_USE_TRITON_POS_EMBED=1
export VLLM_MM_ENCODER_FP8_ATTN=1

echo "Starting job at: $(date)"
echo $SLURM_GPUS_ON_NODE

srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${mounts}" \
    --no-container-mount-home \
    --mpi=pmix \
    bash -c " \
        mlperf-inf-mm-q3vl benchmark nv mpi-dynamo-vllm \
        --use-http-client \
        --max-concurrency=2048 \
        --dynamo.vllm.enable_numa_binding=true \
        --dynamo.frontend.enable_numa_binding=true \
        --dynamo.model.repo_id=nvidia/Qwen3-VL-235B-A22B-Instruct-NVFP4-MLPerf-Inference-Closed-V6.0 \
        --dynamo.model.revision=main \
        ${hf_token_flag} \
        ${wandb_flags} \
        --settings.test.scenario server \
        --settings.test.server_target_qps ${TARGET_QPS} \
        --dynamo.num_warmup_requests_per_vllm_instance 100 \
        --settings.test.mode ${MODE} \
        --settings.test.qsl_rng_seed 2465351861681999779 \
        --settings.test.sample_index_rng_seed 14276810075590677512 \
        --settings.test.schedule_rng_seed 3936089224930324775 \
        --settings.logging.log_output.outdir /output/ \
        --dynamo.vllm.cli=--tensor-parallel-size=1 \
        --dynamo.vllm.cli=--pipeline-parallel-size=1 \
        --dynamo.vllm.cli=--data-parallel-size=1 \
        --dynamo.vllm.cli=--enable-expert-parallel \
        --dynamo.vllm.cli=--all2all-backend=flashinfer_all2allv \
        --dynamo.vllm.cli=--async-scheduling \
        --dynamo.vllm.cli=--max-model-len=32768 \
        --dynamo.vllm.cli=--max-num-seqs=1024 \
        --dynamo.vllm.cli=--mm-encoder-attn-backend=FLASHINFER \
        --dynamo.vllm.cli=--max-num-batched-tokens=4864 \
        --dynamo.vllm.cli=--scheduling-policy=sjf \
        --dynamo.vllm.cli=--compilation-config='{
            \"max_cudagraph_capture_size\": 4864,
            \"cudagraph_capture_sizes\": [
                1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
                136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248,
                256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480,
                496,
                512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024, 1056, 1088, 1120, 1152, 1184, 1216, 1248, 1280, 1312, 1344, 1376, 1408, 1440, 1472, 1504, 1536, 1568, 1600, 1632, 1664, 1696, 1728, 1760, 1792, 1824, 1856, 1888, 1920, 1952, 1984, 2016, 2048, 2080, 2112, 2144, 2176, 2208, 2240, 2272, 2304, 2336, 2368, 2400, 2432, 2464, 2496, 2528, 2560, 2592, 2624, 2656, 2688, 2720, 2752, 2784, 2816, 2848, 2880, 2912, 2944, 2976, 3008, 3040, 3072, 3104, 3136, 3168, 3200, 3232, 3264, 3296, 3328, 3360, 3392, 3424, 3456, 3488, 3520, 3552, 3584, 3616, 3648, 3680, 3712, 3744, 3776, 3808, 3840, 3872, 3904, 3936, 3968, 4000, 4032, 4064, 4096, 4128, 4160, 4192, 4224, 4256, 4288, 4320, 4352, 4384, 4416, 4448, 4480, 4512, 4544, 4576, 4608, 4640, 4672, 4704, 4736, 4768, 4800, 4832,
                4864
            ]
        }' \
        --dynamo.vllm.cli=--override-generation-config='{\"max_new_tokens\": 150}' \
        --dynamo.vllm.cli=--limit-mm-per-prompt.video=0 \
        --dynamo.vllm.cli=--mm-processor-cache-gb=0 \
        --dynamo.vllm.cli=--no-enable-prefix-caching \
        --dynamo.vllm.cli=--enable-multimodal \
        --dynamo.vllm.cli=--connector=none \
        --dynamo.vllm.cli=--kv-events-config='{\"publisher\":\"null\"}' \
        --dynamo.vllm.cli=--distributed-executor-backend=mp; \
        EXIT_CODE=\$?; \
        if [ \$SLURM_LOCALID -eq 0 ]; then \
            if [ \$EXIT_CODE -eq 0 ]; then \
                if [ \"${MODE}\" == \"accuracy_only\" ]; then \
                    mlperf-inf-mm-q3vl evaluate --filename=/output/mlperf_log_accuracy.json; \
                    mv accuracy.txt /output/accuracy.txt; \
                fi; \
            else \
                echo \"Previous numactl command failed with exit code \$EXIT_CODE\"; \
                exit \$EXIT_CODE; \
            fi; \
        fi; \
    "