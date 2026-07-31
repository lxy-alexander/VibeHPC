#!/bin/bash
set -e
set -u

# File descriptor limit is propagated via sbatch --propagate=RLIMIT_NOFILE
# No need to set it here

# Start a CTX or GEN worker for llama2-70b disaggregated inference

role=$1                    # CTX or GEN
instance_id=$2             # Worker instance ID
model_path=$3              # Path to model
port=$4                    # Port for this worker
work_dir=$5                # Work directory
exp_name=${6:-}            # Experiment name
nsys=${7:-0}               # Enable nsys profiling (0 or 1)
nsys_iters=${8:-5000-5100} # Nsys iteration range
trtllm_install_path=${9:-} # Optional: custom TRT-LLM install path

# Debug logging disabled - uncomment if needed for troubleshooting
# script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# if [ -f "${script_dir}/dump_env_debug.sh" ]; then
#     source "${script_dir}/dump_env_debug.sh" "${work_dir}" "${role}_${instance_id}"
# fi

# Install custom TRT-LLM if path provided
if [ -n "$trtllm_install_path" ]; then
    echo "Installing TRT-LLM from $trtllm_install_path"
    pip install -e $trtllm_install_path
fi

unset UCX_TLS
unset UCX_NET_DEVICES  # Allow UCX to use all available devices
export UCX_RNDV_SCHEME=get_zcopy  # Better performance on GB200
export TLLM_LOG_LEVEL=INFO
export TRTLLM_DISABLE_NIXL=1  # Disable NIXL on aarch64 (not available)

# NUMA binding for GB200
numa_bind_cmd="numactl -m 0,1"

# Select config file based on role
if [ "${role}" = "CTX" ]; then
    config_file=${work_dir}/ctx_config.yaml
elif [ "${role}" = "GEN" ]; then
    config_file=${work_dir}/gen_config.yaml
else
    echo "ERROR: Invalid role: ${role}. Must be CTX or GEN."
    exit 1
fi

# Save hostname:port to file (only on first node)
if [ "${SLURM_NODEID}" = "0" ]; then
    mkdir -p ${work_dir}/hostnames/
    echo $(hostname):${port} > ${work_dir}/hostnames/${role}_${instance_id}.txt
fi

# Start worker (with or without nsys profiling)
if (( nsys == 0 )); then
    trtllm-llmapi-launch ${numa_bind_cmd} \
        trtllm-serve ${model_path} \
            --host $(hostname) \
            --port ${port} \
            --extra_llm_api_options ${config_file}
else
    nsys_file_prefix=${work_dir}/${exp_name}_${role}_${instance_id}_${SLURM_NODEID}_${SLURM_PROCID}
    
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
    export TLLM_PROFILE_START_STOP=${nsys_iters}
    
    nsys_prefix="nsys profile \
        -e NSYS_MPI_STORE_TEAMS_PER_RANK=1 \
        --trace=cuda,nvtx \
        --capture-range=cudaProfilerApi \
        --cuda-graph-trace=node \
        --capture-range-end=stop \
        --force-overwrite=true \
        --gpu-metrics-devices=cuda-visible \
        --output=${nsys_file_prefix}_iters_${TLLM_PROFILE_START_STOP}.nsys-rep"
    
    ${nsys_prefix} trtllm-llmapi-launch ${numa_bind_cmd} \
        trtllm-serve ${model_path} \
            --host $(hostname) \
            --port ${port} \
            --extra_llm_api_options ${config_file}
fi

