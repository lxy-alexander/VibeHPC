#!/bin/bash
set -e
set -u

# File descriptor limit is propagated via sbatch --propagate=RLIMIT_NOFILE
# No need to set it here

# Start the disaggregated server that coordinates CTX and GEN workers

echo "=========================================="
echo "start_server.sh started"
echo "=========================================="

num_ctx_servers=$1
num_gen_servers=$2
work_dir=$3
script_dir=$4
num_server_instances=${5:-1}
trtllm_install_path=${6:-}
server_postprocess_workers=${7:-4}
server_workers_per_core=${8:-2}

# Debug logging disabled - uncomment if needed for troubleshooting
# if [ -f "${script_dir}/dump_env_debug.sh" ]; then
#     source "${script_dir}/dump_env_debug.sh" "${work_dir}" "SERVER_${SLURM_PROCID:-0}"
# fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Parameters:"
echo "  num_ctx_servers: ${num_ctx_servers}"
echo "  num_gen_servers: ${num_gen_servers}"
echo "  num_server_instances: ${num_server_instances}"
echo "  server_postprocess_workers: ${server_postprocess_workers}"
echo "  server_workers_per_core: ${server_workers_per_core}"
echo "  work_dir: ${work_dir}"
echo "  script_dir: ${script_dir}"

# Install custom TRT-LLM if path provided
if [ -n "$trtllm_install_path" ]; then
    echo "Installing custom TRT-LLM from: ${trtllm_install_path}"
    pip install -e $trtllm_install_path > /dev/null 2>&1
fi

# Disable NIXL on aarch64 (not available)
export TRTLLM_DISABLE_NIXL=1

# UCX optimizations for disaggregated inference
unset UCX_TLS
unset UCX_NET_DEVICES  # Allow UCX to use all available devices
export UCX_RNDV_SCHEME=get_zcopy  # Better performance on GB200

echo "Environment: TRTLLM_DISABLE_NIXL=1"

# Generate server config(s) (waits for all worker hostnames to be available)
echo ""
echo "Generating server config(s)..."
python3 ${script_dir}/gen_server_config.py \
    --num_ctx_servers ${num_ctx_servers} \
    --num_gen_servers ${num_gen_servers} \
    --num_server_instances ${num_server_instances} \
    --server_postprocess_workers ${server_postprocess_workers} \
    --server_workers_per_core ${server_workers_per_core} \
    --work_dir ${work_dir}

echo ""
if [ ${num_server_instances} -eq 1 ]; then
    echo "Server config generated: ${work_dir}/server_config.yaml"
    echo "Contents:"
    cat ${work_dir}/server_config.yaml
else
    echo "${num_server_instances} server configs generated:"
    for i in $(seq 0 $((num_server_instances - 1))); do
        echo "  ${work_dir}/server_config_${i}.yaml"
    done
    echo "Contents of first config:"
    cat ${work_dir}/server_config_0.yaml
fi

echo ""
echo "Starting ${num_server_instances} disaggregated coordinator server(s)..."
echo ""

# Distributed mode: When srun -n N, each task (SLURM_PROCID) starts one coordinator
if [ -n "${SLURM_PROCID}" ] && [ ${num_server_instances} -gt 1 ]; then
    # This task starts only its assigned coordinator
    server_idx=${SLURM_PROCID}
    my_hostname=$(hostname)
    
    if [ $server_idx -ge $num_server_instances ]; then
        echo "Task ${server_idx} exiting (only ${num_server_instances} coordinators needed)"
        exit 0
    fi
    
    config_file="${work_dir}/server_config_${server_idx}.yaml"
    log_file="${work_dir}/coordinator_${server_idx}.log"
    server_port=$((8300 + server_idx))
    
    echo "Starting coordinator ${server_idx} on ${my_hostname}:${server_port}"
    echo "  (Config binds to 0.0.0.0 for distributed mode)"
    
    # Write actual URL for harness to connect to
    mkdir -p "${work_dir}/coordinator_urls"
    echo "${my_hostname}:${server_port}" > "${work_dir}/coordinator_urls/${server_idx}.txt"
    
    # Run coordinator (foreground - srun manages it)
    trtllm-serve disaggregated -c ${config_file} -t 7200 -r 7200 > ${log_file} 2>&1
    exit $?
else
    # Single node mode: start all coordinators on this node
    pids=()
    my_hostname=$(hostname)
    
    for server_idx in $(seq 0 $((num_server_instances - 1))); do
        if [ ${num_server_instances} -eq 1 ]; then
            config_file="${work_dir}/server_config.yaml"
        else
            config_file="${work_dir}/server_config_${server_idx}.yaml"
            # For multi-coordinator on same node, write actual URLs
            mkdir -p "${work_dir}/coordinator_urls"
            server_port=$((8300 + server_idx))
            echo "${my_hostname}:${server_port}" > "${work_dir}/coordinator_urls/${server_idx}.txt"
        fi
        
        log_file="${work_dir}/coordinator_${server_idx}.log"
        
        echo "Starting coordinator server ${server_idx}..."
        echo "  Config: ${config_file}"
        echo "  Log: ${log_file}"
        
        trtllm-serve disaggregated -c ${config_file} -t 7200 -r 7200 > ${log_file} 2>&1 &
        pids+=($!)
        
        echo "  PID: ${pids[-1]}"
    done
    
    echo ""
    echo "All ${num_server_instances} coordinator(s) started, waiting..."
    
    # Wait for all
    final_exit_code=0
    for pid in "${pids[@]}"; do
        wait $pid
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "ERROR: Coordinator (PID ${pid}) failed: ${exit_code}"
            final_exit_code=$exit_code
        fi
    done
    
    exit ${final_exit_code}
fi

