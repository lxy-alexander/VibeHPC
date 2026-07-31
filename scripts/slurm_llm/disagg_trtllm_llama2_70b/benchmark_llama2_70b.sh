#!/bin/bash

# =============================================================================
# Llama2-70b Disaggregated Inference - Main Benchmark Orchestrator
#
# PURPOSE:
#   This is the core script that:
#   1. Sets up the container environment
#   2. Generates configuration files for CTX and GEN workers
#   3. Launches CTX servers (handle prefill/prompt processing)
#   4. Launches GEN servers (handle decode/token generation)
#   5. Starts the coordinator server (routes requests between CTX/GEN)
#   6. Runs the MLPerf benchmark harness
#
# FLOW:
#   [Benchmark Harness] → [Coordinator] → [CTX Servers] (prefill)
#                                      → [GEN Servers] (decode)
#
# NOTE:
#   - This script is called by llama2_70b_disagg.sh or run_interactive.sh
#   - Don't run this directly - use the wrapper scripts
# =============================================================================

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root/closed/NVIDIA"

# Load common utilities (colored logging, helpers)
source ${repo_root}/closed/NVIDIA/disagg_trtllm_llama2_70b/common.sh

# Check required environment variables
if [ -z "${log_base_dir:-}" ]; then
    echo "ERROR: log_base_dir not set. This script must be called from llama2_70b_disagg.sh or run_interactive.sh"
    exit 1
fi

if [ -z "${exp_tag:-}" ]; then
    echo "ERROR: exp_tag not set. This script must be called from llama2_70b_disagg.sh or run_interactive.sh"
    exit 1
fi

# Set SYSTEM_NAME variants for TRT-LLM servers (x1) and harness (x72)
# These are provided by parent script (llama2_70b_disagg.sh) based on GPU type
export SYSTEM_NAME_X1="${SYSTEM_NAME_X1:-GB200-NVL72_GB200-186GB_aarch64x1}"
export SYSTEM_NAME_X72="${SYSTEM_NAME_X72:-GB200-NVL72_GB200-186GB_aarch64x72}"

# Use x1 for TRT-LLM server launches (atomic system)
export SYSTEM_NAME="${SYSTEM_NAME_X1}"


# Use log_base_dir from parent script (submitted via sbatch or interactive)
# Format: build/logs/{scenario}/{hardware}_ctx{N}_gen{M}_qps{Q}
exp_name="${exp_tag}"
output_dir="${log_base_dir}/${exp_name}"

mkdir -p "$output_dir"
export LOG_DIR=$output_dir

# Create srun commands log file
SRUN_LOG="${output_dir}/srun_commands.log"
: > "$SRUN_LOG"  # Clear file if it exists

# Helper function to log and execute srun commands  
log_srun() {
    # Log the full command to srun_commands.log
    echo "$@" >> "$SRUN_LOG" 2>/dev/null || true
    
    # In dry-run mode, just print the command
    if [ "${DRY_RUN:-0}" = "1" ]; then
        return 0
    fi
    
    # Execute the command
    "$@"
}

# Setup script runs on ALL nodes: TCP opts, ulimit, cleanup
setup_script="${output_dir}/setup_node.sh"
cat > ${setup_script} << 'EOF'
#!/bin/bash
# TCP/Network Optimizations on each node
sudo sysctl -w net.ipv4.ip_local_port_range="1024 65535" >/dev/null 2>&1 || true
sudo sysctl -w net.ipv4.tcp_tw_reuse=1 >/dev/null 2>&1 || true
sudo sysctl -w net.ipv4.tcp_fin_timeout=30 >/dev/null 2>&1 || true
sudo sysctl -w net.core.somaxconn=4096 >/dev/null 2>&1 || true
sudo sysctl -w net.ipv4.tcp_max_syn_backlog=8192 >/dev/null 2>&1 || true
sudo sysctl -w net.ipv4.tcp_fastopen=3 >/dev/null 2>&1 || true

# Try to increase ulimit
ulimit -n 1048576 2>/dev/null || true

# Cleanup stale processes
pkill -9 -f "trtllm-serve" 2>/dev/null || true
pkill -9 -f "run_harness" 2>/dev/null || true
pkill -9 -f "python.*loadgen" 2>/dev/null || true
sudo ss --tcp state CLOSE-WAIT --kill 2>/dev/null || true
EOF
chmod +x ${setup_script}

# Run setup on all nodes
log_srun srun --ntasks=${SLURM_NNODES} --ntasks-per-node=1 bash ${setup_script} >/dev/null 2>&1
sleep 2

# Get ulimit for logging (from current node)
actual_ulimit=$(ulimit -n)
log_info "Setup complete on all ${SLURM_NNODES} nodes (ulimit -n: ${actual_ulimit})"

# Cache transceiver configuration
# Use value passed from parent script or default to 2048
cache_transceiver_max_num_tokens=${cache_transceiver_max_num_tokens:-2048}
cache_transceiver_backend=${cache_transceiver_backend:-UCX}

# Performance stats configuration
enable_iter_stats=${enable_iter_stats:-false}

# Set defaults for coordinator threading if not provided
server_num_postprocess_workers=${server_num_postprocess_workers:-4}
server_workers_per_core=${server_workers_per_core:-2}

# Calculate concurrency (needed for loadgen)
export concurrency=$((num_gen_servers * gen_pp_size * gen_max_batch_size))
export test_mode="${test_mode:-PerformanceOnly}"

# Calculate GPU requirements
ctx_gpus=$((num_ctx_servers * ctx_tp_size * ctx_pp_size))
gen_gpus=$((num_gen_servers * gen_tp_size * gen_pp_size))

# Container setup
container_workdir="/work"
actual_workdir="${repo_root}/closed/NVIDIA"
MLPERF_SCRATCH_PATH="/lustre/share/coreai_mlperf_inference/mlperf_inference_storage_clone"
CONTAINER_NAME="llama2_70b_disagg_${SLURM_JOB_ID}"

# Script directory - point to the scripts subdirectory inside disagg_trtllm_llama2_70b
scripts_dir="/work/scripts/slurm_llm/disagg_trtllm_llama2_70b/scripts"

# Build container mounts (no need to mount scripts separately, they're under /work)
CONTAINER_MOUNTS="$actual_workdir:$container_workdir,$MLPERF_SCRATCH_PATH:/home/mlperf_inference_storage"
if [ -n "$trtllm_install_path" ]; then
    CONTAINER_MOUNTS+=",${trtllm_install_path}:${trtllm_install_path}"
fi
if [ -n "$extra_container_mounts" ]; then
    CONTAINER_MOUNTS+=",${extra_container_mounts}"
fi

# Model path - UPDATE this to match your llama2-70b model location
# You can override by setting MODEL_PATH environment variable before submission
MODEL_DIR="${MODEL_PATH:-/home/mlperf_inference_storage/models/Llama2/fp4-quantized-modelopt/llama2-70b-chat-hf-torch-fp4}"

echo ""
log_section "Llama2-70b Disaggregated Benchmark"
echo ""
log_info "Config: $scenario @ ${target_qps} QPS (${num_reqs} requests)"
log_detail "CTX: ${num_ctx_servers} server(s) × ${ctx_gpus} GPU(s) | Batch=${ctx_max_batch_size} | MaxTokens=${ctx_max_num_tokens}"
log_detail "GEN: ${num_gen_servers} server(s) × ${gen_gpus} GPU(s) | Batch=${gen_max_batch_size} | MaxTokens=${gen_max_num_tokens}"
log_detail "Model: $(basename $MODEL_DIR)"
log_detail "Output: $output_dir"
echo ""

# Start CPU/memory monitoring
log_info "Starting CPU and memory monitoring..."
sar -u -r 1 > ${output_dir}/sar_stats.log 2>&1 &
SAR_PID=$!
log_detail "SAR monitoring PID: $SAR_PID"
echo ""

# Start the container
log_step 1 5 "Starting container..."
log_srun srun --container-image=${CONTAINER_IMAGE} \
        --container-name=${CONTAINER_NAME} \
        --container-mounts=${CONTAINER_MOUNTS} \
        --mpi=pmix \
        --container-mount-home \
        --container-remap-root \
        echo "Container ready" >/dev/null 2>&1
log_success "Container initialized"

# Generate worker config YAML
echo ""
log_step 2 5 "Generating worker configurations..."

log_srun srun -N 1 -n 1 \
        --container-name=${CONTAINER_NAME} \
        --container-mounts=${CONTAINER_MOUNTS} \
        --container-workdir=/work \
        --container-mount-home \
        --container-remap-root \
        --mpi=pmix --overlap \
        python3 ${scripts_dir}/gen_worker_config.py \
                --work_dir ${output_dir} \
                --ctx_tp_size ${ctx_tp_size} \
                --ctx_pp_size ${ctx_pp_size} \
                --ctx_ep_size ${ctx_ep_size} \
                --ctx_max_batch_size ${ctx_max_batch_size} \
                --ctx_max_num_tokens ${ctx_max_num_tokens} \
                --ctx_free_gpu_memory_fraction ${ctx_gpu_frac} \
                --ctx_stream_interval ${ctx_stream_interval} \
                --gen_tp_size ${gen_tp_size} \
                --gen_pp_size ${gen_pp_size} \
                --gen_ep_size ${gen_ep_size} \
                --gen_max_batch_size ${gen_max_batch_size} \
                --gen_max_num_tokens ${gen_max_num_tokens} \
                --gen_gpu_memory_fraction ${gen_gpu_frac} \
                --gen_stream_interval ${gen_stream_interval} \
                --gen_num_postprocess_workers ${gen_num_postprocess_workers} \
                --eplb_num_slots ${eplb_num_slots} \
                --mtp_size ${mtp_size} \
                --cache_transceiver_max_num_tokens ${cache_transceiver_max_num_tokens} \
                --cache_transceiver_backend ${cache_transceiver_backend} \
                $(if [ "${ctx_enable_adp}" = "true" ]; then echo "--ctx_enable_attention_dp"; fi) \
                $(if [ "${gen_enable_adp}" = "true" ]; then echo "--gen_enable_attention_dp"; fi) \
                $(if [ -n "${cg_sizes}" ]; then echo "--cg_sizes ${cg_sizes}"; fi) \
                $(if [ "${enable_iter_stats}" = "true" ]; then echo "--enable_iter_stats"; fi) \
                2>&1 | grep -E "(Config files generated|cuda graph)" | sed 's/^[0-9]*: //g' | sed 's/^/    /'

log_success "CTX config: ${output_dir}/ctx_config.yaml"
log_success "GEN config: ${output_dir}/gen_config.yaml"
echo ""

ntasks_per_node=4  # GB200/GB300 standard

# Calculate GPUs per server
gpus_per_gen_server=$((gen_tp_size * gen_pp_size))
gpus_per_ctx_server=$((ctx_tp_size * ctx_pp_size))

# Calculate total GPUs needed
total_gen_gpus=$((num_gen_servers * gpus_per_gen_server))
total_ctx_gpus=$((num_ctx_servers * gpus_per_ctx_server))
total_gpus=$((total_gen_gpus + total_ctx_gpus))
total_nodes_needed=$(( (total_gpus + ntasks_per_node - 1) / ntasks_per_node ))
total_connections=$((num_server_instances * (num_ctx_servers + num_gen_servers)))

log_info "Configuration: CTX=${num_ctx_servers}, GEN=${num_gen_servers}, Coordinators=${num_server_instances}, Target QPS=${target_qps}"

# ============================================================================
# NODE ALLOCATION STRATEGY
# ============================================================================
# Get all allocated nodes (or use dummy nodes in dry-run mode)
if [ "${DRY_RUN:-0}" = "1" ]; then
    # Generate dummy node names for dry-run
    total_nodes=$(( (total_gpus + ntasks_per_node - 1) / ntasks_per_node ))
    all_nodes=()
    for i in $(seq 0 $((total_nodes - 1))); do
        all_nodes+=("node$(printf '%04d' $i)")
    done
else
    all_nodes=($(scontrol show hostname $SLURM_NODELIST | sort))
    total_nodes=${#all_nodes[@]}
fi

# Calculate nodes needed for workers
num_gen_nodes=$(( (total_gen_gpus + ntasks_per_node - 1) / ntasks_per_node ))
num_ctx_nodes=$(( (total_ctx_gpus + ntasks_per_node - 1) / ntasks_per_node ))
total_worker_nodes=$((num_gen_nodes + num_ctx_nodes))

# GEN workers: nodes 0 to (num_gen_nodes-1)
# CTX workers: nodes num_gen_nodes to (num_gen_nodes+num_ctx_nodes-1)
gen_nodes=("${all_nodes[@]:0:$num_gen_nodes}")
ctx_nodes=("${all_nodes[@]:$num_gen_nodes:$num_ctx_nodes}")

# ============================================================================
# COORDINATOR NODE SELECTION
# Strategy: Place coordinators on CTX nodes first (for proximity), then GEN
# ============================================================================
coordinator_nodes=()

# Place coordinators on CTX nodes first
num_coordinators_on_ctx=$((num_ctx_nodes < num_server_instances ? num_ctx_nodes : num_server_instances))
if [ $num_coordinators_on_ctx -gt 0 ]; then
    coordinator_nodes+=("${ctx_nodes[@]:0:$num_coordinators_on_ctx}")
fi

# If more coordinators needed, use GEN nodes (from the end)
num_coordinators_on_gen=$((num_server_instances - num_coordinators_on_ctx))
if [ $num_coordinators_on_gen -gt 0 ]; then
    # Use last N GEN nodes for coordinators
    gen_coordinator_start=$((num_gen_nodes - num_coordinators_on_gen))
    coordinator_nodes+=("${gen_nodes[@]:$gen_coordinator_start:$num_coordinators_on_gen}")
fi

# ============================================================================
# HARNESS NODE SELECTION
# Strategy: Use a GEN node that doesn't have a coordinator
# ============================================================================
harness_node=""
if [ $num_coordinators_on_gen -gt 0 ]; then
    # Coordinators occupy GEN nodes from (num_gen_nodes - num_coordinators_on_gen) onward
    # Use the node right before the first coordinator-occupied GEN node
    harness_node_idx=$((num_gen_nodes - num_coordinators_on_gen - 1))
    if [ $harness_node_idx -ge 0 ]; then
        harness_node="${gen_nodes[$harness_node_idx]}"
    else
        # Fallback: use last GEN node
        harness_node="${gen_nodes[$((num_gen_nodes - 1))]}"
    fi
else
    # No coordinators on GEN nodes, use last GEN node
    harness_node="${gen_nodes[$((num_gen_nodes - 1))]}"
fi

# Validation
if [ $total_nodes -lt $total_worker_nodes ]; then
    log_error "Insufficient nodes allocated!"
    log_detail "Required: $total_worker_nodes (${num_gen_nodes} GEN + ${num_ctx_nodes} CTX)"
    log_detail "Available: $total_nodes"
    exit 1
fi

# Build coordinator nodelist (comma-separated for srun)
coordinator_nodelist=$(IFS=,; echo "${coordinator_nodes[*]}")

echo ""
log_step 3 5 "Starting workers"
log_info "GEN workers: ${num_gen_servers} workers on ${num_gen_nodes} nodes"
log_info "CTX workers: ${num_ctx_servers} workers on ${num_ctx_nodes} nodes"
log_detail "Harness will run on: ${harness_node}"

# Clean up old configs
rm -rf ${output_dir}/hostnames
rm -rf ${output_dir}/server_config.yaml

# Common srun flags
srun_flags="--container-image=${CONTAINER_IMAGE} --container-name=${CONTAINER_NAME} --container-mounts=${CONTAINER_MOUNTS} --container-workdir=/work --container-remap-root --container-mount-home --mpi=pmix --overlap"

# Start GEN workers
for i in $(seq 0 $((num_gen_servers - 1))); do
    global_gpu_idx=$((i * gpus_per_gen_server))
    start_node_idx=$(( global_gpu_idx / ntasks_per_node ))
    
    if [ $gpus_per_gen_server -le $ntasks_per_node ]; then
        # Intra-node: one or more servers per node
        node_name=${gen_nodes[$start_node_idx]}
        local_start_gpu=$(( global_gpu_idx % ntasks_per_node ))
        local_end_gpu=$(( local_start_gpu + gpus_per_gen_server - 1 ))
        gpu_list=$(seq -s, $local_start_gpu $local_end_gpu)
        
        port=$((8336 + i))
        ipc_port=$((10000 + i))
        ipc_addr="tcp://127.0.0.1:${ipc_port}"
        
        export CUDA_VISIBLE_DEVICES=${gpu_list}
        log_detail "Starting GEN worker ${i} on ${node_name} (GPU ${gpu_list}, port ${port})"
        
        log_srun srun $srun_flags \
            -N 1 \
            --nodelist=${node_name} \
            --ntasks-per-node=${gpus_per_gen_server} \
            --export=ALL,LOG_DIR,TRTLLM_SERVER_DISABLE_GC=1,TRTLLM_WORKER_DISABLE_GC=1,TRTLLM_ENABLE_PDL=1,TRTLLM_DISABLE_NIXL=1,CUDA_SCALE_LAUNCH_QUEUES=4x,UCX_CUDA_IPC_ENABLE_MNNVL=n,UCX_RNDV_SCHEME=get_zcopy,TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR=${ipc_addr},DP_RANK=${i},TLLM_NUMA_AWARE_WORKER_AFFINITY=1,OMPI_MCA_hwloc_base_binding_policy=none,OMPI_MCA_rmaps_base_inherit=1 \
            bash ${scripts_dir}/start_worker.sh "GEN" ${i} ${MODEL_DIR} "${port}" ${output_dir} ${exp_name} ${gen_nsys} ${GEN_TLLM_PROFILE_START_STOP} ${trtllm_install_path} \
            &> ${output_dir}/output_gen_${i}.log &
        log_success "GEN worker ${i} started (log: output_gen_${i}.log)"
    else
        # Multi-node: one server spans multiple nodes
        unset CUDA_VISIBLE_DEVICES
        num_nodes_for_server=$(( gpus_per_gen_server / ntasks_per_node ))
        target_nodes=("${gen_nodes[@]:$start_node_idx:$num_nodes_for_server}")
        node_list=$(IFS=,; echo "${target_nodes[*]}")
        
        port=$((8336 + i))
        log_detail "Starting GEN worker ${i} (multi-node: ${node_list}, port ${port})"
        log_srun srun $srun_flags \
            -N ${num_nodes_for_server} \
            --nodelist=${node_list} \
            --ntasks=${gpus_per_gen_server} \
            --ntasks-per-node=${ntasks_per_node} \
            --export=ALL,LOG_DIR,TRTLLM_SERVER_DISABLE_GC=1,TRTLLM_WORKER_DISABLE_GC=1,TRTLLM_ENABLE_PDL=1,TRTLLM_DISABLE_NIXL=1,CUDA_SCALE_LAUNCH_QUEUES=4x,UCX_CUDA_IPC_ENABLE_MNNVL=n,UCX_RNDV_SCHEME=get_zcopy,TLLM_NUMA_AWARE_WORKER_AFFINITY=1,OMPI_MCA_hwloc_base_binding_policy=none,OMPI_MCA_rmaps_base_inherit=1 \
            bash ${scripts_dir}/start_worker.sh "GEN" ${i} ${MODEL_DIR} "${port}" ${output_dir} ${exp_name} ${gen_nsys} ${GEN_TLLM_PROFILE_START_STOP} ${trtllm_install_path} \
            &> ${output_dir}/output_gen_${i}.log &
        log_success "GEN worker ${i} started (log: output_gen_${i}.log)"
    fi
done

# Start CTX workers
for i in $(seq 0 $((num_ctx_servers - 1))); do
    # Calculate node index within CTX nodes array
    ctx_gpu_idx=$((i * gpus_per_ctx_server))
    start_node_idx=$(( ctx_gpu_idx / ntasks_per_node ))
    
    if [ $gpus_per_ctx_server -le $ntasks_per_node ]; then
        # Intra-node
        node_name=${ctx_nodes[$start_node_idx]}
        # Calculate global GPU index for CUDA_VISIBLE_DEVICES
        global_gpu_idx=$((total_gen_gpus + ctx_gpu_idx))
        local_start_gpu=$(( global_gpu_idx % ntasks_per_node ))
        local_end_gpu=$(( local_start_gpu + gpus_per_ctx_server - 1 ))
        gpu_list=$(seq -s, $local_start_gpu $local_end_gpu)
        
        port=$((8336 + num_gen_servers + i))
        ipc_port=$((10000 + num_gen_servers + i))
        ipc_addr="tcp://127.0.0.1:${ipc_port}"
        
        export CUDA_VISIBLE_DEVICES=${gpu_list}
        log_detail "Starting CTX worker ${i} on ${node_name} (GPU ${gpu_list}, port ${port})"
        
        log_srun srun $srun_flags \
            -N 1 \
            --nodelist=${node_name} \
            --ntasks-per-node=${gpus_per_ctx_server} \
            --export=ALL,LOG_DIR,TRTLLM_SERVER_DISABLE_GC=1,TRTLLM_WORKER_DISABLE_GC=1,TRTLLM_ENABLE_PDL=1,TRTLLM_DISABLE_NIXL=1,CUDA_SCALE_LAUNCH_QUEUES=4x,UCX_CUDA_IPC_ENABLE_MNNVL=n,UCX_RNDV_SCHEME=get_zcopy,TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR=${ipc_addr},DP_RANK=${i},TLLM_NUMA_AWARE_WORKER_AFFINITY=1,OMPI_MCA_hwloc_base_binding_policy=none,OMPI_MCA_rmaps_base_inherit=1 \
            bash ${scripts_dir}/start_worker.sh "CTX" ${i} ${MODEL_DIR} "${port}" ${output_dir} ${exp_name} ${ctx_nsys} ${CTX_TLLM_PROFILE_START_STOP} ${trtllm_install_path} \
            &> ${output_dir}/output_ctx_${i}.log &
        log_success "CTX worker ${i} started (log: output_ctx_${i}.log)"
    else
        # Multi-node
        unset CUDA_VISIBLE_DEVICES
        num_nodes_for_server=$(( gpus_per_ctx_server / ntasks_per_node ))
        target_nodes=("${ctx_nodes[@]:$start_node_idx:$num_nodes_for_server}")
        node_list=$(IFS=,; echo "${target_nodes[*]}")
        
        port=$((8336 + num_gen_servers + i))
        log_detail "Starting CTX worker ${i} (multi-node: ${node_list}, port ${port})"
        log_srun srun $srun_flags \
            -N ${num_nodes_for_server} \
            --nodelist=${node_list} \
            --ntasks=${gpus_per_ctx_server} \
            --ntasks-per-node=${ntasks_per_node} \
            --export=ALL,LOG_DIR,TRTLLM_SERVER_DISABLE_GC=1,TRTLLM_WORKER_DISABLE_GC=1,TRTLLM_ENABLE_PDL=1,TRTLLM_DISABLE_NIXL=1,CUDA_SCALE_LAUNCH_QUEUES=4x,UCX_CUDA_IPC_ENABLE_MNNVL=n,UCX_RNDV_SCHEME=get_zcopy,TLLM_NUMA_AWARE_WORKER_AFFINITY=1,OMPI_MCA_hwloc_base_binding_policy=none,OMPI_MCA_rmaps_base_inherit=1 \
            bash ${scripts_dir}/start_worker.sh "CTX" ${i} ${MODEL_DIR} "${port}" ${output_dir} ${exp_name} ${ctx_nsys} ${CTX_TLLM_PROFILE_START_STOP} ${trtllm_install_path} \
            &> ${output_dir}/output_ctx_${i}.log &
        log_success "CTX worker ${i} started (log: output_ctx_${i}.log)"
    fi
done

echo ""
log_step 4 5 "Starting coordinator server(s)..."
if [ ${num_server_instances} -eq 1 ]; then
    log_detail "Config: ${output_dir}/server_config.yaml"
else
    log_detail "Instances: ${num_server_instances}"
    log_detail "Configs: ${output_dir}/server_config_{0..${num_server_instances}}.yaml"
fi
log_detail "Waiting for ${num_ctx_servers} CTX + ${num_gen_servers} GEN workers to register..."

# Start the coordinator server(s)
# Multi-coordinator: distribute across N nodes (1 per node)
# Single coordinator: use 1 node (like before)

if [ ${num_server_instances} -gt 1 ]; then
    log_detail "Distributing ${num_server_instances} coordinator(s) across specific nodes: ${coordinator_nodelist}"
    log_detail "  - ${num_coordinators_on_ctx} on CTX nodes"
    log_detail "  - ${num_coordinators_on_gen} on GEN nodes"
    log_srun srun --container-name=${CONTAINER_NAME} \
        --container-image=${CONTAINER_IMAGE} \
        --container-mounts=${CONTAINER_MOUNTS} \
        --mpi=pmix --overlap -N ${num_server_instances} -n ${num_server_instances} \
        --nodelist=${coordinator_nodelist} \
        --container-workdir=/work \
        --container-mount-home \
        --container-remap-root \
        --export=LOG_DIR,TRTLLM_SERVER_DISABLE_GC=1,TRTLLM_WORKER_DISABLE_GC=1,TRTLLM_ENABLE_PDL=1,TRTLLM_DISABLE_NIXL=1,TLLM_NUMA_AWARE_WORKER_AFFINITY=1,OMPI_MCA_hwloc_base_binding_policy=none,OMPI_MCA_rmaps_base_inherit=1 \
        bash ${scripts_dir}/start_server.sh ${num_ctx_servers} ${num_gen_servers} ${output_dir} ${scripts_dir} ${num_server_instances} "${trtllm_install_path}" ${server_num_postprocess_workers} ${server_workers_per_core} \
        &> ${output_dir}/output_server.log &
else
    log_srun srun --container-name=${CONTAINER_NAME} \
        --container-image=${CONTAINER_IMAGE} \
        --container-mounts=${CONTAINER_MOUNTS} \
        --mpi=pmix --overlap -N 1 -n 1 \
        --container-workdir=/work \
        --container-mount-home \
        --container-remap-root \
        --export=LOG_DIR,TRTLLM_SERVER_DISABLE_GC=1,TRTLLM_WORKER_DISABLE_GC=1,TRTLLM_ENABLE_PDL=1,TRTLLM_DISABLE_NIXL=1,TLLM_NUMA_AWARE_WORKER_AFFINITY=1,OMPI_MCA_hwloc_base_binding_policy=none,OMPI_MCA_rmaps_base_inherit=1 \
        bash ${scripts_dir}/start_server.sh ${num_ctx_servers} ${num_gen_servers} ${output_dir} ${scripts_dir} ${num_server_instances} "${trtllm_install_path}" ${server_num_postprocess_workers} ${server_workers_per_core} \
        &> ${output_dir}/output_server.log &
fi
log_success "Coordinator server(s) started (log: output_server.log)"

echo ""
# For multi-coordinator, wait for actual hostnames and update server_urls.txt
# For single coordinator, gen_server_config.py already wrote the correct URL
if [ ${num_server_instances} -gt 1 ]; then
    log_detail "Waiting for distributed coordinators to register actual hostnames..."
    sleep 5
    timeout=300
    start_time=$(date +%s)
    
    while true; do
        if [ -d "${output_dir}/coordinator_urls" ]; then
            num_registered=$(ls -1 ${output_dir}/coordinator_urls/*.txt 2>/dev/null | wc -l)
            if [ ${num_registered} -ge ${num_server_instances} ]; then
                # Rebuild server_urls.txt with actual hostnames
                server_urls=""
                for i in $(seq 0 $((num_server_instances - 1))); do
                    url=$(cat ${output_dir}/coordinator_urls/${i}.txt 2>/dev/null)
                    if [ -n "$url" ]; then
                        [ -z "$server_urls" ] && server_urls="$url" || server_urls="${server_urls},${url}"
                    fi
                done
                if [ -n "$server_urls" ]; then
                    echo "$server_urls" > ${output_dir}/server_urls.txt
                    log_success "Distributed coordinators registered: ${server_urls}"
                    break
                fi
            fi
        fi
        
        elapsed=$(( $(date +%s) - start_time ))
        if [ $elapsed -ge $timeout ]; then
            log_warning "Timeout waiting for coordinator URLs"
            break
        fi
        sleep 2
    done
else
    log_detail "Single coordinator - using URL from gen_server_config.py"
fi

echo ""
log_step 5 5 "Running MLPerf benchmark..."
log_detail "Waiting for server health check..."
log_detail "This will block until benchmark completes"
log_detail "Watch progress: tail -f ${output_dir}/harness.log"
log_detail "Harness will run on: ${harness_node}"
echo ""

# In dry-run mode, exit here after showing all commands
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo ""
    echo "[DRY-RUN] All commands shown - would now execute workers, coordinators, and harness"
    echo "[DRY-RUN] Exiting without actual execution"
    exit 0
fi

# Start the benchmark harness (NOTE: No & so this WAITS for completion)
# Health checking is done inside bench_e2e.sh
benchmark_start_time=$(date +%s)
log_srun srun --container-name=${CONTAINER_NAME} \
        --container-image=${CONTAINER_IMAGE} \
        --container-mounts=${CONTAINER_MOUNTS} \
        --container-workdir=/work \
        --container-mount-home \
        --container-remap-root \
        --mpi=pmix --overlap -N 1 -n 1 \
        --nodelist=${harness_node} \
        --export=concurrency,target_qps,num_reqs,scenario,test_mode,warmup_iters,audit,SYSTEM_NAME_X72 \
        bash ${scripts_dir}/bench_e2e.sh ${output_dir} > ${output_dir}/harness.log 2>&1
benchmark_exit_code=$?
benchmark_end_time=$(date +%s)
benchmark_duration=$((benchmark_end_time - benchmark_start_time))

# Stop CPU/memory monitoring
if [ ! -z "$SAR_PID" ]; then
    kill $SAR_PID 2>/dev/null
fi

if [ $benchmark_exit_code -eq 0 ]; then
    log_section "Benchmark Completed Successfully! 🎉"
    log_info "Duration: ${benchmark_duration}s"
else
    log_section "Benchmark Failed"
    log_error "Exit code: $benchmark_exit_code - check ${output_dir}/harness.log"
fi
log_warning "Background workers are still running"
log_info "To clean up:"
log_command "scancel ${SLURM_JOB_ID}"
log_info "Or kill workers directly:"
log_command "pkill -9 trtllm-serve"
echo ""

