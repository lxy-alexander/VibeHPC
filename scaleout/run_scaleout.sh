#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

set -euo pipefail

stage="server"
atomic_system=""
harness_system=""
dp_multiplicity=""
gpus_per_node=""
run_args=""
container_image=""
mlperf_scratch_path="/lustre/share/coreai_mlperf_inference/mlperf_inference_storage_clone"
extra_srun_flags=""
base_port=30000
server_spawn_time=0
dry_run=false
nodelist=""
jobid=""
log_dir=""
gpu_offset=0
harness_target="run_harness"
workspace_override=""

# Helper function to format command with indentation
format_command() {
    local cmd="$1"
    shift
    echo "$cmd \\"
    local last_idx=$(($# - 1))
    local idx=0
    for arg in "$@"; do
        if [[ $idx -eq $last_idx ]]; then
            echo "    $arg"
        else
            echo "    $arg \\"
        fi
        idx=$((idx + 1))
    done
}

# Helper function to execute srun with command printing
run_srun() {
    echo "================================================"
    echo "Executing srun command:"
    echo "================================================"
    format_command "srun" "$@"
    echo ""
    if [[ -n "${HOST_LOG_DIR:-}" ]]; then
        format_command "srun" "$@" >> "${HOST_LOG_DIR}/srun_commands.log"
        echo "" >> "${HOST_LOG_DIR}/srun_commands.log"
        echo "" >> "${HOST_LOG_DIR}/srun_commands.log"
    fi
    if [[ "$dry_run" == "false" ]]; then
        srun "$@"
    fi
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage=*)
            stage="${1#*=}"
            shift
            ;;
        --stage)
            stage=$2
            shift 2
            ;;
        --atomic-system=*)
            atomic_system="${1#*=}"
            shift
            ;;
        --atomic-system)
            atomic_system=$2
            shift 2
            ;;
        --dp-multiplicity=*)
            dp_multiplicity="${1#*=}"
            shift
            ;;
        --dp-multiplicity)
            dp_multiplicity=$2
            shift 2
            ;;
        --harness-system=*)
            harness_system="${1#*=}"
            shift
            ;;
        --harness-system)
            harness_system=$2
            shift 2
            ;;
        --gpus-per-node=*)
            gpus_per_node="${1#*=}"
            shift
            ;;
        --gpus-per-node)
            gpus_per_node=$2
            shift 2
            ;;
        --run-args=*)
            run_args="${1#*=}"
            shift
            ;;
        --run-args)
            run_args=$2
            shift 2
            ;;
        --container-image=*)
            container_image="${1#*=}"
            shift
            ;;
        --container-image)
            container_image=$2
            shift 2
            ;;
        --mlperf-scratch-path=*)
            mlperf_scratch_path="${1#*=}"
            shift
            ;;
        --mlperf-scratch-path)
            mlperf_scratch_path=$2
            shift 2
            ;;
        --extra-srun-flags=*)
            extra_srun_flags="${1#*=}"
            shift
            ;;
        --extra-srun-flags)
            extra_srun_flags=$2
            shift 2
            ;;
        --base-port=*)
            base_port="${1#*=}"
            shift
            ;;
        --base-port)
            base_port=$2
            shift 2
            ;;
        --server-spawn-time=*)
            server_spawn_time="${1#*=}"
            shift
            ;;
        --server-spawn-time)
            server_spawn_time=$2
            shift 2
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        --jobid=*)
            jobid="${1#*=}"
            shift
            ;;
        --jobid)
            jobid=$2
            shift 2
            ;;
        --nodelist=*|-w=*)
            nodelist="${1#*=}"
            shift
            ;;
        --nodelist|-w)
            nodelist=$2
            shift 2
            ;;
        --log-dir=*)
            log_dir="${1#*=}"
            shift
            ;;
        --log-dir)
            log_dir=$2
            shift 2
            ;;
        --gpu-offset=*)
            gpu_offset="${1#*=}"
            shift
            ;;
        --gpu-offset)
            gpu_offset=$2
            shift 2
            ;;
        --audit)
            harness_target="run_audit_harness"
            shift
            ;;
        --workspace=*)
            workspace_override="${1#*=}"
            shift
            ;;
        --workspace)
            workspace_override=$2
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 --stage <server|harness|all> --atomic-system <name> --dp-multiplicity <N> --gpus-per-node <N> --run-args <args> [OPTIONS]"
            echo ""
            echo "Note: All options accept both --key=value and --key value formats"
            echo ""
            echo "Required:"
            echo "  --stage <server|harness|all> Stage: server, harness, or all (runs server then harness)"
            echo "  --atomic-system <name>       Atomic system config (e.g., GB200-NVL72_GB200-186GB_aarch64x8)"
            echo "  --dp-multiplicity <N>        Number of DP ranks"
            echo "  --gpus-per-node <N>          GPUs per node"
            echo "  --run-args <args>            Benchmark arguments"
            echo ""
            echo "Optional:"
            echo "  --harness-system <name>      Override harness system (default: calculated from atomic-system x dp-multiplicity)"
            echo "  --jobid <id>                 SLURM job ID to target (default: use \$SLURM_JOBID from environment)"
            echo "  --container-image <path>     Container image (default: build/sqsh_images/mlperf-inference-\$USER-aarch64-release.sqsh)"
            echo "  --mlperf-scratch-path <path> Scratch path (default: /lustre/share/coreai_mlperf_inference/mlperf_inference_storage_clone)"
            echo "  --extra-srun-flags <flags>   Additional srun flags"
            echo "  --base-port <N>              Base port (default: 30000)"
            echo "  --server-spawn-time <sec>    Sleep time after launching servers in seconds (default: 0)"
            echo "  --nodelist <nodes>           Comma-separated node list (default: use SLURM_JOB_NODELIST)"
            echo "  -w <nodes>                   Alias for --nodelist"
            echo "  --log-dir <path>             Log directory (default: auto-generated with timestamp)"
            echo "  --gpu-offset <N>             Starting GPU index for intra-node deployments (default: 0)"
            echo "  --audit                      Run audit harness (run_audit_harness) instead of regular harness"
            echo "  --dry-run                    Print srun commands without executing them"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# Determine SLURM job ID: use --jobid if provided, otherwise fall back to $SLURM_JOBID
if [[ -n "$jobid" ]]; then
    # User provided explicit job ID
    SLURM_JOBID="$jobid"
    echo "Using explicit job ID: $SLURM_JOBID"
    # Add --jobid to srun flags
    extra_srun_flags="--jobid=$SLURM_JOBID $extra_srun_flags"
elif [[ -n "${SLURM_JOBID:-}" ]]; then
    # Already in SLURM allocation
    echo "Using SLURM_JOBID from environment: $SLURM_JOBID"
else
    echo "ERROR: No SLURM job ID found. Provide --jobid or run inside a SLURM allocation." >&2
    exit 1
fi

# Fetch SLURM_JOB_NODELIST from the job if --nodelist not provided
if [[ -z "$nodelist" ]]; then
    SLURM_JOB_NODELIST=$(scontrol show hostnames $(squeue -j "$SLURM_JOBID" -h -o "%N") | paste -sd "," - 2>/dev/null)
    if [[ -z "$SLURM_JOB_NODELIST" ]]; then
        echo "ERROR: Failed to fetch nodelist for job $SLURM_JOBID. Job may not exist or is not running." >&2
        exit 1
    fi
    echo "Fetched nodelist from job $SLURM_JOBID: $SLURM_JOB_NODELIST"
fi

# Determine workspace (host_vol) - can be overridden via --workspace
if [[ -n "$workspace_override" ]]; then
    # Use explicit workspace override
    host_vol="$(readlink -f "$workspace_override")"
    script_dir="$host_vol/scaleout"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    # sbatch: use the directory from which sbatch was invoked
    script_dir="$SLURM_SUBMIT_DIR/scaleout"
    host_vol="$(readlink -f "$SLURM_SUBMIT_DIR")"
else
    # script: use the directory containing the script
    script_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
    host_vol="$(readlink -f "$script_dir/..")"
fi
container_vol="/work"

# Validate required parameters early
[[ -z "$atomic_system" ]] && { echo "ERROR: --atomic-system is required" >&2; exit 1; }
[[ -z "$dp_multiplicity" ]] && { echo "ERROR: --dp-multiplicity is required" >&2; exit 1; }
[[ -z "$gpus_per_node" ]] && { echo "ERROR: --gpus-per-node is required" >&2; exit 1; }
[[ -z "$run_args" ]] && { echo "ERROR: --run-args is required" >&2; exit 1; }

# Parse GPUs per DP rank from atomic system config (e.g., "...x8" -> 8)
num_gpus_per_dp=$(echo "$atomic_system" | grep -oP 'x\K\d+$')
[[ -z "$num_gpus_per_dp" ]] && { echo "ERROR: Failed to parse GPU count from atomic_system: $atomic_system (expected format: ...x<N>)" >&2; exit 1; }

# Calculate harness system if not provided
if [[ -z "$harness_system" ]]; then
    total_harness_gpus=$((num_gpus_per_dp * dp_multiplicity))
    harness_system=$(echo "$atomic_system" | sed "s/x${num_gpus_per_dp}$/x${total_harness_gpus}/")
    echo "Calculated harness system: $harness_system"
else
    echo "Using user-provided harness system: $harness_system"
fi

# Determine log directory: use --log-dir if provided, otherwise auto-generate
if [[ -n "$log_dir" ]]; then
    # Use provided log directory (absolute path)
    HOST_LOG_DIR="$log_dir"
    LOG_DIR="${log_dir/#${host_vol}/${container_vol}}"
else
    # Auto-generate log directory with timestamp (using harness system)
    TIMESTAMP=$(date +'%Y.%m.%d-%H.%M.%S')
    LOG_DIR="/work/build/logs/scaleout_${harness_system}_slurm-${SLURM_JOBID}_${TIMESTAMP}"
    HOST_LOG_DIR="${LOG_DIR/#${container_vol}/${host_vol}}"
fi
export LOG_DIR
mkdir -p "$HOST_LOG_DIR"

if [[ -z "$container_image" ]]; then
    docker_tag=$(whoami)-aarch64
    container_image="$host_vol/build/sqsh_images/mlperf-inference-$docker_tag-release.sqsh"
fi

# Check for node list
[[ -z "${SLURM_JOB_NODELIST:-}" && -z "$nodelist" ]] && { echo "ERROR: No node list found. Provide --nodelist or run inside a SLURM allocation." >&2; exit 1; }

# Use provided nodelist or fall back to SLURM allocation
if [[ -n "$nodelist" ]]; then
    # User provided explicit nodelist
    job_nodelist="$nodelist"
    echo "Using explicit nodelist: $job_nodelist"
else
    # Use SLURM allocation
    job_nodelist="$SLURM_JOB_NODELIST"
    echo "Using SLURM allocation nodelist: $job_nodelist"
fi

if [[ "$stage" == "server" || "$stage" == "all" ]]; then
    [[ ! "$run_args" =~ --mpi_mode=leader ]] && run_args="$run_args --mpi_mode=leader"
    [[ ! "$run_args" =~ --server_in_foreground ]] && run_args="$run_args --server_in_foreground"
elif [[ "$stage" == "harness" ]]; then
    [[ ! "$run_args" =~ --mpi_mode=leader ]] && run_args="$run_args --mpi_mode=leader"
fi

allocated_node_count=$(scontrol show hostname "$job_nodelist" | wc -l)
allocated_nodes=$(scontrol show hostname "$job_nodelist")

total_gpus_allocated=$((allocated_node_count * gpus_per_node))
total_gpus_required=$((num_gpus_per_dp * dp_multiplicity))

# Validate that allocated resources match DP topology requirements
if [[ $total_gpus_allocated -lt $total_gpus_required ]]; then
    echo "ERROR: Insufficient GPU count" >&2
    echo "  Allocated: $total_gpus_allocated ($allocated_node_count nodes x $gpus_per_node GPUs/node)" >&2
    echo "  Required: $total_gpus_required ($dp_multiplicity DP ranks x $num_gpus_per_dp GPUs/rank)" >&2
    exit 1
fi

echo "============================================"
echo "MLPerf Scaleout - $stage"
echo "============================================"
echo "Atomic system: $atomic_system"
echo "Harness system: $harness_system"
echo "GPUs per DP: $num_gpus_per_dp"
echo "GPUs per node: $gpus_per_node"
echo "DP multiplicity: $dp_multiplicity"
echo "Allocated nodes: $allocated_node_count"
echo "Total GPUs: $total_gpus_allocated"
echo "Log directory: $LOG_DIR"
echo "Harness target: $harness_target"
echo "Server spawn time: ${server_spawn_time}s"
echo "Dry run: $dry_run"

# Generate server URLs for all DP ranks
declare -a server_url_array
if [[ $num_gpus_per_dp -ge $gpus_per_node ]]; then
    # Cross-node: each DP rank spans multiple nodes
    num_nodes_per_dp=$((num_gpus_per_dp / gpus_per_node))
    node_array=($allocated_nodes)
    for ((i=0; i<dp_multiplicity; i++)); do
        server_node_idx=$((i * num_nodes_per_dp))
        server_node=${node_array[$server_node_idx]}
        server_url_array[$i]="${server_node}:${base_port}"
    done
else
    # Intra-node: multiple DP ranks per node, each on different port
    dp_per_node=$((gpus_per_node / num_gpus_per_dp))
    rank_idx=0
    for node_name in $allocated_nodes; do
        for ((j=0; j<dp_per_node; j++)); do
            if [[ $rank_idx -ge $dp_multiplicity ]]; then
                break 2
            fi
            port=$((base_port + j))
            server_url_array[$rank_idx]="${node_name}:${port}"
            rank_idx=$((rank_idx + 1))
        done
    done
fi

if [[ "$stage" == "server" || "$stage" == "all" ]]; then
    if [[ $num_gpus_per_dp -ge $gpus_per_node ]]; then
        num_nodes_per_dp=$((num_gpus_per_dp / gpus_per_node))
        node_array=($allocated_nodes)

        echo "Deployment: Cross-node ($num_nodes_per_dp nodes per DP, $gpus_per_node tasks/node)"
        echo "============================================"

        for ((i=0; i<dp_multiplicity; i++)); do
            server_url="${server_url_array[$i]}"

            # Calculate node list for this DP rank
            start_node_idx=$((i * num_nodes_per_dp))
            end_node_idx=$((start_node_idx + num_nodes_per_dp - 1))
            node_list=""
            for ((n=start_node_idx; n<=end_node_idx; n++)); do
                if [[ -z "$node_list" ]]; then
                    node_list="${node_array[$n]}"
                else
                    node_list="${node_list},${node_array[$n]}"
                fi
            done

            echo "Launching DP rank $i (URL: $server_url, Nodes: $node_list)..."
            server_args="$run_args --trtllm_server_urls=$server_url"
            export DP_RANK=$i
            run_srun \
                --overlap \
                --output="${HOST_LOG_DIR}/slurm_logs/run_llm_server.stdout" \
                --error="${HOST_LOG_DIR}/slurm_logs/run_llm_server.stderr" \
                --export=ALL,MLPERF_SCRATCH_PATH=/home/mlperf_inference_storage \
                --container-image="$container_image" \
                --container-mounts="${host_vol}:${container_vol},${mlperf_scratch_path}:/home/mlperf_inference_storage" \
                --container-workdir="$container_vol" \
                --container-remap-root \
                --nodelist="$node_list" \
                --ntasks-per-node="$gpus_per_node" \
                --nodes="$num_nodes_per_dp" \
                --mpi=pmix \
                $extra_srun_flags \
                make run_llm_server "RUN_ARGS=$server_args" "SYSTEM_NAME=$atomic_system" &
        done
    else
        echo "Deployment: Intra-node ($dp_per_node DP/node, $num_gpus_per_dp tasks/DP)"
        echo "============================================"

        # Launch all DP ranks in parallel with unique IPC addresses per rank
        for ((j=0; j<dp_per_node; j++)); do
            node_idx=0
            # Unique IPC port for each DP rank on the same node
            ipc_port=$((10012 + j))
            ipc_addr="tcp://127.0.0.1:${ipc_port}"
            for node_name in $allocated_nodes; do
                rank_idx=$((node_idx * dp_per_node + j))
                if [[ $rank_idx -ge $dp_multiplicity ]]; then
                    node_idx=$((node_idx + 1))
                    continue
                fi
                start_gpu=$((gpu_offset + j * num_gpus_per_dp))
                gpu_list=$(seq -s, $start_gpu $((start_gpu + num_gpus_per_dp - 1)))
                server_url="${server_url_array[$rank_idx]}"

                echo "Launching DP rank $rank_idx on $node_name (GPUs: $gpu_list, URL: $server_url, IPC: $ipc_addr)..."
                server_args="$run_args --trtllm_server_urls=$server_url"
                export DP_RANK=$rank_idx
                # Use --export to explicitly set NVIDIA_VISIBLE_DEVICES (for container GPU visibility) and IPC address
                export NVIDIA_VISIBLE_DEVICES="${gpu_list}"
                run_srun --overlap \
                    --output="${HOST_LOG_DIR}/slurm_logs/run_llm_server_${node_name}_gpus${gpu_list}.stdout" \
                    --error="${HOST_LOG_DIR}/slurm_logs/run_llm_server_${node_name}_gpus${gpu_list}.stderr" \
                    --export=ALL,NVIDIA_VISIBLE_DEVICES,TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR="$ipc_addr",MLPERF_SCRATCH_PATH=/home/mlperf_inference_storage \
                    --container-image="$container_image" \
                    --container-mounts="${host_vol}:${container_vol},${mlperf_scratch_path}:/home/mlperf_inference_storage" \
                    --container-workdir="$container_vol" \
                    --container-remap-root \
                    --nodes=1 \
                    --nodelist="$node_name" \
                    --ntasks-per-node="$num_gpus_per_dp" \
                    --mpi=pmix \
                    $extra_srun_flags \
                    make run_llm_server "RUN_ARGS=$server_args" "SYSTEM_NAME=$atomic_system" &

                node_idx=$((node_idx + 1))
            done
            # Optional sleep to allow servers to fully initialize
            if [ $server_spawn_time -gt 0 ] && [ "$dry_run" == "false" ]; then
                echo "Sleeping for ${server_spawn_time} seconds to avoid IPC port collision race condition. Increase --server-spawn-time if you see ZMQ errors."
                sleep "$server_spawn_time"
            fi
        done
    fi
    echo "All DP ranks launched in background. Check logs for details."
fi

if [[ "$stage" == "harness" || "$stage" == "all" ]]; then
    server_urls=$(IFS=,; echo "${server_url_array[*]}")

    [[ ! "$run_args" =~ --trtllm_server_urls= ]] && run_args="$run_args --trtllm_server_urls=$server_urls"

    echo "Server URLs: $server_urls"
    echo "============================================"
    echo "Launching harness (target: $harness_target)..."

    run_srun \
        --export=ALL,MLPERF_SCRATCH_PATH=/home/mlperf_inference_storage \
        --container-image="$container_image" \
        --container-mounts="${host_vol}:${container_vol},${mlperf_scratch_path}:/home/mlperf_inference_storage" \
        --container-workdir="$container_vol" \
        --container-remap-root \
        --nodes=1 \
        --overlap \
        $extra_srun_flags \
        make $harness_target "RUN_ARGS=$run_args" "SYSTEM_NAME=$harness_system"
fi
