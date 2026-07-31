#!/bin/bash

# =============================================================================
# Llama2-70b Disaggregated Inference - SLURM Submission Script
#
# PURPOSE:
#   Submit disaggregated inference jobs to SLURM. This separates:
#   - CTX servers: Handle prefill (processing input prompts)
#   - GEN servers: Handle decode (generating output tokens)
#
# USAGE:
#   ./disagg_trtllm_llama2_70b/llama2_70b_disagg.sh [OPTIONS]
#   (Run --help for full options and examples)
#
# =============================================================================

repo_root=$(git rev-parse --show-toplevel)
cd $repo_root/closed/NVIDIA

# Load common utilities (colored logging, helpers)
source scripts/slurm_llm/disagg_trtllm_llama2_70b/common.sh

# =============================================================================
# HELP FUNCTION
# =============================================================================
show_help() {
    cat << 'EOF'
Llama2-70b Disaggregated Inference - SLURM Submission Script

USAGE:
    ./disagg_trtllm_llama2_70b/llama2_70b_disagg.sh [OPTIONS]

DESCRIPTION:
    Submit disaggregated inference jobs to SLURM. This separates:
    - CTX servers: Handle prefill (processing input prompts)
    - GEN servers: Handle decode (generating output tokens)

EXAMPLES:
    # Run with default settings (Interactive, QPS=80, 1 CTX, 2 GEN on GB200)
    ./disagg_trtllm_llama2_70b/llama2_70b_disagg.sh

    # Run with QPS=90 on GB300
    ./disagg_trtllm_llama2_70b/llama2_70b_disagg.sh --gpu GB300 --target-qps 90

    # Run with custom segment value (optional)
    ./disagg_trtllm_llama2_70b/llama2_70b_disagg.sh --segment 2 --target-qps 2900

    # Run with different server configuration
    ./disagg_trtllm_llama2_70b/llama2_70b_disagg.sh --ctx-num-servers 2 --gen-num-servers 4

    # Tune streaming intervals for lower latency
    ./disagg_trtllm_llama2_70b/llama2_70b_disagg.sh --ctx-stream 20 --gen-stream 50

OPTIONS:
    -h, --help                Show this help message

    GPU & System:
    --gpu TYPE                GPU type: GB200, GB300, B200 (default: GB200)
    --partition NAME          SLURM partition name (default: auto-detected from --gpu)
                              Examples: 36x2-a01r, gb200, 36ar, gb300, b200, b200-a01r
    --time HH:MM:SS           SLURM time limit (default: 02:00:00)
    --segment N               SLURM segment value (optional, no default)
    --container-image PATH    Container image path

    CTX Server (Prefill):
    --ctx-num-servers N       Number of CTX servers (default: 1)
    --ctx-tp-size N           Tensor parallelism (default: 1)
    --ctx-pp-size N           Pipeline parallelism (default: 1)
    --ctx-mbs N               Max batch size (default: 4096, auto-syncs --ctx-mnt)
    --ctx-mnt N               Max num tokens (default: 4096, auto-syncs with --ctx-mbs)
    --ctx-gpu-frac F          GPU memory fraction (default: 0.85)
    --ctx-stream N            Stream interval (default: 30, lower = more frequent)
    --ctx-nsys 0|1            Enable nsys profiling (default: 0)

    GEN Server (Decode):
    --gen-num-servers N       Number of GEN servers (default: 2)
    --gen-tp-size N           Tensor parallelism (default: 1)
    --gen-pp-size N           Pipeline parallelism (default: 1)
    --gen-mbs N               Max batch size (default: 768, auto-syncs --gen-mnt)
    --gen-mnt N               Max num tokens (default: 768, auto-syncs with --gen-mbs)
    --gen-gpu-frac F          GPU memory fraction (default: 0.95)
    --gen-stream N            Stream interval (default: 100, lower = more frequent)
    --gen-postprocess N       Postprocess workers per GEN server (default: 4)
    --gen-nsys 0|1            Enable nsys profiling (default: 0)

    Advanced:
    --cg-sizes "N,N,..."      CUDA graph batch sizes (default: auto-generated powers of 2)
                              Example: "1,2,4,8,16,32,64,128,256,512,768"
    --enable-iter-stats       Enable per-iteration perf stats collection (adds overhead, default: disabled)
    --cache-transceiver-mnt N KV cache transfer buffer size (default: 2048)
    --ct-mnt N                Short form of --cache-transceiver-mnt
                              Recommend: set to match or exceed --ctx-mnt for optimal performance
    --backend BACKEND         Cache transceiver backend: DEFAULT, UCX, NCCL (default: UCX)
    --cache-transceiver-backend  Long form of --backend

    Coordinator Server (Master):
    --num-servers N           Number of coordinator server instances (default: 1)
                              Multiple coordinators distribute harness load across CPU cores
    --server-postprocess N    Postprocess workers per coordinator (default: 4)
    --server-workers N        Worker threads per CPU core per coordinator (default: 2)

    Benchmark:
    --scenario MODE           MLPerf scenario: Interactive, Server, Offline (default: Interactive)
    --target-qps N            Target queries per second (default: 80)
    --num-reqs N              Number of requests (default: 8313)
    --warmup-iters N          Number of warmup iterations (default: 0, disabled)
    --test-mode MODE          Test mode: PerformanceOnly, AccuracyOnly (default: PerformanceOnly)
    --audit                   Run audit harness (run_audit_harness) instead of regular harness
    --exp-tag TAG             Custom experiment name (auto-generated if not set)
    --base-dir DIR            Base log directory (default: build/logs)
                              Final path: {base-dir}/{scenario}/{exp-tag}
    --log-dir DIR             Override full log directory (ignores --base-dir and scenario)
                              Final path: {log-dir}/{exp-tag}
    --dry-run                 Print sbatch command instead of executing it

HARDWARE CONFIGURATIONS:
    GB200:  Default Partition: 36x2-a01r,  4 GPUs/node
    GB300:  Default Partition: gb300,       4 GPUs/node
    B200:   Default Partition: b200,        4 GPUs/node
    Note: Override default partition with --partition flag

STREAMING INTERVALS:
    Lower values = more frequent token streaming = lower latency but higher overhead
    Higher values = less frequent streaming = higher latency but lower overhead
    
    CTX default (30): Frequent streaming to start decode phase ASAP
    GEN default (100): Balanced for decode phase overhead/latency

RESOURCE CALCULATION:
    Total GPUs = (ctx_servers × ctx_tp × ctx_pp) + (gen_servers × gen_tp × gen_pp)
    Nodes = ceil(Total GPUs / 4)

LOGS:
    Default: build/logs/{scenario}/{hardware}_ctx{N}_gen{M}_[servers{S}_]qps{Q}_ctxmbs{CMBS}_genmbs{GMBS}/
    With --base-dir: {base-dir}/{scenario}/{hardware}_ctx{N}_gen{M}_...
    With --log-dir: {log-dir}/{hardware}_ctx{N}_gen{M}_...
    
    Note: _servers{S} is only added when using multiple coordinator servers (--num-servers > 1)

EOF
    exit 0
}

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================
# These are the baseline settings optimized for Llama2-70b on GB200.
# All can be overridden via command-line arguments (see --help).

# CTX (Context/Prefill) server configuration
# - CTX servers handle the initial prompt processing (prefill phase)
# - They process all input tokens in parallel
_DEFAULT_CTX_NUM_SERVERS=1        # Number of CTX server instances
_DEFAULT_CTX_TP_SIZE=1            # Tensor Parallelism: splits model across GPUs (1=no split)
_DEFAULT_CTX_PP_SIZE=1            # Pipeline Parallelism: splits layers across GPUs (1=no split)
_DEFAULT_CTX_EP_SIZE=0            # Expert Parallelism: for MoE models (0=not used)
_DEFAULT_CTX_ENABLE_ADP="false"   # Attention Data Parallelism (advanced feature)
_DEFAULT_CTX_MBS=4096             # Max Batch Size: how many tokens can be processed together
_DEFAULT_CTX_MNT=4096             # Max Num Tokens: maximum total tokens in a batch
_DEFAULT_CTX_GPU_FRAC=0.85        # GPU Memory Fraction: 0.85 = use 85% of GPU memory
_DEFAULT_CTX_STREAM=30            # Stream interval: how often to stream tokens back (lower = more frequent)
_DEFAULT_CTX_NSYS=0               # Nsys profiling: 0=off, 1=on (for performance analysis)
_DEFAULT_CTX_TLLM_PROFILE_START_STOP="5000-5100"  # Profiling window (if enabled)

# GEN (Generation/Decode) server configuration
# - GEN servers handle token generation (decode phase)
# - They generate one token at a time, optimized for low latency
# - Usually need more GEN servers than CTX for balanced throughput
_DEFAULT_GEN_NUM_SERVERS=2        # Number of GEN server instances (2 is good starting point)
_DEFAULT_GEN_TP_SIZE=1            # Tensor Parallelism: splits model across GPUs
_DEFAULT_GEN_PP_SIZE=1            # Pipeline Parallelism: splits layers across GPUs
_DEFAULT_GEN_EP_SIZE=0            # Expert Parallelism: for MoE models (0=not used)
_DEFAULT_GEN_ENABLE_ADP="false"   # Attention Data Parallelism
_DEFAULT_GEN_MBS=768              # Max Batch Size: smaller than CTX for decode phase
_DEFAULT_GEN_MNT=1024             # Max Num Tokens: kept at 1024 for flexibility
_DEFAULT_GEN_GPU_FRAC=0.95        # GPU Memory Fraction: 0.95 = use 95% (more aggressive)
_DEFAULT_GEN_STREAM=100           # Stream interval: how often to stream tokens back (lower = more frequent)
_DEFAULT_GEN_NUM_POSTPROCESS_WORKERS=4  # Number of postprocess workers per GEN server
_DEFAULT_GEN_NSYS=0               # Nsys profiling: 0=off, 1=on

# Coordinator (master server) configuration
_DEFAULT_NUM_SERVER_INSTANCES=1            # Number of coordinator server instances
_DEFAULT_SERVER_NUM_POSTPROCESS_WORKERS=4  # Number of postprocess workers for coordinator
_DEFAULT_SERVER_WORKERS_PER_CORE=2         # Worker threads per CPU core for coordinator
_DEFAULT_GEN_TLLM_PROFILE_START_STOP="5000-5100"  # Profiling window

# MLPerf benchmark configuration
# - Scenario: How queries are submitted (Interactive/Server/Offline)
# - QPS: Queries per second (throughput target)
# - Requests: Total number of queries to run
_DEFAULT_MTP_SIZE=0                # MTP (advanced feature, usually 0)
_DEFAULT_EPLB_NUM_SLOTS=0          # Expert Load Balancer slots (MoE only)
_DEFAULT_NUM_REQS=8313             # Number of requests (8313 is MLPerf standard for llama2-70b)
_DEFAULT_SCENARIO="Interactive"    # Interactive: low latency | Server: target QPS | Offline: max throughput
_DEFAULT_TARGET_QPS=80             # Target queries/second (80 is a good starting point)
_DEFAULT_WARMUP_ITERS=0            # Warmup iterations: 0=disabled, 10-100=typical values
_DEFAULT_TEST_MODE="PerformanceOnly"  # Test mode: PerformanceOnly or AccuracyOnly
_DEFAULT_AUDIT="false"             # Audit mode: run_audit_harness instead of run_harness
_DEFAULT_BASE_DIR="build/logs"     # Base directory for logs

# Container and system paths
# - Update these paths if your environment is different
_DEFAULT_CONTAINER_IMAGE=""  # Set via --container-image parameter or environment variable
_DEFAULT_GPU="GB200"  # Options: GB200, GB300, B200
_DEFAULT_TIME="02:00:00"
_DEFAULT_SEGMENT=""  # SLURM segment value (only set if passed)
_DEFAULT_TRTLLM_INSTALL_PATH=""
_DEFAULT_EXTRA_CONTAINER_MOUNTS=""
_DEFAULT_CG_SIZES=""
_DEFAULT_ENABLE_ITER_STATS="false"
_DEFAULT_CACHE_TRANSCEIVER_MAX_NUM_TOKENS=2048  # KV cache transfer buffer size
_DEFAULT_CACHE_TRANSCEIVER_BACKEND="UCX"    # KV cache transfer backend (DEFAULT, UCX, NCCL)
_DEFAULT_DRY_RUN="false"

# Parse command line arguments (optional overrides)
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) show_help ;;
        --exp-tag) exp_tag="$2"; shift 2 ;;
        --base-dir) base_dir="$2"; shift 2 ;;
        --log-dir) log_dir="$2"; shift 2 ;;
        --gpu) gpu="$2"; shift 2 ;;
        --partition) partition="$2"; shift 2 ;;
        --time) time="$2"; shift 2 ;;
        --segment) segment="$2"; shift 2 ;;
        --container-image) CONTAINER_IMAGE="$2"; shift 2 ;;
        --ctx-num-servers) num_ctx_servers="$2"; shift 2 ;;
        --ctx-tp-size) ctx_tp_size="$2"; shift 2 ;;
        --ctx-pp-size) ctx_pp_size="$2"; shift 2 ;;
        --ctx-mbs|--ctx-max-batch-size) ctx_max_batch_size="$2"; shift 2 ;;
        --ctx-mnt|--ctx-max-num-tokens) ctx_max_num_tokens="$2"; shift 2 ;;
        --ctx-gpu-frac) ctx_gpu_frac="$2"; shift 2 ;;
        --ctx-stream|--ctx-stream-interval) ctx_stream_interval="$2"; shift 2 ;;
        --ctx-nsys) ctx_nsys="$2"; shift 2 ;;
        --gen-num-servers) num_gen_servers="$2"; shift 2 ;;
        --gen-tp-size) gen_tp_size="$2"; shift 2 ;;
        --gen-pp-size) gen_pp_size="$2"; shift 2 ;;
        --gen-mbs|--gen-max-batch-size) gen_max_batch_size="$2"; shift 2 ;;
        --gen-mnt|--gen-max-num-tokens) gen_max_num_tokens="$2"; shift 2 ;;
        --gen-gpu-frac) gen_gpu_frac="$2"; shift 2 ;;
        --gen-stream|--gen-stream-interval) gen_stream_interval="$2"; shift 2 ;;
        --gen-postprocess|--gen-postprocess-workers) gen_num_postprocess_workers="$2"; shift 2 ;;
        --gen-nsys) gen_nsys="$2"; shift 2 ;;
        --cg-sizes) cg_sizes="$2"; shift 2 ;;
        --enable-iter-stats) enable_iter_stats="true"; shift 1 ;;
        --cache-transceiver-mnt|--ct-mnt) cache_transceiver_max_num_tokens="$2"; shift 2 ;;
        --backend|--cache-transceiver-backend) cache_transceiver_backend="$2"; shift 2 ;;
        --num-servers|--num-server-instances) num_server_instances="$2"; shift 2 ;;
        --server-postprocess|--server-postprocess-workers) server_num_postprocess_workers="$2"; shift 2 ;;
        --server-workers|--server-workers-per-core) server_workers_per_core="$2"; shift 2 ;;
        --target-qps) target_qps="$2"; shift 2 ;;
        --num-reqs) num_reqs="$2"; shift 2 ;;
        --warmup-iters|--warmup-iterations) warmup_iters="$2"; shift 2 ;;
        --scenario) scenario="$2"; shift 2 ;;
        --test-mode) test_mode="$2"; shift 2 ;;
        --audit) audit="true"; shift 1 ;;
        --log-dir|--output-dir) log_dir="$2"; shift 2 ;;
        --dry-run) dry_run="true"; shift 1 ;;
        *) log_error "Unknown argument: $1"; echo "Use -h or --help for usage information"; exit 1 ;;
    esac
done

# Apply defaults
export num_ctx_servers=${num_ctx_servers:=$_DEFAULT_CTX_NUM_SERVERS}
export ctx_tp_size=${ctx_tp_size:=$_DEFAULT_CTX_TP_SIZE}
export ctx_pp_size=${ctx_pp_size:=$_DEFAULT_CTX_PP_SIZE}
export ctx_ep_size=${ctx_ep_size:=$_DEFAULT_CTX_EP_SIZE}
export ctx_enable_adp=${ctx_enable_adp:=$_DEFAULT_CTX_ENABLE_ADP}
export ctx_max_batch_size=${ctx_max_batch_size:=$_DEFAULT_CTX_MBS}
export ctx_max_num_tokens=${ctx_max_num_tokens:=$_DEFAULT_CTX_MNT}
export ctx_gpu_frac=${ctx_gpu_frac:=$_DEFAULT_CTX_GPU_FRAC}
export ctx_stream_interval=${ctx_stream_interval:=$_DEFAULT_CTX_STREAM}

export num_gen_servers=${num_gen_servers:=$_DEFAULT_GEN_NUM_SERVERS}
export gen_tp_size=${gen_tp_size:=$_DEFAULT_GEN_TP_SIZE}
export gen_pp_size=${gen_pp_size:=$_DEFAULT_GEN_PP_SIZE}
export gen_ep_size=${gen_ep_size:=$_DEFAULT_GEN_EP_SIZE}
export gen_enable_adp=${gen_enable_adp:=$_DEFAULT_GEN_ENABLE_ADP}
export gen_max_batch_size=${gen_max_batch_size:=$_DEFAULT_GEN_MBS}
export gen_max_num_tokens=${gen_max_num_tokens:=$_DEFAULT_GEN_MNT}
export gen_gpu_frac=${gen_gpu_frac:=$_DEFAULT_GEN_GPU_FRAC}
export gen_stream_interval=${gen_stream_interval:=$_DEFAULT_GEN_STREAM}
export gen_num_postprocess_workers=${gen_num_postprocess_workers:=$_DEFAULT_GEN_NUM_POSTPROCESS_WORKERS}

export mtp_size=${mtp_size:=$_DEFAULT_MTP_SIZE}
export eplb_num_slots=${eplb_num_slots:=$_DEFAULT_EPLB_NUM_SLOTS}

export num_reqs=${num_reqs:=$_DEFAULT_NUM_REQS}
export scenario="${scenario:=$_DEFAULT_SCENARIO}"
export target_qps=${target_qps:=$_DEFAULT_TARGET_QPS}
export warmup_iters=${warmup_iters:=$_DEFAULT_WARMUP_ITERS}
export test_mode="${test_mode:=$_DEFAULT_TEST_MODE}"
export audit="${audit:=$_DEFAULT_AUDIT}"

export ctx_nsys=${ctx_nsys:=$_DEFAULT_CTX_NSYS}
export gen_nsys=${gen_nsys:=$_DEFAULT_GEN_NSYS}
export CTX_TLLM_PROFILE_START_STOP="${CTX_TLLM_PROFILE_START_STOP:=$_DEFAULT_CTX_TLLM_PROFILE_START_STOP}"
export GEN_TLLM_PROFILE_START_STOP="${GEN_TLLM_PROFILE_START_STOP:=$_DEFAULT_GEN_TLLM_PROFILE_START_STOP}"

export num_server_instances=${num_server_instances:=$_DEFAULT_NUM_SERVER_INSTANCES}
export server_num_postprocess_workers=${server_num_postprocess_workers:=$_DEFAULT_SERVER_NUM_POSTPROCESS_WORKERS}
export server_workers_per_core=${server_workers_per_core:=$_DEFAULT_SERVER_WORKERS_PER_CORE}

export CONTAINER_IMAGE="${CONTAINER_IMAGE:=$_DEFAULT_CONTAINER_IMAGE}"
export gpu="${gpu:=$_DEFAULT_GPU}"
export time=${time:=$_DEFAULT_TIME}
export segment=${segment:=$_DEFAULT_SEGMENT}
export trtllm_install_path=${trtllm_install_path:=$_DEFAULT_TRTLLM_INSTALL_PATH}
export extra_container_mounts=${extra_container_mounts:=$_DEFAULT_EXTRA_CONTAINER_MOUNTS}
export cg_sizes=${cg_sizes:=$_DEFAULT_CG_SIZES}
export enable_iter_stats=${enable_iter_stats:=$_DEFAULT_ENABLE_ITER_STATS}
export cache_transceiver_max_num_tokens=${cache_transceiver_max_num_tokens:=$_DEFAULT_CACHE_TRANSCEIVER_MAX_NUM_TOKENS}
export cache_transceiver_backend=${cache_transceiver_backend:=$_DEFAULT_CACHE_TRANSCEIVER_BACKEND}
export dry_run=${dry_run:=$_DEFAULT_DRY_RUN}

# Note: MBS and MNT are independent - no auto-syncing
# CTX MNT default: 4096
# GEN MNT default: 1024 (kept at 1024 unless explicitly specified)

# Map GPU type to partition (if not already set), hardware name, and system name base
case "${gpu^^}" in  # Convert to uppercase for case-insensitive matching
    GB200)
        partition="${partition:-36x2-a01r}"  # Default: 36x2-a01r, but can override with --partition
        hardware="GB200"
        system_name_base="GB200-NVL72_GB200-186GB_aarch64"
        ;;
    GB300)
        partition="${partition:-gb300}"
        hardware="GB300"
        system_name_base="GB300-NVL72_GB300-288GB_aarch64"
        ;;
    B200)
        partition="${partition:-b200}"  # Default: b200, but can override with --partition
        hardware="B200"
        system_name_base="B200-NVL72_B200-96GB_aarch64"
        ;;
    *)
        log_error "Invalid GPU type: $gpu"
        log_info "Valid options: GB200, GB300, B200"
        exit 1
        ;;
esac

# Calculate total servers and build system names
total_servers=$((num_ctx_servers + num_gen_servers))
export SYSTEM_NAME_X1="${system_name_base}x1"
export SYSTEM_NAME_X72="${system_name_base}x${total_servers}"

# Build experiment directory structure
# Priority: --log-dir > --base-dir/{scenario} > default (build/logs/{scenario})
if [ -n "${log_dir:-}" ]; then
    # --log-dir overrides everything (full path control)
    export log_base_dir="${log_dir}"
else
    # Use --base-dir or default, then append scenario
    base_dir="${base_dir:-$_DEFAULT_BASE_DIR}"
    export log_base_dir="${base_dir}/${scenario}"
fi

# Build exp_tag with optional _serversN suffix if multiple coordinators
if [ "$num_server_instances" -gt 1 ]; then
    export exp_tag="${exp_tag:-${hardware}_ctx${num_ctx_servers}_gen${num_gen_servers}_servers${num_server_instances}_qps${target_qps}_ctxmbs${ctx_max_batch_size}_genmbs${gen_max_batch_size}}"
else
    export exp_tag="${exp_tag:-${hardware}_ctx${num_ctx_servers}_gen${num_gen_servers}_qps${target_qps}_ctxmbs${ctx_max_batch_size}_genmbs${gen_max_batch_size}}"
fi
mkdir -p ${log_base_dir}

# Calculate resource requirements
gpus_per_node=4  # GB200/GB300 systems have 4 GPUs per node

ctx_total_gpus=$((num_ctx_servers * ctx_tp_size * ctx_pp_size))
gen_total_gpus=$((num_gen_servers * gen_tp_size * gen_pp_size))
total_gpus=$((ctx_total_gpus + gen_total_gpus))

# Calculate total nodes needed (all GPUs packed efficiently)
# Harness runs on an existing GEN node (no extra node needed)
server_nodes=$(( (total_gpus + gpus_per_node - 1) / gpus_per_node ))
total_nodes=$(( server_nodes ))  # Harness shares a GEN node without coordinator

# Job name (following required format: coreai_mlperf_inference-<subproject>.<details>)
job_name="coreai_mlperf_inference-llama2_70b.${exp_tag}"

log_section "Submitting Llama2-70b Disaggregated Job"
echo ""
log_info "Experiment: $exp_tag"
log_info "Scenario: $scenario | Target QPS: $target_qps | Requests: $num_reqs | Warmup: $warmup_iters"
if [ "$audit" = "true" ]; then
    log_detail "⚠ AUDIT MODE ENABLED - Using run_audit_harness"
fi
if [ -n "$cg_sizes" ]; then
    log_detail "Chunked prefill enabled: $cg_sizes"
fi
echo ""

log_info "CTX (Prefill) Workers: $num_ctx_servers"
log_detail "Parallelism: TP=$ctx_tp_size × PP=$ctx_pp_size = ${ctx_total_gpus} GPUs"
log_detail "Batch: ${ctx_max_batch_size} tokens/batch, Max: ${ctx_max_num_tokens} tokens"
log_detail "Memory: ${ctx_gpu_frac}, Stream: ${ctx_stream_interval} (lower=frequent, higher=less overhead)"
if [ "$ctx_nsys" != "0" ]; then
    log_detail "⚠ Nsys profiling ENABLED (iterations: $ctx_nsys)"
fi
echo ""

log_info "GEN (Decode) Workers: $num_gen_servers"
log_detail "Parallelism: TP=$gen_tp_size × PP=$gen_pp_size = ${gen_total_gpus} GPUs"
log_detail "Batch: ${gen_max_batch_size} tokens/batch, Max: ${gen_max_num_tokens} tokens"
log_detail "Memory: ${gen_gpu_frac}, Stream: ${gen_stream_interval}, Postprocess workers: ${gen_num_postprocess_workers}"
if [ "$gen_nsys" != "0" ]; then
    log_detail "⚠ Nsys profiling ENABLED (iterations: $gen_nsys)"
fi
echo ""

log_info "Coordinators: $num_server_instances (distributes harness load across CPU cores)"
log_detail "CPU workers: ${server_num_postprocess_workers} postprocess, ${server_workers_per_core} workers/core"
log_detail "Placement: Prioritize CTX nodes, then GEN nodes (1 coordinator per node)"
echo ""

# Calculate node distribution
gpus_per_node=4  # Assuming 4 GPUs per node for GB200/GB300
gen_nodes_needed=$(( (gen_total_gpus + gpus_per_node - 1) / gpus_per_node ))
ctx_nodes_needed=$(( (ctx_total_gpus + gpus_per_node - 1) / gpus_per_node ))
num_coordinators_on_ctx=$((ctx_nodes_needed < num_server_instances ? ctx_nodes_needed : num_server_instances))
num_coordinators_on_gen=$((num_server_instances - num_coordinators_on_ctx))

log_info "Node Allocation (${total_nodes} nodes total):"
log_detail "GEN nodes:   ${gen_nodes_needed} nodes (nodes 0-$((gen_nodes_needed-1)))"
log_detail "CTX nodes:   ${ctx_nodes_needed} nodes (nodes ${gen_nodes_needed}-$((gen_nodes_needed+ctx_nodes_needed-1)))"
log_detail "Coordinators: ${num_coordinators_on_ctx} on CTX nodes + ${num_coordinators_on_gen} on GEN nodes"
log_detail "Harness:     1 node (dedicated GEN node without coordinator)"
echo ""

log_info "Port Allocation:"
log_detail "Coordinators: 8300-$((8300+num_server_instances-1)) (${num_server_instances} ports)"
log_detail "GEN workers:  8336-$((8336+num_gen_servers-1)) (${num_gen_servers} ports)"
log_detail "CTX workers:  $((8336+num_gen_servers))-$((8336+num_gen_servers+num_ctx_servers-1)) (${num_ctx_servers} ports)"
echo ""

log_info "Optimization Flags:"
log_detail "TRT-LLM: GC disabled, PDL enabled, NIXL disabled, CUDA queues=4x"
log_detail "UCX: IPC MNNVL disabled, RNDV scheme=get_zcopy"
log_detail "NUMA: Aware worker affinity enabled"
log_detail "Cache transceiver: backend=${cache_transceiver_backend}, max_tokens=${cache_transceiver_max_num_tokens}"
if [ "$enable_iter_stats" = "true" ]; then
    log_detail "⚠ Iteration stats ENABLED (verbose logging)"
fi
echo ""

log_info "SLURM Configuration:"
log_detail "Partition: $partition"
if [ -n "$segment" ]; then
    log_detail "Segment: $segment"
fi
log_detail "Time limit: $time"
log_detail "Container: $(basename $CONTAINER_IMAGE)"
log_detail "Output dir: ${log_base_dir}/"
echo ""

# Submit the job - build segment argument conditionally
SEGMENT_ARG=""
if [ -n "$segment" ]; then
    SEGMENT_ARG="--segment ${segment}"
fi

SBATCH_CMD="sbatch \
    --nodes=${total_nodes} \
    ${SEGMENT_ARG} \
    --comment \"MLPerf llama2-70b disaggregated inference\" \
    --partition=${partition} \
    --time=${time} \
    --account=coreai_mlperf_inference \
    --job-name=${job_name} \
    --output=${log_base_dir}/slurm-%j-${exp_tag}.txt \
    --propagate=RLIMIT_NOFILE \
    --export=ALL,log_base_dir,audit,SYSTEM_NAME_X1,SYSTEM_NAME_X72 \
    ${PWD}/scripts/slurm_llm/disagg_trtllm_llama2_70b/benchmark_llama2_70b.sh"

if [ "$dry_run" = "true" ]; then
    # Create log directory and save command for dry-run
    mkdir -p "${log_base_dir}/${exp_tag}"
    DRY_RUN_LOG="${log_base_dir}/${exp_tag}/dry_run_command.txt"
    
    echo "$SBATCH_CMD" > "$DRY_RUN_LOG"
    
    log_info "DRY RUN MODE - Simulating execution without submitting to SLURM"
    echo ""
    log_info "Would execute sbatch command:"
    echo "$SBATCH_CMD"
    echo ""
    log_success "Command saved to:"
    log_detail "$DRY_RUN_LOG"
    echo ""
    log_info "Now showing commands that benchmark script would execute:"
    echo "=============================================================="
    echo ""
    
    # Run benchmark script in dry-run mode to show all commands
    export DRY_RUN=1
    bash scripts/slurm_llm/disagg_trtllm_llama2_70b/benchmark_llama2_70b.sh
    
    echo ""
    echo "=============================================================="
    log_info "DRY RUN COMPLETE - No job was submitted"
    log_info "Logs would appear in:"
    log_detail "${log_base_dir}/${exp_tag}/"
    exit 0
fi

eval "$SBATCH_CMD"

log_success "Job submitted!"
echo ""
log_info "Monitor status:"
log_command "squeue -u $USER"
echo ""
log_info "Logs will appear in:"
log_detail "${log_base_dir}/${exp_tag}/"
echo ""

