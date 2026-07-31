import code.common.constants as C
import code.llmlib.fields as llm_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.harness as harness_fields

import os
os.environ['TRTLLM_SERVER_DISABLE_GC'] = '1' #Server 
os.environ['OMPI_MCA_coll_ucc_enable'] = '0' #Hitting "Deserialization failed: invalid load key" 
os.environ['TLLM_NUMA_AWARE_WORKER_AFFINITY'] = '1' #perf improved from 180->187tps/gpu
os.environ['CUDA_SCALE_LAUNCH_QUEUES'] = '4x'  #perf improved from 152->177 along with cuda graph and mnt1024 changes
os.environ['TRTLLM_GEMM_ALLREDUCE_FUSION_ENABLED']= '1'
os.environ['TRTLLM_LLAMA_EAGER_FUSION_DISABLED']= '1'
os.environ['TRTLLM_DISABLE_NVFP4_LAYERNORM_FUSION']= '0'

harness_config = {
    model_fields.input_dtype: 'int32',
    llm_fields.llm_gen_config_path: 'code/llama3_1-405b/tensorrt/generation_config.json',
    loadgen_fields.min_duration: 600000,
    loadgen_fields.min_query_count: 8313,  
    loadgen_fields.server_target_qps: 2.3,
    model_fields.precision: 'fp4',
    harness_fields.tensor_path: 'build/preprocessed_data/llama3.1-405b/',
    llm_fields.traffic_distribution_policy: 'isl_load_balancing',
    llm_fields.trtllm_checkpoint_flags: {
        'kv_cache_dtype': 'fp8',
    },
    llm_fields.trtllm_runtime_flags: {
        'workers_per_core': 1,
        'max_concurrency': 2048,
    },
    llm_fields.use_token_latencies: True,
    llm_fields.readiness_timeout: 900,
}


EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): harness_config,
}

# Atomic configs for leader mode with multiple variants
# Usage: --mpi_mode=leader --config_id=<variant_name>
ATOMIC_EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        "default": harness_config,
    },
}
