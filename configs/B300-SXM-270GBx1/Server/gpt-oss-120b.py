import code.common.constants as C
import code.llmlib.fields as llm_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.harness as harness_fields

import os
os.environ['TRTLLM_MOE_ENABLE_ALLTOALL_WITHOUT_ALLGATHER'] = '1'
os.environ['MLPINF_HTTP_USE_COMPLETIONS'] = '1'
os.environ['TRTLLM_SERVER_DISABLE_GC'] = '1'
os.environ['TLLM_PROFILE_START_STOP'] = '12000-12100'

# Base config (PerformanceOnly)
# Based on GB300-NVL72_GB300-288GB_aarch64x1 Server (1-GPU config, no TP)
# Scaled for 8 GPUs with power adjustment
ifb_config = {
    llm_fields.llm_gen_config_path: 'code/gpt-oss-120b/tensorrt/generation_config_performance.json',
    harness_fields.tensor_path: 'build/data/gpt-oss/v4/perf',

    loadgen_fields.min_duration: 600000,
    loadgen_fields.min_query_count: 6396 * 4,
    llm_fields.warmup_iterations: 0,
    llm_fields.use_token_latencies: True,
    llm_fields.trtllm_build_flags: {
        'max_beam_width': 1,
        'remove_input_padding': 'enable',
        'max_num_tokens': 4638 * 3,
        'max_input_len': 15536,
        'max_seq_len': 15536 + 10240,  # 25776
        'enable_attention_dp': False,
        'torch_compile_config': {
            'enable_fullgraph': True,
            'enable_piecewise_cuda_graph': True,
            'enable_userbuffers': False,
            'capture_num_tokens': [512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4352, 4608, 4864, 5120, 5376, 5632, 5888, 6144, 6400, 6656, 6912, 7168, 7424, 7680, 7936, 8192, 8704, 9216, 9728, 10240, 11264, 12288, 13312, 13914],
        },
    },
    llm_fields.trtllm_runtime_flags: {
        'exclude_input_from_output': True,
        'use_inflight_batching': True,
        'max_num_tokens': 4638 * 3,
        'batch_scheduler_policy': 'max_util',
        'context_chunking_policy': 'first_come_first_served',
        'kvcache_free_gpu_mem_frac': 0.98,
        'enable_chunked_context': True,
        'max_concurrency': 512 + 1024,
        'cuda_graph_batch_sizes': [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896],
        'cuda_graph_padding_enabled': True,
        'moe_backend': 'TRTLLM',
        "adp_balancing_enable": False,
        "adp_balancing_batching_wait_iters": 10,
        "adp_balancing_timeout_iters": 500,
        'num_postprocess_workers': 4,
        "stream_interval": 20,
        "sampler_type": "TRTLLMSampler",
    },
    harness_fields.use_graphs: True,

    llm_fields.trtllm_checkpoint_flags: {
        'kv_cache_dtype': 'fp8',
    },
    model_fields.precision: 'fp4',
    model_fields.input_dtype: 'int32',

    model_fields.gpu_batch_size: {
        'gpt-oss-120b': 128 * 7,
    },
    loadgen_fields.server_target_qps: 10.6,

    # 1-GPU config (no TP) - 8 independent instances
    llm_fields.tensor_parallelism: 1,
    llm_fields.pipeline_parallelism: 1,
    llm_fields.moe_expert_parallelism: 1,

    harness_fields.vboost_slider: 1,
}


EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): ifb_config,
}

# Accuracy-specific overrides (deep merged when test_mode=AccuracyOnly)
ACCURACY_OVERRIDES = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        llm_fields.llm_gen_config_path: 'code/gpt-oss-120b/tensorrt/generation_config_accuracy.json',
        harness_fields.tensor_path: 'build/data/gpt-oss/v4/acc',
        loadgen_fields.min_query_count: 4395,
        llm_fields.trtllm_build_flags: {
            'max_input_len': 3072,
            'max_seq_len': 3072 + 32768,  # 35840
        },
    },
}

# Compliance test overrides (keyed by AuditTest)
COMPLIANCE_OVERRIDES = {
    C.AuditTest.TEST07: {
        C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
            llm_fields.llm_gen_config_path: 'code/gpt-oss-120b/tensorrt/generation_config_performance.json',
            harness_fields.tensor_path: 'build/data/gpt-oss/v4/compliance/test07',
            loadgen_fields.min_query_count: 990,
        },
    },
    C.AuditTest.TEST09: {
        C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
            loadgen_fields.min_query_count: 6396,
            loadgen_fields.min_duration: 0,
        },
    },
}

ATOMIC_EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        "default": ifb_config,
    },
}
