import code.common.constants as C
import code.llmlib.fields as llm_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.harness as harness_fields
import os

os.environ['TRTLLM_SERVER_DISABLE_GC'] = '1'
os.environ['TRTLLM_WORKER_DISABLE_GC'] = '1'
os.environ['TRTLLM_ENABLE_PDL'] = '1'

ifb_config = {
    model_fields.gpu_batch_size: {
        'llama2-70b': 3072,
    },
    model_fields.input_dtype: 'int32',
    llm_fields.llm_gen_config_path: 'code/llama2-70b/tensorrt/generation_config.json',
    loadgen_fields.min_duration: 1800000,
    model_fields.precision: 'fp4',
    loadgen_fields.server_target_qps: 53,
    harness_fields.tensor_path: 'build/preprocessed_data/llama2-70b/',
    llm_fields.tensor_parallelism: 1,
    llm_fields.pipeline_parallelism: 1,
    llm_fields.trtllm_build_flags: {
        'max_beam_width': 1,
        'kv_cache_type': 'paged',
        'remove_input_padding': 'enable',
        'multiple_profiles': 'enable',
        'use_fused_mlp': 'enable',
        'context_fmha': 'enable',
        'max_num_tokens': 4608,
        'max_input_len': 1024,
        'max_seq_len': 2048,
        'use_fp8_context_fmha': 'enable',
        'use_paged_context_fmha': 'enable',
        'tokens_per_block': 32,
        'norm_quant_fusion': 'enable',
        'torch_compile_config': {
            'enable_fullgraph': True,
            'enable_inductor': False,
            'enable_piecewise_cuda_graph': True,
            'enable_userbuffers': True,
            'capture_num_tokens': [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4352, 4608],
        },
    },
    llm_fields.trtllm_checkpoint_flags: {
        'kv_cache_dtype': 'fp8',
    },
    llm_fields.trtllm_runtime_flags: {
        'exclude_input_from_output': True,
        'use_inflight_batching': True,
        'max_num_tokens': 4608,
        'batch_scheduler_policy': 'max_util',
        'context_chunking_policy': 'first_come_first_served',
        'kvcache_free_gpu_mem_frac': 0.95,
        'enable_chunked_context': True,
        'max_concurrency': 3072,
        'num_postprocess_workers': 4,
        'stream_interval': 50,
        'sampler_type': 'TRTLLMSampler',
    },
    llm_fields.traffic_distribution_policy: 'isl_load_balancing',
    harness_fields.use_graphs: False,
    llm_fields.use_token_latencies: True,
    harness_fields.enable_metrics: False,
    llm_fields.harness_use_hf_tokenizer: False,
    harness_fields.vboost_slider: 1,
}

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): ifb_config,
}

ATOMIC_EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        "default": ifb_config,
    },
}
