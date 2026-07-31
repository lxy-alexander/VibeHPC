import code.common.constants as C
import code.llmlib.fields as llm_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.harness as harness_fields


EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        model_fields.gpu_batch_size: {
            'llama2-70b': 768,
        },
        model_fields.input_dtype: 'int32',
        llm_fields.llm_gen_config_path: 'code/llama2-70b/tensorrt/generation_config.json',
        loadgen_fields.min_duration: 1200000,
        model_fields.precision: 'fp4',
        loadgen_fields.server_target_qps: 220,
        harness_fields.tensor_path: 'build/preprocessed_data/llama2-70b/',
        llm_fields.trtllm_build_flags: {
            'max_beam_width': 1,
            'kv_cache_type': 'paged',
            'remove_input_padding': 'enable',
            'multiple_profiles': 'enable',
            'use_fused_mlp': 'enable',
            'context_fmha': 'enable',
            'max_num_tokens': 768,
            'max_input_len': 1024,
            'max_seq_len': 2048,
            'use_fp8_context_fmha': 'enable',
            'use_paged_context_fmha': 'enable',
            'tokens_per_block': 32,
            'gemm_plugin': 'fp4',
        },
        llm_fields.tensor_parallelism: 1,
        llm_fields.pipeline_parallelism: 1,
        llm_fields.trtllm_runtime_flags: {
            'exclude_input_from_output': True,
            'use_inflight_batching': True,
            'max_num_tokens': 768,
            'batch_scheduler_policy': 'max_util',
            'context_chunking_policy': 'first_come_first_served',
            'kvcache_free_gpu_mem_frac': 0.95,
            'enable_chunked_context': True,
        },
        harness_fields.use_graphs: False,
        llm_fields.use_token_latencies: True,
    },
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.999), C.PowerSetting.MaxP): {
        model_fields.gpu_batch_size: {
            'llama2-70b': 768,
        },
        model_fields.input_dtype: 'int32',
        llm_fields.llm_gen_config_path: 'code/llama2-70b/tensorrt/generation_config.json',
        loadgen_fields.min_duration: 1200000,
        model_fields.precision: 'fp4',
        loadgen_fields.server_target_qps: 220,
        harness_fields.tensor_path: 'build/preprocessed_data/llama2-70b/',
        llm_fields.trtllm_build_flags: {
            'max_beam_width': 1,
            'kv_cache_type': 'paged',
            'remove_input_padding': 'enable',
            'multiple_profiles': 'enable',
            'use_fused_mlp': 'enable',
            'context_fmha': 'enable',
            'max_num_tokens': 768,
            'max_input_len': 1024,
            'max_seq_len': 2048,
            'use_fp8_context_fmha': 'enable',
            'use_paged_context_fmha': 'enable',
            'tokens_per_block': 32,
            'gemm_plugin': 'fp4',
        },
        llm_fields.tensor_parallelism: 1,
        llm_fields.pipeline_parallelism: 1,
        llm_fields.trtllm_runtime_flags: {
            'exclude_input_from_output': True,
            'use_inflight_batching': True,
            'max_num_tokens': 768,
            'batch_scheduler_policy': 'max_util',
            'context_chunking_policy': 'first_come_first_served',
            'kvcache_free_gpu_mem_frac': 0.95,
            'enable_chunked_context': True,
        },
        harness_fields.use_graphs: False,
        llm_fields.use_token_latencies: True,
    },
}
