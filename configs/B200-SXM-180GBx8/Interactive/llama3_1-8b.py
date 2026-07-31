import code.common.constants as C
import code.llmlib.fields as llm_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.harness as harness_fields
import code.fields.triton as triton_fields


base = {
    # Data paths. You should not need to change this unless you know what you are doing.
    llm_fields.llm_gen_config_path: 'code/llama3_1-8b/tensorrt/generation_config.json',
    harness_fields.tensor_path: 'build/preprocessed_data/llama3.1-8b/',

    # Length limits and beam width are set by MLCommons rules and should not be changed.
    loadgen_fields.min_duration: 600000,
    harness_fields.use_graphs: True,
    llm_fields.use_token_latencies: True,
    llm_fields.trtllm_build_flags: {
        'max_beam_width': 1,
        'kv_cache_type': 'paged',
        'remove_input_padding': 'enable',
        'multiple_profiles': 'enable',
        'use_fused_mlp': 'enable',
        'context_fmha': 'enable',
        'max_num_tokens': 12000,
        'max_input_len': 2540,
        'max_seq_len': 2668,
        'use_fp8_context_fmha': 'enable',
        'use_paged_context_fmha': 'enable',
        'tokens_per_block': 32,
    },
    llm_fields.trtllm_runtime_flags: {
        'exclude_input_from_output': True,
        'use_inflight_batching': True,
        'max_num_tokens': 12000,
        'batch_scheduler_policy': 'max_util',
        'context_chunking_policy': 'first_come_first_served',
        'kvcache_free_gpu_mem_frac': 0.9,  # Progressively lower by 0.1/0.05 if you hit OOM errors.
        'enable_chunked_context': True,
        'cuda_graph_padding_enabled': True,
        'workers_per_core': 12,
        'max_concurrency': -1,
        'disable_gc': True,
    },

    # Precision fields. You should not need to change these.
    llm_fields.trtllm_checkpoint_flags: {
        'kv_cache_dtype': 'fp8',
    },
    model_fields.precision: 'fp4',
    model_fields.input_dtype: 'int32',

    # Tune me! If you hit an OOM, decrease the batch size.
    model_fields.gpu_batch_size: {
        'llama3_1-8b': 6000,
    },
    loadgen_fields.server_target_qps: 800,

    # You can try increasing these if you have multiple GPUs.
    llm_fields.tensor_parallelism: 1,
    llm_fields.pipeline_parallelism: 1,

    # Only supported on Hopper GPUs. On other GPUs, this will not do anything.
    harness_fields.vboost_slider: 1,
}


EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): base
}
