import code.common.constants as C
import code.llmlib.fields as llm_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.harness as harness_fields
import os

os.environ['TRTLLM_SERVER_DISABLE_GC'] = '1'
os.environ['TRTLLM_WORKER_DISABLE_GC'] = '1'
os.environ['TRTLLM_ENABLE_PDL'] = '1'

harness_config = {
    llm_fields.llm_gen_config_path: 'code/llama2-70b/tensorrt/generation_config.json',
    harness_fields.tensor_path: 'build/preprocessed_data/llama2-70b/',
    loadgen_fields.min_duration: 1800000,
    loadgen_fields.server_target_qps: 400,
    llm_fields.traffic_distribution_policy: 'isl_load_balancing',
    llm_fields.trtllm_runtime_flags: {
        'max_concurrency': 3072,
        'sampler_type': 'TRTLLMSampler',
        'num_postprocess_workers': 4,
    },
    llm_fields.trtllm_checkpoint_flags: {
        'kv_cache_dtype': 'fp8',
    },
    model_fields.precision: 'fp4',
    model_fields.input_dtype: 'int32',
    harness_fields.use_graphs: False,
    harness_fields.enable_metrics: False,
    llm_fields.harness_use_hf_tokenizer: False,
    harness_fields.vboost_slider: 1,
}

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): harness_config,
}

ATOMIC_EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        "default": harness_config,
    },
}
