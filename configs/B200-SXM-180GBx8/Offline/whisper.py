import code.common.constants as C
import code.llmlib.fields as llm_fields
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields
import code.fields.harness as harness_fields
import code.fields.power as power_fields
from importlib import import_module
from nvmitten.constants import Precision
whisper_fields = import_module("code.whisper.tensorrt.fields")

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        model_fields.gpu_batch_size: {
            'encoder': 320,
            'decoder': 320,
        },
        model_fields.input_dtype: Precision.FP32,
        llm_fields.llm_gen_config_path: 'code/whisper/tensorrt/generation_config.json',
        loadgen_fields.offline_expected_qps: 770,
        model_fields.precision: Precision.FP16,
        harness_fields.tensor_path: 'build/preprocessed_data/whisper-large-v3/',
        llm_fields.tensor_parallelism: 1,
        llm_fields.pipeline_parallelism: 1,
        whisper_fields.whisper_encoder_build_flags: {
            'max_beam_width': 1,
            'max_batch_size': 320,
            'kv_cache_type': 'paged',
            'remove_input_padding': 'enable',
            'moe_plugin': 'disable',
            'gemm_plugin': 'disable',
            'max_input_len': 3000,
            'max_seq_len': 3000,
            'bert_attention_plugin': 'float16',
        },
        whisper_fields.whisper_decoder_build_flags: {
            'max_beam_width': 1,
            'max_batch_size': 320,
            'kv_cache_type': 'paged',
            'remove_input_padding': 'enable',
            'moe_plugin': 'disable',
            'max_input_len': 14,
            'max_seq_len': 174,
            'max_encoder_input_len': 3000,
            'gpt_attention_plugin': 'float16',
            'gemm_plugin': 'float16',
        },
        llm_fields.trtllm_runtime_flags: {
            'exclude_input_from_output': True,
            'use_inflight_batching': False,
            'max_num_tokens': 3000,
            'enable_chunked_context': False,
        },
        harness_fields.use_graphs: False,
        llm_fields.use_token_latencies: False,
        harness_fields.vboost_slider: 1,
    },
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxQ): {
        model_fields.gpu_batch_size: {
            'encoder': 320,
            'decoder': 320,
        },
        model_fields.input_dtype: Precision.FP32,
        llm_fields.llm_gen_config_path: 'code/whisper/tensorrt/generation_config.json',
        loadgen_fields.offline_expected_qps: 14,
        power_fields.power_limit: 450,
        model_fields.precision: Precision.FP16,
        harness_fields.tensor_path: 'build/preprocessed_data/whisper-large-v3/',
        llm_fields.tensor_parallelism: 1,
        llm_fields.pipeline_parallelism: 1,
        whisper_fields.whisper_encoder_build_flags: {
            'max_beam_width': 1,
            'max_batch_size': 320,
            'kv_cache_type': 'paged',
            'remove_input_padding': 'enable',
            'moe_plugin': 'disable',
            'gemm_plugin': 'disable',
            'max_input_len': 3000,
            'max_seq_len': 3000,
            'bert_attention_plugin': 'float16',
        },
        whisper_fields.whisper_decoder_build_flags: {
            'max_beam_width': 1,
            'max_batch_size': 320,
            'kv_cache_type': 'paged',
            'remove_input_padding': 'enable',
            'moe_plugin': 'disable',
            'max_input_len': 14,
            'max_seq_len': 174,
            'max_encoder_input_len': 3000,
            'gpt_attention_plugin': 'float16',
            'gemm_plugin': 'float16',
        },
        llm_fields.trtllm_runtime_flags: {
            'exclude_input_from_output': True,
            'use_inflight_batching': False,
            'max_num_tokens': 3000,
            'enable_chunked_context': False,
        },
        harness_fields.use_graphs: False,
        llm_fields.use_token_latencies: False,
        harness_fields.vboost_slider: 1,
    },
}
