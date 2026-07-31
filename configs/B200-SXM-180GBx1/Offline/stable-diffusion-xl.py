import code.common.constants as C
import code.fields.models as model_fields
import code.fields.harness as harness_fields
import code.fields.loadgen as loadgen_fields


base = {
    model_fields.gpu_batch_size: {
        'clip1': 64,
        'clip2': 64,
        'unet': 64,
        'vae': 8,
    },
    harness_fields.gpu_copy_streams: 1,
    harness_fields.gpu_inference_streams: 1,
    model_fields.input_dtype: 'int32',
    model_fields.input_format: 'linear',
    loadgen_fields.offline_expected_qps: 3.75,
    model_fields.precision: {
        'clip1': C.Precision.FP32,
        'clip2': C.Precision.FP32,
        'unet': C.Precision.FP8,
        'vae': C.Precision.FP32,
    },
    harness_fields.tensor_path: 'build/preprocessed_data/coco2014-tokenized-sdxl/5k_dataset_final/',
    harness_fields.use_graphs: False,
}

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): base,
}

# Atomic configs for leader mode with multiple variants
# Usage: --mpi_mode=leader --config_id=<variant_name>
ATOMIC_EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        "default": base,
    },
}
