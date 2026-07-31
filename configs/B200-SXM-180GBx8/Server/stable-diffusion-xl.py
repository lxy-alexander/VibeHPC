import code.common.constants as C
import code.fields.models as model_fields
import code.fields.harness as harness_fields
from importlib import import_module
sdxl_fields = import_module("code.stable-diffusion-xl.tensorrt.fields")
import code.fields.loadgen as loadgen_fields


EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        model_fields.gpu_batch_size: {
            'clip1': 16,
            'clip2': 16,
            'unet': 16,
            'vae': 8,
        },
        harness_fields.gpu_copy_streams: 1,
        harness_fields.gpu_inference_streams: 1,
        model_fields.input_dtype: 'int32',
        model_fields.input_format: 'linear',
        model_fields.precision: {
            'clip1': C.Precision.FP32,
            'clip2': C.Precision.FP32,
            'unet': C.Precision.FP8,
            'vae': C.Precision.FP32,
        },
        sdxl_fields.batcher_time_limit: 5,
        loadgen_fields.server_target_qps: 28.8,
        harness_fields.tensor_path: 'build/preprocessed_data/coco2014-tokenized-sdxl/5k_dataset_final/',
        harness_fields.use_graphs: False,
    },
}