import code.common.constants as C
import code.fields.models as model_fields
import code.fields.loadgen as loadgen_fields


EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        model_fields.gpu_batch_size: {
            'rgat': 11000,
        },
        model_fields.input_dtype: 'int64',
        model_fields.input_format: 'linear',
        loadgen_fields.offline_expected_qps: 650000,
        model_fields.precision: 'fp16',
    },
}
