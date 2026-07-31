# Quantization script for mixtral-8x7b model
This folder containes code using NVIDIA ModelOpt that converts the Mixtral-8x7B-Instruct-v0.1 HuggingFace checkpoint (which is in float16) to TRTLLM quantized checkpoints (fp8+fp16, fp4+fp16)

## Example usage

### B200 checkpoint generation
```
python3 -m code.mixtral-8x7b.modelopt.main \
    --model_path=/work/build/models/Mixtral/Mixtral-8x7B-Instruct-v0.1 \
    --quantized_checkpoint_path=/raid/data/shobhitv/tllm_ckpnt_moe_fp4 \
    --calib_dataset_path=/work/build/data/moe/mlperf_mixtral8x7b_moe_calibration_dataset_1k.pkl \
    --effective_bits=6 \
    --tp_size=1 \
    --pp_size=1 \
    --num_calib_steps=16 \
    --num_score_steps=4 \
    --fp4=True
```

### H200 Checkpoint generation
```
python3 -m code.mixtral-8x7b.modelopt.main \
    --model_path=/work/build/models/Mixtral/Mixtral-8x7B-Instruct-v0.1 \
    --quantized_checkpoint_path=/raid/data/shobhitv/mlperf_moe_fp8_ckpnt_sm90/fp8-quantized-modelopt/mixtral-8x7b-instruct-v0.1-tp1pp1-fp8 \
    --calib_dataset_path=/work/build/data/moe/mlperf_mixtral8x7b_moe_calibration_dataset_1k.pkl \
    --effective_bits=8.5 \
    --tp_size=1 \
    --pp_size=1 \
    --num_calib_steps=16 \
    --num_score_steps=4 \
    --fp4=False
```


The checkpoint generation should be invoked automatically in the `make generate_engines` workflow, provided the `--checkpoint_dir` does not have the `rank0.safetensors`

```
make generate_engines RUN_ARGS="--benchmarks=moe --scenarios=offline --checkpoint_dir=/path/to/checkpoint --engine_dir=/path/to/engine"
```
If `/path/to/checkpoint/rank0.safetensors` exists, checkpoint generation will be skipped and engine will be generated in /path/to/engine
If `/path/to/checkpoint/rank0.safetensors` does not exist, checkpoint will be generated first, then engine will be generated from that.
