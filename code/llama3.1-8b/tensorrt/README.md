# Llama3.1 8B

## Getting started

### Download Model

Please download model files by following the mlcommons README.md with instructions:

```bash
# following steps: https://github.com/mlcommons/inference/tree/master/language/llama3.1-8b#get-model
# If your company has license concern, please download the model from the following link: https://llama3-1.mlcommons.org/
export CHECKPOINT_PATH=build/models/Llama3.1-8B/Meta-Llama-3.1-8B-Instruct
git lfs install
git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct ${CHECKPOINT_PATH}
cd ${CHECKPOINT_PATH} && git checkout 0e9e39f249a16976918f6564b8830bc894c89659
```

Please untar the quantized checkpoint packaged with the container:

```bash
tar -xzf /opt/fp4-quantized-modelopt/llama3_1-8b-instruct-hf-torch-fp4.tar.gz -C $(BUILD_DIR)/models/Llama3.1-8B/fp4-quantized-modelopt/
```

### Download and Prepare Data

Please download data files by following the mlcommons README.md with instructions.
Please move the downloaded json into expected path and follow steps to run the required data pre-processing:

```bash
# follow: https://github.com/mlcommons/inference/tree/master/language/llama3.1-8b#get-dataset
# to download file: cnn_eval.json, cnn_dailymail_calibration.json

# make sure you are in mlperf's container
make prebuild ENV=release BENCHMARKS=llama

# move into right directory
mv cnn_eval.json build/data/llama3.1-8b/cnn_eval.json
mv cnn_dailymail_calibration.json build/data/llama3.1-8b/cnn_dailymail_calibration.json

# run pre-process step for llama3
python3 code/llama3_1-8b/tensorrt/preprocess_data.py --data_dir build/data/ --preprocessed_data_dir build/preprocessed-data
```

Make sure after the 2 steps above, you have:

1. model downloaded at: `build/models/Llama3.1-8B/Meta-Llama-3.1-8B-Instruct/`
2. preprocessed data at `build/preprocessed_data/llama3.1-8b/`:

- `build/preprocessed_data/llama3.1-8b/input_lens.npy`
- `build/preprocessed_data/llama3.1-8b/input_ids_padded.npy`
- `build/preprocessed_data/llama3.1-8b/mlperf_llama3.1-8b_calibration_1k/data.parquet`

## Build and run the benchmarks

Please follow the steps below in MLPerf container. Note that the quantization is done in the generate_engines step, so you don't need to do it separately.

```bash
# make sure you are in mlperf's container
make prebuild ENV=release BENCHMARKS=llama

# if not, make sure you already built TRTLLM as well as mlperf harnesses are needed.
# No longer needed if it's release container
# make build

# Please update configs/llama3_1-8b to include your custom machine config before building the engine
make run_llm_server RUN_ARGS="--core_type=trtllm_endpoint --benchmarks=llama3.1-8b --scenarios=Offline"
# Wait for few minutes for server to spin up, see logs
make run_harness RUN_ARGS="--core_type=trtllm_endpoint --benchmarks=llama3.1-8b --scenarios=Offline"
```
