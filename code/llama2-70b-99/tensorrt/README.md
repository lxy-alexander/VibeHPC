# Llama2

## Getting started

Please first download the data, model and preprocess the data folloing the steps below _within the mlperf container_. Note that if you have already downloaded the model and data prior to v4.1, you don't need to redo them. But you _need to re-run_ the preprocess_data step for the updated calibration data.

### Download Model

Please download model files by following the mlcommons README.md with instructions:

```bash
# following steps: https://github.com/mlcommons/inference/blob/master/language/llama2-70b/README.md#get-dataset
```

### Download NVFP4 quantized model

(Public, need agreement) https://huggingface.co/centml/llama2-70b-chat-hf-torch-fp4_mlperf-inf-v6.0

Please pull the checkpoint, and point --llm_quantizer_outdir to this dir (or place it in the corresponding default directory).

### Download and Prepare Data

Please download data files by following the mlcommons README.md with instructions.
Please move the downloaded pickle into expected path and follow steps to run the required data pre-processing:

```bash
# follow: https://github.com/mlcommons/inference/blob/master/language/llama2-70b/README.md#get-dataset
# to download file: open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz, open_orca_gpt4_tokenized_llama.calibration_1000.pkl.gz

# unzip files
gzip -dk open_orca_gpt4_tokenized_llama.sampled_24576.pkl.gz
gzip -dk open_orca_gpt4_tokenized_llama.calibration_1000.pkl.gz

# make sure you are in mlperf's container
make prebuild ENV=release BENCHMARKS=llama

# move into right directory
mv open_orca_gpt4_tokenized_llama.*.pkl build/data/llama2-70b/

# run pre-process step for llama2
python3 code/llama2-70b/tensorrt/preprocess_data.py --data_dir build/data/ --preprocessed_data_dir build/preprocessed-data
```

Make sure after the 2 steps above, you have:

1. model downloaded at: `build/models/Llama2/Llama-2-70b-chat-hf/`,
2. preprocessed data at `build/preprocessed_data/llama2-70b/`:

- `build/preprocessed_data/llama2-70b/input_lens.npy`
- `build/preprocessed_data/llama2-70b/input_ids_padded.npy`
- `build/preprocessed_data/llama2-70b/mlperf_llama2_openorca_calibration_1k/data.parquet`

## Build and run the benchmarks

Please follow the steps below in MLPerf container.

```bash
# make sure you are in mlperf's container
make prebuild ENV=release BENCHMARKS=llama

# if not, make sure you already built TRTLLM 
# No longer needed in release container
# make build


# Please update configs/llama2-70b to include your custom machine config before running
# Start the LLM server
make run_llm_server RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --core_type=trtllm_endpoint"

# Wait for the server to be ready, then run the harness in a separate terminal
make run_harness RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --core_type=trtllm_endpoint"

# For accuracy test
make run_harness RUN_ARGS="--benchmarks=llama2-70b --scenarios=Offline --core_type=trtllm_endpoint --accuracy_target=.999 --test_mode=AccuracyOnly"
```

For multi-node runs (NVL72), use the scaleout scripts. See `scaleout/REPRODUCE.md` for detailed commands.

For a general rule of thumb, GPUs with:

- ~40GB of VMEM needs tensor parallelism of 4
- ~80GB of VMEM needs tensor parallelism of 2
- > 90GB of VMEM can run tensor parallelism of 1.

You should expect to get the following results (the detailed number might be different):

```
   accuracy: [PASSED] ROUGE1: 44.495 (Threshold=43.836) | [PASSED] ROUGE2: 22.089 (Threshold=21.689) | [PASSED] ROUGEL: 28.694 (Threshold=28.222) | [PASSED] TOKENS_PER_SAMPLE: 293.100 (Threshold=263.970)
```


Note: If you would like to quantize the LLama2-70B reference model yourself instead of using the provided huggingface NVFP4 quantized model, please update the quantizater code as follows. 
In `closed/NVIDIA/code/llmlib/hf_quantize.py`, please update lines 530-531 as follows:
```
if hasattr(tokenizer, '_tokenizer') and hasattr(tokenizer._tokenizer, 'no_truncation'):
  tokenizer._tokenizer.no_truncation()
tokenizer.save_pretrained(export_path)
```
