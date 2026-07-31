#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
import torch
import pandas as pd
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--ref_pkl_path", type=str, default="build/data/moe/mlperf_mixtral8x7b_moe_dataset_15k.pkl",
                    help="HF ref pickle file")
parser.add_argument("--model_path", type=str, default="/raid/data/mlperf-llm/Mixtral-8x7B-Instruct-v0.1",
                    help="mixtral hf model path")
parser.add_argument("--bs", type=int, default=64,
                    help="batch size")
parser.add_argument("--samples", type=int, default=15000,
                    help="Number of samples to run. Max 15000")
args = parser.parse_args()

G_MAX_INPUT_SEQLEN = 2048
G_MAX_OUTPUT_SEQLEN = 1024

"""
dataset                                                           GSM8K
id                                                            train.548
question              Gary manages two Amazon distribution centers. ...
input                 <s> [INST] As an expert problem solver solve s...
ref_output            The first center processes 10000 packages per ...
gt_output                                                         14000
tok_input             [1, 1, 28705, 733, 16289, 28793, 1136, 396, 75...
tok_ref_output        [415, 907, 4982, 9537, 28705, 28740, 28734, 28...
stop_sequence                                                      </s>
tok_stop_sequence                                                   [2]
tok_input_len                                                       662
tok_ref_output_len                                                  174
Name: 0, dtype: object
"""
df = pd.read_pickle(args.ref_pkl_path)

device = "cuda"  # the device to load the model onto

# Load the model from local if possible.
model_path = Path(args.model_path)
if not model_path.exists():
    model_path = Path("build/models/Mixtral/Mixtral-8x7B-Instruct-v0.1")

tokenizer = AutoTokenizer.from_pretrained(
    model_path, padding_side="left", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# gen parameter. We stop at 1024
gen_kwargs = {
    "max_new_tokens": G_MAX_OUTPUT_SEQLEN,
    "do_sample": False,
    "temperature": None,
    "top_p": None,
}

# Start inference
BS = args.bs
bidx = 0
model.eval()

input_tokens = []
input_tokens_lens = []
output_tokens = []
output_tokens_lens = []
output_texts = []

tic = time.time()
n_samples = min(len(df), args.samples)
for idx in range(0, n_samples, BS):
    tac = time.time()
    print(f"Processing {idx}/{n_samples}, time: {tac - tic}s")
    sidx = idx
    eidx = min(sidx + BS, n_samples)

    # We use batch_encode_plus for batch inference.
    batch_texts = df['input'][sidx:eidx].tolist()
    batch_ids = tokenizer.batch_encode_plus(batch_texts, return_tensors="pt", padding=True)
    tok_input_length = batch_ids['attention_mask'].sum(
        axis=1).to(torch.int32).tolist()
    input_tokens_lens += tok_input_length
    tok_input_id = batch_ids['input_ids'].to(torch.int32).tolist()
    # Remove eos from the input id
    tok_input_id = [[element for element in sublist if element !=
                    tokenizer.eos_token_id] for sublist in tok_input_id]
    input_tokens += tok_input_id

    batch_ids = batch_ids.to(device)
    _, length = batch_ids.input_ids.shape
    outputs = model.generate(**batch_ids, num_return_sequences=1,
                             **gen_kwargs)

    output_ids = outputs[:, length:].cpu().tolist()
    output_tokens += output_ids

    # Filter out EOS
    id_filtered = [[num for num in sublist if num !=
                    tokenizer.eos_token_id] for sublist in output_ids]
    output_id_len = [len(out) for out in id_filtered]
    output_tokens_lens += output_id_len

    # Detokenizer
    output_msgs = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True)
    output_texts += output_msgs
    bidx += 1

# Assemble the output
output_df = df[:len(output_tokens)].copy()
output_df["nv_ref_output"] = output_texts
output_df["nv_tok_ref_output"] = output_tokens
output_df["nv_tok_ref_output_length"] = output_tokens_lens

fname = f"mixtral_8x7b_{len(output_tokens)}_BS{args.bs}_greedy_hf_ref_fp16.pkl"
output_df.to_pickle(fname)
print(f"Output pickle generated at: {fname}")
