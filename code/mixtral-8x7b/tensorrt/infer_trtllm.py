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
import pandas as pd
import numpy as np
from pathlib import Path

import torch
import traceback
import time
import argparse

import tensorrt_llm
import tensorrt_llm.profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp

parser = argparse.ArgumentParser()
parser.add_argument("--tllm_engine_dir", type=str,
                    help="The path to the engine")
parser.add_argument("--ref_pkl_path", type=str, default="build/data/moe/mlperf_mixtral8x7b_moe_dataset_15k.pkl",
                    help="HF ref pickle file")
parser.add_argument("--model_path", type=str, default="/raid/data/mlperf-llm/Mixtral-8x7B-Instruct-v0.1",
                    help="mixtral hf model path")
parser.add_argument("--bs", type=int, default=128,
                    help="batch size")
parser.add_argument("--samples", type=int, default=15000,
                    help="Number of samples to run. Max 15000")
args = parser.parse_args()

G_MAX_INPUT_SEQLEN = 2048
G_MAX_OUTPUT_SEQLEN = 1024

df = pd.read_pickle(args.ref_pkl_path)
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

# Load the model from local if possible.
model_path = Path(args.model_path)
if not model_path.exists():
    model_path = Path("build/models/Mixtral/Mixtral-8x7B-Instruct-v0.1")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path, padding_side="left", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Prepare runner
runtime_rank = tensorrt_llm.mpi_rank()
runner_kwargs = dict(engine_dir=args.tllm_engine_dir,
                     rank=runtime_rank,
                     max_batch_size=args.bs,
                     max_input_len=G_MAX_INPUT_SEQLEN,
                     max_output_len=G_MAX_OUTPUT_SEQLEN,
                     max_beam_width=1,)

runner = ModelRunnerCpp.from_dir(**runner_kwargs)

st = time.time()
output_tokens = []
output_tokens_lens = []
output_texts = []

bidx = 0
n_samples = min(len(df), args.samples)
try:
    for idx in range(0, n_samples, args.bs):
        tac = time.time()
        print(f"Processing {idx}/{n_samples}, time: {tac - st}s")
        sidx = idx
        eidx = min(sidx + args.bs, n_samples)

        # Read from pickle
        batch_input_ids = df['tok_input'][sidx:eidx].tolist()
        batch_input_ids = [[element for element in sublist if element !=
                            tokenizer.eos_token_id] for sublist in batch_input_ids]

        batch_input_ids = [
            torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
        ]
        input_lengths = [x.size(0) for x in batch_input_ids]

        # trtllm expects List[List[int]] for each query
        # valid list eg: [ [[2]], [[1, 2, 3, 4]] ]
        stop_words_list = [
            np.expand_dims(np.array(_list), 0).tolist()
            for _list in df['tok_stop_sequence'][sidx:eidx].to_list()
        ]

        with torch.no_grad():
            outputs = runner.generate(
                batch_input_ids,
                max_new_tokens=G_MAX_OUTPUT_SEQLEN,
                temperature=None,
                end_id=tokenizer.eos_token_id,
                pad_id=tokenizer.pad_token_id,
                stop_words_list=stop_words_list,
            )

        torch.cuda.synchronize()
        output_ids = outputs.cpu().tolist()

        # Strip input, take beam 0
        output_ids_no_input = [ids[0][input_lengths[idx]:]
                               for (idx, ids) in enumerate(output_ids)]
        output_tokens += output_ids_no_input

        # Filter out EOS
        id_filtered = [[num for num in sublist if num !=
                        tokenizer.eos_token_id] for sublist in output_ids_no_input]
        output_id_len = [len(out) for out in id_filtered]
        output_tokens_lens += output_id_len

        # Detokenizer
        output_msgs = tokenizer.batch_decode(
            output_ids_no_input, skip_special_tokens=True)
        output_texts += output_msgs

        bidx += 1
except Exception as e:
    print(f"Exception caught!!! will not raise for saving the stuff")
    print(e)
    traceback.print_exc()

output_df = df[:len(output_tokens)].copy()
output_df["nv_tllm_ref_output"] = output_texts
output_df["nv_tllm_tok_ref_output"] = output_tokens
output_df["nv_tllm_tok_ref_output_length"] = output_tokens_lens

fname = f"trtllm_mixtral_8x7b_{len(output_tokens)}_BS{args.bs}_greedy.pkl"
output_df.to_pickle(fname)
print(f"dataframe dumped to {fname}")
