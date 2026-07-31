#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to preprocess the data for BERT."""

import argparse
import os

import numpy as np
import pandas as pd
from pathlib import Path
from code.common import logging
from datasets import Dataset

G_MAX_INPUT_TOK_LEN = 2048
G_MAX_STOP_WORDS_TOK_LEN = 4  # based on mbxp dataset
G_MIXTRAL_EOS = 2

"""
>>> df.loc[0]
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
tok_input_len                                                       657
tok_ref_output_len                                                  174
Name: 0, dtype: object
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir", "-d",
        help="Path to the input open_orca pickle file",
        default="build/data"
    )
    parser.add_argument(
        "--preprocessed_data_dir", "-o",
        help="Output directory for the preprocessed data.",
        default="build/preprocessed_data"
    )
    args = parser.parse_args()
    pkl_path = Path(args.data_dir) / "moe/mlperf_mixtral8x7b_moe_dataset_15k.pkl"
    output_dir = Path(args.preprocessed_data_dir) / "moe"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(pkl_path)
    inp_toks = df['tok_input'].to_list()
    stop_toks = df['tok_stop_sequence'].to_list()

    toks_np = np.ones((len(inp_toks), G_MAX_INPUT_TOK_LEN), dtype=np.int32) * G_MIXTRAL_EOS
    tok_len_np = df['tok_input_len'].to_numpy().astype(np.int32)

    for i, q in enumerate(inp_toks):
        toks_np[i, :len(q)] = q
        assert len(q) == tok_len_np[i]

    stop_toks_np = np.ones((len(stop_toks), G_MAX_STOP_WORDS_TOK_LEN), dtype=np.int32) * G_MIXTRAL_EOS
    for i, q in enumerate(stop_toks):
        stop_toks_np[i, :len(q)] = q
        assert len(q) <= G_MAX_STOP_WORDS_TOK_LEN

    np.save(Path(output_dir) / "input_ids_padded.npy", toks_np)
    np.save(Path(output_dir) / "input_lens.npy", tok_len_np)
    np.save(Path(output_dir) / "stop_ids_padded", stop_toks_np)

    # Preprocess calibration pickle files into Dataset files for TRTLLM
    calib_pkl_path = Path(args.data_dir) / "moe/mlperf_mixtral8x7b_moe_calibration_dataset_1k.pkl"
    calib_df = pd.read_pickle(calib_pkl_path)

    # Ensure the DataFrame has the correct structure
    if 'input' not in calib_df.columns:
        raise ValueError("The DataFrame does not contain an 'input' column.")
    hf_dataset = Dataset.from_pandas(calib_df[['input']])
    hf_dataset = hf_dataset.rename_column("input", "text")
    # Note that you have to save as a parquet to use "load_dataset".
    # # See: https://github.com/huggingface/datasets/issues/6703
    dataset_dir = output_dir / 'mlperf_mixtral8x7b_calibration_1k_dataset'
    dataset_dir.mkdir(exist_ok=True)
    hf_dataset.to_parquet(dataset_dir / "data.parquet")

    logging.info(f"Done preprocessing Mixtral dataset at {output_dir}")


if __name__ == '__main__':
    main()
