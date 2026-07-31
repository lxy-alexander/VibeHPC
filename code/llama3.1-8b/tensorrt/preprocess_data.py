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

"""Script to preprocess the data for Llama3.1-8b."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset

G_LLAMA3_1_8B_MAX_INPUT_SEQLEN = 2540
G_LLAMA3_1_8B_EOS = 128009
G_PROMPT_INPUT = (
    "Summarize the following news article in 128 tokens. "
    "Please output the summary only, without any other text.\n\nArticle:\n{input}\n\nSummary:"
)


def preprocess_cnndailymail_prompt(data):
    sources = [G_PROMPT_INPUT.format_map(
        example) for example in data]
    targets = [f"{example['output']}" for example in data]

    return sources, targets


def preprocess_cnndailymail_gptj6b(data_dir, preprocessed_data_dir):
    cnn_val_json_path = os.path.join(
        data_dir, "llama3.1-8b", "cnn_eval.json")
    output_dir = Path(preprocessed_data_dir) / "llama3.1-8b"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Reading CNN dailymail examples...")
    df = pd.read_json(cnn_val_json_path)
    toks = df['tok_input'].to_list()
    toks_np = np.ones((len(toks), G_LLAMA3_1_8B_MAX_INPUT_SEQLEN), dtype=np.int32) * G_LLAMA3_1_8B_EOS
    tok_len_np = np.array([len(tok) for tok in toks])

    for i, q in enumerate(toks):
        toks_np[i, :len(q)] = q
        assert len(q) == tok_len_np[i]

    np.save(os.path.join(output_dir, "input_ids_padded.npy"), toks_np)
    np.save(os.path.join(output_dir, "input_lens.npy"), tok_len_np)

    print("Done saving preprocessed data.")


def preprocess_calibration(data_dir, preprocessed_data_dir):
    cnn_calib_json_path = os.path.join(
        data_dir, "llama3.1-8b", "cnn_dailymail_calibration.json")

    with open(cnn_calib_json_path, 'r') as fh:
        data = json.load(fh)
    sources, _ = preprocess_cnndailymail_prompt(data)
    sources = [{"text": row} for row in sources]
    assert len(sources) == 1000, "The length of the calibration list is not 1000!"

    hf_dataset = Dataset.from_list(sources)
    # Cannot have "cnn_dailymail" in the path because of TRTLLM quantization rule.
    dataset_dir = Path(preprocessed_data_dir) / "llama3.1-8b" / "mlperf_llama3.1-8b_calibration_1k"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    hf_dataset.to_parquet(dataset_dir / "data.parquet")

    print(f"Finished processing calibration dataset at {dataset_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir", "-d",
        help="Directory containing the input data.",
        default="build/data"
    )
    parser.add_argument(
        "--preprocessed_data_dir", "-o",
        help="Output directory for the preprocessed data.",
        default="build/preprocessed_data"
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    preprocessed_data_dir = args.preprocessed_data_dir

    preprocess_cnndailymail_gptj6b(data_dir, preprocessed_data_dir)
    preprocess_calibration(data_dir, preprocessed_data_dir)

    print("Done!")


if __name__ == '__main__':
    main()
