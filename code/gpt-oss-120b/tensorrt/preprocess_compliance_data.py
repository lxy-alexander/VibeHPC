#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Preprocess GPQA compliance data for GPT-OSS-120B TEST07.

Creates input_ids_padded.npy and input_lens.npy from the compliance parquet file.

Usage:
    python preprocess_compliance_data.py \
        --input-file build/data/gpt-oss/v4/acc/acc_eval_compliance_gpqa.parquet \
        --output-dir build/data/gpt-oss/v4/compliance/test07
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# Constants (match existing GPT-OSS preprocessing)
G_MAX_INPUT_TOK_LEN = 3072  # Same as accuracy config
G_OSS_EOS = 2  # EOS token for GPT-OSS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", "-i", required=True,
                        help="Path to compliance parquet file")
    parser.add_argument("--output-dir", "-o",
                        default="build/data/gpt-oss/v4/compliance/test07",
                        help="Output directory")
    parser.add_argument("--tokenizer", "-t",
                        default="openai/gpt-oss-120b",
                        help="HuggingFace tokenizer name/path (only used if tokens not in parquet)")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")

    # Check for token columns (prefer pre-tokenized data)
    if 'tok_input' in df.columns:
        print("Using pre-tokenized 'tok_input' column")
        toks = df['tok_input'].to_list()
        if 'tok_input_len' in df.columns:
            tok_len_np = df['tok_input_len'].to_numpy().astype(np.int32)
        else:
            tok_len_np = np.array([len(t) for t in toks], dtype=np.int32)
    elif 'input_tokens' in df.columns:
        print("Using pre-tokenized 'input_tokens' column")
        toks = df['input_tokens'].to_list()
        tok_len_np = np.array([len(t) for t in toks], dtype=np.int32)
    else:
        # Need to tokenize from raw text
        print("No token column found, tokenizing from text...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

        # Find text column
        text_col = None
        for col in ['templated_text_input', 'input', 'text', 'prompt']:
            if col in df.columns:
                text_col = col
                break
        if text_col is None:
            raise ValueError(f"No text column found. Available: {df.columns.tolist()}")

        print(f"Tokenizing '{text_col}' column with {args.tokenizer}...")
        toks = [tokenizer.encode(text) for text in df[text_col]]
        tok_len_np = np.array([len(t) for t in toks], dtype=np.int32)

    # Determine max length from data
    max_len = max(len(t) for t in toks)
    pad_len = max(G_MAX_INPUT_TOK_LEN, max_len)
    print(f"Max token length in data: {max_len}, padding to: {pad_len}")

    # Create padded array
    toks_np = np.ones((len(toks), pad_len), dtype=np.int32) * G_OSS_EOS
    for i, q in enumerate(toks):
        toks_np[i, :len(q)] = q
        assert len(q) == tok_len_np[i], f"Length mismatch at {i}: {len(q)} != {tok_len_np[i]}"

    # Save files
    np.save(output_dir / "input_ids_padded.npy", toks_np)
    np.save(output_dir / "input_lens.npy", tok_len_np)

    print(f"\nDone preprocessing at {output_dir}")
    print(f"  - Samples: {len(toks)}")
    print(f"  - Padded length: {pad_len}")
    print(f"  - Token length stats: min={tok_len_np.min()}, max={tok_len_np.max()}, mean={tok_len_np.mean():.1f}")
    print(f"  - Files created:")
    print(f"    - input_ids_padded.npy: shape={toks_np.shape}")
    print(f"    - input_lens.npy: shape={tok_len_np.shape}")


if __name__ == "__main__":
    main()
