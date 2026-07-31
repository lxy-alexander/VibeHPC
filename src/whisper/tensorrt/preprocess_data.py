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

import argparse
import os
import subprocess


def to_absolute_path(path):
    assert (len(path) > 0)

    if path[0] == "/":
        return path
    return os.path.join(os.getcwd(), path)


def preprocess_whisper(data_dir, preprocessed_data_dir):
    custom_env = os.environ.copy()  # Start with a copy of the current environment
    custom_env["DATA_DIR"] = data_dir
    custom_env["OUTPUT_DIR"] = preprocessed_data_dir

    subprocess.run(["bash", "code/whisper/tensorrt/download_dataset.sh"], env=custom_env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", "-d",
        help="Specifies the directory containing the input data.",
        default="build/data/whipser-large-v3/"
    )
    parser.add_argument(
        "--preprocessed_data_dir", "-o",
        help="Specifies the output directory for the preprocessed data.",
        default="build/preprocessed_data/whisper-large-v3/"
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    preprocessed_data_dir = args.preprocessed_data_dir

    preprocess_whisper(data_dir, preprocessed_data_dir)
    print(f"Ouptut Preprocessed Dataset: {preprocessed_data_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
