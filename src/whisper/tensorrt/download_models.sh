#!/bin/bash
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

source code/common/file_downloads.sh


# Make sure the script is executed inside the container
if [ -e /work/code/whisper/tensorrt/download_models.sh ]
then
    echo "Inside container, start downloading..."
else
    echo "WARNING: Please enter the MLPerf container (make prebuild) before downloading Whisper model."
    echo "WARNING: Whisper model is NOT downloaded! Exiting..."
    exit 1
fi


MODEL_DIR=/work/build/models

mkdir -p ${MODEL_DIR}

if [ -e /work/build/models/whisper-large-v3/multilingual.tiktoken ]
then
    echo "multilingual.tiktoken already exist."
else
    echo "download multilingual.tiktoken ..."
    download_file models whisper-large-v3 https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken multilingual.tiktoken

fi

if [ -e /work/build/models/whisper-large-v3/mel_filters.npz ]
then
    echo "mel_filters.npz already exist."
else
    echo "download mel_filters.npz ..."
    download_file models whisper-large-v3 https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz mel_filters.npz
fi

if [ -e /work/build/models/whisper-large-v3/large-v3.pt ]
then
    echo "large-v3.pt already exist."
else
    echo "download large-v3.pt ..."
    download_file models whisper-large-v3 https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt large-v3.pt

fi

md5sum ${MODEL_DIR}/whisper-large-v3/multilingual.tiktoken | grep "da95f6601b7c4327d4464d081b9dcf09"
if [ $? -ne 0 ]; then
    echo "Whisper multilingual.tiktoken md5sum mismatch"
    exit -1
fi

md5sum ${MODEL_DIR}/whisper-large-v3/mel_filters.npz | grep "61b070b259b27b8f8550e632d5300c8b"
if [ $? -ne 0 ]; then
    echo "Whisper mel_filters.npz md5sum mismatch"
    exit -1
fi

md5sum ${MODEL_DIR}/whisper-large-v3/large-v3.pt | grep "017baacdaada84d0d5cb030140875b65"
if [ $? -ne 0 ]; then
    echo "Whisper large-v3.pt md5sum mismatch"
    exit -1
fi

echo "Whisper large v3 model download complete!"
