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
if [ -e /work/code/stable-diffusion-xl/tensorrt/download_model.sh ]
then
    echo "Inside container, start downloading..."
else
    echo "WARNING: Please enter the MLPerf container (make prebuild) before downloading SDXL model."
    echo "WARNING: SDXL model is NOT downloaded! Exiting..."
    exit 1
fi

MODEL_DIR=/work/build/models
DATA_DIR=/work/build/data

# Download the fp16 raw weights of MLCommon hosted HF checkpoints
mkdir -p $MODEL_DIR/SDXL/official_pytorch/fp16/stable_diffusion_fp16/

rclone copy mlc-inference:mlcommons-inference-wg-public/stable_diffusion_fp16 $MODEL_DIR/SDXL/official_pytorch/fp16/stable_diffusion_fp16/ -P

md5sum ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/text_encoder/model.safetensors | grep "81b87e641699a4cd5985f47e99e71eeb"
if [ $? -ne 0 ]; then
    echo "SDXL CLIP1 fp16 model md5sum mismatch"
    exit -1
fi

md5sum ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/text_encoder_2/model.safetensors | grep "5e540a9d92f6f88d3736189fd28fa6cd"
if [ $? -ne 0 ]; then
    echo "SDXL CLIP2 fp16 model md5sum mismatch"
    exit -1
fi

md5sum ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/unet/diffusion_pytorch_model.safetensors | grep "edfa956683fb6121f717d095bf647f53"
if [ $? -ne 0 ]; then
    echo "SDXL UNet fp16 model md5sum mismatch"
    exit -1
fi

md5sum ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/vae/diffusion_pytorch_model.safetensors | grep "25fe90074af9a0fe36d4a713ad5a3a29"
if [ $? -ne 0 ]; then
    echo "SDXL VAE fp16 model md5sum mismatch"
    exit -1
fi
echo "SDXL model download complete!"

# Run onnx generation script
python3 -m code.stable-diffusion-xl.tensorrt.create_onnx_model
echo "SDXL model generation complete!"

if [ "${BUILD_CONTEXT}" != "aarch64-Orin" ]
then
    # Datacenter: Use FP8 Unet and INT8 VAE engine
    if [ -e /opt/unetxl.fp8 ]
    then
        echo "Prebaked image for partners detected!"
        mv -rv /opt/unetxl.fp8 ${MODEL_DIR}/SDXL/modelopt_models/unetxl.fp8
    else
        mkdir -p ${MODEL_DIR}/SDXL/modelopt_models/unetxl.fp8/
        echo "Runing SDXL UNet fp8 quantization and export on 500 calibration captions. The process will take ~1 hour on DGX H100 and won't work on Orin for prohibitively long running time"
        python3 /work/code/stable-diffusion-xl/modelopt/quantize_unet.py --model ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/ --batch-size 1 \
            --calib-size 500 --n_steps 20 --calib-data /work/code/stable-diffusion-xl/modelopt/captions.tsv --latent /work/code/stable-diffusion-xl/modelopt/latents.pt \
            --exp_name ${MODEL_DIR}/SDXL/modelopt_models/unetxl.fp8/unetxl.fp8.pt --onnx_dir ${MODEL_DIR}/SDXL/modelopt_models/unetxl.fp8
        echo "Finished SDXL UNet fp8 quantization and export"
    fi

    mkdir -p ${MODEL_DIR}/SDXL/modelopt_models/vae.int8/
    echo "Runing SDXL VAE quantization on 64 calibration captions. The process will take ~5 mins on DGX H100 or a few hours on Orin"
    python3 /work/code/stable-diffusion-xl/modelopt/quantize_vae.py --model ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/ --batch-size 1 \
        --format int8 --calib-size 64 --n_steps 20 --calib-data /work/code/stable-diffusion-xl/modelopt/captions.tsv --alpha 1.0  \
        --quant-level 3.0 --latent /work/code/stable-diffusion-xl/modelopt/latents.pt --exp_name ${MODEL_DIR}/SDXL/modelopt_models/vae.int8/vae.int8.pt
    echo "Exporting SDXL fp32-int8 VAE onnx. The process will take ~1 min on DGX H100 or a few hours on Orin"
    python /work/code/stable-diffusion-xl/modelopt/export_vae.py --model ${MODEL_DIR}/SDXL/official_pytorch/fp16/stable_diffusion_fp16/checkpoint_pipe/ \
        --quantized-ckpt ${MODEL_DIR}/SDXL/modelopt_models/vae.int8/vae.int8.pt --format int8 --quant-level 3.0 --onnx-dir ${MODEL_DIR}/SDXL/modelopt_models/vae.int8
    echo "Finished SDXL VAE int8 quantization and export"
    echo "SDXL model quantization complete!"
else
    # Orin: Use INT8 Unet
    mkdir -p ${MODEL_DIR}/SDXL/modelopt_models/
    if [ -e /opt/unetxl.int8 ]
    then
        echo "Prebaked image for partners detected!"
        mv -rv /opt/unetxl.int8 ${MODEL_DIR}/SDXL/modelopt_models/unetxl.int8
    else
        # TODO yihengz: update int8 unet recipe
        echo "Int8 Unet quantization is not needed for v4.1 and is not implemented. For people want to run int8 unet, please use 4.0 NVIDIA submission code (https://github.com/mlcommons/inference_results_v4.0/tree/main/closed/NVIDIA) to generate the model"
    fi
fi
