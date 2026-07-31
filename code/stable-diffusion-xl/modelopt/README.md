# Stable Diffusion XL 1.0 Base Quantization with NVIDIA TensorRT Model Optimizer

This folder containes code using NVIDIA ModelOpt to calibrate and quantize the UNet and VAE part of the SDXL to fp8 and int8 accordingly for MLPerf Inference.

## Get Started

The quantization scripts are automatically invoked during model downloading stage through `make download_model BENCHMARKS="stable-diffusion-xl"`. Below are the steps to run the quantization manually

### SDXL FP8 UNet
#### Quantization and ONNX Export

```sh
python3 quantize_unet.py --batch-size 1 --calib-size 500 --n_steps 20 \
    --calib-data ./captions.tsv --latent ./latents.pt \
    --exp_name {UNET_QUANTIZED_CHECKPOINT_PATH} --onnx_dir {UNET_ONNX_EXPORT_DIR}
```

### SDXL INT8 VAE
#### Quantization

```sh
python quantize_vae.py \
    --model stabilityai/stable-diffusion-xl-base-1.0 \
    --batch-size 1 --format int8 --calib-size 64 \
    --n_steps 20 --calib-data ./captions.tsv \
    --alpha 1.0  --quant-level 3.0 --latent ./latents.pt \
    --exp_name {VAE_QUANTIZED_CHECKPOINT_PATH}
```

#### ONNX Export

```sh
python export_vae.py \
    --model stabilityai/stable-diffusion-xl-base-1.0 \
    --quantized-ckpt {.PT_QUANTIZED_CHECKPOINT_PATH} \
    --format int8 --quant-level 3.0 \
    --onnx-dir {VAE_ONNX_EXPORT_DIR}
```
