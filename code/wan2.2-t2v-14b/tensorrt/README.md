# WAN2.2-A14B Text-to-Video Benchmark

This benchmark performs text-to-video generation using the [WAN2.2-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) model and evaluates video quality using VBench metrics.

:warning: **IMPORTANT**: Please use `closed/NVIDIA` as the working directory when running the below commands. :warning:

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Run Inference through LoadGen](#run-inference-through-loadgen)
- [Accuracy Evaluation (VBench)](#accuracy-evaluation-vbench)
- [Internal Development Tools](#internal-development-tools)
- [Configuration Options](#configuration-options)

---

## Overview

WAN2.2-A14B is a 14 billion parameter text-to-video diffusion model. The benchmark generates videos from text prompts and evaluates them using VBench metrics including:

- Subject Consistency
- Background Consistency
- Motion Smoothness
- Dynamic Degree
- Appearance Style
- Scene

---

## Model Architecture

- **Transformer (DiT)**: VflyWanTransformer3DModel with 40 attention heads
- **VAE**: VflyWanAutoencoderKL (spatial downscale 8x, temporal: (frames-1)/4+1)
- **Text Encoder**: T5-based encoder
- **Default Output**: 81 frames at 720x1280 resolution

---
## Launch Container
```bash
BENCHMARKS=wan22 make prebuild
```
---
## Preprocessing

To preprocess the prompts and generate fixed latents for reproducible results:

```bash
BENCHMARKS=wan22-a14b make preprocess_data
```

This copy the prompts and fixed latent to scratch space:
- `$MLPERF_SCRATCH_PATH/preprocessed_data/wan22-a14b/prompts.txt`
- `$MLPERF_SCRATCH_PATH/preprocessed_data/wan22-a14b/fixed_latent.pt`

---

## Run Inference

Run the following commands from within the container to run inference through LoadGen:

### Performance Mode

```bash
make run_harness RUN_ARGS="--benchmarks=wan-2.2-t2v-a14b --scenarios=Offline --test_mode=PerformanceOnly"
```

## Accuracy Evaluation (VBench)

### Running Accuracy Mode

```bash
make run_harness RUN_ARGS="--benchmarks=wan-2.2-t2v-a14b --scenarios=Offline --test_mode=AccuracyOnly"
```

### Automatic Accuracy Check

After running in AccuracyOnly mode, accuracy is automatically evaluated using VBench. The accuracy checker:

1. Extracts generated videos from the log directory (`<log_dir>/video/`)
2. Sets up a VBench virtual environment (at `/work/.vbench-venv`) if not already present
3. Runs VBench evaluation on the generated videos
4. Reports the average score across all dimensions

#### VBench Dimensions

| Dimension | Description |
|-----------|-------------|
| `subject_consistency` | Consistency of the main subject across frames |
| `background_consistency` | Consistency of background elements |
| `motion_smoothness` | Smoothness of motion between frames |
| `dynamic_degree` | Amount of motion/dynamics in the video |
| `appearance_style` | Visual style quality |
| `scene` | Scene composition quality |

#### Reference Accuracy

| Metric | Value |
|--------|-------|
| Reference Accuracy (BF16) | **70.48** |
| Accuracy Threshold (99%) | **69.7752** |


---



## Configuration Options

### Video Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `wan22_model_path` | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | Path to model weights |
| `wan22_num_frames` | `81` | Number of frames to generate |
| `wan22_height` | `720` | Video height in pixels |
| `wan22_width` | `1280` | Video width in pixels |
| `wan22_num_inference_steps` | `20` | Number of diffusion steps |
| `wan22_guidance_scale` | `4.0` | Classifier-free guidance scale |
| `wan22_guidance_scale_2` | `3.0` | Classifier-free guidance scale |
| `wan22_attn_type` | `te-fp8` | Attention implementation type |
| `wan22_linear_type` | `te-fp8-per-tensor` | Linear layer type for quantization |

### Negative Prompt

The default negative prompt helps avoid common artifacts:

```
vivid colors, overexposed, static, blurry details, subtitles, style,
work of art, painting, picture, still, overall grayish, worst quality,
low quality, JPEG artifacts, ugly, deformed, extra fingers, poorly drawn hands,
poorly drawn face, deformed, disfigured, deformed limbs, fused fingers,
static image, cluttered background, three legs, many people in the background,
walking backwards
```

---
## Troubleshooting
GB200/GB300 vbench venv setup issue
```
rm -rf .venv-vbench
python3 -m venv --system-site-packages .venv-vbench
.venv-vbench/bin/pip install -r code/wan22-a14b/tensorrt/accuracy/vbench_requirements.aarch64.txt
```

NoResult error in submission checker
Double `accuracy.txt`
contains
```
'vbench_score': <your scores, ex. 70.01>
```

---

## File Structure

```
code/wan22-a14b/tensorrt/
├── accuracy   # not used in this round
│   ├── samples.txt
│   ├── vbench_requirements.aarch64.txt
│   └── vbench_requirements.x86_64.txt
├── constants.py
├── dataset.py
├── fields.py
├── harness.py
├── __init__.py
├── preprocess_data.sh
├── README.md
├── utils.py
└── worker.py
```
