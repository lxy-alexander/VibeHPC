# VibeHPC MLPerf Inference v6.1 Submission

This repository contains VibeHPC's MLPerf Inference v6.1 submission materials.
It is derived from NVIDIA MLPerf Inference code and keeps NVIDIA and third-party
copyright, license, and attribution notices.

VibeHPC modifications are provided by VibeHPC. Upstream NVIDIA code, NVIDIA
containers, MLCommons code, model artifacts, datasets, and other third-party
components remain subject to their own license terms. Use of NVIDIA names in
this repository identifies upstream technologies, hardware, and derived code; it
does not transfer ownership of NVIDIA or third-party materials to VibeHPC.

This README is not legal advice. Before final submission or redistribution,
confirm that VibeHPC has the right to use and submit all code, models, datasets,
containers, and generated artifacts.

## Required Notices

Keep these files and notices in the submitted tree:

- `attributions.txt`
- `documentation/licenses_and_attributions.adoc`
- source file copyright headers
- license files included with downloaded model or dataset artifacts
- benchmark-specific README files under `code/<benchmark>/`

Do not remove NVIDIA, MLCommons, PyTorch, TensorFlow, Caffe, Caffe2, model, or
dataset notices.

## Directory Layout

```text
.
├── code/                  # Benchmark implementations and harness code
├── configs/               # System, benchmark, and scenario configuration
├── documentation/         # Submission documentation
├── systems/               # VibeHPC system description JSON files
├── results/               # Generated MLPerf result logs
├── scripts/               # Utility scripts
├── attributions.txt       # Third-party attribution notices
└── VERSION                # Submission version
```

Current VibeHPC system descriptions:

- `systems/B200-SXM-180GBx1_TRT.json`
- `systems/B200-SXM-180GBx8_TRT.json`
- `systems/B300-SXM-270GBx1_TRT.json`
- `systems/B300-SXM-270GBx8_TRT.json`
- `systems/system_descriptions.tsv`

## Setup

Set the submitter and scratch path:

```bash
export SUBMITTER=VibeHPC
export MLPERF_SCRATCH_PATH=/path/to/mlperf_scratch
mkdir -p "$MLPERF_SCRATCH_PATH"/{data,models,preprocessed_data,logs}
```

Clone third-party dependencies if they are not already present:

```bash
mkdir -p 3rdparty
git clone --depth 1 https://github.com/mlcommons/inference.git 3rdparty/mlc-inference
git clone --depth 1 https://github.com/NVIDIA/TensorRT-LLM.git 3rdparty/trtllm
```

Model weights and datasets are not assumed to be redistributable. Obtain them
only through the official benchmark instructions and applicable license terms.
See the README under each benchmark directory for exact preparation steps.

## Running

Use benchmark-specific README files for model and dataset preparation. A typical
single-node flow is:

```bash
make link_dirs
make generate_engines RUN_ARGS="--benchmarks=<benchmark> --scenarios=<scenario>"
make run_harness RUN_ARGS="--benchmarks=<benchmark> --scenarios=<scenario> --test_mode=AccuracyOnly"
make run_harness RUN_ARGS="--benchmarks=<benchmark> --scenarios=<scenario> --test_mode=PerformanceOnly"
```

For LLM server workflows:

```bash
make run_llm_server SYSTEM_NAME=<single-gpu-system> RUN_ARGS="--benchmarks=<benchmark> --scenarios=<scenario>"
make run_harness SYSTEM_NAME=<full-system> RUN_ARGS="--benchmarks=<benchmark> --scenarios=<scenario>"
```

For multi-node or Slurm-based runs, use the scaleout documentation:

- `scaleout/README.md`
- `scaleout/REPRODUCE.md`
- `scripts/slurm_llm/dynamo_disagg/REPRODUCE.md`

## Submission Artifacts

Generate and stage results through the NVIDIA-derived submission flow:

```bash
make stage_results
make stage_compliance
make truncate_results
make check_submission
make pack_submission
```

The official MLPerf Inference submission checker for the v6.1 submission round
is the final authority for required layout, file names, and compliance checks.
Do not treat local scripts or this README as a replacement for the official
rules and checker.

## Documentation

- `documentation/README.md`: submission documentation index
- `documentation/calibration.adoc`: calibration and quantization notes
- `documentation/bandwidth.adoc`: bandwidth evidence methodology
- `documentation/licenses_and_attributions.adoc`: legal notices and attribution
  guidance

Before final packaging, verify that `systems/`, `documentation/`, `results/`,
and generated scenario-level files match the official MLPerf Inference v6.1
requirements.
