#!/bin/bash

# Usage: sbatch --nodes=5 --account coreai_mlperf_inference --time 2:0:0 -p backfill \
#        scripts/slurm_llm/dynamo_disagg/example_batch_script.sh
#
# This script sets up a Python venv and launches a disaggregated serving cluster.

# Use SLURM_SUBMIT_DIR (directory where sbatch was run) as the workspace
# Note: SCRIPT_DIR doesn't work because sbatch copies the script to /var/spool/
cd "${SLURM_SUBMIT_DIR:-.}"
echo "Working directory: $(pwd)"

# Set up venv with pip
python3 -m venv --clear /tmp/disagg_venv
source /tmp/disagg_venv/bin/activate
python3 -m ensurepip --upgrade
pip install --upgrade pip
pip install -r scripts/slurm_llm/dynamo_disagg/requirements.txt

# Use -u for unbuffered output so logs appear in slurm output file immediately
# Option 1: Using YAML config
# python3 -u scripts/slurm_llm/dynamo_disagg/launch_disagg_cluster.py \
#   --config scripts/slurm_llm/dynamo_disagg/config/sample/disagg_deployment_minimal.yaml \
#   --container-image /path/to/image.sqsh

# Option 2: Using MLPerf system config (default)
python3 -u scripts/slurm_llm/dynamo_disagg/launch_disagg_cluster.py \
  --system GB200-NVL72_GB200-186GB_aarch64x20 \
  --benchmark deepseek-r1 --scenario Interactive \
  --container-image /lustre/fsw/coreai_mlperf_inference/mlperf_inference_images/mlpinf+mlperf-inference+dynamo-v0.8.0-tensorrt_llm_release-feat-1.2-mlpinf-5ad8cf6_sms_90_100_103_120-mlperf_llm_aarch64+latest.sqsh

# Option 3: Accuracy run (uses _accuracy.yml overrides for server, appends to harness args)
# python3 -u scripts/slurm_llm/dynamo_disagg/launch_disagg_cluster.py \
#   --system GB200-NVL72_GB200-186GB_aarch64x20 \
#   --benchmark deepseek-r1 --scenario Interactive \
#   --container-image /path/to/image.sqsh \
#   --accuracy

# Option 4: Accuracy run with harness (--accuracy appends --test_mode=AccuracyOnly to harness args)
# python3 -u scripts/slurm_llm/dynamo_disagg/launch_disagg_cluster.py \
#   --system GB200-NVL72_GB200-186GB_aarch64x20 \
#   --benchmark deepseek-r1 --scenario Interactive \
#   --container-image /path/to/image.sqsh \
#   --accuracy \
#   --run-harness-args="--benchmarks=deepseek-r1 --scenarios=Interactive"
