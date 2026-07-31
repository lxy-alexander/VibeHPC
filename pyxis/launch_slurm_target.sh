#!/bin/bash
set -e

if [ -z "$SBATCH_TEMPLATE_FILE" ]; then
    echo "SBATCH_TEMPLATE_FILE is not set. Please set it to the template file to use."
    exit 1
fi

if [ -z "$ARCH" ]; then
    echo "ARCH is not set. Please set it to 'x86_64' or 'aarch64'."
    exit 1
else
    echo -e "\033[31m\$ARCH is set to $ARCH. Please make sure you have the correct architecture to match the compute node.\033[0m"
fi

if [ -z "$MLPERF_SCRATCH_PATH" ]; then
    echo "MLPERF_SCRATCH_PATH is not set. Please set it to the scratch path for MLPerf."
    exit 1
fi


if [ -z "$SBATCH_CONTAINER_IMAGE" ]; then
    echo "SBATCH_CONTAINER_IMAGE is not set."
    read -e -p "Please provide a path to the sqsh file or docker image to pull: " container_image
    export SBATCH_CONTAINER_IMAGE="$container_image"
fi



# Convert spaces to commas in PREBUILD_ENV_VARS
export SBATCH_EXPORT=$(echo "$PREBUILD_ENV_VARS" | tr ' ' ',')
SBATCH_EXPORT="$SBATCH_EXPORT,INSTALL_TRTLLM=$INSTALL_TRTLLM,INSTALL_LOADGEN=$INSTALL_LOADGEN"

# Set all required SBATCH variables
export SBATCH_TEMPLATE_FILE="$SBATCH_TEMPLATE_FILE"
export SBATCH_JOB_NAME="mlperf_inf_slurm_job"
export SBATCH_STDOUT="slurm-%j-mlperf_inf_slurm_job.out"
export SBATCH_STDERR="slurm-%j-mlperf_inf_slurm_job.err"
export SBATCH_PARTITION="$SBATCH_PARTITION"
export SBATCH_CONTAINER_IMAGE="$SBATCH_CONTAINER_IMAGE"
export SBATCH_CONTAINER_MOUNTS="${HOST_VOL}:${CONTAINER_VOL},${MLPERF_SCRATCH_PATH}:/home/mlperf_inference_storage"
export SBATCH_CONTAINER_WORKDIR="${CONTAINER_VOL}"
export SBATCH_CONTAINER_SAVE="$SBATCH_CONTAINER_SAVE"
export SBATCH_ACCOUNT="$SBATCH_ACCOUNT"
export SBATCH_NODELIST="$SBATCH_NODELIST"
export EXTRA_SRUN_FLAGS="$EXTRA_SRUN_FLAGS"

# Run the template script
"$(dirname $0)/run_template_sbatch.sh"
