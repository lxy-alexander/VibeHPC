#!/bin/bash

#SBATCH --job-name=%{sbatch_job_name}
#SBATCH --output=%{sbatch_stdout}
#SBATCH --error=%{sbatch_stderr}
#SBATCH --partition=%{sbatch_partition}
#SBATCH --export=%{sbatch_export}
#SBATCH --account=%{sbatch_account}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --container-image=%{sbatch_container_image}
#SBATCH --container-mounts=%{sbatch_container_mounts}
#SBATCH --container-workdir=%{sbatch_container_workdir}
#SBATCH --container-save=%{sbatch_container_save}
#SBATCH --nodelist=%{sbatch_nodelist}
#SBATCH --container-mount-home
#SBATCH --container-env=%{sbatch_container_env}

set -e

# Assert that we are running on a single node
if [ "$SLURM_JOB_NUM_NODES" != "1" ]; then
    echo "Error: This script must be run on a single node (SLURM_JOB_NUM_NODES = $SLURM_JOB_NUM_NODES)"
    exit 1
fi

# Define base directory for common scripts
base_dir="/work/docker/common"

# Check required environment variables
required_vars=("ENV" "BUILD_CONTEXT" "MITTEN_HASH" "INSTALL_RGAT_DEPS")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
        echo "Error: The following required environment variable is not set: $var"
        echo "Please set ${required_vars[@]} before running the script."
        exit 1
    fi
done

MITTEN_GIT_URL=https://github.com/NVIDIA/mitten.git

# Create a temporary directory for our work
TEMP_DIR=$(mktemp -d)
cd $TEMP_DIR

# Function to source a script if it exists
source_if_exists() {
    local script=$1
    if [ -f "$script" ]; then
        script_name=$(basename "$script")
        echo "Running $script and logging to /work/${script_name}.log ..."
        bash "$script" > "/work/${script_name}.log" 2>&1
    else
        echo "Warning: $script not found, skipping..."
    fi
}

echo "Installing apt dependencies..."
source_if_exists "$base_dir/install_apt_deps.sh"

echo "Installing pip dependencies..."
cp $base_dir/requirements.${BUILD_CONTEXT}.* .
source_if_exists "$base_dir/install_pip_deps.sh"

echo "Installing miscellaneous dependencies..."
source_if_exists "$base_dir/install_misc_deps.sh"

echo "Installing mitten..."
git clone $MITTEN_GIT_URL /tmp/mitten
source_if_exists "$base_dir/install_mitten.sh"

echo "Installing Tensorrt..."
source_if_exists "$base_dir/install_tensorrt.sh"

echo "Installing MPI4Py..."
source_if_exists "$base_dir/install_mpi4py.sh"

echo "Installing nixl..."
source_if_exists "$base_dir/install_nixl.sh"

echo "Installing ccache..."
source_if_exists "$base_dir/install_ccache.sh"


if [ "$INSTALL_RGAT_DEPS" = "1" ]; then
    echo "Installing RGAT dependencies..."
    if [ -d "$base_dir/patches" ]; then
        cp -r "$base_dir/patches" .
    fi
    source_if_exists "$base_dir/install_rgat_deps.sh"
fi

cd /work
rm -rf $TEMP_DIR

if [ "$INSTALL_TRTLLM" = "1" ]; then
    echo "Installing LLM dependencies..."
    export ENV=$ENV
    make clone_trt_llm && make build_trt_llm 
fi

if [ "$INSTALL_LOADGEN" = "1" ]; then
    echo "Installing LoadGen dependencies..."
    export ENV=$ENV
    make clone_loadgen && make build_loadgen
fi


echo "Installation completed successfully!"
