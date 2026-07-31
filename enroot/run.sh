#!/bin/bash

# Recommended: change $image_dir to your own path
git_dir=$(git rev-parse --show-toplevel)
image_dir=$git_dir/closed/NVIDIA/enroot/images
trtllm_build_arch="100-real;103-real"
stage="build"
mounts="$git_dir:$git_dir,$git_dir/closed/NVIDIA:/work,$HOME/.bash_history:$HOME/.bash_history"
run_cmd="--pty bash"
extra_srun_flags=""
container_image=""
container_uri=""
pull_stage=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "    --help|-h: Show this help message"
            echo "    --image-dir: Set the image directory"
            echo "    --trtllm-build-arch: Set the TRTLLM build architecture"
            echo "    --stage: Set the stage (build/run)"
            echo "    --run-cmd: Set the run command"
            exit 0
            ;;
        --image-dir)
            image_dir=$2
            shift 2
            ;;
        --trtllm-build-arch)
            trtllm_build_arch=$2
            shift 2
            ;;
        --stage)
            stage=$2
            shift 2
            ;;
        --run-cmd)
            run_cmd=$2
            shift 2
            ;;
        --mount)
            mounts="$mounts,$2"
            shift 2
            ;;
        --container-image)
            container_image=$2
            shift 2
            ;;
        --pull)
            pull_stage=$2
            container_uri=$3
            stage="pull"
            shift 3
            ;;
        *)
            extra_srun_flags="$extra_srun_flags $1"
            shift 1
            ;;
    esac
done

mkdir -p $image_dir

# Recommended: build only for your own architecture
# Example for arch, use `100-real;103-real` to build sm100 and sm103 (B200, B300)
# "all" will 
# - build all architectures, will take more time 
# - however, the resultant sqsh will be able to run trtllm for all architectures (preferable for release)
export TRTLLM_BUILD_ARCH=$trtllm_build_arch

# Set image paths
export TRTLLM_DEVEL_SQSH=$image_dir/trtllm_devel.sqsh
export TRTLLM_REL_SQSH=$image_dir/trtllm_rel.sqsh
export MLPERF_REL_SQSH=$image_dir/mlperf_rel.sqsh

echo "================================================"
echo "Image directory: $image_dir"
echo "|--- TRTLLM devel sqsh: trtllm_devel.sqsh"
echo "|--- TRTLLM rel sqsh: trtllm_rel.sqsh"
echo "|--- MLPerf rel sqsh: mlperf_rel.sqsh"
echo ""
echo "TRTLLM build architecture: $trtllm_build_arch"
echo "================================================"

if [ -z "$container_image" ]; then
    container_image=$MLPERF_REL_SQSH
fi

# Build the images
cd $git_dir/closed/NVIDIA
# Validate stage parameter
if [[ "$stage" != "build" && "$stage" != "run" && "$stage" != "pull" ]]; then
    echo "Error: Invalid stage '$stage'. Must be 'build' or 'run' or 'pull'"
    exit 1
fi

if [ "$stage" = "build" ]; then
    make -C enroot build_mlperf_rel || {
        echo "Error: Failed to build MLPerf release image"
        exit 1
    }
elif [ "$stage" = "run" ]; then
    make -C enroot run_srun_step \
        RUN_CMD="$run_cmd" \
        CONTAINER_IMAGE=$container_image \
        CONTAINER_MOUNTS="$mounts" \
        CONTAINER_WORKDIR=/work \
        EXTRA_SRUN_FLAGS="$extra_srun_flags" \
        || {
        echo "Error: Failed to run srun step"
        exit 1
    }
elif [ "$stage" = "pull" ]; then
    if [ -f $image_dir/$pull_stage.sqsh ]; then
        echo "Error: $image_dir/$pull_stage.sqsh already exists - will not overwrite"
        exit 1
    fi
    srun enroot import -o $image_dir/$pull_stage.sqsh docker://$container_uri || {
        echo "Error: Failed to pull $container_uri"
        exit 1
    }
    echo "Successfully pulled $container_uri to $image_dir/$pull_stage.sqsh"
fi
