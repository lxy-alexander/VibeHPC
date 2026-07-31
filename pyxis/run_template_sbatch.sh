#!/bin/bash
set -e

if [ -z "$SLURM_MODE" ]; then
    echo "SLURM_MODE is not set. Please set it to 'sbatch' or 'srun'."
    exit 1
fi


# Check if SBATCH_PARTITION is set, prompt if not
if [ -z "$SBATCH_PARTITION" ]; then
    echo "====== Available partitions ======"
    sinfo -h -o "%P" | sort | uniq
    echo "====== End of available partitions ======"
    read -p "SBATCH_PARTITION is not defined. Please enter partition name: " partition
    export SBATCH_PARTITION="$partition"
fi

# Check if SBATCH_CONTAINER_SAVE is set, prompt if not
if [ -z "$SBATCH_CONTAINER_SAVE" ]; then
    read -e -p "SBATCH_CONTAINER_SAVE is not defined. Please enter path to save sqsh file, leave blank to not save: " container_save
    export SBATCH_CONTAINER_SAVE="$container_save"
fi

# Check if SBATCH_ACCOUNT is set, prompt if not
if [ -z "$SBATCH_ACCOUNT" ]; then
    echo "====== Available accounts ======"
    sacctmgr show user where name=$USER format=user,account%40,defaultaccount%40
    echo "====== End of available accounts ======"
    read -e -p "SBATCH_ACCOUNT is not defined. Please enter the slurm account to use. Leave blank to not specify account: " slurm_account
    export SBATCH_ACCOUNT="$slurm_account"
fi


# Check required environment variables
required_vars=(
    "SBATCH_TEMPLATE_FILE"
    "SBATCH_JOB_NAME"
    "SBATCH_STDOUT"
    "SBATCH_STDERR"
    "SBATCH_PARTITION"
    "SBATCH_EXPORT"
    "SBATCH_CONTAINER_IMAGE"
    "SBATCH_CONTAINER_MOUNTS"
    "SBATCH_CONTAINER_WORKDIR"
)

# Check if all required variables are set
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: $var is not defined"
        exit 1
    fi
done

# Extract just the keys from SBATCH_EXPORT
export SBATCH_CONTAINER_ENV=$(echo "$SBATCH_EXPORT" | tr ',' '\n' | cut -d'=' -f1 | tr '\n' ',' | sed 's/,$//')

# Review variables
echo -e "\n\nReview the following variables:"
echo "SBATCH_TEMPLATE_FILE: $SBATCH_TEMPLATE_FILE"

echo -e "\n====== Regular SLURM variables ======"
echo "SBATCH_JOB_NAME: $SBATCH_JOB_NAME"
echo "SBATCH_ACCOUNT: $SBATCH_ACCOUNT"
echo "SBATCH_STDOUT: $SBATCH_STDOUT"
echo "SBATCH_STDERR: $SBATCH_STDERR"
echo "SBATCH_PARTITION: $SBATCH_PARTITION"
echo "SBATCH_EXPORT: $SBATCH_EXPORT"

echo -e "\n====== PYXIS (Container) variables ======"
echo "SBATCH_CONTAINER_IMAGE: $SBATCH_CONTAINER_IMAGE"
echo "SBATCH_CONTAINER_SAVE: $SBATCH_CONTAINER_SAVE"
echo "SBATCH_CONTAINER_MOUNTS: $SBATCH_CONTAINER_MOUNTS"
echo "SBATCH_CONTAINER_WORKDIR: $SBATCH_CONTAINER_WORKDIR"
echo "SBATCH_CONTAINER_ENV: $SBATCH_CONTAINER_ENV"

if [ "$SLURM_MODE" = "sbatch" ]; then
    # Create temporary file and copy template
    GENERATED_SRC=$(mktemp "/tmp/${SBATCH_TEMPLATE_FILE##*/}.XXXX")
    cp "$SBATCH_TEMPLATE_FILE" "$GENERATED_SRC"

    # Replace placeholders in the template
    sed -i "s#%{sbatch_job_name}#$SBATCH_JOB_NAME#g" "$GENERATED_SRC"
    sed -i "s#%{sbatch_stdout}#$SBATCH_STDOUT#g" "$GENERATED_SRC"
    sed -i "s#%{sbatch_stderr}#$SBATCH_STDERR#g" "$GENERATED_SRC"
    sed -i "s#%{sbatch_partition}#$SBATCH_PARTITION#g" "$GENERATED_SRC"
    sed -i "s#%{sbatch_export}#$SBATCH_EXPORT#g" "$GENERATED_SRC"
    sed -i "s#%{sbatch_container_image}#$SBATCH_CONTAINER_IMAGE#g" "$GENERATED_SRC"
    sed -i "s#%{sbatch_container_mounts}#$SBATCH_CONTAINER_MOUNTS#g" "$GENERATED_SRC"
    sed -i "s#%{sbatch_container_workdir}#$SBATCH_CONTAINER_WORKDIR#g" "$GENERATED_SRC"
    sed -i "s#%{sbatch_container_env}#$SBATCH_CONTAINER_ENV#g" "$GENERATED_SRC"

    if [ ! -z "$SBATCH_CONTAINER_SAVE" ]; then
        sed -i "s#%{sbatch_container_save}#$SBATCH_CONTAINER_SAVE#g" "$GENERATED_SRC"
    else
        sed -i "/%{sbatch_container_save}/d" "$GENERATED_SRC"
    fi

    if [ ! -z "$SBATCH_ACCOUNT" ]; then
        sed -i "s#%{sbatch_account}#$SBATCH_ACCOUNT#g" "$GENERATED_SRC"
    else
        sed -i "/%{sbatch_account}/d" "$GENERATED_SRC"
    fi

    if [ ! -z "$SBATCH_NODELIST" ]; then
        sed -i "s#%{sbatch_nodelist}#$SBATCH_NODELIST#g" "$GENERATED_SRC"
    else
        sed -i "/%{sbatch_nodelist}/d" "$GENERATED_SRC"
    fi

    echo "The generated sbatch script is at $GENERATED_SRC"
    SLURM_CMD="sbatch $GENERATED_SRC"
    echo "Running command: $SLURM_CMD"

elif [ "$SLURM_MODE" = "srun" ]; then
    # srun mode - build and run srun command
    SRUN_FLAGS="--job-name=$SBATCH_JOB_NAME \
        --partition=$SBATCH_PARTITION \
        --export=$SBATCH_EXPORT \
        --container-image=$SBATCH_CONTAINER_IMAGE \
        --container-mounts=$SBATCH_CONTAINER_MOUNTS \
        --container-workdir=$SBATCH_CONTAINER_WORKDIR \
        --container-env=$SBATCH_CONTAINER_ENV \
        --container-mount-home"

    if [ ! -z "$SBATCH_NODELIST" ]; then
        SRUN_FLAGS="$SRUN_FLAGS --nodelist=$SBATCH_NODELIST"
    fi

    if [ ! -z "$SBATCH_ACCOUNT" ]; then
        SRUN_FLAGS="$SRUN_FLAGS --account=$SBATCH_ACCOUNT"
    fi

    if [ ! -z "$SBATCH_CONTAINER_SAVE" ]; then
        SRUN_FLAGS="$SRUN_FLAGS --container-save=$SBATCH_CONTAINER_SAVE"
    fi

    SLURM_CMD="srun $SRUN_FLAGS $EXTRA_SRUN_FLAGS $SBATCH_TEMPLATE_FILE"

    echo "Running srun command: $SLURM_CMD"
else
    echo "Error: SLURM_MODE is not set"
    exit 1
fi

# Ask for confirmation
read -p "Continue? [y/n] " answer
if [ "$answer" != "y" ]; then
    echo "Aborting..."
    exit 1
fi

eval "$SLURM_CMD"
