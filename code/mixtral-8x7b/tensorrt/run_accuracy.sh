#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

if ! service docker status > /dev/null; then
  sudo service docker start
  sudo service docker status
  sleep 1
fi

IMG_NAME=mixtral-accuracy_checker
ACCURACY_FILES_DIR=/tmp/

CHECKPOINT_PATH=""
MLPERF_ACCURACY_FILE=""
DATASET_FILE=""
MODULE_PATH=""

# Parse named arguments
while [ $# -gt 0 ]; do
  case "$1" in
    --module-path=*)
      MODULE_PATH="${1#*=}"
      ;;
    --checkpoint-path=*)
      CHECKPOINT_PATH="${1#*=}"
      ;;
    --mlperf-accuracy-file=*)
      MLPERF_ACCURACY_FILE="${1#*=}"
      ;;
    --dataset-file=*)
      DATASET_FILE="${1#*=}"
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
  shift
done

echo "Module Path: $MODULE_PATH"
echo "Checkpoint Path: $CHECKPOINT_PATH"
echo "MLPerf Accuracy File: $MLPERF_ACCURACY_FILE"
echo "Dataset File: $DATASET_FILE"

TMP_DIR=$(mktemp -d -p $ACCURACY_FILES_DIR -t accuracy-check-XXXXXXXXXX)
CMD="python3.9 /work/$(basename $TMP_DIR)/$(basename $MODULE_PATH) --checkpoint-path $CHECKPOINT_PATH --mlperf-accuracy-file /work/$(basename $TMP_DIR)/$(basename $MLPERF_ACCURACY_FILE) --dataset-file $DATASET_FILE --dtype int32 --n_workers 2"

sudo docker build -q -t $IMG_NAME --network host - < /work/code/mixtral-8x7b/tensorrt/Dockerfile.accuracy
CONTAINER=$(sudo docker run -d -it -w /work -v ${MLPERF_SCRATCH_PATH}:${MLPERF_SCRATCH_PATH} $IMG_NAME)

cp $(dirname $MODULE_PATH)/* $TMP_DIR/
cp $MLPERF_ACCURACY_FILE $TMP_DIR/
sudo docker cp $TMP_DIR/ $CONTAINER:/work/

echo "Running command in accuracy container: ${CMD}"
sudo docker exec -it $CONTAINER python -c "import nltk; nltk.download('punkt_tab')"
sudo docker exec -it $CONTAINER bash -i -c "$CMD"

sudo docker stop $CONTAINER > /dev/null
rm -rf ${TMP_DIR}
