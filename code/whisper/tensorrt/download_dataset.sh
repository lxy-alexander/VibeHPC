
source code/common/file_downloads.sh

DATA_DIR=${DATA_DIR:-build/data}

export WORKSPACE_DIR=/work
export DATA_DIR=/work/build/data/whisper-large-v3
export LIBRISPEECH_DIR=${DATA_DIR}/LibriSpeech
export UTILS_DIR=/work/code/whisper/tensorrt/utils
export OUTPUT_DIR=/work/build/preprocessed_data/whisper-large-v3
mkdir -p ${LIBRISPEECH_DIR}

cd ${WORKSPACE_DIR}

# Downloads all Librispeech dev paritions
python ${UTILS_DIR}/download_librispeech.py \
    ${UTILS_DIR}/inference_librispeech.csv \
    ${LIBRISPEECH_DIR} \
    -e ${DATA_DIR}

# Consolidates all Librispeech paritions into common dir
mkdir -p ${LIBRISPEECH_DIR}/dev-all
cp -r ${LIBRISPEECH_DIR}/dev-clean/* \
      ${LIBRISPEECH_DIR}/dev-other/* \
      ${LIBRISPEECH_DIR}/dev-all/

# Coverts original Librispeech flac to wav and creates manifest file
python ${UTILS_DIR}/convert_librispeech.py \
   --input_dir ${LIBRISPEECH_DIR}/dev-all \
   --dest_dir ${DATA_DIR}/dev-all \
   --output_json ${DATA_DIR}/dev-all.json

# Repackages Librispeech samples into samples approaching 30s
python ${UTILS_DIR}/repackage_librispeech.py --manifest ${DATA_DIR}/dev-all.json \
	                              --data_dir ${DATA_DIR} \
				      --output_dir ${OUTPUT_DIR}/dev-all-repack \
				      --output_json ${OUTPUT_DIR}/dev-all-repack.json
