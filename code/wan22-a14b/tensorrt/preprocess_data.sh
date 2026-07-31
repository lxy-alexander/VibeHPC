
mkdir -p $MLPERF_SCRATCH_PATH/preprocessed_data/wan22-a14b

cp 3rdparty/mlc-inference/text_to_video/wan-2.2-t2v-a14b/data/vbench_prompts.txt $MLPERF_SCRATCH_PATH/preprocessed_data/wan22-a14b/prompts.txt
cp 3rdparty/mlc-inference/text_to_video/wan-2.2-t2v-a14b/data/fixed_latent.pt $MLPERF_SCRATCH_PATH/preprocessed_data/wan22-a14b/fixed_latent.pt 
