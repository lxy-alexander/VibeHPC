git clone ${TRTLLM_URL} /opt/trtllm
cd /opt/trtllm
git checkout ${TRTLLM_HASH}
python scripts/build_wheel.py --use_ccache --benchmarks -a="100-real" --install
