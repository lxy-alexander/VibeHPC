# NVIDIA MLPerf Inference Benchmarks

## List of Benchmarks

Please refer to the `README.md` in each benchmark directory for implementation details.

### LLM Benchmarks
- [deepseek-r1](deepseek-r1/tensorrt/README.md) - DeepSeek-R1 671B MoE
- [gpt-oss-120b](gpt-oss-120b/tensorrt/README.md) - GPT-OSS 120B
- [llama2-70b](llama2-70b/tensorrt/README.md) - Llama 2 70B
- [llama3.1-8b](llama3_1-8b/tensorrt/README.md) - Llama 3.1 8B
- [llama3.1-405b](llama3_1-405b/tensorrt/README.md) - Llama 3.1 405B

### Vision Benchmarks
- [stable-diffusion-xl](stable-diffusion-xl/tensorrt/README.md) - Stable Diffusion XL (Text-to-Image)
- [qwen3-vl-235b-a22b](qwen3-vl-235b-a22b/tensorrt/README.md) - Qwen3-VL 235B (Vision-Language)
- [wan22-a14b](wan22-a14b/tensorrt/README.md) - WAN 2.2 T2V 14B (Text-to-Video)

### Other Benchmarks
- [whisper](whisper/tensorrt/README.md) - Whisper Large v3 (Speech Recognition)
- [rgat](rgat/tensorrt/README.md) - R-GAT (Graph Neural Network)
- [dlrm-v3](dlrm-v3/README.md) - DLRM v3 (Recommendation)

## Other Directories

- [common](common) - holds shared scripts to generate TensorRT optimized plan files and to run the harnesses.
- [harness](harness) - holds source codes of the harness interfacing with LoadGen.
- [plugin](plugin) - holds source codes of TensorRT plugins used by the benchmarks.
