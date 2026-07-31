# MLPerf Inference Calibration and Quantization Details
## NVIDIA MLPerf Quantization

Post-training quantization (PTQ) requires a dynamic range for each weight and activation tensor. Quantization is symmetric for both.

### Weights

Dynamic range values are generally per-channel (or per-row for matrix multiply). In a few cases, a per-tensor value is used. We find the maximum absolute value `t` of any element of the channel or tensor, and the dynamic range is then `[-t,t]`.

### Activations (for TRT implicit quantization)

For each activation tensor, we use a distinct dynamic range that applies across the entire tensor. We invoke the model on a set of representative inputs in FP32 precision, and create a per-tensor histogram of absolute values. The histogram initially uses 1024 equal-range bins whose range is set by the initial batch, but dynamically resizes by doubling the number of bins as necessary to accommodate the range of subsequent batches. Call this histogram, which has [`power-of-2`] bins, where all data elements are guaranteed to fall into one of the bins, the "starting histogram". We then apply one of two methods, as chosen by the application.

- Fractional: we compute some user-specified fraction (1, or very close to 1) of the maximum absolute value of the tensor.
- Entropy: for each bin B in the starting histogram, we compute a divergence value as follows:
    - Create a truncated histogram where each bin has the same range and count as the original, except that all elements in bins beyond B are considered to be in B, and all bins beyond B are removed.
    - Create a coarse histogram by discretizing the truncated histogram into 127 bins of equal range between 0 and the midpoint of B, placing all elements in the final bin of the truncated histogram into the final bin of the coarse histogram.
    - Compute the KL-divergence between the distributions represented by the coarse histogram and the truncated histogram.
    - The dynamic range chosen is the center of the bin which minimizes divergence.

### Additional Details

A number of minor modifications are applied to this basic algorithm, including discarding the first bin in the histogram (which typically contains a huge number of noise activations) immediately after it has been built how empty bins are treated when computing divergence. For some operations which are not expected to change dynamic range (e.g. max-pooling, concatenation) we propagate dynamic range from the output to the input(s).

### Quantization in Plugins

NVIDIA's closed division submissions primarily use TensorRT, which implements the scheme described above. Where plugins are used, weight quantization is performed as described above, and activation quantization uses dynamic range values computed using TensorRT on the original network. The plugins access these values through TensorRT's calibration cache.

### LLM Quantization (Explicit quantization)

LLM submissions use FP8 or FP4 if the NVIDIA accelerator supports that feature. Quantization details for such submissions:

All FP8 quantization (including weight quantization) is symmetric, per-tensor. For FP4 quantization, a per-block quantization is added additionally to provide better accuracy. A block is defined as a group of consecutive value within the tensor. 

The dynamic range for per-tensor quantization is defined to be the 99.9 percentile value observed in the values of that tensor when the model is executed in FP16 or FP32 on the calibration dataset. For per-block quantization, the dynamic range is computed dynamically during runtime. For a tensor/block with dynamic range dr, the quantized value x_q is computed from the unquantized value x as:

```
x_q = round(clip(x / dr * m, -m, m))
```
where m is the max of the format, for example 448 for FP8, and ties are rounded to even.

When quantizing Llama3.1 8b, Llama2-70B and Llama3.1-405B, the following tensors are quantized in each decoder.

- Linear (including dense and QKV linear) layer inputs and weights
- Attention: Q, K inputs after RoPE, and V inputs
- MLP Layer inputs and weights
- KV Cache entries
- On accelerators which support FP4, the following layers are quantized:
    - Selected linear and MLP layers* inputs and weights for transformer layer.
    - KV Cache entries

When quantizing DeepSeek-R1 for accelerators which support FP4, for each decoder layer:

- We use bf16 for MLA GEMM's input and weights
    - except WO_GEMM in which the weight may be quantized to nvfp4 if applicable
- The weights and activations of MLP in the first 3 layers are in nvfp4
- MOE layer: experts' weights and activations are in nvfp4
- KV-Cache entries are in fp8

Note: *the quantization is done through NVIDIA ModelOpt, applied based on ModelOpt heuristic search.

### WAN-2.2-T2V-A14B Quantization

WAN-2.2-T2V-A14B is a text-to-video diffusion model. NVIDIA's submission uses FP8 quantization for the DiT (Diffusion Transformer) component:

- ViT attention layers: Q, K, V linear inputs and weights are FP8 quantized
- GEMM operations: All major GEMM operations in the transformer blocks are FP8 quantized
- The VAE and text encoder remain in higher precision (BF16/FP16) to maintain output quality

The quantization is performed using NVIDIA ModelOpt with per-tensor FP8 scaling factors derived from calibration on a representative set of prompts.

### Qwen3-VL-235B-A22B Quantization

Qwen3-VL-235B-A22B is a vision-language model with Mixture-of-Experts (MoE) architecture. NVIDIA's submission uses FP4 quantization:

- GEMM operations: All linear layer weights in the transformer blocks are quantized to NVFP4 (NVIDIA FP4 format)
- Per-block quantization is applied additionally to provide better accuracy
- Vision encoder remains in higher precision to maintain visual understanding quality

The quantization is performed using NVIDIA ModelOpt. For detailed quantization instructions, see `code/qwen3-vl-235b-a22b/vllm/README.md`.

### Open Division Quantization

If applicable, for Open Division submissions, quantization details are in the READMEs attached to each individual Open Division submission.
