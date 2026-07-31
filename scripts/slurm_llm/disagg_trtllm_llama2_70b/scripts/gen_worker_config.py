import argparse
import os
import yaml


def generate_config_files(work_dir: str,
                    cg_sizes: str,
                    ctx_tp_size: int,
                    ctx_ep_size: int,
                    ctx_pp_size: int,
                    ctx_max_batch_size: int,
                    ctx_max_num_tokens: int,
                    ctx_max_seq_len: int,
                    ctx_free_gpu_memory_fraction: float,
                    ctx_enable_attention_dp: bool,
                    gen_tp_size: int,
                    gen_ep_size: int,
                    gen_pp_size: int,
                    gen_max_batch_size: int,
                    gen_max_num_tokens: int,
                    gen_max_seq_len: int,
                    gen_enable_attention_dp: bool,
                    gen_gpu_memory_fraction: float,
                    eplb_num_slots: int,
                    mtp_size: int = 0,
                    cache_transceiver_max_num_tokens: int = 1024,
                    cache_transceiver_backend: str = 'UCX',
                    ctx_stream_interval: int = 30,
                    gen_stream_interval: int = 100,
                    gen_num_postprocess_workers: int = 4,
                    enable_iter_stats: bool = False) -> None:
    """
    Generate configuration YAML files for llama2-70b disaggregated inference.
    """
    ctx_config = {
        'build_config': {
            'max_batch_size': ctx_max_batch_size,
            'max_num_tokens': ctx_max_num_tokens,
            'max_seq_len': ctx_max_seq_len,
        },
        'max_batch_size': ctx_max_batch_size,
        'max_num_tokens': ctx_max_num_tokens,
        'max_seq_len': ctx_max_seq_len,
        'tensor_parallel_size': ctx_tp_size,
        'enable_attention_dp': True if ctx_enable_attention_dp else False,
        'pipeline_parallel_size': ctx_pp_size,
        'print_iter_log': True,
        'enable_iter_perf_stats': enable_iter_stats,
        'cuda_graph_config': None,
        'disable_overlap_scheduler': True,
        'enable_chunked_prefill': True,
        'stream_interval': ctx_stream_interval,  # How often to stream tokens back (lower = more frequent streaming)
        'kv_cache_config': {
            'enable_block_reuse': False,
            'free_gpu_memory_fraction': ctx_free_gpu_memory_fraction,
            'dtype': 'fp8',
        },
        'cache_transceiver_config': {
            'max_tokens_in_buffer': cache_transceiver_max_num_tokens,
            'backend': cache_transceiver_backend,
        },
        'sampler_type': 'TRTLLMSampler',
        'scheduler_config': {
            'capacity_scheduler_policy': 'MAX_UTILIZATION', 
            'context_chunking_policy': 'FIRST_COME_FIRST_SERVED',
        },
    }

    # CUDA Graph batch sizes for GEN servers
    # These are the batch sizes that will be pre-compiled into CUDA graphs for fast execution
    # Default: powers of 2 up to the max batch size, PLUS the max batch size itself
    # Example: for batch_size=768 → [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768]
    if cg_sizes:
        gen_cuda_graph_batch_sizes = list(map(int, cg_sizes.split(',')))
    else:
        # Generate powers of 2 up to (but not exceeding) max batch size
        default_capture_sizes = [1 << i for i in range(gen_max_batch_size.bit_length())]
        # Always include the max batch size if it's not already there
        if gen_max_batch_size not in default_capture_sizes:
            default_capture_sizes.append(gen_max_batch_size)
        gen_cuda_graph_batch_sizes = sorted(default_capture_sizes)

    gen_config = {
        'build_config': {
            'max_batch_size': gen_max_batch_size,
            'max_num_tokens': gen_max_num_tokens,
            'max_seq_len': gen_max_seq_len,
        },
        'tensor_parallel_size': gen_tp_size,
        'enable_attention_dp': True if gen_enable_attention_dp else False,
        'pipeline_parallel_size': gen_pp_size,
        'max_batch_size': gen_max_batch_size,
        'max_num_tokens': gen_max_num_tokens,
        'max_seq_len': gen_max_seq_len,
        'cuda_graph_config': {
            'enable_padding': True,
            'batch_sizes': gen_cuda_graph_batch_sizes,
        },
        'print_iter_log': True,
        'enable_iter_perf_stats': enable_iter_stats,
        'kv_cache_config': {
            'enable_block_reuse': False,
            'free_gpu_memory_fraction': gen_gpu_memory_fraction,
            'dtype': 'fp8',
        },
        'cache_transceiver_config': {
            'max_tokens_in_buffer': cache_transceiver_max_num_tokens,
            'backend': cache_transceiver_backend,
        },
        'stream_interval': gen_stream_interval,  # How often to stream tokens back (lower = more frequent streaming)
        'num_postprocess_workers': gen_num_postprocess_workers,
        'sampler_type': 'TRTLLMSampler',
        'scheduler_config': {
            'capacity_scheduler_policy': 'MAX_UTILIZATION', 
            'context_chunking_policy': 'FIRST_COME_FIRST_SERVED',
        },
    }

    # MoE config not needed for llama2-70b (not a MoE model)
    
    if mtp_size > 0:
        ctx_config['speculative_config'] = {
            'decoding_type': 'MTP',
            'num_nextn_predict_layers': mtp_size
        }
        gen_config['speculative_config'] = {
            'decoding_type': 'MTP',
            'num_nextn_predict_layers': mtp_size
        }

    ctx_config_file = os.path.join(work_dir, "ctx_config.yaml")
    gen_config_file = os.path.join(work_dir, "gen_config.yaml")
    
    with open(ctx_config_file, "w") as f:
        yaml.dump(ctx_config, f, default_flow_style=False, sort_keys=False)
    with open(gen_config_file, "w") as f:
        yaml.dump(gen_config, f, default_flow_style=False, sort_keys=False)

    print(f"Config files generated: {ctx_config_file}, {gen_config_file}")
    print(f"Gen cuda graph batch sizes: {gen_cuda_graph_batch_sizes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--work_dir", type=str, default="logs", help="Work directory")
    parser.add_argument("--ctx_tp_size", type=int, default=1, help="Tensor parallel size for context servers")
    parser.add_argument("--ctx_ep_size", type=int, default=0, help="Expert parallel size for context servers")
    parser.add_argument("--ctx_pp_size", type=int, default=1, help="Pipeline parallel size for context servers")
    parser.add_argument("--ctx_max_batch_size", type=int, default=4096, help="Max batch size for context servers")
    parser.add_argument("--ctx_max_num_tokens", type=int, default=4096, help="Max number of tokens for context servers")
    parser.add_argument("--ctx_max_seq_len", type=int, default=2048, help="Max sequence length for context servers")
    parser.add_argument("--ctx_free_gpu_memory_fraction", type=float, default=0.85, help="Free GPU memory fraction for context servers")
    parser.add_argument("--ctx_enable_attention_dp", dest='ctx_enable_attention_dp', action='store_true', help="Enable attention DP for context servers")
    parser.add_argument("--gen_tp_size", type=int, default=1, help="Tensor parallel size for generation servers")
    parser.add_argument("--gen_ep_size", type=int, default=0, help="Expert parallel size for generation servers")
    parser.add_argument("--gen_pp_size", type=int, default=1, help="Pipeline parallel size for generation servers")
    parser.add_argument("--gen_max_batch_size", type=int, default=768, help="Max batch size for generation servers")
    parser.add_argument("--gen_max_num_tokens", type=int, default=768, help="Max number of tokens for generation servers")
    parser.add_argument("--gen_max_seq_len", type=int, default=2048, help="Max sequence length for generation servers")
    parser.add_argument("--gen_enable_attention_dp", dest='gen_enable_attention_dp', action='store_true', help="Enable attention DP for generation servers")
    parser.add_argument("--gen_gpu_memory_fraction", type=float, default=0.95, help="GPU memory fraction for generation servers")
    parser.add_argument("--eplb_num_slots", type=int, default=0, help="Number of slots for eplb")
    parser.add_argument("--mtp_size", type=int, default=0, help="Number of nextn layers for MTP")
    parser.add_argument("--cache_transceiver_max_num_tokens", type=int, default=1024, help="Max number of tokens for cache transceiver")
    parser.add_argument("--cache_transceiver_backend", type=str, default="UCX", help="Cache transceiver backend: DEFAULT, UCX, or NCCL")
    parser.add_argument("--cg_sizes", type=str, default="", help="Override the default cuda graph batch sizes")
    parser.add_argument("--ctx_stream_interval", type=int, default=30, help="CTX stream interval (lower = more frequent streaming)")
    parser.add_argument("--gen_stream_interval", type=int, default=100, help="GEN stream interval (lower = more frequent streaming)")
    parser.add_argument("--gen_num_postprocess_workers", type=int, default=4, help="Number of postprocess workers per GEN server")
    parser.add_argument("--enable_iter_stats", dest='enable_iter_stats', action='store_true', help="Enable per-iteration performance stats collection (adds overhead)")

    args = parser.parse_args()

    generate_config_files(
        args.work_dir, args.cg_sizes, args.ctx_tp_size, args.ctx_ep_size, args.ctx_pp_size,
        args.ctx_max_batch_size, args.ctx_max_num_tokens, args.ctx_max_seq_len, 
        args.ctx_free_gpu_memory_fraction, args.ctx_enable_attention_dp, 
        args.gen_tp_size, args.gen_ep_size, args.gen_pp_size, args.gen_max_batch_size,
        args.gen_max_num_tokens, args.gen_max_seq_len, args.gen_enable_attention_dp, 
        args.gen_gpu_memory_fraction, args.eplb_num_slots, args.mtp_size,
        args.cache_transceiver_max_num_tokens, args.cache_transceiver_backend,
        args.ctx_stream_interval, args.gen_stream_interval,
        args.gen_num_postprocess_workers, args.enable_iter_stats
    )

