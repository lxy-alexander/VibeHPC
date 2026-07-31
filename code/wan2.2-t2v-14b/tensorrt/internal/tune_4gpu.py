#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
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

"""
Tuning script for WAN22-A14B on 4-GPU machine.

This script searches for optimal parallelism configurations by testing
different combinations of:
- ulysses_size: Sequence parallelism (must divide latent_height AND attention_heads)
- cfg_size: CFG (classifier-free guidance) parallelism
- dp_size: Data parallelism
- tp_size: Tensor parallelism
- fsdp_size: Fully Sharded Data Parallel

WAN22-A14B Constraints (720p, 81 frames):
- latent_height = 720 // 8 = 90
- num_attention_heads = 40
- ulysses_size must divide BOTH 90 and 40
- Valid ulysses_size values: 1, 2, 5, 10

Usage:
    python code/wan22-a14b/tensorrt/scripts/tune_4gpu.py [--results_file results.csv] [--quick]
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import List, Optional, Tuple


# WAN22-A14B model constants (720p resolution)
LATENT_HEIGHT = 720 // 8  # = 90
NUM_ATTENTION_HEADS = 40
# Valid ulysses_size values: must divide both 90 and 40
# Divisors of 90: 1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, 90
# Divisors of 40: 1, 2, 4, 5, 8, 10, 20, 40
# Common: 1, 2, 5, 10
VALID_ULYSSES_SIZES = [1, 2, 5, 10]

# cp_size and ring_size must divide latent_height (90)
# For 4-GPU configs, practical values are those ≤ 4 that divide 90
# Divisors of 90 that are ≤ 4: 1, 2, 3
# Note: 4 does NOT divide 90 (90/4 = 22.5)
VALID_CP_SIZES = [1, 2, 3]
VALID_RING_SIZES = [1, 2, 3]


@dataclass
class ParallelConfig:
    """Parallelism configuration for WAN22-A14B."""
    ulysses_size: int = 1
    cfg_size: int = 1
    dp_size: int = 1
    tp_size: int = 1
    fsdp_size: int = 1
    cp_size: int = 1
    ring_size: int = 1
    
    @property
    def sequence_parallel_size(self) -> int:
        """Sequence parallelism GPUs = ulysses * cfg * ring * cp."""
        return self.ulysses_size * self.cfg_size * self.ring_size * self.cp_size
    
    def validate(self) -> Tuple[bool, str]:
        """
        Validate configuration against WAN22-A14B constraints.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check ulysses_size constraint
        if self.ulysses_size not in VALID_ULYSSES_SIZES:
            return False, (
                f"ulysses_size={self.ulysses_size} invalid. "
                f"Must divide both latent_height ({LATENT_HEIGHT}) and "
                f"num_attention_heads ({NUM_ATTENTION_HEADS}). "
                f"Valid values: {VALID_ULYSSES_SIZES}"
            )
        
        # Check cp_size constraint - must divide latent_height (90)
        if self.cp_size not in VALID_CP_SIZES:
            return False, (
                f"cp_size={self.cp_size} invalid. "
                f"Must divide latent_height ({LATENT_HEIGHT}). "
                f"Valid values for 4-GPU: {VALID_CP_SIZES}"
            )
        
        # Check ring_size constraint - must divide latent_height (90)
        if self.ring_size not in VALID_RING_SIZES:
            return False, (
                f"ring_size={self.ring_size} invalid. "
                f"Must divide latent_height ({LATENT_HEIGHT}). "
                f"Valid values for 4-GPU: {VALID_RING_SIZES}"
            )
        
        # cfg_size typically 1 or 2 (for positive/negative CFG batches)
        if self.cfg_size > 2:
            return False, f"cfg_size={self.cfg_size} unusual, typically 1 or 2"
        
        return True, ""
    
    def to_args(self) -> str:
        """Convert config to command line arguments."""
        args = []
        if self.ulysses_size != 1:
            args.append(f"--wan22_ulysses_size={self.ulysses_size}")
        if self.cfg_size != 1:
            args.append(f"--wan22_cfg_size={self.cfg_size}")
        if self.dp_size != 1:
            args.append(f"--wan22_dp_size={self.dp_size}")
        if self.tp_size != 1:
            args.append(f"--wan22_tp_size={self.tp_size}")
        if self.fsdp_size != 1:
            args.append(f"--wan22_fsdp_size={self.fsdp_size}")
        if self.cp_size != 1:
            args.append(f"--wan22_cp_size={self.cp_size}")
        if self.ring_size != 1:
            args.append(f"--wan22_ring_size={self.ring_size}")
        return " ".join(args) if args else ""
    
    def __str__(self) -> str:
        parts = []
        if self.ulysses_size != 1:
            parts.append(f"ulysses={self.ulysses_size}")
        if self.cfg_size != 1:
            parts.append(f"cfg={self.cfg_size}")
        if self.dp_size != 1:
            parts.append(f"dp={self.dp_size}")
        if self.tp_size != 1:
            parts.append(f"tp={self.tp_size}")
        if self.fsdp_size != 1:
            parts.append(f"fsdp={self.fsdp_size}")
        if self.cp_size != 1:
            parts.append(f"cp={self.cp_size}")
        if self.ring_size != 1:
            parts.append(f"ring={self.ring_size}")
        return ", ".join(parts) if parts else "baseline (all=1)"


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    config: ParallelConfig
    qps: float
    latency_ms: float
    success: bool
    error_msg: str = ""
    duration_sec: float = 0.0


def generate_4gpu_configs() -> List[ParallelConfig]:
    """
    Generate valid parallelism configurations for 4 GPUs.
    
    WAN22-A14B constraints (720p, 81 frames):
    - latent_height = 90, num_attention_heads = 40
    - Valid ulysses_size: 1, 2, 5, 10
    - cfg_size: typically 1 or 2
    """
    configs = []
    
    # === Sequence Parallelism Strategies ===
    # These use ulysses * cfg * ring * cp GPUs
    
    # Strategy 1: ulysses=2, cfg=2 -> 4 GPUs (recommended for diffusion)
    configs.append(ParallelConfig(ulysses_size=2, cfg_size=2))
    
    # Strategy 2: ulysses=2, cp=2 -> 4 GPUs
    configs.append(ParallelConfig(ulysses_size=2, cp_size=2))
    
    # Strategy 3: ulysses=2, ring=2 -> 4 GPUs
    configs.append(ParallelConfig(ulysses_size=2, ring_size=2))
    
    # Strategy 4: cfg=2, cp=2 -> 4 GPUs
    configs.append(ParallelConfig(cfg_size=2, cp_size=2))
    
    # Strategy 5: cfg=2, ring=2 -> 4 GPUs
    configs.append(ParallelConfig(cfg_size=2, ring_size=2))
    
    # Strategy 6: cp=2, ring=2 -> 4 GPUs
    configs.append(ParallelConfig(cp_size=2, ring_size=2))
    
    # Strategy 7: Context parallelism + data parallel (cp=2, dp=2 -> 4 GPUs)
    # Note: cp_size=4 is INVALID because 90 (latent_height) is not divisible by 4
    configs.append(ParallelConfig(cp_size=2, dp_size=2))
    
    # Strategy 8: Ring attention + data parallel (ring=2, dp=2 -> 4 GPUs)
    # Note: ring_size=4 is INVALID because 90 (latent_height) is not divisible by 4
    configs.append(ParallelConfig(ring_size=2, dp_size=2))
    
    # === Data/Model Parallelism Strategies ===
    # These don't use sequence parallelism GPUs the same way
    
    # Strategy 9: Pure data parallelism (independent inference)
    configs.append(ParallelConfig(dp_size=4))
    
    # Strategy 10: Tensor parallelism (model split across GPUs)
    configs.append(ParallelConfig(tp_size=4))
    
    # Strategy 11: FSDP (sharded model parameters)
    configs.append(ParallelConfig(fsdp_size=4))
    
    # === Mixed Strategies ===
    
    # Strategy 12: ulysses + data parallel
    configs.append(ParallelConfig(ulysses_size=2, dp_size=2))
    
    # Strategy 13: cfg + data parallel
    configs.append(ParallelConfig(cfg_size=2, dp_size=2))
    
    # Strategy 14: tensor + data parallel
    configs.append(ParallelConfig(tp_size=2, dp_size=2))
    
    # Strategy 15: ulysses + tensor parallel
    configs.append(ParallelConfig(ulysses_size=2, tp_size=2))
    
    # Strategy 16: cfg + tensor parallel
    configs.append(ParallelConfig(cfg_size=2, tp_size=2))
    
    # Strategy 17: FSDP + data parallel
    configs.append(ParallelConfig(fsdp_size=2, dp_size=2))
    
    # Strategy 18: Baseline single GPU (for comparison)
    configs.append(ParallelConfig())
    
    # Validate all configs
    valid_configs = []
    for config in configs:
        is_valid, error = config.validate()
        if is_valid:
            valid_configs.append(config)
        else:
            print(f"Skipping invalid config {config}: {error}")
    
    return valid_configs


def generate_quick_configs() -> List[ParallelConfig]:
    """Generate a smaller set of configs for quick testing."""
    configs = [
        #ParallelConfig(),  # Baseline single GPU
        ParallelConfig(ulysses_size=2, cfg_size=1),  # Recommended
        ParallelConfig(dp_size=4),  # Simple data parallel
        ParallelConfig(cfg_size=2, dp_size=1),  # CFG + data
        ParallelConfig(tp_size=2),  # Tensor parallel
    ]
    
    # Validate
    valid_configs = []
    for config in configs:
        is_valid, error = config.validate()
        if is_valid:
            valid_configs.append(config)
        else:
            print(f"Skipping invalid config {config}: {error}")
    
    return valid_configs


def parse_qps_from_output(output: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse QPS and latency from LoadGen output.
    
    Returns:
        Tuple of (qps, latency_ms) or (None, None) if not found
    """
    qps = None
    latency = None
    
    # Look for Offline QPS pattern
    qps_patterns = [
        r"Samples per second:\s*([\d.]+)",
        r"Result is:\s*([\d.]+)",
        r"Offline results:\s*([\d.]+)\s*QPS",
        r"samples_per_second\s*:\s*([\d.]+)",
        r"Scheduled samples per second\s*:\s*([\d.]+)",
    ]
    
    for pattern in qps_patterns:
        match = re.search(pattern, output)
        if match:
            qps = float(match.group(1))
            break
    
    # Look for latency pattern
    latency_patterns = [
        r"Mean latency \(ns\)\s*:\s*([\d.]+)",
        r"mean_latency_ns\s*:\s*([\d.]+)",
        r"Latency\s*:\s*([\d.]+)\s*ms",
    ]
    
    for pattern in latency_patterns:
        match = re.search(pattern, output)
        if match:
            latency = float(match.group(1))
            # Convert ns to ms if needed
            if "ns" in pattern:
                latency = latency / 1e6
            break
    
    return qps, latency


def run_benchmark(
    config: ParallelConfig,
    min_query_count: int = 4,
    min_duration_ms: int = 60000,
    timeout_sec: int = 6000,
    accuracy_mode: bool = False,
    dump_videos: bool = False,
    video_output_dir: Optional[str] = None,
    video_fps: int = 16,
) -> BenchmarkResult:
    """
    Run a single benchmark with the given parallelism configuration.
    
    Args:
        config: Parallelism configuration to test
        min_query_count: Minimum queries to run
        min_duration_ms: Minimum duration in milliseconds
        timeout_sec: Timeout in seconds
    
    Returns:
        BenchmarkResult with QPS and success status
    """
    print(f"\n{'='*60}")
    print(f"Testing config: {config}")
    print(f"{'='*60}")
    
    # Validate config first
    is_valid, error = config.validate()
    if not is_valid:
        print(f"INVALID CONFIG: {error}")
        return BenchmarkResult(
            config=config,
            qps=0.0,
            latency_ms=0.0,
            success=False,
            error_msg=f"Invalid config: {error}",
        )
    
    parallel_args = config.to_args()
    
    test_mode = "AccuracyOnly" if accuracy_mode else "PerformanceOnly"
    
    # Build video dump args
    video_args = ""
    if dump_videos:
        video_args = " --wan22_dump_videos=true"
        if video_output_dir:
            video_args += f" --wan22_video_output_dir={video_output_dir}"
        if video_fps != 16:
            video_args += f" --wan22_video_fps={video_fps}"
    
    cmd = (
        f'make run_harness RUN_ARGS="'
        f'--benchmarks=wan22-a14b '
        f'--scenarios=offline '
        f'--test_mode={test_mode} '
        f'--min_query_count={min_query_count} '
        f'--min_duration={min_duration_ms} '
        f'{parallel_args}{video_args}"'
    )
    
    print(f"Command: {cmd}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        
        duration = time.time() - start_time
        output = result.stdout + result.stderr
        
        # Check for errors
        if result.returncode != 0:
            error_msg = "Command failed with non-zero exit code"
            if "CUDA out of memory" in output:
                error_msg = "CUDA OOM"
            elif "RuntimeError" in output:
                # Extract actual error
                match = re.search(r"RuntimeError: (.+?)(?:\n|$)", output)
                if match:
                    error_msg = f"RuntimeError: {match.group(1)[:100]}"
                else:
                    error_msg = "RuntimeError"
            elif "must be divisible" in output:
                # Extract the specific divisibility error
                match = re.search(r"(\w+)'s chunk dimension \d+ must be divisible by .+", output)
                if match:
                    error_msg = f"Chunk divisibility error: {match.group(0)[:100]}"
                else:
                    error_msg = "Parallelism constraint violation"
            
            print(f"FAILED: {error_msg}")
            return BenchmarkResult(
                config=config,
                qps=0.0,
                latency_ms=0.0,
                success=False,
                error_msg=error_msg,
                duration_sec=duration,
            )
        
        # Parse QPS from output
        qps, latency = parse_qps_from_output(output)
        
        if qps is None:
            print("WARNING: Could not parse QPS from output")
            # Print last part of output for debugging
            lines = output.strip().split('\n')
            print("Output tail:")
            for line in lines[-20:]:
                print(f"  {line}")
            qps = 0.0
        
        if latency is None:
            latency = 0.0
        
        print(f"SUCCESS: QPS={qps:.4f}, Latency={latency:.2f}ms, Duration={duration:.1f}s")
        
        return BenchmarkResult(
            config=config,
            qps=qps,
            latency_ms=latency,
            success=True,
            duration_sec=duration,
        )
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"TIMEOUT after {timeout_sec}s")
        return BenchmarkResult(
            config=config,
            qps=0.0,
            latency_ms=0.0,
            success=False,
            error_msg=f"Timeout after {timeout_sec}s",
            duration_sec=duration,
        )
    except Exception as e:
        duration = time.time() - start_time
        print(f"ERROR: {str(e)}")
        return BenchmarkResult(
            config=config,
            qps=0.0,
            latency_ms=0.0,
            success=False,
            error_msg=str(e),
            duration_sec=duration,
        )


def save_results(results: List[BenchmarkResult], filename: str):
    """Save benchmark results to CSV file."""
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ulysses_size", "cfg_size", "dp_size", "tp_size", "fsdp_size",
            "cp_size", "ring_size", "qps", "latency_ms", "success",
            "error_msg", "duration_sec"
        ])
        
        for r in results:
            writer.writerow([
                r.config.ulysses_size,
                r.config.cfg_size,
                r.config.dp_size,
                r.config.tp_size,
                r.config.fsdp_size,
                r.config.cp_size,
                r.config.ring_size,
                r.qps,
                r.latency_ms,
                r.success,
                r.error_msg,
                r.duration_sec,
            ])
    
    print(f"\nResults saved to: {filename}")


def print_summary(results: List[BenchmarkResult]):
    """Print summary of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print(f"\nTotal configs tested: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        # Sort by QPS descending
        successful.sort(key=lambda x: x.qps, reverse=True)
        
        print("\n" + "-" * 80)
        print("TOP CONFIGURATIONS (by QPS):")
        print("-" * 80)
        print(f"{'Rank':<5} {'Config':<45} {'QPS':<12} {'Latency(ms)':<12}")
        print("-" * 80)
        
        for i, r in enumerate(successful[:10], 1):
            config_str = str(r.config)[:43]
            print(f"{i:<5} {config_str:<45} {r.qps:<12.4f} {r.latency_ms:<12.2f}")
        
        print("\n" + "-" * 80)
        print("BEST CONFIGURATION:")
        print("-" * 80)
        best = successful[0]
        print(f"  Config: {best.config}")
        print(f"  QPS: {best.qps:.4f}")
        print(f"  Latency: {best.latency_ms:.2f} ms")
        print(f"  Duration: {best.duration_sec:.1f} s")
        
        # Print recommended command
        args = best.config.to_args()
        print("\nRecommended command:")
        if args:
            print(f'  make run_harness RUN_ARGS="--benchmarks=wan22-a14b --scenarios=offline '
                  f'--test_mode=PerformanceOnly {args}"')
        else:
            print(f'  make run_harness RUN_ARGS="--benchmarks=wan22-a14b --scenarios=offline '
                  f'--test_mode=PerformanceOnly"')
    
    if failed:
        print("\n" + "-" * 80)
        print("FAILED CONFIGURATIONS:")
        print("-" * 80)
        for r in failed:
            print(f"  {r.config}")
            print(f"    Error: {r.error_msg}")


def main():
    parser = argparse.ArgumentParser(
        description="Tune WAN22-A14B parallelism for 4-GPU machine"
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default=f"wan22_a14b_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        help="Output CSV file for results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test with fewer configurations",
    )
    parser.add_argument(
        "--min_query_count",
        type=int,
        default=4,
        help="Minimum query count for each test (default: 4)",
    )
    parser.add_argument(
        "--min_duration",
        type=int,
        default=60000,
        help="Minimum duration in ms (default: 60000 = 1 min)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=6000,
        help="Timeout per test in seconds (default: 6000)",
    )
    parser.add_argument(
        "--config_index",
        type=int,
        default=None,
        help="Run only specific config by index (for debugging)",
    )
    parser.add_argument(
        "--list_configs",
        action="store_true",
        help="List all configurations and exit",
    )
    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="Run accuracy test instead of performance test",
    )
    parser.add_argument(
        "--dump_videos",
        action="store_true",
        help="Dump generated videos as mp4 (only with --accuracy)",
    )
    parser.add_argument(
        "--video_output_dir",
        type=str,
        default=None,
        help="Directory to save videos (default: build/logs/.../videos)",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=16,
        help="FPS for saved mp4 videos (default: 16)",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("WAN22-A14B 4-GPU Parallelism Tuning Script")
    print("=" * 80)
    print(f"\nModel constraints (720p, 81 frames):")
    print(f"  latent_height = {LATENT_HEIGHT}")
    print(f"  num_attention_heads = {NUM_ATTENTION_HEADS}")
    print(f"  valid ulysses_size = {VALID_ULYSSES_SIZES}")
    
    # Generate configurations
    if args.quick:
        configs = generate_quick_configs()
        print("\nRunning QUICK test with reduced configurations")
    else:
        configs = generate_4gpu_configs()
    
    print(f"\n{len(configs)} valid configurations:")
    print("-" * 80)
    for i, config in enumerate(configs):
        print(f"  [{i:2d}] {config}")
    
    if args.list_configs:
        return
    
    print(f"\nResults file: {args.results_file}")
    print(f"Min query count: {args.min_query_count}")
    print(f"Min duration: {args.min_duration} ms")
    print(f"Timeout: {args.timeout} s")
    
    # Filter to specific config if requested
    if args.config_index is not None:
        if 0 <= args.config_index < len(configs):
            configs = [configs[args.config_index]]
            print(f"\nRunning only config index {args.config_index}")
        else:
            print(f"Error: config_index {args.config_index} out of range [0, {len(configs)-1}]")
            sys.exit(1)
    
    # Run benchmarks
    results = []
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running benchmark...")
        result = run_benchmark(
            config,
            min_query_count=args.min_query_count,
            min_duration_ms=args.min_duration,
            timeout_sec=args.timeout,
            accuracy_mode=args.accuracy,
            dump_videos=args.dump_videos,
            video_output_dir=args.video_output_dir,
            video_fps=args.video_fps,
        )
        results.append(result)
        
        # Save intermediate results
        save_results(results, args.results_file)
    
    # Print summary
    print_summary(results)
    
    print(f"\nFull results saved to: {args.results_file}")


if __name__ == "__main__":
    main()

