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

"""
Request lifecycle tracking utils.
"""

from __future__ import annotations
from datetime import datetime
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict


class RequestMetrics:
    """
    Lightweight class for tracking per-request lifecycle metrics.

    Tracks complete lifecycle from ZMQ recv to ZMQ push with minimal overhead.
    Only uses timestamp recording during request processing.
    """
    __slots__ = ('request_id', 'input_seq_len', 'output_seq_len',
                 'zmq_recv_time', 'http_post_time', 'first_token_time',
                 'chunk_times', 'chunk_texts', 'chunk_token_counts',
                 'final_token_time', 'zmq_push_time')

    def __init__(self, request_id: int, input_seq_len: int, zmq_recv_time: float):
        self.request_id = request_id
        self.input_seq_len = input_seq_len
        self.output_seq_len = 0
        self.zmq_recv_time = zmq_recv_time
        self.http_post_time = None
        self.first_token_time = None
        self.chunk_times = []  # Track all chunk arrival times
        self.chunk_texts = []  # Store chunk text for verification/debugging
        self.chunk_token_counts = None  # Calculated from tokenized output at finalization
        self.final_token_time = None
        self.zmq_push_time = None

    def mark_http_post(self):
        """Mark when HTTP POST is issued."""
        if self.http_post_time is None:
            self.http_post_time = time.time()

    def mark_first_token(self, chunk_text: str):
        """Mark when first token arrives and store chunk text."""
        if self.first_token_time is None:
            self.first_token_time = time.time()
            self.chunk_times.append(self.first_token_time)
            self.chunk_texts.append(chunk_text)

    def mark_chunk(self, chunk_text: str):
        """Mark when a chunk arrives and store chunk text (no tokenization overhead)."""
        self.chunk_times.append(time.time())
        self.chunk_texts.append(chunk_text)

    def mark_completion(self, output_seq_len: int):
        """Mark completion after ZMQ push."""
        completion_time = time.time()
        self.final_token_time = completion_time
        self.zmq_push_time = completion_time
        self.output_seq_len = output_seq_len


    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with derived metrics for pandas DataFrame."""
        # Helper to convert timestamp to datetime string
        def to_dt_str(ts):
            return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f') if ts else None

        # Raw timestamps (as datetime strings) and counts
        data = {
            'request_id': self.request_id,
            'input_seq_len': self.input_seq_len,
            'output_seq_len': self.output_seq_len,
            'num_chunks': len(self.chunk_times),
            'zmq_recv_time': to_dt_str(self.zmq_recv_time),
            'http_post_time': to_dt_str(self.http_post_time),
            'first_token_time': to_dt_str(self.first_token_time),
            'final_token_time': to_dt_str(self.final_token_time),
            'zmq_push_time': to_dt_str(self.zmq_push_time),
        }

        # Derived metrics
        if self.zmq_push_time:
            data['total_time_s'] = self.zmq_push_time - self.zmq_recv_time

        if self.http_post_time:
            data['http_queue_latency_s'] = self.http_post_time - self.zmq_recv_time

        if self.first_token_time and self.http_post_time:
            data['ttft_s'] = self.first_token_time - self.http_post_time

        # TPOT LoadGen Estimate: (final_time - first_time) / (OSL - 1)
        if self.first_token_time and self.final_token_time and self.output_seq_len > 1:
            time_after_first_token = self.final_token_time - self.first_token_time
            data['tpot_loadgen_estimate_ms'] = time_after_first_token / (self.output_seq_len - 1) * 1000

        # TPOT Chunk-Based: Calculated offline if chunk_token_counts available
        if self.chunk_token_counts and len(self.chunk_token_counts) == len(self.chunk_times) > 1:
            weighted_tpots = []
            for i in range(1, len(self.chunk_times)):
                time_delta = self.chunk_times[i] - self.chunk_times[i-1]
                num_tokens = self.chunk_token_counts[i]
                if num_tokens > 0:
                    weighted_tpots.append((time_delta / num_tokens * 1000, num_tokens))

            if weighted_tpots:
                total_tokens = sum(w for _, w in weighted_tpots)
                data['tpot_chunk_based_ms'] = sum(tpot * w for tpot, w in weighted_tpots) / total_tokens
                data['tokens_per_chunk_mean'] = sum(self.chunk_token_counts) / len(self.chunk_token_counts)

        # Inter-Chunk Latency
        if len(self.chunk_times) > 1:
            inter_chunk_latencies = [
                (self.chunk_times[i] - self.chunk_times[i-1]) * 1000  # ms
                for i in range(1, len(self.chunk_times))
            ]
            data['chunk_latency_mean_ms'] = sum(inter_chunk_latencies) / len(inter_chunk_latencies)

        return data


class WorkerMetricsCollector:
    """
    Collector for worker metrics with pandas DataFrame export.

    Always collects metrics (minimal overhead), dumps to pickle at shutdown.
    Optionally logs metrics to worker log when verbose=True.
    """

    def __init__(self, verbose: bool, log_dir: str, worker_id: int = 0):
        self.verbose = verbose
        self.log_dir = Path(log_dir)
        self.worker_id = worker_id
        self.active_requests: Dict[int, RequestMetrics] = {}

    def start_request(self, request_id: int, input_seq_len: int) -> RequestMetrics:
        """Create and track a new request (captures ZMQ recv time)."""
        zmq_recv_time = time.time()
        metrics = RequestMetrics(request_id, input_seq_len, zmq_recv_time)
        self.active_requests[request_id] = metrics
        return metrics

    def finalize_request(self, request_id: int, output_seq_len: int):
        """Mark request completion after ZMQ push."""
        if request_id not in self.active_requests:
            return

        metrics = self.active_requests[request_id]
        metrics.mark_completion(output_seq_len)

        # Only log if verbose (chunk-based TPOT will be in pickle only)
        if self.verbose:
            logging.info(f"[{request_id}] {metrics.to_dict()}")

    def dump_to_pickle(self, tokenizer):
        """Dump all metrics to pandas DataFrame pickle file with offline chunk-based TPOT."""
        if not self.active_requests:
            return

        try:
            # Calculate chunk token counts offline for chunk-based TPOT
            for metrics in self.active_requests.values():
                if metrics.chunk_texts:
                    metrics.chunk_token_counts = [len(tokenizer.encode(text).ids) for text in metrics.chunk_texts]

            metrics_data = [m.to_dict() for m in self.active_requests.values()]
            if not metrics_data:
                return

            df = pd.DataFrame(metrics_data)
            # Save to same directory as worker logs (use worker_id for cleaner naming)
            pickle_file = self.log_dir / f"request_metrics_worker_{self.worker_id}.pkl"
            df.to_pickle(pickle_file)
            logging.info(f"Dumped {len(metrics_data)} request metrics to {pickle_file}")

        except Exception as e:
            logging.warning(f"Failed to dump metrics to pickle: {e}")


def aggregate_and_plot_worker_metrics(log_dir: str):
    """Aggregate worker metrics and create plots similar to progress display."""
    log_path = Path(log_dir)
    pickle_files = list(log_path.glob("request_metrics_worker_*.pkl"))

    if not pickle_files:
        return

    # Load and combine all worker metrics
    dfs = []
    for pickle_file in pickle_files:
        try:
            dfs.append(pd.read_pickle(pickle_file))
        except Exception as e:
            logging.warning(f"Failed to load {pickle_file}: {e}")

    if not dfs:
        return

    df = pd.concat(dfs, ignore_index=True)
    df['zmq_recv_ts'] = pd.to_datetime(df['zmq_recv_time'])
    df = df.sort_values('zmq_recv_ts').reset_index(drop=True)

    logging.info(f"Aggregated {len(df)} requests from {len(pickle_files)} workers")

    # Create plots
    df_plot = df[df['ttft_s'].notna()].copy()
    if df_plot.empty:
        return

    request_indices = list(range(len(df_plot)))
    http_queue_ms = df_plot['http_queue_latency_s'].fillna(0).values * 1000
    ttft_ms = df_plot['ttft_s'].fillna(0).values * 1000
    tpot_ms = df_plot['tpot_loadgen_estimate_ms'].fillna(0).values

    _, axs = plt.subplots(2, 1, figsize=(12, 10))

    # TTFT Breakdown
    axs[0].bar(request_indices, http_queue_ms, label='HTTP Queue', color='#FFB347')
    axs[0].bar(request_indices, ttft_ms, bottom=http_queue_ms, label='TTFT', color='#6495ED')
    axs[0].set_ylabel('Latency (ms)')
    axs[0].set_title(f'TTFT Breakdown (P99={df_plot["ttft_s"].quantile(0.99)*1000:.1f}ms)')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # TPOT
    axs[1].bar(request_indices, tpot_ms, label='TPOT', color='#95E1D3')
    axs[1].set_xlabel('Request Index (by issue time)')
    axs[1].set_ylabel('TPOT (ms)')
    axs[1].set_title(f'TPOT per Request (P99={df_plot["tpot_loadgen_estimate_ms"].quantile(0.99):.1f}ms)')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = log_path / "request_metrics_breakdown.png"
    plt.savefig(output_file)
    logging.info(f"Request metrics plot saved to: {output_file}")
    plt.close()
