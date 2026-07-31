#!/usr/bin/env python3
"""
Token Statistics Visualization Script

Analyzes and visualizes token statistics from endpoint harness logs (JSONL format).
Provides insights into token processing, memory usage, latency, and KV cache performance.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

# Handle optional imports gracefully
try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Install with: pip install pandas")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: matplotlib and numpy are required. Install with: pip install matplotlib numpy")
    sys.exit(1)

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available, using basic matplotlib styling")


class TokenStatsAnalyzer:
    """Analyzer for token statistics from endpoint harness logs"""

    def __init__(self, jsonl_path: str):
        self.jsonl_path = jsonl_path
        self.data = None
        self.stats_summary = {}

    def load_data(self) -> bool:
        """Load and parse JSONL data"""
        try:
            print(f"Loading data from: {self.jsonl_path}")

            records = []
            with open(self.jsonl_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line.strip())
                        records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed JSON on line {line_num}: {e}")
                        continue

            if not records:
                print("Error: No valid records found in file")
                return False

            # Convert to DataFrame
            self.data = pd.DataFrame(records)

            # Parse timestamps
            if 'timestamp' in self.data.columns:
                self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], format='%m-%d-%Y %H:%M:%S.%f')
                self.data['time_elapsed'] = (self.data['timestamp'] - self.data['timestamp'].min()).dt.total_seconds()

            print(f"Loaded {len(self.data)} records")
            print(f"Time span: {self.data['time_elapsed'].max():.1f} seconds")

            return True

        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def extract_nested_metrics(self):
        """Extract nested metrics from JSON structures"""
        if self.data is None:
            return

        # Extract inflight batching stats
        if 'inflightBatchingStats' in self.data.columns:
            inflight_stats = pd.json_normalize(self.data['inflightBatchingStats'])
            for col in inflight_stats.columns:
                self.data[f'inflight_{col}'] = inflight_stats[col]

        # Extract KV cache stats
        if 'kvCacheStats' in self.data.columns:
            kv_stats = pd.json_normalize(self.data['kvCacheStats'])
            for col in kv_stats.columns:
                self.data[f'kv_{col}'] = kv_stats[col]

        # Extract static batching stats
        if 'staticBatchingStats' in self.data.columns:
            static_stats = pd.json_normalize(self.data['staticBatchingStats'])
            for col in static_stats.columns:
                self.data[f'static_{col}'] = static_stats[col]

    def calculate_derived_metrics(self):
        """Calculate derived metrics"""
        if self.data is None:
            return

        # Convert memory to GB
        if 'gpuMemUsage' in self.data.columns:
            self.data['gpuMemUsage_GB'] = self.data['gpuMemUsage'] / (1024**3)

        # Calculate tokens per second
        if 'inflight_numCtxTokens' in self.data.columns and 'iterLatencyMS' in self.data.columns:
            self.data['tokens_per_second'] = (self.data['inflight_numCtxTokens'] /
                                            (self.data['iterLatencyMS'] / 1000.0)).replace([np.inf, -np.inf], 0)

        # Calculate KV cache utilization
        if 'kv_usedNumBlocks' in self.data.columns and 'kv_maxNumBlocks' in self.data.columns:
            self.data['kv_utilization_pct'] = (self.data['kv_usedNumBlocks'] /
                                             self.data['kv_maxNumBlocks'] * 100)

        # Calculate total requests (context + generation)
        if 'inflight_numContextRequests' in self.data.columns and 'inflight_numGenRequests' in self.data.columns:
            self.data['total_requests'] = (self.data['inflight_numContextRequests'].fillna(0) +
                                         self.data['inflight_numGenRequests'].fillna(0))

    def print_summary_stats(self):
        """Print comprehensive summary statistics"""
        if self.data is None:
            print("No data available")
            return

        print("\n" + "="*80)
        print("TOKEN STATISTICS SUMMARY")
        print("="*80)

        print(f"File: {self.jsonl_path}")
        print(f"Total records: {len(self.data):,}")
        print(f"Time span: {self.data['time_elapsed'].max():.1f} seconds")
        print(f"Average iteration latency: {self.data['iterLatencyMS'].mean():.1f} ms")

        # Token statistics
        if 'inflight_numCtxTokens' in self.data.columns:
            ctx_tokens = self.data['inflight_numCtxTokens'].fillna(0)
            print(f"\nCONTEXT TOKEN STATISTICS:")
            print(f"  Total context tokens processed: {ctx_tokens.sum():,}")
            print(f"  Average context tokens per iteration: {ctx_tokens.mean():.1f}")
            print(f"  Max context tokens in single iteration: {ctx_tokens.max():,}")

        if 'inflight_numGenRequests' in self.data.columns:
            gen_requests = self.data['inflight_numGenRequests'].fillna(0)
            print(f"\nGENERATION REQUEST STATISTICS:")
            print(f"  Average generation requests per iteration: {gen_requests.mean():.1f}")
            print(f"  Max generation requests in single iteration: {gen_requests.max():,}")
            print(f"  Total iterations with generation requests: {(gen_requests > 0).sum():,}")

        # Request statistics
        if 'numActiveRequests' in self.data.columns:
            print(f"\nREQUEST STATISTICS:")
            print(f"  Average active requests: {self.data['numActiveRequests'].mean():.1f}")
            print(f"  Max active requests: {self.data['numActiveRequests'].max():,}")
            print(f"  Average queued requests: {self.data['numQueuedRequests'].mean():.1f}")

        # KV Cache block statistics
        if all(col in self.data.columns for col in ['kv_allocTotalBlocks', 'kv_freeNumBlocks', 'kv_maxNumBlocks', 'kv_usedNumBlocks']):
            print(f"\nKV CACHE BLOCK STATISTICS:")
            print(f"  Max blocks available: {self.data['kv_maxNumBlocks'].iloc[0]:,}")
            print(f"  Average allocated total blocks: {self.data['kv_allocTotalBlocks'].mean():.0f}")
            print(f"  Average used blocks: {self.data['kv_usedNumBlocks'].mean():.0f}")
            print(f"  Average free blocks: {self.data['kv_freeNumBlocks'].mean():.0f}")
            print(f"  Peak used blocks: {self.data['kv_usedNumBlocks'].max():,}")

            # Validation check
            sample_used = self.data['kv_usedNumBlocks'].iloc[-1]
            sample_free = self.data['kv_freeNumBlocks'].iloc[-1]
            sample_max = self.data['kv_maxNumBlocks'].iloc[-1]
            print(f"  Validation (latest): {sample_used} + {sample_free} = {sample_used + sample_free} (Max: {sample_max})")

        # KV Cache statistics
        if 'kv_utilization_pct' in self.data.columns:
            print(f"\nKV CACHE STATISTICS:")
            print(f"  Average cache utilization: {self.data['kv_utilization_pct'].mean():.1f}%")
            print(f"  Max cache utilization: {self.data['kv_utilization_pct'].max():.1f}%")
            if 'kv_cacheHitRate' in self.data.columns:
                print(f"  Average cache hit rate: {self.data['kv_cacheHitRate'].mean():.1f}%")

    def create_token_dashboard(self, save_path: Optional[str] = None):
        """Create comprehensive token statistics dashboard"""
        if self.data is None:
            print("No data available for visualization")
            return

        # Create 2x3 dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='lightgray')
        fig.suptitle(f'Token Statistics Dashboard\n{self.jsonl_path}', fontsize=16, fontweight='bold')

        # Set grey background for all subplots
        for ax_row in axes:
            for ax in ax_row:
                ax.set_facecolor('lightgray')

        # Top-left: Context tokens over time
        if 'inflight_numCtxTokens' in self.data.columns:
            ctx_tokens = self.data['inflight_numCtxTokens'].fillna(0)
            axes[0, 0].plot(self.data['time_elapsed'], ctx_tokens, color='blue', linewidth=1, alpha=0.8)
            axes[0, 0].fill_between(self.data['time_elapsed'], ctx_tokens, alpha=0.3, color='blue')
            axes[0, 0].set_title('Context Tokens Over Time', fontweight='bold')
            axes[0, 0].set_ylabel('Context Tokens')
            axes[0, 0].grid(True, alpha=0.3)

        # Top-middle: Active requests over time
        if 'numActiveRequests' in self.data.columns:
            axes[0, 1].plot(self.data['time_elapsed'], self.data['numActiveRequests'],
                           color='green', linewidth=1, alpha=0.8)
            axes[0, 1].fill_between(self.data['time_elapsed'], self.data['numActiveRequests'],
                                   alpha=0.3, color='green')
            axes[0, 1].set_title('Active Requests Over Time', fontweight='bold')
            axes[0, 1].set_ylabel('Number of Requests')
            axes[0, 1].grid(True, alpha=0.3)

        # Top-right: Generation requests over time
        if 'inflight_numGenRequests' in self.data.columns:
            gen_requests = self.data['inflight_numGenRequests'].fillna(0)
            axes[0, 2].plot(self.data['time_elapsed'], gen_requests,
                           color='red', linewidth=1, alpha=0.8)
            axes[0, 2].fill_between(self.data['time_elapsed'], gen_requests, alpha=0.3, color='red')
            axes[0, 2].set_title('Generation Requests Over Time', fontweight='bold')
            axes[0, 2].set_ylabel('Number of Gen Requests')
            axes[0, 2].grid(True, alpha=0.3)

        # Bottom-left: KV cache blocks over time
        if all(col in self.data.columns for col in ['kv_allocTotalBlocks', 'kv_freeNumBlocks', 'kv_maxNumBlocks', 'kv_usedNumBlocks']):
            axes[1, 0].plot(self.data['time_elapsed'], self.data['kv_maxNumBlocks'],
                           color='blue', linewidth=2, alpha=0.8, label='Max Blocks')
            axes[1, 0].plot(self.data['time_elapsed'], self.data['kv_usedNumBlocks'],
                           color='orange', linewidth=1, alpha=0.8, label='Used Blocks')
            axes[1, 0].plot(self.data['time_elapsed'], self.data['kv_freeNumBlocks'],
                           color='green', linewidth=1, alpha=0.8, label='Free Blocks')

            # Add fill areas for better visualization
            axes[1, 0].fill_between(self.data['time_elapsed'], 0, self.data['kv_usedNumBlocks'],
                                   alpha=0.3, color='orange', label='Used Area')
            axes[1, 0].fill_between(self.data['time_elapsed'], self.data['kv_usedNumBlocks'],
                                   self.data['kv_maxNumBlocks'], alpha=0.3, color='green', label='Free Area')

            axes[1, 0].set_title('KV Cache Blocks Over Time', fontweight='bold')
            axes[1, 0].set_ylabel('Number of Blocks')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend(fontsize=8)

            # Add validation text
            sample_idx = len(self.data) // 2  # Middle sample
            if sample_idx < len(self.data):
                used = self.data.iloc[sample_idx]['kv_usedNumBlocks']
                free = self.data.iloc[sample_idx]['kv_freeNumBlocks']
                max_blocks = self.data.iloc[sample_idx]['kv_maxNumBlocks']
                sum_check = used + free
                axes[1, 0].text(0.02, 0.98, f'Validation: {used} + {free} = {sum_check} (Max: {max_blocks})',
                               transform=axes[1, 0].transAxes, fontsize=8,
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Bottom-middle: KV cache utilization
        if 'kv_utilization_pct' in self.data.columns:
            axes[1, 1].plot(self.data['time_elapsed'], self.data['kv_utilization_pct'],
                           color='orange', linewidth=1, alpha=0.8)
            axes[1, 1].fill_between(self.data['time_elapsed'], self.data['kv_utilization_pct'],
                                   alpha=0.3, color='orange')
            axes[1, 1].set_title('KV Cache Utilization Over Time', fontweight='bold')
            axes[1, 1].set_ylabel('Utilization (%)')
            axes[1, 1].set_ylim(0, 100)
            axes[1, 1].grid(True, alpha=0.3)

        # Bottom-right: Tokens per second (throughput)
        if 'tokens_per_second' in self.data.columns:
            # Filter out extreme values for better visualization
            tps_data = self.data['tokens_per_second'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(tps_data) > 0:
                tps_filtered = tps_data[tps_data <= tps_data.quantile(0.95)]  # Remove top 5% outliers
                time_filtered = self.data.loc[tps_filtered.index, 'time_elapsed']

                axes[1, 2].plot(time_filtered, tps_filtered, color='brown', linewidth=1, alpha=0.8)
                axes[1, 2].set_title('Token Throughput Over Time', fontweight='bold')
                axes[1, 2].set_ylabel('Tokens/Second')
                axes[1, 2].grid(True, alpha=0.3)

        # Set common x-axis labels
        for ax in axes.flat:
            ax.set_xlabel('Time (seconds)')

        plt.tight_layout()

        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='lightgray')
                print(f"Token dashboard saved to: {save_path}")
            except Exception as e:
                print(f"Warning: Could not save dashboard plot: {e}")

        try:
            plt.show()
        except Exception as e:
            print(f"Warning: Could not display dashboard plot: {e}")
            plt.close(fig)



def main():
    parser = argparse.ArgumentParser(description='Visualize token statistics from endpoint harness logs')

    parser.add_argument('jsonl_file', help='JSONL file containing token statistics')
    parser.add_argument('--output-dir', '-o', default='token_stats_plots',
                       help='Output directory for plots (default: token_stats_plots)')
    parser.add_argument('--summary-only', action='store_true',
                       help='Only print summary statistics (no plots)')

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.jsonl_file):
        print(f"Error: File not found: {args.jsonl_file}")
        return

    # Create output directory
    if not args.summary_only:
        os.makedirs(args.output_dir, exist_ok=True)

    # Initialize analyzer
    analyzer = TokenStatsAnalyzer(args.jsonl_file)

    if not analyzer.load_data():
        return

    # Extract and calculate metrics
    analyzer.extract_nested_metrics()
    analyzer.calculate_derived_metrics()

    # Print summary statistics
    analyzer.print_summary_stats()

    if not args.summary_only:
        print(f"\nGenerating visualizations in {args.output_dir}/...")

        # Generate token dashboard
        dashboard_path = os.path.join(args.output_dir, 'token_dashboard.png')
        analyzer.create_token_dashboard(save_path=dashboard_path)

        print(f"\nPlot saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
