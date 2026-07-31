#!/usr/bin/env python3
"""
NSYS SQLite Analysis Tool
Analyzes GPU kernel execution data from nsys-exported SQLite databases.
Focuses on kernel execution time share and performance metrics.

Requirements:
    pip install -r requirements.txt

Usage:
    python nsys_sqlite.py database.sqlite
"""

import sys
import os
import sqlite3
import argparse
from pathlib import Path
import math

PHI_BAR = (1 + math.sqrt(5)) / 2

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

try:
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    print("Warning: seaborn not available, using basic matplotlib styling")
    plt.style.use('default')
    HAS_SEABORN = False

class NSysAnalyzer:
    """Analyzer for NSYS SQLite database"""

    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.kernel_data = None
        self.total_execution_time = 0

    def connect(self):
        """Connect to the SQLite database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"Connected to database: {self.db_path}")
            return True
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            return False

    def load_kernel_data(self):
        """Load and process kernel execution data"""
        if not self.conn:
            print("No database connection")
            return False

        query = """
        SELECT
            s.value as kernel_name,
            k.start,
            k.end,
            (k.end - k.start) as duration,
            k.deviceId,
            k.streamId,
            k.gridX,
            k.gridY,
            k.gridZ,
            k.blockX,
            k.blockY,
            k.blockZ,
            k.registersPerThread,
            k.staticSharedMemory,
            k.dynamicSharedMemory
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        ORDER BY k.start
        """

        try:
            print("Loading kernel execution data...")
            self.kernel_data = pd.read_sql_query(query, self.conn)

            # Convert timestamps to relative time (nanoseconds to seconds)
            min_time = self.kernel_data['start'].min()
            self.kernel_data['start_sec'] = (self.kernel_data['start'] - min_time) / 1e9
            self.kernel_data['end_sec'] = (self.kernel_data['end'] - min_time) / 1e9
            self.kernel_data['duration_ms'] = self.kernel_data['duration'] / 1e6  # Convert to milliseconds

            # Calculate total execution time
            self.total_execution_time = self.kernel_data['duration'].sum()

            print(f"Loaded {len(self.kernel_data)} kernel executions")
            print(f"Total execution time: {self.total_execution_time / 1e9:.3f} seconds")

            return True

        except Exception as e:
            print(f"Error loading kernel data: {e}")
            return False

    def get_kernel_summary(self, top_n=20):
        """Get summary statistics for top kernels by execution time"""
        if self.kernel_data is None:
            return None

        summary = self.kernel_data.groupby('kernel_name').agg({
            'duration': ['count', 'sum', 'mean', 'std'],
            'duration_ms': ['mean', 'min', 'max']
        }).round(3)

        # Flatten column names
        summary.columns = ['count', 'total_duration_ns', 'avg_duration_ns', 'std_duration_ns',
                          'avg_duration_ms', 'min_duration_ms', 'max_duration_ms']

        # Calculate percentage of total execution time
        summary['time_share_percent'] = (summary['total_duration_ns'] / self.total_execution_time) * 100

        # Sort by total duration and get top N
        summary = summary.sort_values('total_duration_ns', ascending=False).head(top_n)

        return summary

    def create_kernel_time_share_plot(self, save_path):
        """Create kernel execution time share visualization"""
        if self.kernel_data is None:
            print("No kernel data available")
            return

        # Use all unique kernels
        unique_kernels = self.kernel_data['kernel_name'].nunique()
        summary = self.get_kernel_summary(unique_kernels)

        fig = plt.figure(figsize=(14, 8), facecolor='lightgray')  # Grey background
        fig.suptitle(f'Kernel Execution Analysis\n{self.db_path}', fontsize=16, fontweight='bold')

        # Create grid layout: legend on left, pie chart on right
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 2])

        # Pie chart on the right
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.set_facecolor('lightgray')  # Grey background for pie chart

        # Use contrasting adjacent colors - create a custom colormap
        n_colors = len(summary)
        contrasting_colors = []

        # Generate contrasting colors using HSV color space
        import colorsys
        for i in range(n_colors):
            hue = (i * (PHI_BAR-1)) % 1.0  # Golden ratio for good distribution
            saturation = 0.7 + (i % 2) * 0.3  # Alternate between 0.7 and 1.0
            value = 0.8 + (i % 3) * 0.1  # Vary brightness slightly
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            contrasting_colors.append(rgb)

        # Include "Others" category for remaining kernels
        others_percent = 100 - summary['time_share_percent'].sum()

        # Create labels for legend with percentages
        legend_labels = []
        for i, (name, row) in enumerate(summary.iterrows()):
            percent = row['time_share_percent']
            if len(name) > 30:
                truncated_name = name[:30] + '...'
            else:
                truncated_name = name
            legend_labels.append(f'{truncated_name} ({percent:.1f}%)')

        sizes = list(summary['time_share_percent'])
        pie_colors = contrasting_colors

        if others_percent > 0.1:
            sizes.append(others_percent)
            legend_labels.append(f'Others ({others_percent:.1f}%)')
            pie_colors.append('darkgray')

        # Create pie chart with conditional percentage labels (only for slices > 1%)
        def autopct_func(pct):
            return f'{pct:.1f}%' if pct > 1.0 else ''

        wedges, texts, autotexts = ax1.pie(sizes, colors=pie_colors, startangle=90,
                                          autopct=autopct_func)
        ax1.set_title(f'All {unique_kernels} Kernels by Execution Time Share', fontweight='bold')

        # Style the percentage text on the pie chart
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_color('white')
            autotext.set_weight('bold')

        # Create legend on the left
        ax_legend = fig.add_subplot(gs[0, 0])
        ax_legend.axis('off')
        ax_legend.set_facecolor('lightgray')  # Grey background for legend

        # Create legend patches
        legend_patches = []
        for i, color in enumerate(pie_colors):
            legend_patches.append(plt.Rectangle((0, 0), 1, 1, facecolor=color))

        ax_legend.legend(legend_patches, legend_labels, loc='center', fontsize=9,
                        title='Kernel Names & Time Share', title_fontsize=11, frameon=False)

        plt.tight_layout()

        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='lightgray')
            print(f"Kernel time share plot saved to: {save_path}")
        except Exception as e:
            print(f"Warning: Could not save dashboard plot: {e}")
            print("Try reducing data size or using --summary-only option")
        finally:
            plt.close()

    def _detect_kernel_sequence(self, data, min_sequence_length=50):
        """Detect repeating kernel execution sequences that span the entire trace"""
        # Create a sequence of kernel names in execution order
        kernel_sequence = data['kernel_name'].tolist()
        total_kernels = len(kernel_sequence)

        # Try different sequence lengths to find repeating patterns
        for seq_len in range(min_sequence_length, min(500, total_kernels // 2)):
            pattern = kernel_sequence[:seq_len]

            # Check if this pattern repeats throughout the entire sequence
            is_valid_pattern = True
            repeats = 0

            # Check every position where the pattern could start
            for start_idx in range(0, total_kernels, seq_len):
                end_idx = min(start_idx + seq_len, total_kernels)
                current_chunk = kernel_sequence[start_idx:end_idx]

                # For the last chunk, it might be incomplete
                if len(current_chunk) == seq_len:
                    if current_chunk == pattern:
                        repeats += 1
                    else:
                        is_valid_pattern = False
                        break
                else:
                    # Check if the incomplete last chunk matches the beginning of the pattern
                    if current_chunk == pattern[:len(current_chunk)]:
                        repeats += 1  # Count partial match as valid
                    else:
                        is_valid_pattern = False
                        break

            # If pattern repeats throughout the entire sequence, it's valid
            if is_valid_pattern and repeats >= 2:
                coverage_percent = (repeats * seq_len) / total_kernels * 100
                print(f"Found repeating sequence of {seq_len} kernels, repeated {repeats} times")
                print(f"Pattern covers {coverage_percent:.1f}% of the entire trace")
                return seq_len, repeats

        # If no clear pattern found, use a reasonable chunk size
        print("No repeating pattern found throughout the entire trace")
        return min(200, len(kernel_sequence) // 2), 1

    def _print_iteration_sequence_summary(self, sequence_data, seq_length, num_repeats):
        """Print summary of the iteration kernel execution sequence"""
        print("\n" + "="*80)
        print("ITERATION KERNEL EXECUTION SEQUENCE")
        print("="*80)

        sequence_duration = (sequence_data['end_sec'] - sequence_data['start_sec'].min()).max()
        total_sequence_time = sequence_data['duration'].sum() / 1e6  # Convert to ms

        print(f"Sequence length: {seq_length} kernels")
        print(f"Sequence duration: {sequence_duration:.3f} seconds")
        print(f"Total compute time in sequence: {total_sequence_time:.3f} ms")
        print(f"Number of repetitions in full trace: {num_repeats}")
        print(f"Total iterations detected: {num_repeats}")

        # Analyze kernel composition in the sequence
        kernel_counts = sequence_data['kernel_name'].value_counts()
        kernel_times = sequence_data.groupby('kernel_name')['duration'].sum() / 1e6  # Convert to ms

        print(f"\nKERNEL COMPOSITION IN ONE ITERATION:")
        print("-" * 80)
        print(f"{'Kernel Name':<50} {'Count':<8} {'Time (ms)':<12} {'% of Iter':<10}")
        print("-" * 80)

        for kernel_name in kernel_counts.index[:15]:  # Top 15 kernels in sequence
            count = kernel_counts[kernel_name]
            time_ms = kernel_times[kernel_name]
            percent = (time_ms / total_sequence_time) * 100

            display_name = kernel_name[:47] + '...' if len(kernel_name) > 50 else kernel_name
            print(f"{display_name:<50} {count:<8} {time_ms:<12.3f} {percent:<10.1f}%")

        # Device utilization in sequence
        device_usage = sequence_data.groupby('deviceId').agg({
            'duration': 'sum',
            'kernel_name': 'count'
        })
        device_usage['duration_ms'] = device_usage['duration'] / 1e6
        device_usage['percent'] = (device_usage['duration_ms'] / total_sequence_time) * 100

        print(f"\nDEVICE UTILIZATION IN ONE ITERATION:")
        print("-" * 80)
        print(f"{'Device':<10} {'Kernels':<10} {'Time (ms)':<12} {'% of Iter':<10}")
        print("-" * 80)

        for device_id in device_usage.index:
            kernels = device_usage.loc[device_id, 'kernel_name']
            time_ms = device_usage.loc[device_id, 'duration_ms']
            percent = device_usage.loc[device_id, 'percent']
            print(f"Device {device_id:<3} {kernels:<10} {time_ms:<12.3f} {percent:<10.1f}%")

        print(f"\nITERATION PERFORMANCE METRICS:")
        print("-" * 80)
        avg_kernel_duration = sequence_data['duration_ms'].mean()
        max_kernel_duration = sequence_data['duration_ms'].max()
        min_kernel_duration = sequence_data['duration_ms'].min()

        print(f"Average kernel duration: {avg_kernel_duration:.3f} ms")
        print(f"Longest kernel in iteration: {max_kernel_duration:.3f} ms")
        print(f"Shortest kernel in iteration: {min_kernel_duration:.3f} ms")
        print(f"Kernel launch frequency: {seq_length/sequence_duration:.1f} kernels/second")

        if num_repeats > 1:
            total_iteration_time = num_repeats * sequence_duration
            print(f"Total time for all iterations: {total_iteration_time:.3f} seconds")
            print(f"Iteration throughput: {num_repeats/total_iteration_time:.2f} iterations/second")

    def create_timeline_analysis(self, save_path, sample_duration=10.0):
        """Create kernel launch density analysis"""
        if self.kernel_data is None:
            print("No kernel data available")
            return

        full_data = self.kernel_data.copy()

        if len(full_data) == 0:
            print("No kernel data available")
            return

        # Create single plot for kernel launch density
        fig, ax = plt.subplots(1, 1, figsize=(16, 6), facecolor='lightgray')
        fig.suptitle(f'Kernel Launch Density Analysis\n{self.db_path}',
                    fontsize=16, fontweight='bold')

        # Set grey background
        ax.set_facecolor('lightgray')

        # Kernel execution density over specified duration of full trace
        density_duration = min(sample_duration, full_data['start_sec'].max())
        density_data = full_data[full_data['start_sec'] <= density_duration].copy()

        time_bins = np.linspace(0, density_duration, 100)
        kernel_counts = []

        for i in range(len(time_bins)-1):
            start_time = time_bins[i]
            end_time = time_bins[i+1]
            count = len(density_data[(density_data['start_sec'] >= start_time) &
                                   (density_data['start_sec'] < end_time)])
            kernel_counts.append(count)

        ax.plot(time_bins[:-1], kernel_counts, linewidth=2, color='blue')
        ax.fill_between(time_bins[:-1], kernel_counts, alpha=0.3, color='blue')
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Kernel Launch Count', fontsize=12)
        ax.set_title(f'Kernel Launch Density Over Time (First {density_duration:.1f}s of trace)', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add statistics text
        total_kernels = len(density_data)
        avg_density = total_kernels / density_duration if density_duration > 0 else 0
        max_density = max(kernel_counts) if kernel_counts else 0

        stats_text = f'Total Kernels: {total_kernels:,}\nAvg Density: {avg_density:.1f} kernels/sec\nPeak Density: {max_density} kernels/bin'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='lightgray')
            print(f"Kernel launch density analysis saved to: {save_path}")
        except Exception as e:
            print(f"Warning: Could not save dashboard plot: {e}")
            print("Try reducing data size or using --summary-only option")
        finally:
            plt.close()

    def load_gpu_metrics(self):
        """Load GPU utilization metrics data"""
        if not self.conn:
            print("No database connection")
            return False

        query = """
        SELECT
            gm.timestamp,
            tigm.metricName,
            gm.value
        FROM GPU_METRICS gm
        JOIN TARGET_INFO_GPU_METRICS tigm ON gm.metricId = tigm.metricId
        ORDER BY gm.timestamp, tigm.metricName
        """

        try:
            self.gpu_metrics = pd.read_sql_query(query, self.conn)
            if len(self.gpu_metrics) > 0:
                print(f"Loaded {len(self.gpu_metrics)} GPU metric samples")
                return True
            else:
                print("No GPU metrics data found")
                return False
        except Exception as e:
            print(f"Error loading GPU metrics: {e}")
            return False

    def analyze_gpu_utilization(self):
        """Analyze GPU utilization metrics and provide comprehensive summary"""
        if not hasattr(self, 'gpu_metrics') or self.gpu_metrics is None:
            print("No GPU metrics data available")
            return

        print("\n" + "="*80)
        print("GPU UTILIZATION ANALYSIS")
        print("="*80)

        # Basic metrics info
        total_samples = len(self.gpu_metrics)
        unique_metrics = self.gpu_metrics['metricName'].nunique()
        time_span = (self.gpu_metrics['timestamp'].max() - self.gpu_metrics['timestamp'].min()) / 1e9

        print(f"Database: {self.db_path}")
        print(f"Total metric samples: {total_samples:,}")
        print(f"Unique metrics: {unique_metrics}")
        print(f"Time span: {time_span:.3f} seconds")
        print(f"Sampling rate: ~{total_samples/time_span/unique_metrics:.0f} samples/second per metric")

        # Analyze key utilization metrics
        key_metrics = [
            'SMs Active [Throughput %]',
            'Tensor Active [Throughput %]',
            'GR Active [Throughput %]',
            'SM Issue [Throughput %]',
            'DRAM Read Bandwidth [Throughput %]',
            'DRAM Write Bandwidth [Throughput %]'
        ]

        print(f"\nKEY UTILIZATION METRICS:")
        print("-" * 80)
        print(f"{'Metric':<40} {'Min':<8} {'Max':<8} {'Mean':<8} {'Std':<8} {'Samples':<10}")
        print("-" * 80)

        utilization_stats = {}
        for metric in key_metrics:
            metric_data = self.gpu_metrics[self.gpu_metrics['metricName'] == metric]['value']
            if len(metric_data) > 0:
                stats = {
                    'min': metric_data.min(),
                    'max': metric_data.max(),
                    'mean': metric_data.mean(),
                    'std': metric_data.std(),
                    'count': len(metric_data)
                }
                utilization_stats[metric] = stats

                display_name = metric.replace(' [Throughput %]', '').replace(' [', ' [')
                if len(display_name) > 37:
                    display_name = display_name[:37] + '...'

                print(f"{display_name:<40} {stats['min']:<8.1f} {stats['max']:<8.1f} "
                      f"{stats['mean']:<8.1f} {stats['std']:<8.1f} {stats['count']:<10,}")

        # Clock frequency analysis
        clock_metrics = [
            'GPC Clock Frequency [MHz]',
            'SYS Clock Frequency [MHz]'
        ]

        print(f"\nCLOCK FREQUENCY ANALYSIS:")
        print("-" * 80)
        print(f"{'Clock Type':<40} {'Min (MHz)':<12} {'Max (MHz)':<12} {'Mean (MHz)':<12} {'Samples':<10}")
        print("-" * 80)

        for metric in clock_metrics:
            metric_data = self.gpu_metrics[self.gpu_metrics['metricName'] == metric]['value']
            if len(metric_data) > 0:
                # Convert to MHz if needed (values might be in Hz)
                if metric_data.mean() > 1e6:
                    metric_data = metric_data / 1e6

                display_name = metric.replace(' [MHz]', '').replace('GPC', 'Graphics').replace('SYS', 'System')
                print(f"{display_name:<40} {metric_data.min():<12.0f} {metric_data.max():<12.0f} "
                      f"{metric_data.mean():<12.0f} {len(metric_data):<10,}")


        # NVLink analysis if available
        nvlink_metrics = [m for m in self.gpu_metrics['metricName'].unique() if 'NVLink' in m]
        if nvlink_metrics:
            print(f"\nNVLINK UTILIZATION:")
            print("-" * 80)

            for metric in nvlink_metrics[:4]:  # Show top 4 NVLink metrics
                metric_data = self.gpu_metrics[self.gpu_metrics['metricName'] == metric]['value']
                if len(metric_data) > 0:
                    display_name = metric.replace('NVLink ', '').replace(' [Throughput %]', '')
                    if len(display_name) > 35:
                        display_name = display_name[:35] + '...'
                    print(f"{display_name:<40} Mean: {metric_data.mean():<6.1f}% Max: {metric_data.max():<6.1f}%")

        print("\n" + "="*80)

    def create_gpu_utilization_dashboard(self, save_path, max_points=2000):
        """Create comprehensive GPU utilization dashboard with 4 key metrics over time"""
        if not hasattr(self, 'gpu_metrics') or self.gpu_metrics is None:
            print("No GPU metrics data available. Loading...")
            if not self.load_gpu_metrics():
                return

        # Prepare data for plotting
        df = self.gpu_metrics.copy()

        # Convert timestamp to seconds from start
        min_time = df['timestamp'].min()
        df['time_sec'] = (df['timestamp'] - min_time) / 1e9

        # Aggregate duplicate entries by taking the mean
        df_agg = df.groupby(['time_sec', 'metricName'])['value'].mean().reset_index()

        # Apply downsampling using 0.03% window size with mean aggregation
        max_time = df_agg['time_sec'].max()
        min_time = df_agg['time_sec'].min()
        timeline_duration = max_time - min_time
        window_size = timeline_duration * 0.0003  # 0.03% of timeline

        print(f"Applying downsampling with window size: {window_size:.4f} seconds (0.03% of {timeline_duration:.3f}s timeline)")

        def efficient_window_downsample(times, values, window_size):
            """Efficient downsampling using sliding windows with mean aggregation"""
            if len(times) == 0:
                return [], []

            result_times = []
            result_values = []

            current_window_start = times[0]
            window_values = []
            window_times = []

            for i in range(len(times)):
                current_time = times[i]

                # If current point is within the window, add it
                if current_time <= current_window_start + window_size:
                    window_values.append(values[i])
                    window_times.append(current_time)
                else:
                    # Process current window if it has data
                    if window_values:
                        # Use mean of values and mean of times for this window
                        result_values.append(sum(window_values) / len(window_values))
                        result_times.append(sum(window_times) / len(window_times))

                    # Start new window
                    current_window_start = current_time
                    window_values = [values[i]]
                    window_times = [current_time]

            # Process the last window
            if window_values:
                result_values.append(sum(window_values) / len(window_values))
                result_times.append(sum(window_times) / len(window_times))

            return result_times, result_values

        # Apply efficient downsampling to each metric separately
        df_downsampled_list = []

        for metric in df_agg['metricName'].unique():
            metric_data = df_agg[df_agg['metricName'] == metric].sort_values('time_sec')

            # Extract sorted arrays for efficient processing
            times = metric_data['time_sec'].values
            values = metric_data['value'].values

            # Apply efficient window-based downsampling
            downsampled_times, downsampled_values = efficient_window_downsample(times, values, window_size)

            # Create downsampled dataframe for this metric
            if downsampled_times:  # Only add if we have data
                metric_downsampled = pd.DataFrame({
                    'time_sec': downsampled_times,
                    'metricName': metric,
                    'value': downsampled_values
                })
                df_downsampled_list.append(metric_downsampled)

        df_downsampled = pd.concat(df_downsampled_list, ignore_index=True) if df_downsampled_list else pd.DataFrame()

        original_points = len(df_agg)
        downsampled_points = len(df_downsampled)
        reduction_factor = original_points / downsampled_points if downsampled_points > 0 else 1

        print(f"Downsampling completed: {original_points:,} → {downsampled_points:,} points (reduction factor: {reduction_factor:.1f}x)")

        # Pivot data for easier plotting
        df_pivot = df_downsampled.pivot(index='time_sec', columns='metricName', values='value')

        # Create the dashboard
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='lightgray')
        fig.suptitle(f'GPU Utilization Dashboard\n{self.db_path}', fontsize=16, fontweight='bold')

        # Set grey background for all subplots
        for ax_row in axes:
            for ax in ax_row:
                ax.set_facecolor('lightgray')

        # Top-left: SM Active % over time
        if 'SMs Active [Throughput %]' in df_pivot.columns:
            sm_data = df_pivot['SMs Active [Throughput %]'].dropna()
            axes[0, 0].plot(sm_data.index, sm_data.values, color='blue', linewidth=1, alpha=0.8)
            axes[0, 0].fill_between(sm_data.index, sm_data.values, alpha=0.3, color='blue')
            axes[0, 0].set_title('SM Active Utilization', fontweight='bold', fontsize=12)
            axes[0, 0].set_ylabel('Utilization (%)', fontsize=10)
            axes[0, 0].set_ylim(0, 100)
            axes[0, 0].grid(True, alpha=0.3)

            # Add mean line
            mean_val = sm_data.mean()
            axes[0, 0].axhline(mean_val, color='red', linestyle='--', alpha=0.7,
                              label=f'Mean: {mean_val:.1f}%')
            axes[0, 0].legend(fontsize=9)

        # Top-right: Tensor Active % over time
        if 'Tensor Active [Throughput %]' in df_pivot.columns:
            tensor_data = df_pivot['Tensor Active [Throughput %]'].dropna()
            axes[0, 1].plot(tensor_data.index, tensor_data.values, color='green', linewidth=1, alpha=0.8)
            axes[0, 1].fill_between(tensor_data.index, tensor_data.values, alpha=0.3, color='green')
            axes[0, 1].set_title('Tensor Active Utilization', fontweight='bold', fontsize=12)
            axes[0, 1].set_ylabel('Utilization (%)', fontsize=10)
            axes[0, 1].set_ylim(0, 100)
            axes[0, 1].grid(True, alpha=0.3)

            # Add mean line
            mean_val = tensor_data.mean()
            axes[0, 1].axhline(mean_val, color='red', linestyle='--', alpha=0.7,
                              label=f'Mean: {mean_val:.1f}%')
            axes[0, 1].legend(fontsize=9)

        # Bottom-left: Memory Bandwidth (DRAM Read/Write stacked area)
        dram_read = df_pivot.get('DRAM Read Bandwidth [Throughput %]', pd.Series())
        dram_write = df_pivot.get('DRAM Write Bandwidth [Throughput %]', pd.Series())

        if not dram_read.empty and not dram_write.empty:
            # Align the data by time index
            common_index = dram_read.dropna().index.intersection(dram_write.dropna().index)
            if len(common_index) > 0:
                read_aligned = dram_read.loc[common_index]
                write_aligned = dram_write.loc[common_index]

                axes[1, 0].fill_between(common_index, 0, read_aligned, alpha=0.7, color='orange', label='Read')
                axes[1, 0].fill_between(common_index, read_aligned, read_aligned + write_aligned,
                                       alpha=0.7, color='red', label='Write')
                axes[1, 0].set_title('DRAM Bandwidth Utilization', fontweight='bold', fontsize=12)
                axes[1, 0].set_ylabel('Bandwidth (%)', fontsize=10)
                axes[1, 0].set_ylim(0, max(100, (read_aligned + write_aligned).max() * 1.1))
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend(fontsize=9)

                # Add total bandwidth line
                total_bw = read_aligned + write_aligned
                mean_total = total_bw.mean()
                axes[1, 0].plot(common_index, total_bw, color='black', linestyle='-', alpha=0.8, linewidth=1)
                axes[1, 0].text(0.02, 0.98, f'Total Mean: {mean_total:.1f}%',
                               transform=axes[1, 0].transAxes, fontsize=9,
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Bottom-right: Clock Frequencies (dual y-axis)
        gpc_clock = df_pivot.get('GPC Clock Frequency [MHz]', pd.Series())
        sys_clock = df_pivot.get('SYS Clock Frequency [MHz]', pd.Series())

        if not gpc_clock.empty:
            # Convert to MHz if needed
            if gpc_clock.mean() > 1e6:
                gpc_clock = gpc_clock / 1e6
            if not sys_clock.empty and sys_clock.mean() > 1e6:
                sys_clock = sys_clock / 1e6

            gpc_clean = gpc_clock.dropna()
            ax_clock = axes[1, 1]

            # Plot GPC clock
            line1 = ax_clock.plot(gpc_clean.index, gpc_clean.values, color='purple',
                                 linewidth=1, alpha=0.8, label='Graphics Clock')
            ax_clock.set_ylabel('Graphics Clock (MHz)', color='purple', fontsize=10)
            ax_clock.tick_params(axis='y', labelcolor='purple')
            ax_clock.set_title('Clock Frequencies', fontweight='bold', fontsize=12)
            ax_clock.grid(True, alpha=0.3)

            # Create second y-axis for system clock
            if not sys_clock.empty:
                ax_clock2 = ax_clock.twinx()
                sys_clean = sys_clock.dropna()
                line2 = ax_clock2.plot(sys_clean.index, sys_clean.values, color='brown',
                                      linewidth=1, alpha=0.8, label='System Clock')
                ax_clock2.set_ylabel('System Clock (MHz)', color='brown', fontsize=10)
                ax_clock2.tick_params(axis='y', labelcolor='brown')

                # Combined legend
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax_clock.legend(lines, labels, loc='upper right', fontsize=9)
            else:
                ax_clock.legend(fontsize=9)

        # Set common x-axis labels
        for ax in axes.flat:
            ax.set_xlabel('Time (seconds)', fontsize=10)

            plt.tight_layout()

        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='lightgray')
            print(f"GPU utilization dashboard saved to: {save_path}")
        except Exception as e:
            print(f"Warning: Could not save dashboard plot: {e}")
            print("Try reducing data size or using --summary-only option")
        finally:
            plt.close()

    def print_summary_stats(self):
        """Print comprehensive summary statistics"""
        if self.kernel_data is None:
            print("No kernel data available")
            return

        # Calculate derived metrics for summary
        self._calculate_derived_metrics()

        print("\n" + "="*80)
        print("NSYS KERNEL EXECUTION SUMMARY")
        print("="*80)

        print(f"Database: {self.db_path}")
        print(f"Total kernels executed: {len(self.kernel_data):,}")
        print(f"Unique kernel types: {self.kernel_data['kernel_name'].nunique()}")
        print(f"Total execution time: {self.total_execution_time / 1e9:.3f} seconds")
        print(f"Average kernel duration: {self.kernel_data['duration_ms'].mean():.3f} ms")
        print(f"Timeline span: {self.kernel_data['start_sec'].max():.3f} seconds")

        # Device information
        devices = self.kernel_data['deviceId'].unique()
        print(f"Devices used: {len(devices)} ({list(devices)})")

        print("\nTOP 10 KERNELS BY EXECUTION TIME:")
        print("-" * 80)
        summary = self.get_kernel_summary(10)

        for kernel_name, row in summary.iterrows():
            print(f"{kernel_name[:60]:<60} {row['time_share_percent']:>6.2f}% "
                  f"({row['count']:>6,} calls, {row['avg_duration_ms']:>8.3f}ms avg)")


    def _calculate_derived_metrics(self):
        """Calculate derived metrics like grid_size, block_size, etc."""
        if self.kernel_data is None:
            return

        # Calculate grid size and occupancy metrics
        self.kernel_data['grid_size'] = (self.kernel_data['gridX'] *
                                       self.kernel_data['gridY'] *
                                       self.kernel_data['gridZ'])
        self.kernel_data['block_size'] = (self.kernel_data['blockX'] *
                                        self.kernel_data['blockY'] *
                                        self.kernel_data['blockZ'])
        self.kernel_data['total_threads'] = self.kernel_data['grid_size'] * self.kernel_data['block_size']

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze NSYS SQLite database for kernel execution metrics')

    parser.add_argument('database', help='SQLite database file from nsys export')
    parser.add_argument('--output-dir', '-o', default='nsys_analysis',
                       help='Output directory for plots (default: nsys_analysis)')
    parser.add_argument('--timeline-duration', '-t', type=float, default=10.0,
                       help='Duration for timeline analysis in seconds (default: 10.0)')
    parser.add_argument('--max-plot-points', type=int, default=2000,
                       help='Maximum data points per metric for plotting (default: 2000)')
    parser.add_argument('--summary-only', action='store_true',
                       help='Only print summary statistics (no plots)')

    args = parser.parse_args()

    # Check if database exists
    if not os.path.exists(args.database):
        print(f"Error: Database file not found: {args.database}")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize analyzer
    analyzer = NSysAnalyzer(args.database)

    if not analyzer.connect():
        return

    if not analyzer.load_kernel_data():
        analyzer.close()
        return

    # Print kernel summary statistics
    analyzer.print_summary_stats()

    # Load and analyze GPU metrics for utilization analysis
    if analyzer.load_gpu_metrics():
        analyzer.analyze_gpu_utilization()

    if not args.summary_only:
        print(f"\nGenerating visualizations in {args.output_dir}/...")

        # Generate GPU utilization dashboard
        dashboard_path = os.path.join(args.output_dir, 'gpu_utilization_dashboard.png')
        analyzer.create_gpu_utilization_dashboard(save_path=dashboard_path, max_points=args.max_plot_points)

        # Generate kernel analysis plots
        time_share_path = os.path.join(args.output_dir, 'kernel_time_share.png')
        analyzer.create_kernel_time_share_plot(save_path=time_share_path)

        # Generate kernel launch density analysis
        timeline_path = os.path.join(args.output_dir, 'execution_timeline.png')
        analyzer.create_timeline_analysis(save_path=timeline_path, sample_duration=args.timeline_duration)

        print(f"\nAll plots saved to: {args.output_dir}")

        analyzer.close()

if __name__ == '__main__':
    main()
