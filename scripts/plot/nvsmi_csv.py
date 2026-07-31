#!/usr/bin/env python3
"""
NVIDIA SMI Log Visualization Tool
Visualizes power, temperature, memory usage, and GPU utilization from NVIDIA SMI CSV files.
Supports single or multiple CSV files for comparative analysis.

Requirements:
    pip install -r requirements.txt

Usage:
    # Single CSV file
    python nvidia_smi.py --files /path/to/nvsmi_dump.csv

    # Multiple CSV files
    python nvidia_smi.py --files file1.csv file2.csv file3.csv

    # Pattern matching
    # Single CSV file
    python nvidia_smi.py /path/to/nvsmi_dump.csv

    # With custom output directory
    python nvidia_smi.py /path/to/nvsmi_dump.csv --output-dir my_plots

    # Power analysis only
    python nvidia_smi.py /path/to/nvsmi_dump.csv --power-only
"""

import sys
import os
import glob
import argparse
from pathlib import Path
from datetime import datetime

# Check for required packages
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

try:
    import seaborn as sns
    # Set style for better looking plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    print("Warning: seaborn not available, using basic matplotlib styling")
    plt.style.use('default')
    HAS_SEABORN = False

def parse_nvsmi_file(filepath):
    """Parse a single NVIDIA SMI CSV file and group by GPU"""
    print(f"Parsing: {filepath}")

    try:
        # Read CSV file
        df = pd.read_csv(filepath)

        # Clean column names
        df.columns = df.columns.str.strip()

        # Extract numeric values from power, temperature, etc.
        def process_column(col_src, col_dst, extract_val=None):
            """Process a column by extracting/converting to numeric values"""
            if col_src in df.columns:
                processed = extract_val(df[col_src]) if extract_val else df[col_src]
            elif col_dst in df.columns:
                processed = df[col_dst]
            else:
                return
            df[col_dst] = pd.to_numeric(processed, errors='coerce')

        process_column('power.draw [W]', 'power_draw', lambda c: c.str.extract(r'(\d+\.?\d*)')[0])
        process_column('clocks.current.sm [MHz]', 'sm_clock', lambda c: c.str.extract(r'(\d+)')[0])
        process_column('clocks.current.memory [MHz]', 'memory_clock', lambda c: c.str.extract(r'(\d+)')[0])
        process_column('temperature.gpu', 'temperature')
        process_column('memory.total [MiB]', 'memory_total', lambda c: c.str.extract(r'(\d+)')[0])
        process_column('memory.used [MiB]', 'memory_used', lambda c: c.str.extract(r'(\d+)')[0])
        process_column('utilization.gpu [%]', 'gpu_util', lambda c: c.str.extract(r'(\d+)')[0])

        # Calculate memory utilization percentage if both total and used are available
        if 'memory_total' in df.columns and 'memory_used' in df.columns:
            df['memory_util'] = (df['memory_used'] / df['memory_total']) * 100

        # Identify GPU by uuid (preferred) or index column
        gpu_id_col = None
        for primary_key in ["uuid", "index"]:
            if primary_key in df.columns:
                df["gpu_id"] = df[primary_key]
                gpu_id_col = primary_key
                break
        if not gpu_id_col:
            df["gpu_id"] = "GPU-0"

        # Group data by GPU and create per-GPU time indices
        per_gpu_data = {}
        for gpu_id in df['gpu_id'].unique():
            gpu_df = df[df['gpu_id'] == gpu_id].copy().reset_index(drop=True)

            # Try to use actual timestamps if available
            timestamp_col = None
            for col in ['timestamp', 'time', 'datetime']:
                if col in gpu_df.columns:
                    timestamp_col = col
                    break
            if timestamp_col:
                # Parse timestamps and calculate elapsed time
                try:
                    gpu_df['timestamp_parsed'] = pd.to_datetime(gpu_df[timestamp_col])
                    start_time = gpu_df['timestamp_parsed'].iloc[0]
                    gpu_df['time_seconds'] = (gpu_df['timestamp_parsed'] - start_time).dt.total_seconds()
                    gpu_df['time_minutes'] = gpu_df['time_seconds'] / 60.0
                except Exception as e:
                    print(f"Warning: Failed to parse timestamps for GPU {gpu_id}: {e}")
                    print("Falling back to polling interval calculation")
                    timestamp_col = None
            if not timestamp_col:
                # Fall back to row index with actual polling rate
                # Get polling interval from environment variable (default: 200ms)
                polling_interval_ms = int(os.environ.get('NVSMI_REFRESH_RATE', 200))
                polling_interval_sec = polling_interval_ms / 1000.0
                gpu_df['time_seconds'] = gpu_df.index * polling_interval_sec
                gpu_df['time_minutes'] = gpu_df['time_seconds'] / 60.0

            per_gpu_data[gpu_id] = gpu_df

        return {
            'per_gpu_data': per_gpu_data,
            'filepath': filepath,
            'gpu_count': len(per_gpu_data),
            'gpu_ids': list(per_gpu_data.keys())
        }

    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

class MetricPlot:
    """Handles plotting for a single metric/column"""

    def __init__(self, column_name, display_name, unit, color='blue', limits=None, value_format='.1f'):
        """
        Initialize a metric plot.

        Args:
            column_name: Name of the column in the dataframe
            display_name: Human-readable name for display
            unit: Unit of measurement (e.g., 'W', 'MHz', 'MiB', '%')
            color: Color for the plot
            limits: Tuple of (min, max) limits for the plot
            value_format: Format string for displaying values (default: '.1f')
        """
        self.column_name = column_name
        self.display_name = display_name
        self.unit = unit
        self.color = color
        self.limits = limits
        self.value_format = value_format

    def _calculate_effective_limits(self, data, limits):
        """Calculate effective limits based on data range and specified limits"""
        actual_min = data.min()
        actual_max = data.max()

        if limits:
            min_limit, max_limit = limits
            if actual_max < min_limit or actual_min > max_limit:
                # Data is completely outside specified range, use actual data range
                return actual_min, actual_max
            else:
                # Use intersection of specified limits and actual data range
                return max(min_limit, actual_min), min(max_limit, actual_max)
        else:
            return actual_min, actual_max

    def create_histogram(self, ax, data, limits=None, num_bins=50):
        """Create histogram with statistics on given axes"""
        effective_min, effective_max = self._calculate_effective_limits(data, limits)
        filtered_data = data[(data >= effective_min) & (data <= effective_max)]

        if len(filtered_data) == 0:
            ax.text(0.5, 0.5, 'No data in range', ha='center', va='center',
                   transform=ax.transAxes)
            return filtered_data

        # Create histogram
        bins = np.linspace(effective_min, effective_max, num_bins)
        ax.hist(filtered_data, bins=bins, alpha=0.7, color=self.color, edgecolor='black')

        # Add mean line and statistics
        mean_val = filtered_data.mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5,
                  label=f'Mean: {mean_val:{self.value_format}}{self.unit}')
        ax.legend(fontsize=8)

        ax.set_xlabel(f'{self.display_name} ({self.unit})', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.grid(True, alpha=0.3)

        return filtered_data

    def create_timeline(self, ax, df, time_col, time_label, limits=None):
        """Create timeline plot on given axes"""
        if self.column_name not in df.columns or time_col not in df.columns:
            ax.text(0.5, 0.5, f'{self.display_name} data not available',
                   ha='center', va='center', transform=ax.transAxes)
            return

        data = df[self.column_name].dropna()
        if len(data) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                   transform=ax.transAxes)
            return

        effective_min, effective_max = self._calculate_effective_limits(data, limits)
        mask = (df[self.column_name] >= effective_min) & (df[self.column_name] <= effective_max)
        filtered_df = df[mask]

        if len(filtered_df) > 0:
            ax.plot(filtered_df[time_col], filtered_df[self.column_name],
                   alpha=0.8, color=self.color, linewidth=0.8)
            ax.set_ylim(effective_min, effective_max)

        ax.set_xlabel(time_label, fontsize=8)
        ax.set_ylabel(f'{self.display_name} ({self.unit})', fontsize=8)
        ax.grid(True, alpha=0.3)

    def create_detailed_analysis(self, df, time_col, time_label, gpu_idx, gpu_id,
                                file_path, limits=None, save_dir=None):
        """Create detailed analysis with histogram and timeline for a single GPU"""
        if self.column_name not in df.columns:
            print(f"No {self.display_name} data found for GPU {gpu_id}!")
            return

        data = df[self.column_name].dropna()
        if len(data) == 0:
            print(f"No valid {self.display_name} data for GPU {gpu_id}, skipping...")
            return

        effective_min, effective_max = self._calculate_effective_limits(data, limits)
        filtered_data = data[(data >= effective_min) & (data <= effective_max)]

        # Print statistics
        print(f"\n{self.display_name} Analysis Statistics for GPU {gpu_idx} ({gpu_id}):")
        print(f"  Actual range: {data.min():{self.value_format}}{self.unit} - "
              f"{data.max():{self.value_format}}{self.unit}")
        print(f"  Analysis range: {effective_min:{self.value_format}}{self.unit} - "
              f"{effective_max:{self.value_format}}{self.unit}")
        print(f"  Total samples: {len(data)}")
        print(f"  Samples in range: {len(filtered_data)} "
              f"({len(filtered_data)/len(data)*100:.1f}%)")

        # Create figure with histogram and timeline
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'{self.display_name} Analysis - GPU {gpu_idx}\n{file_path}\n{gpu_id}',
                    fontsize=14, fontweight='bold')

        # Histogram
        bins = np.linspace(effective_min, effective_max, 50)
        ax1.hist(filtered_data, bins=bins, alpha=0.7, color=self.color, edgecolor='black')
        ax1.set_title(f'{self.display_name} Distribution Histogram '
                     f'({effective_min:{self.value_format}}{self.unit} - '
                     f'{effective_max:{self.value_format}}{self.unit})',
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel(f'{self.display_name} ({self.unit})', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Add statistics
        mean_val = filtered_data.mean()
        std_val = filtered_data.std()
        ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_val:{self.value_format}}{self.unit}')
        ax1.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7,
                   label=f'±1σ: {std_val:{self.value_format}}{self.unit}')
        ax1.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7)
        ax1.legend()

        # Timeline
        if time_col in df.columns:
            mask = (df[self.column_name] >= effective_min) & (df[self.column_name] <= effective_max)
            filtered_df = df[mask]
            if len(filtered_df) > 0:
                ax2.plot(filtered_df[time_col], filtered_df[self.column_name],
                        alpha=0.8, color=self.color, linewidth=1)

        ax2.set_title(f'{self.display_name} Over Time (Filtered: '
                     f'{effective_min:{self.value_format}}{self.unit} - '
                     f'{effective_max:{self.value_format}}{self.unit})',
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel(time_label, fontsize=11)
        ax2.set_ylabel(f'{self.display_name} ({self.unit})', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(effective_min, effective_max)

        plt.tight_layout()

        if save_dir:
            safe_name = self.column_name.replace('.', '_').replace(' ', '_')
            save_path = os.path.join(save_dir, f'{safe_name}_analysis_gpu{gpu_idx}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"{self.display_name} analysis saved to: {save_path}")
            plt.close(fig)
        else:
            plt.show()

        # Print detailed statistics
        print(f"Detailed Statistics:")
        print(f"  Mean: {mean_val:{self.value_format}}{self.unit}")
        print(f"  Std Dev: {std_val:{self.value_format}}{self.unit}")
        print(f"  Median: {filtered_data.median():{self.value_format}}{self.unit}")
        print(f"  Min: {filtered_data.min():{self.value_format}}{self.unit}")
        print(f"  Max: {filtered_data.max():{self.value_format}}{self.unit}")

class NVIDIASMIVisualizer:
    """Visualizer for NVIDIA SMI data"""

    def __init__(self, per_gpu_data, filepath, time_unit='seconds'):
        self.per_gpu_data = per_gpu_data
        self.filepath = filepath
        self.time_unit = time_unit
        self.time_col = 'time_seconds' if time_unit == 'seconds' else 'time_minutes'
        self.time_label = 'Time (seconds)' if time_unit == 'seconds' else 'Time (minutes)'

    def create_sm_utilization_analysis(self, util_limits=(0, 100), save_dir=None):
        """Create SM utilization histogram and line graph per GPU"""
        if not self.per_gpu_data:
            print("No data to visualize")
            return

        per_gpu_data = self.per_gpu_data
        file_path = self.filepath

        print(f"\nGenerating SM Clock Analysis for {len(per_gpu_data)} GPU(s)...")

        # Create MetricPlot for SM clock
        sm_plot = MetricPlot('sm_clock', 'SM Clock', 'MHz', color='green',
                            limits=util_limits, value_format='.0f')

        # Create plots for each GPU
        for gpu_idx, (gpu_id, df) in enumerate(per_gpu_data.items()):
            sm_plot.create_detailed_analysis(df, self.time_col, self.time_label,
                                            gpu_idx, gpu_id, file_path,
                                            limits=util_limits, save_dir=save_dir)


    def create_master_power_summary(self, power_limits=(800, 1400), save_path=None):
        """Create master power summary plot with all GPUs (2 cols: power hist, power timeline)"""
        if not self.per_gpu_data:
            print("No data to visualize")
            return

        per_gpu_data = self.per_gpu_data
        file_path = self.filepath

        num_gpus = len(per_gpu_data)
        print(f"\nGenerating Master Power Summary for {num_gpus} GPU(s)...")

        if num_gpus == 0:
            print("Warning: No GPUs found in data, skipping master power summary")
            return

        # Create MetricPlot for power
        power_plot = MetricPlot('power_draw', 'Power', 'W', color='blue',
                               limits=power_limits, value_format='.0f')

        # Create figure with 2 columns and num_gpus rows
        fig, axes = plt.subplots(num_gpus, 2, figsize=(14, 4 * num_gpus))

        # Handle single GPU case (axes won't be 2D)
        if num_gpus == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'Power Summary (All GPUs)\n{file_path}',
                    fontsize=16, fontweight='bold', y=0.995)

        for gpu_idx, (gpu_id, df) in enumerate(per_gpu_data.items()):
            # Shortened GPU ID for display
            short_gpu_id = str(gpu_id)[-12:] if len(str(gpu_id)) > 12 else str(gpu_id)

            # Column 0: Power Histogram
            if 'power_draw' in df.columns:
                data = df['power_draw'].dropna()
                if len(data) > 0:
                    power_plot.create_histogram(axes[gpu_idx, 0], data, limits=power_limits, num_bins=30)

            axes[gpu_idx, 0].set_ylabel(f'GPU {gpu_idx}\n{short_gpu_id}', fontsize=9, fontweight='bold')
            axes[gpu_idx, 0].tick_params(labelsize=7)

            # Column 1: Power Timeline
            power_plot.create_timeline(axes[gpu_idx, 1], df, self.time_col, self.time_label, limits=power_limits)
            axes[gpu_idx, 1].tick_params(labelsize=7)

        # Add column titles to top row
        axes[0, 0].set_title('Power Distribution', fontsize=10, fontweight='bold', pad=10)
        axes[0, 1].set_title('Power Over Time', fontsize=10, fontweight='bold', pad=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Master power summary saved to: {save_path}")
            plt.close(fig)
        else:
            plt.show()

    def create_master_sm_summary(self, sm_limits=(0, 3000), save_path=None):
        """Create master SM clock summary plot with all GPUs (2 cols: sm hist, sm timeline)"""
        if not self.per_gpu_data:
            print("No data to visualize")
            return

        per_gpu_data = self.per_gpu_data
        file_path = self.filepath

        num_gpus = len(per_gpu_data)
        print(f"\nGenerating Master SM Clock Summary for {num_gpus} GPU(s)...")

        if num_gpus == 0:
            print("Warning: No GPUs found in data, skipping master SM clock summary")
            return

        # Create MetricPlot for SM clock
        sm_plot = MetricPlot('sm_clock', 'SM Clock', 'MHz', color='green',
                            limits=sm_limits, value_format='.0f')

        # Create figure with 2 columns and num_gpus rows
        fig, axes = plt.subplots(num_gpus, 2, figsize=(14, 4 * num_gpus))

        # Handle single GPU case (axes won't be 2D)
        if num_gpus == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'SM Clock Summary (All GPUs)\n{file_path}',
                    fontsize=16, fontweight='bold', y=0.995)

        for gpu_idx, (gpu_id, df) in enumerate(per_gpu_data.items()):
            # Shortened GPU ID for display
            short_gpu_id = str(gpu_id)[-12:] if len(str(gpu_id)) > 12 else str(gpu_id)

            # Column 0: SM Clock Histogram
            if 'sm_clock' in df.columns:
                data = df['sm_clock'].dropna()
                if len(data) > 0:
                    sm_plot.create_histogram(axes[gpu_idx, 0], data, limits=sm_limits, num_bins=30)

            axes[gpu_idx, 0].set_ylabel(f'GPU {gpu_idx}\n{short_gpu_id}', fontsize=9, fontweight='bold')
            axes[gpu_idx, 0].tick_params(labelsize=7)

            # Column 1: SM Clock Timeline
            sm_plot.create_timeline(axes[gpu_idx, 1], df, self.time_col, self.time_label, limits=sm_limits)
            axes[gpu_idx, 1].tick_params(labelsize=7)

        # Add column titles to top row
        axes[0, 0].set_title('SM Clock Distribution', fontsize=10, fontweight='bold', pad=10)
        axes[0, 1].set_title('SM Clock Over Time', fontsize=10, fontweight='bold', pad=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Master SM clock summary saved to: {save_path}")
            plt.close(fig)
        else:
            plt.show()

    def create_power_histogram(self, power_limits=(800, 1400), save_dir=None):
        """Create power histogram with specified limits per GPU"""
        if not self.per_gpu_data:
            print("No data to visualize")
            return

        per_gpu_data = self.per_gpu_data
        file_path = self.filepath

        print(f"\nGenerating Power Analysis for {len(per_gpu_data)} GPU(s)...")

        # Create MetricPlot for power
        power_plot = MetricPlot('power_draw', 'Power Draw', 'W', color='blue',
                               limits=power_limits, value_format='.1f')

        # Create plots for each GPU
        for gpu_idx, (gpu_id, df) in enumerate(per_gpu_data.items()):
            power_plot.create_detailed_analysis(df, self.time_col, self.time_label,
                                               gpu_idx, gpu_id, file_path,
                                               limits=power_limits, save_dir=save_dir)

    def create_master_memory_summary(self, save_path=None):
        """Create master memory summary plot with all GPUs (3 cols: mem used, mem free, mem util)"""
        if not self.per_gpu_data:
            print("No data to visualize")
            return

        per_gpu_data = self.per_gpu_data
        file_path = self.filepath

        num_gpus = len(per_gpu_data)
        print(f"\nGenerating Master Memory Summary for {num_gpus} GPU(s)...")

        if num_gpus == 0:
            print("Warning: No GPUs found in data, skipping master memory summary")
            return

        # Create figure with 3 columns and num_gpus rows
        fig, axes = plt.subplots(num_gpus, 3, figsize=(18, 4 * num_gpus))

        # Handle single GPU case (axes won't be 2D)
        if num_gpus == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'Memory Summary (All GPUs)\n{file_path}',
                    fontsize=16, fontweight='bold', y=0.995)

        for gpu_idx, (gpu_id, df) in enumerate(per_gpu_data.items()):
            # Shortened GPU ID for display
            short_gpu_id = str(gpu_id)[-12:] if len(str(gpu_id)) > 12 else str(gpu_id)

            # === Column 0: Memory Used Timeline ===
            ax_used = axes[gpu_idx, 0]
            if 'memory_used' in df.columns and self.time_col in df.columns:
                mem_used = df['memory_used'].dropna()
                if len(mem_used) > 0:
                    ax_used.plot(df[self.time_col], df['memory_used'],
                                alpha=0.8, color='blue', linewidth=1)
                    ax_used.set_title(f'GPU {gpu_idx} ({short_gpu_id})\nMemory Used',
                                     fontsize=10, fontweight='bold')
                    ax_used.set_xlabel(self.time_label, fontsize=9)
                    ax_used.set_ylabel('Memory Used (MiB)', fontsize=9)
                    ax_used.grid(True, alpha=0.3)

                    # Add mean line
                    mean_used = mem_used.mean()
                    ax_used.axhline(mean_used, color='red', linestyle='--',
                                   linewidth=1, alpha=0.7, label=f'Mean: {mean_used:.0f} MiB')
                    ax_used.legend(fontsize=8)
                else:
                    ax_used.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                               transform=ax_used.transAxes)
            else:
                ax_used.text(0.5, 0.5, 'Memory used data not available',
                           ha='center', va='center', transform=ax_used.transAxes)

            # === Column 1: Memory Free Timeline ===
            ax_free = axes[gpu_idx, 1]
            if 'memory_total' in df.columns and 'memory_used' in df.columns and self.time_col in df.columns:
                df_temp = df[['memory_total', 'memory_used']].copy()
                df_temp['memory_free'] = df_temp['memory_total'] - df_temp['memory_used']
                mem_free = df_temp['memory_free'].dropna()

                if len(mem_free) > 0:
                    ax_free.plot(df[self.time_col], df_temp['memory_free'],
                               alpha=0.8, color='green', linewidth=1)
                    ax_free.set_title(f'GPU {gpu_idx} ({short_gpu_id})\nMemory Free',
                                    fontsize=10, fontweight='bold')
                    ax_free.set_xlabel(self.time_label, fontsize=9)
                    ax_free.set_ylabel('Memory Free (MiB)', fontsize=9)
                    ax_free.grid(True, alpha=0.3)

                    # Add mean line
                    mean_free = mem_free.mean()
                    ax_free.axhline(mean_free, color='red', linestyle='--',
                                  linewidth=1, alpha=0.7, label=f'Mean: {mean_free:.0f} MiB')
                    ax_free.legend(fontsize=8)
                else:
                    ax_free.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                               transform=ax_free.transAxes)
            else:
                ax_free.text(0.5, 0.5, 'Memory free data not available',
                           ha='center', va='center', transform=ax_free.transAxes)

            # === Column 2: Memory Utilization Timeline ===
            ax_util = axes[gpu_idx, 2]
            if 'memory_util' in df.columns and self.time_col in df.columns:
                mem_util = df['memory_util'].dropna()
                if len(mem_util) > 0:
                    ax_util.plot(df[self.time_col], df['memory_util'],
                               alpha=0.8, color='purple', linewidth=1)
                    ax_util.set_title(f'GPU {gpu_idx} ({short_gpu_id})\nMemory Utilization',
                                    fontsize=10, fontweight='bold')
                    ax_util.set_xlabel(self.time_label, fontsize=9)
                    ax_util.set_ylabel('Memory Utilization (%)', fontsize=9)
                    ax_util.set_ylim(0, 100)
                    ax_util.grid(True, alpha=0.3)

                    # Add mean line
                    mean_util = mem_util.mean()
                    ax_util.axhline(mean_util, color='red', linestyle='--',
                                  linewidth=1, alpha=0.7, label=f'Mean: {mean_util:.1f}%')
                    ax_util.legend(fontsize=8)
                else:
                    ax_util.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                               transform=ax_util.transAxes)
            else:
                ax_util.text(0.5, 0.5, 'Memory utilization data not available',
                           ha='center', va='center', transform=ax_util.transAxes)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Master memory summary saved to: {save_path}")
            plt.close(fig)
        else:
            plt.show()

    def create_memory_analysis(self, save_dir=None):
        """Create detailed memory analysis plots per GPU"""
        if not self.per_gpu_data:
            print("No data to visualize")
            return

        per_gpu_data = self.per_gpu_data
        file_path = self.filepath

        print(f"\nGenerating Memory Analysis for {len(per_gpu_data)} GPU(s)...")

        # Create plots for each GPU
        for gpu_idx, (gpu_id, df) in enumerate(per_gpu_data.items()):
            has_mem_data = ('memory_used' in df.columns or 'memory_total' in df.columns or
                          'memory_util' in df.columns)

            if not has_mem_data:
                print(f"No memory data found for GPU {gpu_id}!")
                continue

            # Create figure with 3 rows: used, free, utilization
            fig, axes = plt.subplots(3, 1, figsize=(14, 12))

            # Add file path and GPU ID as main title
            fig.suptitle(f'Memory Analysis - GPU {gpu_idx}\n{file_path}\n{gpu_id}',
                        fontsize=14, fontweight='bold')

            # === Row 0: Memory Used ===
            if 'memory_used' in df.columns and self.time_col in df.columns:
                mem_used = df['memory_used'].dropna()
                if len(mem_used) > 0:
                    axes[0].plot(df[self.time_col], df['memory_used'],
                               alpha=0.8, color='blue', linewidth=1)
                    axes[0].set_title('Memory Used Over Time', fontsize=12, fontweight='bold')
                    axes[0].set_xlabel(self.time_label, fontsize=11)
                    axes[0].set_ylabel('Memory Used (MiB)', fontsize=11)
                    axes[0].grid(True, alpha=0.3)

                    # Statistics
                    mean_used = mem_used.mean()
                    axes[0].axhline(mean_used, color='red', linestyle='--', linewidth=2,
                                  label=f'Mean: {mean_used:.0f} MiB')
                    axes[0].legend()

                    print(f"  GPU {gpu_idx} Memory Used - Mean: {mean_used:.0f} MiB, "
                          f"Min: {mem_used.min():.0f} MiB, Max: {mem_used.max():.0f} MiB")

            # === Row 1: Memory Free ===
            if 'memory_total' in df.columns and 'memory_used' in df.columns and self.time_col in df.columns:
                df_temp = df[['memory_total', 'memory_used']].copy()
                df_temp['memory_free'] = df_temp['memory_total'] - df_temp['memory_used']
                mem_free = df_temp['memory_free'].dropna()

                if len(mem_free) > 0:
                    axes[1].plot(df[self.time_col], df_temp['memory_free'],
                               alpha=0.8, color='green', linewidth=1)
                    axes[1].set_title('Memory Free Over Time', fontsize=12, fontweight='bold')
                    axes[1].set_xlabel(self.time_label, fontsize=11)
                    axes[1].set_ylabel('Memory Free (MiB)', fontsize=11)
                    axes[1].grid(True, alpha=0.3)

                    # Statistics
                    mean_free = mem_free.mean()
                    axes[1].axhline(mean_free, color='red', linestyle='--', linewidth=2,
                                  label=f'Mean: {mean_free:.0f} MiB')
                    axes[1].legend()

                    print(f"  GPU {gpu_idx} Memory Free - Mean: {mean_free:.0f} MiB, "
                          f"Min: {mem_free.min():.0f} MiB, Max: {mem_free.max():.0f} MiB")

            # === Row 2: Memory Utilization ===
            if 'memory_util' in df.columns and self.time_col in df.columns:
                mem_util = df['memory_util'].dropna()
                if len(mem_util) > 0:
                    axes[2].plot(df[self.time_col], df['memory_util'],
                               alpha=0.8, color='purple', linewidth=1)
                    axes[2].set_title('Memory Utilization Over Time', fontsize=12, fontweight='bold')
                    axes[2].set_xlabel(self.time_label, fontsize=11)
                    axes[2].set_ylabel('Memory Utilization (%)', fontsize=11)
                    axes[2].set_ylim(0, 100)
                    axes[2].grid(True, alpha=0.3)

                    # Statistics
                    mean_util = mem_util.mean()
                    axes[2].axhline(mean_util, color='red', linestyle='--', linewidth=2,
                                  label=f'Mean: {mean_util:.1f}%')
                    axes[2].legend()

                    print(f"  GPU {gpu_idx} Memory Util - Mean: {mean_util:.1f}%, "
                          f"Min: {mem_util.min():.1f}%, Max: {mem_util.max():.1f}%")

            plt.tight_layout()

            if save_dir:
                save_path = os.path.join(save_dir, f'memory_analysis_gpu{gpu_idx}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Memory analysis saved to: {save_path}")
                plt.close(fig)
            else:
                plt.show()

def parse_power_limits(limits_str):
    """Parse power limits from string format 'min,max'"""
    try:
        parts = limits_str.split(',')
        if len(parts) != 2:
            raise ValueError("Power limits must be in format 'min,max'")
        return float(parts[0]), float(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid power limits '{limits_str}': {e}")

def parse_sm_limits(limits_str):
    """Parse SM clock limits from string format 'min,max'"""
    try:
        parts = limits_str.split(',')
        if len(parts) != 2:
            raise ValueError("SM limits must be in format 'min,max'")
        return float(parts[0]), float(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid SM limits '{limits_str}': {e}")

def main():
    parser = argparse.ArgumentParser(description='Visualize NVIDIA SMI CSV file')

    # Single file input
    parser.add_argument('file', help='CSV file to visualize')

    parser.add_argument('--output-dir', '-o',
                       default='nvsmi',
                       help='Output directory for plots')
    parser.add_argument('--time-unit', '-t', choices=['seconds', 'minutes'],
                       default='seconds',
                       help='Time unit for x-axis (default: seconds)')
    parser.add_argument('--power-limits', type=parse_power_limits, default='800,1400',
                       help='Power limits for histogram as min,max (default: 800,1400)')
    parser.add_argument('--sm-limits', type=parse_sm_limits, default='0,3000',
                       help='SM clock limits for analysis as min,max MHz (default: 0,3000)')
    parser.add_argument('--power-only', action='store_true',
                       help='Only generate power analysis')
    parser.add_argument('--sm-only', action='store_true',
                       help='Only generate SM clock analysis')
    parser.add_argument('--memory-only', action='store_true',
                       help='Only generate memory analysis')
    parser.add_argument('--show-plots', '-s', action='store_true',
                       help='Show plots interactively')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse data
    print(f"Parsing NVIDIA SMI CSV file: {args.file}")

    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        return

    data = parse_nvsmi_file(args.file)

    if not data:
        print("No valid data found in file!")
        return

    total_records = sum(len(gpu_df) for gpu_df in data['per_gpu_data'].values())
    print(f"Loaded: {total_records} records, {data['gpu_count']} GPU(s)")
    print(f"GPU IDs: {', '.join(str(gid) for gid in data['gpu_ids'])}")

    # Create visualizations
    visualizer = NVIDIASMIVisualizer(data['per_gpu_data'], data['filepath'], time_unit=args.time_unit)

    if args.power_only:
        print(f"\nGenerating power histogram with limits {args.power_limits[0]:.0f}W - {args.power_limits[1]:.0f}W...")
        # Power histogram only
        visualizer.create_power_histogram(power_limits=args.power_limits, save_dir=args.output_dir)

    elif args.sm_only:
        print(f"\nGenerating SM clock analysis with limits {args.sm_limits[0]:.0f}MHz - {args.sm_limits[1]:.0f}MHz...")
        # SM clock analysis only
        visualizer.create_sm_utilization_analysis(util_limits=args.sm_limits, save_dir=args.output_dir)

    elif args.memory_only:
        print(f"\nGenerating memory analysis...")
        # Memory analysis only
        visualizer.create_memory_analysis(save_dir=args.output_dir)

    else:
        print(f"\nGenerating visualizations with time in {args.time_unit}...")

        # Master power summary (all GPUs in one plot)
        master_power_path = os.path.join(args.output_dir, 'master_power_summary.png')
        visualizer.create_master_power_summary(power_limits=args.power_limits, save_path=master_power_path)

        # Master SM clock summary (all GPUs in one plot)
        master_sm_path = os.path.join(args.output_dir, 'master_sm_summary.png')
        visualizer.create_master_sm_summary(sm_limits=args.sm_limits, save_path=master_sm_path)

        # Master memory summary (all GPUs in one plot)
        master_memory_path = os.path.join(args.output_dir, 'master_memory_summary.png')
        visualizer.create_master_memory_summary(save_path=master_memory_path)

        # Power histogram
        visualizer.create_power_histogram(power_limits=args.power_limits, save_dir=args.output_dir)

        # SM clock analysis
        visualizer.create_sm_utilization_analysis(util_limits=args.sm_limits, save_dir=args.output_dir)

        # Memory analysis
        visualizer.create_memory_analysis(save_dir=args.output_dir)

    print(f"\nAll plots saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
