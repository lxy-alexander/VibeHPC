# Plot Generation Guide

This directory contains scripts to generate three types of performance plots from MLPerf runs.

## Prerequisites

```bash
pip install -r scripts/plot/requirements.txt
```

---

## 1. NVIDIA-SMI Trace Plots

Visualizes GPU power, temperature, memory usage, and SM clocks from nvidia-smi CSV logs.

**Steps:**
Using the `--verbose_nvsmi` RUN_ARG is going to trigger a `nvidia-smi --query-gpu=...` process in the background. In the end of the make target, the plotting script is invoked to visualize the stats
```bash
make run_harness RUN_ARGS="... --verbose_nvsmi"
```

**Output:** Power histograms, SM clock analysis, memory utilization plots (per GPU + master summaries)

**Environment Variables:**
- `NVSMI_REFRESH_RATE` - Polling interval in milliseconds (default: 200ms)
  ```bash
  # Use custom 500ms polling interval
  NVSMI_REFRESH_RATE=500 make run_harness RUN_ARGS="... --verbose_nvsmi"
  ```

**Options:**
- `--power-only` - Generate only power analysis
- `--sm-only` - Generate only SM clock analysis
- `--memory-only` - Generate only memory analysis
- `--truncate-tail` - Remove inactive tail from data
  - `--tail-threshold` - The variation to identify as tail. By default, this is 0.01 (defines tail as time period having less than 1% variation)

---

## 2. Nsys Profiling Plots

Analyzes CUDA kernel execution and GPU utilization from nsys profiling data.

**Steps:**

1. Collect nsys profile during your run using `--nsys_options`:
   ```bash
   # Example run command with nsys profiling
   make run_harness RUN_ARGS="... --nsys_options="/path/to/nsys_options.yml"
   ```

2. **Export nsys-rep to SQLite** (manual step required):
   ```bash
   nsys export --sqlite=/path/to/output.sqlite /path/to/output.nsys-rep
   ```

3. Generate plots:
   ```bash
   python nsys_sqlite.py /path/to/output.sqlite --output-dir nsys_plots
   ```

**Output:** Kernel time share pie chart, launch density timeline, GPU utilization dashboard

**Options:**
- `--timeline-duration 10.0` - Duration for timeline analysis (seconds)
- `--summary-only` - Print statistics only, no plots

---

## 3. Token Statistics Plots

Visualizes token processing, KV cache usage, and throughput from endpoint harness logs.

**Steps:**

1. Extract token stats from endpoint harness logs (JSONL format):
  - `--core_type=trtllm_endpoint`: Iteration stats is at `$LOG_DIR/endpoint_harness_logs/metrics_0_0_0_0_$port.jsonl`
  - `--core_type=trtllm_executor`: Iteration stats is at `$LOG_DIR/$system_name/$workload_name/$scenario/harness_iteration_stats.log`
2. Generate plots:
   ```bash
   python token_stats.py /path/to/token_stats.jsonl --output-dir token_plots
   ```

**Output:** Dashboard with context tokens, active requests, KV cache blocks/utilization, throughput over time

**Options:**
- `--summary-only` - Print statistics only, no plots

---

## Notes

- All scripts save plots as high-resolution PNG files (300 dpi)
- Use `--help` with any script for full options list
- For nsys: the manual export step (nsys-rep → sqlite) is required before plotting
