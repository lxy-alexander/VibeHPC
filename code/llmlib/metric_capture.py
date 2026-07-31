#!/usr/bin/env python3
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
Standalone script to capture metrics from trtllm-serve /metrics and /perf_metrics endpoints.
This script runs as a background process during harness execution.
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any

import requests

# NOTE(vir|ryan): WAR
# dynamo /metrics endpoint returns plaintext in prometheus format, not JSON format
DYNAMO_OVERRIDE_NOUSE_JSON_METRICS = os.getenv('MLPINF_USE_DYNAMO', '0') == '1'


class MetricsCapture:
    """Captures metrics from trtllm-serve /metrics endpoint.

    Polls endpoints indefinitely, waiting for each response to complete
    before polling again at the specified interval.
    """

    def __init__(self,
                 endpoint_urls: List[str],
                 output_dir: Path,
                 poll_interval: float = 1.0,
                 server_logs_dir: Optional[Path] = None):
        """
        Initialize metrics capture.

        Args:
            endpoint_urls: List of trtllm-serve endpoint URLs (e.g., ["localhost:8000"])
            output_dir: Directory to save metrics logs
            poll_interval: Time between polls in seconds
            server_logs_dir: Directory containing server logs to monitor
        """
        self.endpoint_urls = endpoint_urls
        self.output_dir = Path(output_dir)
        self.poll_interval = poll_interval
        self.server_logs_dir = Path(server_logs_dir) if server_logs_dir else None
        self.running = True

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Setup HTTP session
        self.session = requests.Session()

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Initialize server log tracking
        self.server_log_positions = {}
        self.server_log_files = {}
        self.log_copy_thread = None
        if self.server_logs_dir and self.server_logs_dir.exists():
            self._initialize_server_logs()
            # Start log copy thread
            self.log_copy_thread = threading.Thread(target=self._log_copy_worker, daemon=True)
            self.log_copy_thread.start()
            logging.info("Started server log copy thread")

        # Initialize performance metrics capture
        # Store metrics in memory and dump as JSON array on shutdown
        self.perf_metrics_data = []
        self.perf_metrics_lock = threading.Lock()
        self.perf_metrics_thread = threading.Thread(target=self._perf_metrics_worker, daemon=True)
        self.perf_metrics_thread.start()
        logging.info("Started performance metrics capture thread")

    def _setup_logging(self):
        """Configure logging to file."""
        log_file = self.output_dir / "metrics_capture.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logging.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def _initialize_server_logs(self):
        """Initialize server log tracking by capturing current line positions."""
        server_log_pattern = "trtllm_serve_*.log"
        server_logs = sorted(self.server_logs_dir.glob(server_log_pattern))

        logging.info(f"Found {len(server_logs)} server log files in {self.server_logs_dir}")

        for log_file in server_logs:
            try:
                # Count current lines in the file
                with open(log_file, 'r') as f:
                    line_count = sum(1 for _ in f)
                    self.server_log_positions[log_file] = line_count

                # Create output file for this server log
                log_idx = log_file.stem.split('_')[-1]  # Extract number from trtllm_serve_X.log
                output_file = self.output_dir / f"server_log_{log_idx}.log"
                self.server_log_files[log_file] = open(output_file, 'w')

                logging.info(f"Tracking {log_file.name} from line {line_count}, output to {output_file.name}")
            except Exception as e:
                logging.warning(f"Failed to initialize tracking for {log_file}: {e}")

    def _log_copy_worker(self):
        """Worker thread that continuously copies new server log lines."""
        logging.info("Server log copy thread started")
        while self.running:
            try:
                self._copy_new_server_logs()
                time.sleep(self.poll_interval)
            except Exception as e:
                logging.error(f"Error in log copy thread: {e}", exc_info=True)
                time.sleep(self.poll_interval)  # Continue even on error
        logging.info("Server log copy thread stopped")

    def _copy_new_server_logs(self):
        """Copy new lines from server logs to output files."""
        for log_file, current_pos in self.server_log_positions.items():
            try:
                with open(log_file, 'r') as f:
                    # Skip to the last known position
                    for _ in range(current_pos):
                        f.readline()

                    # Read new lines
                    new_lines = []
                    for line in f:
                        new_lines.append(line)

                    # Update position
                    self.server_log_positions[log_file] = current_pos + len(new_lines)

                    # Write new lines to output
                    if new_lines and log_file in self.server_log_files:
                        output_file = self.server_log_files[log_file]
                        output_file.writelines(new_lines)
                        output_file.flush()

                        if len(new_lines) > 0:
                            logging.debug(f"Copied {len(new_lines)} new lines from {log_file.name}")

            except Exception as e:
                logging.warning(f"Failed to copy logs from {log_file}: {e}")

    def _perf_metrics_worker(self):
        """Worker thread that continuously captures performance metrics from /perf_metrics endpoint."""
        logging.info("Performance metrics capture thread started")
        
        # Use first endpoint (appropriate for disaggregated serving)
        endpoint_url = self.endpoint_urls[0]
        if len(self.endpoint_urls) > 1:
            logging.info(f"Multiple endpoints provided, using first for perf_metrics: {endpoint_url}")
        
        temp_metrics_file = self.output_dir / "perf_metrics_temp.json"
        
        while self.running:
            try:
                self._capture_perf_metrics(endpoint_url, temp_metrics_file)
                time.sleep(self.poll_interval)
            except Exception as e:
                logging.error(f"Error in perf metrics capture thread: {e}", exc_info=True)
                time.sleep(self.poll_interval)  # Continue even on error
        
        # Clean up temp file if it exists
        if temp_metrics_file.exists():
            temp_metrics_file.unlink()
        
        logging.info("Performance metrics capture thread stopped")

    def _capture_perf_metrics(self, endpoint_url: str, temp_file: Path):
        """
        Capture performance metrics and store in memory.
        
        Args:
            endpoint_url: The endpoint URL (e.g., "localhost:8000")
            temp_file: Temporary file for curl output
        """
        target_url = f"http://{endpoint_url}/perf_metrics"
        
        curl_cmd = [
            "curl", "-s", "-o", str(temp_file), target_url
        ]
        
        try:
            subprocess.run(curl_cmd, check=True, capture_output=True, timeout=10)
            
            # Read the metrics
            with open(temp_file, 'r') as f:
                metrics_data = json.load(f)
            
            # Store metrics in memory (flatten if it's a list)
            with self.perf_metrics_lock:
                if isinstance(metrics_data, list):
                    self.perf_metrics_data.extend(metrics_data)
                else:
                    self.perf_metrics_data.append(metrics_data)
            
            logging.debug(f"Captured {len(metrics_data) if isinstance(metrics_data, list) else 1} perf metrics")
            
        except subprocess.TimeoutExpired:
            logging.warning(f"Timeout fetching perf_metrics from {endpoint_url}")
        except subprocess.CalledProcessError as e:
            logging.warning(f"Failed to capture perf_metrics from {endpoint_url}: {e}")
        except json.JSONDecodeError as e:
            logging.warning(f"Failed to parse perf_metrics JSON from {endpoint_url}: {e}")
        except Exception as e:
            logging.warning(f"Error processing perf_metrics from {endpoint_url}: {e}")

    def _dump_perf_metrics(self):
        """Dump collected performance metrics to JSON file (time_breakdown.py compatible format)."""
        output_file = self.output_dir / "perf_metrics.json"
        
        with self.perf_metrics_lock:
            metrics_count = len(self.perf_metrics_data)
            if metrics_count > 0:
                with open(output_file, 'w') as f:
                    json.dump(self.perf_metrics_data, f, indent=2)
                logging.info(f"Dumped {metrics_count} performance metrics to {output_file}")
            else:
                logging.info("No performance metrics captured")

    def _poll_metrics(self, endpoint_url: str) -> Optional[List[Dict[str, Any]]]:
        """
        Poll metrics from a single endpoint. Waits indefinitely for response.

        Args:
            endpoint_url: The endpoint URL (e.g., "localhost:8000")

        Returns:
            Metrics data or None if request failed
        """
        try:
            url = f"http://{endpoint_url}/metrics"
            # No timeout - wait indefinitely for the server to return all data
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.warning(f"Failed to fetch metrics from {endpoint_url}: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.warning(f"Invalid JSON response from {endpoint_url}: {e}")
            if DYNAMO_OVERRIDE_NOUSE_JSON_METRICS:
                return {}
            return None

    def run(self):
        """Main loop to continuously poll metrics."""
        logging.info(f"Starting metrics capture for endpoints: {self.endpoint_urls}")
        logging.info(f"Output directory: {self.output_dir}")
        logging.info(f"Poll interval: {self.poll_interval}s")

        # Open metric files for each endpoint
        metric_files = {}
        for endpoint_url in self.endpoint_urls:
            # Create safe filename from endpoint URL
            safe_name = endpoint_url.replace(":", "_").replace(".", "_")
            metric_file = self.output_dir / f"metrics_{safe_name}.jsonl"
            metric_files[endpoint_url] = open(metric_file, 'w')
            logging.info(f"Writing metrics for {endpoint_url} to {metric_file}")

        try:
            poll_count = 0
            while self.running:
                # Poll each endpoint
                for endpoint_url in self.endpoint_urls:
                    metrics = self._poll_metrics(endpoint_url)

                    if metrics is not None:
                        # Add poll count and write to file
                        for metric in metrics:
                            metric['poll_count'] = poll_count

                        # Write as JSON lines
                        metric_file = metric_files[endpoint_url]
                        for metric in metrics:
                            json.dump(metric, metric_file)
                            metric_file.write('\n')
                        metric_file.flush()

                        # Log summary
                        if metrics:
                            latest = metrics[-1]
                            logging.debug(
                                f"[{endpoint_url}] Iter: {latest.get('iter', 'N/A')}, "
                                f"GPU Mem: {latest.get('gpuMemUsage', 0) / 1e9:.2f}GB, "
                                f"Latency: {latest.get('iterLatencyMS', 0):.2f}ms"
                            )

                poll_count += 1

                if self.running:
                    time.sleep(self.poll_interval)

        finally:
            # Wait for log copy thread to finish
            if self.log_copy_thread and self.log_copy_thread.is_alive():
                logging.info("Waiting for log copy thread to finish...")
                self.log_copy_thread.join(timeout=5.0)

            # Wait for perf metrics thread to finish
            if self.perf_metrics_thread and self.perf_metrics_thread.is_alive():
                logging.info("Waiting for perf metrics thread to finish...")
                self.perf_metrics_thread.join(timeout=5.0)

            # Dump performance metrics as JSON array (time_breakdown.py compatible)
            self._dump_perf_metrics()

            # Close all metric files
            for metric_file in metric_files.values():
                metric_file.close()

            # Close all server log files
            for log_file in self.server_log_files.values():
                log_file.close()

            logging.info("Metrics capture stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Capture metrics from trtllm-serve endpoints")
    parser.add_argument(
        "--endpoints",
        type=str,
        help="Comma-separated list of endpoint URLs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default='/tmp/trtllm_metrics/',
        help="Output directory for metrics logs"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--server-logs-dir",
        type=str,
        default=None,
        help="Directory containing server logs to monitor"
    )

    args = parser.parse_args()

    # Parse endpoints
    endpoint_urls = [url.strip() for url in args.endpoints.split(",") if url.strip()]

    if not endpoint_urls:
        logging.error("No endpoints specified")
        sys.exit(1)

    # Create and run metrics capture
    capture = MetricsCapture(
        endpoint_urls=endpoint_urls,
        output_dir=Path(args.output_dir),
        poll_interval=args.poll_interval,
        server_logs_dir=Path(args.server_logs_dir) if args.server_logs_dir else None,
    )

    try:
        capture.run()
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
