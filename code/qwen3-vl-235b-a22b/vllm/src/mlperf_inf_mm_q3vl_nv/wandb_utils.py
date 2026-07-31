"""Weights & Biases (wandb) utilities for logging benchmark configurations and results."""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Final

import wandb
from loguru import logger
from mlperf_inf_mm_q3vl.schema import Settings

# Nanoseconds to milliseconds conversion factor
NS_TO_MS = 1_000_000


class MultipleLogFilesFoundError(RuntimeError):
    """Raised when multiple log files match the expected pattern."""

    def __init__(self, pattern: str, matches: list[str]) -> None:
        """Initialize the exception.

        Args:
            pattern: The glob pattern used to search for files.
            matches: The list of matching file paths.
        """
        super().__init__(
            f"Found {len(matches)} files matching pattern '{pattern}', expected 1. "
            f"Matches: {matches}",
        )


# MLPerf log prefix
MLPERF_LOG_PREFIX: Final[str] = ":::MLLOG"

# Environment variable prefixes/names to capture for logging
RELEVANT_ENV_PREFIXES: tuple[str, ...] = (
    "CUDA_",
    "DYN_",
    "ETCD_",
    "MLPERF_",
    "MPI_",
    "NATS_",
    "NCCL_",
    "NVIDIA_",
    "NV_",
    "OMPI_",
    "OMP_",
    "TORCH_",
    "TRITON_",
    "UCX_",
    "VLLM_",
)


def parse_cmd_to_dict(cmd: list[str]) -> dict[str, str | bool]:
    """Parse a command list into a dictionary format.

    Converts CLI arguments into key-value pairs:
    - `--variable value` becomes `{"variable": "value"}`
    - `--variable=value` becomes `{"variable": "value"}`
    - `--flag` (store_true flag) becomes `{"flag": True}`
    - `-tp 4` becomes `{"tp": "4"}` (multi-letter short options)

    Single-letter short options like `-m` are skipped.

    Args:
        cmd: The command list to parse.

    Returns:
        A dictionary with parsed arguments.
    """
    result: dict[str, str | bool] = {}
    i = 0
    while i < len(cmd):
        arg = cmd[i]
        """
            When the arg does not start with "--" or "-", the cmd may contain 
            "python3 xxx", which would be skipped to scan the next arg
        """

        if arg.startswith("--"):
            prefix = "--"
        elif arg.startswith("-") and len(arg) > 2 and not arg[1].isdigit():
            prefix = "-"
        else:
            # Skip args that don't start with - or -- (like "python3")
            i += 1
            continue

        # Handle `--key=value` or `-k=value` format
        if "=" in arg:
            key, value = arg[len(prefix) :].split("=", 1)
            result[key] = value
        # Handle `--key value` or `-k value` format
        elif i + 1 < len(cmd) and not cmd[i + 1].startswith("-"):
            # Next arg is a value
            key = arg[len(prefix) :]
            result[key] = cmd[i + 1]
            i += 1  # Skip the value
        else:
            # It's a store_true flag
            key = arg[len(prefix) :]
            result[key] = True

        i += 1

    return result


def filter_relevant_env(env: dict[str, str]) -> dict[str, str]:
    """Filter environment variables to only include relevant/user-specified ones.

    Args:
        env: The full environment dictionary.

    Returns:
        A filtered dictionary with only relevant environment variables.
    """
    return {
        k: v
        for k, v in env.items()
        if any(k.startswith(prefix) for prefix in RELEVANT_ENV_PREFIXES)
    }


def _filter_sensitive_args(cmd: list[str]) -> list[str]:
    """Filter out sensitive arguments from a command list.

    Args:
        cmd: The command list to filter.

    Returns:
        A new list with sensitive arguments removed.
    """
    sensitive_args = {"--hf-token", "--api-key", "--token"}
    filtered_cmd = []
    i = 0

    while i < len(cmd):
        arg = cmd[i]

        # Check for --sensitive-arg=value format
        k, v = arg.split("=", 1) if "=" in arg else (None, None)

        # Check for --sensitive-arg value format
        if arg in sensitive_args or k in sensitive_args:
            if v is None and i + 1 < len(cmd):  # --arg value format
                i += 1  # Skip the next argument (the value)
            i += 1  # Move to next argument
            continue

        filtered_cmd.append(arg)
        i += 1

    return filtered_cmd


def log_launch_config_to_wandb(
    name: str,
    cmd: list[str],
    env: dict[str, str] | None = None,
) -> None:
    """Log launch configuration to wandb.

    Args:
        name: The name of the process being launched.
        cmd: The command being executed.
        env: The environment variables for the process. Optional, defaults to empty dict.
    """
    if wandb.run is None:
        logger.warning("wandb run is not initialized, skipping launch config logging")
        return

    # Filter out sensitive arguments before parsing
    filtered_cmd = _filter_sensitive_args(cmd)

    # Parse cmd into a dictionary format
    cmd_dict = parse_cmd_to_dict(filtered_cmd)

    # Filter env to only include relevant variables
    filtered_env = filter_relevant_env(env) if env else {}

    config_update = {
        f"{name}/cmd_parsed": cmd_dict,
    }

    # Only add env if it's not empty
    if filtered_env:
        config_update[f"{name}/env"] = filtered_env

    wandb.config.update(config_update, allow_val_change=True)
    logger.info("Logged {} launch config to wandb", name)


def parse_mlperf_detail_log(file_content: str) -> dict[str, Any]:
    """Parse MLPerf LoadGen detail log file into a structured dictionary.

    The detail log contains lines in the format:
    :::MLLOG {"key": "metric_name", "value": metric_value, ...}

    Each line after the :::MLLOG prefix is a valid JSON object.

    Args:
        file_content: The content of the mlperf_log_detail.txt file.

    Returns:
        A dictionary mapping metric keys to their latest values.
    """
    lines = file_content.strip().split("\n")

    results: dict[str, list[Any]] = defaultdict(list)
    for line in lines:
        if line.startswith(MLPERF_LOG_PREFIX):
            # Strip the :::MLLOG prefix and the space after it
            json_str = line[len(MLPERF_LOG_PREFIX) + 1 :]
            try:
                # Replace null characters that may violate JSON standard
                entry = json.loads(json_str.replace("\x00", "").replace("\u0000", ""))
                key = entry["key"]
                results[key].append(entry["value"])
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to parse MLPerf log entry: {} - {}", line, e)
                continue

    # Return only the latest value for each key
    return {k: v[-1] for k, v in results.items()}


def find_mlperf_log_detail(settings: Settings) -> Path | None:
    """Find the mlperf_log_detail.txt file in the log directory.

    Searches for files matching the pattern:
    {prefix}*detail*{suffix}.txt

    Args:
        settings: The benchmark settings containing log output configuration.

    Returns:
        Path to the detail file if found, None otherwise.
    """
    log_dir = Path(settings.logging.log_output.outdir)
    prefix = settings.logging.log_output.prefix
    suffix = settings.logging.log_output.suffix
    key = "detail"

    # Build the filename pattern
    filename_pattern = f"{prefix}*{key}*{suffix}.txt"

    # Search in log_dir
    pattern = str(log_dir / "**" / filename_pattern)
    matches = glob.glob(pattern, recursive=True)

    if not matches:
        return None

    if len(matches) > 1:
        raise MultipleLogFilesFoundError(filename_pattern, matches)

    return Path(matches[0])


def read_and_log_mlperf_detail_to_wandb(settings: Settings) -> None:
    """Read mlperf_log_detail.txt and log its contents to wandb.

    The detail log file contains JSON-formatted log entries that are
    easier to parse than the summary text file.

    Args:
        settings: The benchmark settings containing the log output directory.
    """
    if wandb.run is None:
        return

    detail_path = find_mlperf_log_detail(settings)
    if detail_path is None:
        logger.warning(
            "Could not find mlperf_log_detail.txt in {}",
            settings.logging.log_output.outdir,
        )
        return

    logger.info("Reading MLPerf detail log from {}", detail_path)

    try:
        detail_content = detail_path.read_text()
        metrics = parse_mlperf_detail_log(detail_content)
        if metrics:
            wandb.log(metrics)
            logger.info(
                "Logged {} metrics to wandb from mlperf_log_detail.txt",
                len(metrics),
            )
        else:
            logger.warning("Could not parse any metrics from mlperf_log_detail.txt")

        # Save all log files in the log directory as artifacts
        log_dir = Path(settings.logging.log_output.outdir)
        log_files = [f for f in log_dir.glob("*") if f.is_file()]
        for log_file in log_files:
            wandb.save(str(log_file), base_path=str(log_dir), policy="now")
        logger.info(
            "Saved {} log files from {} as wandb artifacts",
            len(log_files),
            log_dir,
        )

    except Exception as e:
        logger.error("Failed to read/parse mlperf_log_detail.txt: {}", e)
        raise
