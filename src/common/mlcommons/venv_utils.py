# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Shared virtual environment utilities for accuracy checkers and compliance tests."""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import sys
from pathlib import Path


def ensure_venv_ready(venv_path: Path, requirements_file: Path) -> Path:
    """Ensure a virtual environment exists with required dependencies installed.

    The venv is considered ready if:
    1. The venv python3 binary exists
    2. A marker file exists with the hash of the requirements file (to detect changes)

    Args:
        venv_path: Path where the venv should be created/verified.
        requirements_file: Path to requirements.txt file for pip install.

    Returns:
        Path to the venv directory.

    Raises:
        FileNotFoundError: If requirements_file doesn't exist.
        RuntimeError: If venv creation or pip install fails.
    """
    if not requirements_file.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_file}")

    venv_python = venv_path / "bin" / "python3"
    marker_file = venv_path / ".requirements_hash"
    requirements_hash = _hash_file(requirements_file)

    # Check if venv is ready
    if venv_path.exists():
        if not venv_python.exists():
            logging.warning(f"Venv at {venv_path} is corrupted (python not found). Recreating...")
            shutil.rmtree(venv_path)
        elif not marker_file.exists() or marker_file.read_text().strip() != requirements_hash:
            logging.warning(f"Requirements have changed. Recreating venv at {venv_path}...")
            shutil.rmtree(venv_path)
        else:
            logging.info(f"Venv ready at {venv_path}")
            return venv_path

    # Create new venv
    _create_venv(venv_path, requirements_file, requirements_hash)
    return venv_path


def _hash_file(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _create_venv(venv_path: Path, requirements_file: Path, requirements_hash: str) -> None:
    """Create a new venv and install requirements."""
    logging.info(f"Creating venv at {venv_path}...")
    subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)

    logging.info(f"Installing requirements from {requirements_file}...")
    pip_path = venv_path / "bin" / "pip"
    result = subprocess.run(
        [str(pip_path), "install", "-r", str(requirements_file)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logging.error(f"Failed to install requirements: {result.stderr}")
        raise RuntimeError(
            f"Failed to install requirements from {requirements_file}. "
            f"Please manually run: {pip_path} install -r {requirements_file}\n"
            f"Error: {result.stderr}"
        )

    # Write marker file with requirements hash
    marker_file = venv_path / ".requirements_hash"
    marker_file.write_text(requirements_hash)
    logging.info(f"Successfully created venv at {venv_path}")
