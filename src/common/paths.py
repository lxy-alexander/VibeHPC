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


from pathlib import Path
from typing import Final

import os


def _verify_path(p: Path, create_if_missing: bool = False):
    if not p.exists():
        if create_if_missing:
            p.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(f"Core file {p} does not exist.")
    return p


# __file__ is path/to/code/common/paths.py
PROJECT_BASE_DIR: Final[Path] = _verify_path(Path(__file__).parent.parent.parent)
WORKING_DIR: Final[Path] = Path.cwd()

VERSION_FILE: Final[Path] = _verify_path(PROJECT_BASE_DIR / "VERSION")

CODE_DIR: Final[Path] = _verify_path(PROJECT_BASE_DIR / "code")
BUILD_DIR: Final[Path] = _verify_path(Path(os.environ.get("BUILD_DIR", PROJECT_BASE_DIR / "build")), create_if_missing=True)
MODEL_DIR: Final[Path] = BUILD_DIR / "models"
SUBMODULES_DIR: Final[Path] = _verify_path(PROJECT_BASE_DIR / "3rdparty")
TRTLLM_DIR: Final[Path] = _verify_path(SUBMODULES_DIR / "trtllm")
MLCOMMONS_INF_REPO: Final[Path] = _verify_path(SUBMODULES_DIR / "mlc-inference")
RESULTS_SUBMISSION_DIR: Final[Path] = Path(os.environ.get("ARTIFACTS_DIR", BUILD_DIR / "artifacts"))
RESULTS_STAGING_DIR: Final[Path] = Path(os.environ.get("ARTIFACTS_STAGING", BUILD_DIR / "submission-staging"))

MLPERF_SCRATCH_PATH: Final[Path] = Path(os.environ.get("MLPERF_SCRATCH_PATH", "/home/mlperf_inference_storage"))
