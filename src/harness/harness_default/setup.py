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

import os
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

from code.common.paths import MLCOMMONS_INF_REPO

LOADGEN_INCLUDE_DIR = str(MLCOMMONS_INF_REPO / "loadgen")
LWIS_INCLUDE_DIR = "/work/code/harness/lwis/include"
cxx_flags = os.environ.get("CXX_FLAGS", "").split()


class LWISBuildExt(build_ext):
    def run(self):
        self.build_lib = "/work/build/harness/lib"
        os.makedirs(self.build_lib, exist_ok=True)
        super().run()


lwis_module = Pybind11Extension(
    "lwis_api",
    sources=[
        "lwis_pybind.cc",
        "../common/logger.cpp"
    ],
    include_dirs=[
        pybind11.get_include(),
        LOADGEN_INCLUDE_DIR,
        LWIS_INCLUDE_DIR,
        "/work/code/harness/common",
        "/usr/include",
        "/usr/local/cuda/include",
    ],
    library_dirs=[
        str(MLCOMMONS_INF_REPO / "loadgen/build"),
        "/work/build/harness/lwis",
        "/usr/local/cuda/lib64",
        "/usr/local/lib",
    ],
    libraries=[
        "lwis",
        "mlperf_loadgen",
        "nvinfer",
        "nvinfer_plugin",
        "rt",
        "dl",
        "pthread",
        "cudart_static",
        "glog",
        "numa",
    ],
    extra_compile_args=cxx_flags,
)

setup(
    name="harness_default",
    cmdclass={"build_ext": LWISBuildExt},
    ext_modules=[lwis_module],
)
