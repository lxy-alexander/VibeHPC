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

import ctypes
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List

from ..common.constants import Benchmark
from ..common import paths
from ..common.systems.system_list import DETECTED_SYSTEM


@dataclass
class LoadablePlugin:
    """
    A dataclass that represents a loadable TensorRT plugin with associated constraints.

    This class manages the loading of TensorRT plugins, including checking if they can be loaded
    based on system constraints and actually loading them when appropriate.

    Attributes:
        path (str): Path to the TensorRT plugin library file
        constraints (List[Callable[[], bool]]): List of constraint functions that determine
            whether the plugin can be loaded. Each function takes a dictionary of arguments
            and returns a boolean indicating if the constraint is satisfied.
    """

    path: str
    """str: Path to the TRT plugin library"""

    constraints: List[Callable[[], bool]] = field(default_factory=list)
    """List[Callable[[], bool]: list of constraints that describes whether the plugin can be loaded """

    def get_full_path(self) -> str:
        """
        Get the full path to the plugin library by combining the build directory and plugin path.

        Returns:
            str: The complete path to the plugin library file
        """
        return paths.BUILD_DIR / "plugins" / self.path

    def load(self, args: dict) -> None:
        """
        Load the TensorRT plugin if all constraints are satisfied.

        Args:
            args (dict): Dictionary of arguments used to evaluate constraints

        Note:
            Prints a message when loading the plugin and uses ctypes to load the shared library
        """
        if self.can_load(args):
            print(f"Loading TensorRT plugin from {self.get_full_path()}")
            ctypes.CDLL(str(self.get_full_path()))

    def can_load(self, args: dict) -> bool:
        """
        Check if the plugin can be loaded by evaluating all constraints.

        Args:
            args (dict): Dictionary of arguments used to evaluate constraints

        Returns:
            bool: True if all constraints are satisfied, False otherwise
        """
        for constraint in self.constraints:
            if not constraint(args):
                return False
        return True


class LoadablePlugins(Enum):
    """
    Enumeration of all available TensorRT plugins that can be loaded.

    Each enum member represents a specific TensorRT plugin with its associated path and constraints.
    The plugins are system-specific and may have different implementations or constraints based on
    the detected system architecture (e.g., Hopper, Ada).

    Attributes:
        DLRMv2EmbeddingLookupPlugin: Plugin for DLRM v2 embedding lookup operations
        NMSOptPlugin: Plugin for optimized Non-Maximum Suppression
        RetinaNetConcatOutputPlugin: Plugin for concatenating RetinaNet outputs
    """

    DLRMv2EmbeddingLookupPlugin = LoadablePlugin("DLRMv2EmbeddingLookupPlugin/libdlrmv2embeddinglookupplugin.so")
    NMSOptPlugin = LoadablePlugin("NMSOptPlugin/libnmsoptplugin.so")
    RetinaNetConcatOutputPlugin = LoadablePlugin("retinanetConcatPlugin/libretinanetconcatplugin.so")


base_plugin_map = {
    Benchmark.DLRMv2: [LoadablePlugins.DLRMv2EmbeddingLookupPlugin],
    Benchmark.Retinanet: [LoadablePlugins.NMSOptPlugin, LoadablePlugins.RetinaNetConcatOutputPlugin],
}
