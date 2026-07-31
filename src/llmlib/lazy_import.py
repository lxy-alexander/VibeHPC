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

"""Lightweight lazy import utility.

This module is kept separate from utils.py to avoid importing heavy
dependencies (matplotlib, tqdm, etc.) when only LazyImport is needed.
"""


class LazyImport:
    """
    Lazy import wrapper that only imports when accessed (via __call__()) or when _load() is called.
    """

    def __init__(self, module_name: str, attribute_name: str = None):
        self.module_name = module_name
        self.attribute_name = attribute_name
        self._module = None
        self._attribute = None

    def _load(self):
        if self._module is None:
            try:
                import importlib
                self._module = importlib.import_module(self.module_name)
                if self.attribute_name:
                    self._attribute = getattr(self._module, self.attribute_name)
            except ImportError as e:
                raise ImportError(
                    f"Failed to import {self.module_name}"
                    f"{f'.{self.attribute_name}' if self.attribute_name else ''}: {e}\n"
                ) from e
        return self._attribute if self.attribute_name else self._module

    def __getattr__(self, name):
        obj = self._load()
        return getattr(obj, name)

    def __call__(self, *args, **kwargs):
        obj = self._load()
        return obj(*args, **kwargs)
