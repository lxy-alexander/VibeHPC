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
"""This module provides context managers for managing MLPerf LoadGen Query Sample Library (QSL) and System Under Test (SUT) resources.

The module contains two main classes:
- ScopedQSL: Manages the lifecycle of a QSL instance
- ScopedSUT: Manages the lifecycle of a SUT instance

These classes ensure proper resource cleanup using Python's context manager protocol.
"""

from typing import List

from code.common import logging

# pylint: disable=c-extension-no-member
import mlperf_loadgen as lg


class ScopedQSL:
    """Context manager for managing MLPerf LoadGen Query Sample Library (QSL) resources.

    This class handles the creation and destruction of QSL instances, ensuring proper resource
    cleanup. It implements the context manager protocol (__enter__ and __exit__) to allow
    usage with Python's 'with' statement.

    Attributes:
        total_sample_count (int): Total number of samples in the QSL
        performance_sample_count (int): Number of samples to use for performance testing
        _qsl: Internal reference to the QSL instance
    """

    def __init__(self,
                 total_sample_count: int,
                 performance_sample_count: int):
        """Initialize the ScopedQSL instance.

        Args:
            total_sample_count (int): Total number of samples in the QSL
            performance_sample_count (int): Number of samples to use for performance testing
        """
        self.total_sample_count = total_sample_count
        self.performance_sample_count = performance_sample_count

        self._qsl = None

    def load_samples_to_ram(self, samples: List[lg.QuerySample]):
        """Load samples into RAM.

        Args:
            samples (List[lg.QuerySample]): List of query samples to load into RAM
        """
        pass  # pylint: disable=unnecessary-pass

    def unload_samples_from_ram(self, samples: List[lg.QuerySample]):
        """Unload samples from RAM.

        Args:
            samples (List[lg.QuerySample]): List of query samples to unload from RAM
        """
        pass  # pylint: disable=unnecessary-pass

    def __enter__(self):
        """Create and return a QSL instance.

        Returns:
            The constructed QSL instance

        Raises:
            AssertionError: If a QSL instance already exists
        """
        assert self._qsl is None, "QSL state already created, was it not destroyed properly?"
        self._qsl = lg.ConstructQSL(self.total_sample_count,
                                    self.performance_sample_count,
                                    self.load_samples_to_ram,
                                    self.unload_samples_from_ram)
        logging.debug("Initialized QSL with %d samples (%d performance sample count)",
                      self.total_sample_count, self.performance_sample_count)
        return self._qsl

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up the QSL instance.

        Args:
            exc_type: Type of exception if any
            exc_val: Exception value if any
            exc_tb: Exception traceback if any

        Raises:
            Any exception that was caught during the context
        """
        lg.DestroyQSL(self._qsl)
        logging.debug("Destroyed QSL")
        if exc_val:
            raise exc_val


class ScopedSUT:
    """Context manager for managing MLPerf LoadGen System Under Test (SUT) resources.

    This class handles the creation and destruction of SUT instances, ensuring proper resource
    cleanup. It implements the context manager protocol (__enter__ and __exit__) to allow
    usage with Python's 'with' statement.

    Attributes:
        _sut: Internal reference to the SUT instance
    """

    def __init__(self):
        """Initialize the ScopedSUT instance."""
        self._sut = None

    def issue_queries(self, query_samples: List[lg.QuerySample]):
        """Issue queries to the SUT.

        Args:
            query_samples (List[lg.QuerySample]): List of query samples to process
        """
        pass  # pylint: disable=unnecessary-pass

    def flush_queries(self):
        """Flush pending queries from the SUT.
        """
        pass  # pylint: disable=unnecessary-pass

    def __enter__(self):
        """Create and return a SUT instance.

        Returns:
            The constructed SUT instance

        Raises:
            AssertionError: If a SUT instance already exists
        """
        assert self._sut is None, "SUT state already created, was it not destroyed properly?"
        self._sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self._sut

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up the SUT instance.

        Args:
            exc_type: Type of exception if any
            exc_val: Exception value if any
            exc_tb: Exception traceback if any

        Raises:
            Any exception that was caught during the context
        """
        lg.DestroySUT(self._sut)
        logging.debug("Destroyed SUT")
        if exc_val:
            raise exc_val
