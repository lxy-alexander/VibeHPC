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
Utilities for multiprocess or multithreaded worker management and state.
"""

import atexit
import logging
import multiprocessing as mp
import queue
import signal
import time
import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
import threading

from tokenizers import Tokenizer as AutoTokenizer

# Share tokenizers across worker threads in threading mode
_tokenizer_cache = {}


def get_cached_tokenizer(model_name: str, model_revision: str) -> Any:
    """Get or create cached tokenizer instance (to share among async tasks in event-loop)"""
    tokenizer_key = (model_name, model_revision)
    if tokenizer_key not in _tokenizer_cache:
        _tokenizer_cache[tokenizer_key] = AutoTokenizer.from_pretrained(
            model_name, revision=model_revision
        )
    return _tokenizer_cache[tokenizer_key]


class WorkerProcessManager:
    """
    Manager for worker processes.
    This class provides state-tracking and APIs for worker processes for multiprocess mode.

    It is responsible for:
    - Track active worker processes
    - Start worker processes and wait for them to be ready
    - Shutdown worker processes gracefully (on signal or exit)
    """

    # Singletone pattern
    # tracking of all WorkerProcessManager instances
    _signal_handlers_registered = False
    _all_managers = []
    _managers_lock = threading.Lock()
    _original_handlers = {}
    _cleanup_in_progress = False

    def __init__(self):
        """Initialize the manager with process tracking"""
        self._active_worker_processes: List[mp.Process] = []
        self._pre_cleanup_callbacks = []

        # Register this instance with the class-level tracking
        with WorkerProcessManager._managers_lock:
            WorkerProcessManager._all_managers.append(self)

        # Register global cleanup handlers (only once)
        self._register_global_cleanup_handlers()

    def _register_global_cleanup_handlers(self):
        """Register cleanup handlers once globally for all WorkerProcessManager instances."""
        with WorkerProcessManager._managers_lock:
            if WorkerProcessManager._signal_handlers_registered:
                return

            # Register atexit handler globally
            atexit.register(WorkerProcessManager._global_cleanup_all)

            # Register signal handlers globally and save true originals
            for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGTSTP]:
                WorkerProcessManager._original_handlers[sig] = signal.signal(
                    sig, WorkerProcessManager._global_signal_handler
                )

            WorkerProcessManager._signal_handlers_registered = True
            logging.debug("Global cleanup handlers registered for worker process management")

    @classmethod
    def _global_signal_handler(cls, signum, frame):
        """Global signal handler that cleans up all WorkerProcessManager instances."""
        if cls._cleanup_in_progress:
            return  # Prevent re-entrance

        cls._cleanup_in_progress = True
        logging.info(f"Received signal {signum}. Cleaning up all worker processes...")

        # Clean up all manager instances
        cls._global_cleanup_all()

        # Restore original handler and re-raise signal
        if signum in cls._original_handlers:
            signal.signal(signum, cls._original_handlers[signum])
            os.kill(os.getpid(), signum)

    @classmethod
    def _global_cleanup_all(cls):
        """Clean up all WorkerProcessManager instances."""
        with cls._managers_lock:
            # Create a copy to iterate over (cleanup might modify the list)
            managers_to_cleanup = cls._all_managers.copy()

        for manager in managers_to_cleanup:
            try:
                manager._cleanup_worker_processes()
            except Exception as e:
                logging.error(f"Error during manager cleanup: {e}")

    def add_pre_cleanup_callback(self, callback: Callable[[], None]):
        """Add a callback to be executed before worker cleanup."""
        self._pre_cleanup_callbacks.append(callback)

    def _cleanup_worker_processes(self):
        """Clean up all active worker processes."""
        # Execute pre-cleanup callbacks first (e.g., for ZMQ shutdown signals)
        for callback in self._pre_cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logging.error(f"Pre-cleanup callback failed: {e}")

        if self._active_worker_processes:
            logging.debug(f"Cleaning up {len(self._active_worker_processes)} worker processes")
            self._terminate_processes(self._active_worker_processes)
            self._active_worker_processes.clear()

    def cleanup(self):
        """Explicitly cleanup all managed worker processes."""
        self._cleanup_worker_processes()

    def get_active_worker_count(self) -> int:
        """Get the current number of active worker processes"""
        return len(self._active_worker_processes)

    @staticmethod
    def _terminate_processes(processes: List[mp.Process], timeout: float = 0.1) -> None:
        """Terminate a list of processes. """
        for process in processes:
            if process.is_alive():
                process.terminate()
                try:
                    process.join(timeout=timeout)
                except TimeoutError:
                    process.kill()
                    process.join()

    @staticmethod
    def _worker_wrapper(worker_main_func, init_args, worker_id, readiness_queue, log_dir):
        """Wrapper function that sets up logging before calling the actual worker function."""
        # Set up logging if log_dir is provided
        if log_dir is not None:
            setup_logging_for_worker(worker_id=worker_id, log_dir=log_dir)

        # Call the actual worker function
        worker_main_func(*init_args, worker_id, readiness_queue)

    def start_worker_processes(
        self,
        worker_count: int,
        worker_main_func: Callable,
        init_args: List[Any],
        process_name_prefix: str = "Worker",
        log_dir: Optional[str] = None
    ) -> List[mp.Process]:
        """
        Start worker processes and wait for them to be ready.

        Args:
            worker_count: Number of worker processes to start
            worker_main_func: The main function for workers to run
            init_args: Arguments to pass to worker_main_func
            process_name_prefix: Prefix for process names
            log_dir: Optional directory for worker logs. If provided, logging will be automatically configured.

        Returns:
            List of worker processes.
        """
        readiness_queue = mp.Queue()
        worker_processes = []

        for i in range(worker_count):
            process = mp.Process(
                target=WorkerProcessManager._worker_wrapper,
                args=(worker_main_func, init_args, i, readiness_queue, log_dir),
                name=f"{process_name_prefix}-{i}",
                daemon=False
            )
            process.start()
            worker_processes.append(process)
            self._active_worker_processes.append(process)

        # Wait for all workers to signal readiness
        ready_workers = 0
        start_time = time.time()

        try:
            while ready_workers < worker_count:
                message = readiness_queue.get(timeout=30.0)
                if message is True:
                    ready_workers += 1
                else:
                    # Worker reported an error
                    raise RuntimeError(f"Worker initialization failed: {message}")
        except (queue.Empty, RuntimeError) as e:
            # On failure, terminate all started processes
            logging.error(f"Worker startup failed: {e}. Terminating workers.")

            # Clean up the readiness queue before handling the error
            readiness_queue.close()
            readiness_queue.join_thread()

            self._terminate_processes(worker_processes)
            # Remove failed processes from active list
            for p in worker_processes:
                if p in self._active_worker_processes:
                    self._active_worker_processes.remove(p)

            # Re-raise the exception
            if isinstance(e, queue.Empty):
                raise TimeoutError("Worker initialization timed out after 30s") from e
            raise e

        elapsed = time.time() - start_time
        logging.debug(f"All {worker_count} worker processes ready in {elapsed:.2f}s")

        # Clean up the readiness queue to prevent semaphore leaks
        readiness_queue.close()
        readiness_queue.join_thread()

        return worker_processes

    def shutdown_workers(
        self,
        worker_processes: List[mp.Process],
        request_queues: List[mp.Queue]
    ) -> None:
        """Shutdown worker processes gracefully"""
        # Signal all workers to stop
        for queue in request_queues:
            queue.put(None)

        # Terminate all processes
        self._terminate_processes(worker_processes)

        # Remove from active list
        for p in worker_processes:
            if p in self._active_worker_processes:
                self._active_worker_processes.remove(p)

    def __del__(self):
        """Remove this instance from the global tracking when garbage collected."""
        with WorkerProcessManager._managers_lock:
            if self in WorkerProcessManager._all_managers:
                WorkerProcessManager._all_managers.remove(self)


def setup_logging_for_worker(worker_id: int, log_dir: str):
    """Common logging setup for worker processes to write to a dedicated file."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Silence verbose loggers
    logging.getLogger('openai').setLevel(logging.CRITICAL)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)

    # Assert that log_dir is valid
    assert log_dir is not None and Path(log_dir).exists(), f"valid log_dir must be provided, given: {log_dir}"

    # Create log file based on PID
    pid = os.getpid()
    log_file = os.path.join(log_dir, f"worker_{worker_id}_pid_{pid}.log")

    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [Worker-%(worker_id)s/PID-%(process)d] - %(message)s',
        defaults={'worker_id': worker_id}
    )
    file_handler.setFormatter(formatter)

    # Remove existing handlers and add the file handler
    logger.handlers.clear()
    logger.addHandler(file_handler)

    # Also add console handler for critical errors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Worker {worker_id} (PID: {pid}) logging initialized")
