# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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


import contextlib
import multiprocessing as mp
import time
import torch


class SharedBuffer:
    """A Read-Only-Once shared memory buffer. Essentially a faster version of the special case of mp.Queue(1).
    """

    def __init__(self,
                 size: int,
                 typecode: str = 'i',
                 access_timeout: float = 0.01):
        self.buff = mp.Array(typecode, size)
        self.n_values = mp.Value('i', 0)

        self.data_update_event = mp.Event()
        self.data_stale_event = mp.Event()
        self.data_stale_event.set()  # Buffer starts out stale

        self.compute_lock = mp.Lock()
        self.stop_event = mp.Event()

        self._cv = mp.Condition(self.compute_lock)

    @property
    def stopped(self):
        return self.stop_event.is_set()

    def write(self, L):
        with self.write_context(len(L)):
            self.buff[:len(L)] = L

    def stop(self):
        if self.stopped:
            return

        with self._cv:
            self.stop_event.set()
            self.data_stale_event.set()
            self.data_update_event.set()

            self._cv.notify_all()

    def is_writable(self):
        return self.stop_event.is_set() or \
                (self.data_stale_event.is_set() and (not self.data_update_event.is_set()))

    def is_readable(self):
        return self.stop_event.is_set() or \
                (self.data_update_event.is_set() and (not self.data_stale_event.is_set()))

    @contextlib.contextmanager
    def write_context(self, data_len: int):
        if self.stopped:
            yield None
        else:
            with self._cv:
                if not self.is_writable():
                    self._cv.wait_for(self.is_writable)
                if self.stopped:
                    yield None
                else:
                    try:
                        yield self.buff
                    finally:
                        self.n_values.value = data_len

                        self.data_stale_event.clear()
                        self.data_update_event.set()
                        self._cv.notify_all()

    @contextlib.contextmanager
    def read_context(self):
        if self.stopped:
            yield None
        else:
            with self._cv:
                if not self.is_readable():
                    self._cv.wait_for(self.is_readable)
                if self.stopped:
                    yield None
                else:
                    try:
                        yield self.buff[:self.n_values.value]
                    finally:
                        self.data_update_event.clear()
                        self.data_stale_event.set()
                        self._cv.notify_all()


class SharedQueue:
    def __init__(self, max_size: int):
        self.max_size = max_size

        self.buff = mp.Array('Q', max_size)
        self.pop_idx = mp.Value('Q', 0)
        self.append_idx = mp.Value('Q', 0)

        self.is_full = mp.Event()
        self.has_values = mp.Event()
        self._stopped = mp.Event()

        self._cv = mp.Condition()

    def stop(self):
        self._stopped.set()

        with self._cv:
            self._cv.notify_all()

    def _has_space(self) -> bool:
        return self._stopped.is_set() or not self.is_full.is_set()

    def _has_values(self) -> bool:
        return self._stopped.is_set() or self.has_values.is_set()

    def append(self, val):
        with self._cv:
            if self.is_full.is_set():
                self._cv.wait_for(self._has_space)

                if self._stopped.is_set():
                    return

            curr = self.append_idx.value
            self.buff[curr] = val

            _i = (curr + 1) % self.max_size
            self.append_idx.value = _i

            if _i == self.pop_idx.value:
                self.is_full.set()
            else:
                self.is_full.clear()
            self.has_values.set()
            self._cv.notify_all()

    def pop(self):
        with self._cv:
            if not self.has_values.is_set():
                self._cv.wait_for(self._has_values)

                if self._stopped.is_set():
                    return

            curr = self.pop_idx.value
            val = self.buff[curr]

            _i = (curr + 1) % self.max_size
            self.pop_idx.value = _i

            if _i == self.append_idx.value:
                self.has_values.clear()
            self.is_full.clear()
            self._cv.notify_all()

            return val


class SharedBufferCollection:
    """Manager wrapper around a collection of SharedBuffers.
    """
    def __init__(self, n_buffers, *args, **kwargs):
        self.buffers = dict()
        self.writable = SharedQueue(n_buffers)
        self.readable = SharedQueue(n_buffers)

        for _ in range(n_buffers):
            buff = SharedBuffer(*args, **kwargs)
            self.buffers[id(buff)] = buff
            self.writable.append(id(buff))

        self.stop_event = mp.Event()

    @property
    def stopped(self):
        return self.stop_event.is_set()

    def stop(self):
        if self.stopped:
            return

        self.stop_event.set()
        self.writable.stop()
        self.readable.stop()
        for buf in self.buffers.values():
            buf.stop()

    @contextlib.contextmanager
    def write_context(self, data_len: int):
        if self.stopped:
            yield None
        else:
            buff_ref = self.writable.pop()
            if self.stop_event.is_set():
                yield None
            else:
                buff = self.buffers[buff_ref]

                try:
                    with buff.write_context(data_len) as _ptr:
                        yield _ptr
                finally:
                    self.readable.append(buff_ref)

    @contextlib.contextmanager
    def read_context(self):
        if self.stopped:
            yield None
        else:
            buff_ref = self.readable.pop()
            if self.stop_event.is_set():
                yield None
            else:
                buff = self.buffers[buff_ref]

                try:
                    with buff.read_context() as dat:
                        yield dat
                finally:
                    self.writable.append(buff_ref)
