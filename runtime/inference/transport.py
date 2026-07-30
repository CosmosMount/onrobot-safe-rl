"""Shared-memory transport between runtime and train."""

from __future__ import annotations

import hashlib
import pickle
import struct
import time
from multiprocessing import resource_tracker, shared_memory
from typing import Any

_MAGIC = 0x47524C53
_HEADER = struct.Struct("<IQQI")
_DEFAULT_SIZE = 1 << 20


def _open_existing(name: str) -> shared_memory.SharedMemory:
    """Attach without claiming ownership of another process's mailbox."""
    shm = shared_memory.SharedMemory(name, create=False)
    # Python <=3.12 registers every attachment for process-exit unlinking.
    # Only the process that created the mailbox may own that lifecycle.
    resource_tracker.unregister(shm._name, "shared_memory")
    return shm


def _name(key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]
    return f"go2_rl_{digest}"


class SharedMemoryMailbox:
    def __init__(self, key: str, *, size: int = _DEFAULT_SIZE):
        self.key = key
        self.name = _name(key)
        self.size = size
        self._shm: shared_memory.SharedMemory | None = None
        self._seq = 0
        self._last_read_seq = 0

    @property
    def socket_path(self) -> str:
        return self.key

    def create(self) -> None:
        if self._shm is not None:
            return
        try:
            self._shm = shared_memory.SharedMemory(self.name, create=True, size=self.size)
            self.clear()
        except FileExistsError:
            self._shm = _open_existing(self.name)

    def open(self) -> None:
        if self._shm is None:
            self._shm = _open_existing(self.name)

    def wait_ready(self, *, timeout: float = 120.0, retry_interval: float = 0.05) -> None:
        deadline = time.time() + timeout
        while True:
            try:
                self.open()
                return
            except FileNotFoundError:
                if time.time() >= deadline:
                    raise TimeoutError(f"Shared memory mailbox not found: {self.name}")
                time.sleep(retry_interval)

    def clear(self) -> None:
        if self._shm is None:
            self.create()
        assert self._shm is not None
        self._seq += 2
        _HEADER.pack_into(self._shm.buf, 0, _MAGIC, self._seq, 0, 1)

    def write(self, message: Any) -> None:
        if self._shm is None:
            self.create()
        assert self._shm is not None
        if isinstance(message, dict) and message.get("clear"):
            self.clear()
            return
        payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload) > self.size - _HEADER.size:
            raise ValueError(f"Shared memory message too large: {len(payload)} bytes")
        self._seq += 1
        if self._seq % 2 == 0:
            self._seq += 1
        _HEADER.pack_into(self._shm.buf, 0, _MAGIC, self._seq, 0, 0)
        self._shm.buf[_HEADER.size:_HEADER.size + len(payload)] = payload
        self._seq += 1
        _HEADER.pack_into(self._shm.buf, 0, _MAGIC, self._seq, len(payload), 0)

    def read_latest(self, *, consume: bool) -> Any | None:
        if self._shm is None:
            self.open()
        assert self._shm is not None
        for _ in range(3):
            magic1, seq1, length, flags = _HEADER.unpack_from(self._shm.buf, 0)
            if magic1 != _MAGIC or flags & 1 or length == 0 or seq1 % 2:
                return None
            payload = bytes(self._shm.buf[_HEADER.size:_HEADER.size + length])
            magic2, seq2, length2, flags2 = _HEADER.unpack_from(self._shm.buf, 0)
            if magic2 == magic1 and seq2 == seq1 and length2 == length and flags2 == flags:
                if consume and seq1 == self._last_read_seq:
                    return None
                self._last_read_seq = seq1
                return pickle.loads(payload)
            time.sleep(0.001)
        return None

    def close(self) -> None:
        if self._shm is not None:
            self._shm.close()
            self._shm = None


class SharedMemoryReceiver:
    def __init__(self, key: str):
        self.socket_path = key
        self._mailbox = SharedMemoryMailbox(key)

    def bind(self) -> None:
        self._mailbox.create()

    def recv(self, *, timeout: float | None = None) -> Any:
        self.bind()
        deadline = None if timeout is None else time.time() + timeout
        while True:
            message = self._mailbox.read_latest(consume=True)
            if message is not None:
                return message
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError(f"Timed out waiting for shared memory mailbox: {self._mailbox.name}")
            time.sleep(0.002)

    def recv_latest(self) -> Any | None:
        self.bind()
        return self._mailbox.read_latest(consume=True)

    def clear(self) -> None:
        self._mailbox.clear()

    def close(self) -> None:
        self._mailbox.close()


class SharedMemorySender:
    def __init__(self, key: str):
        self.socket_path = key
        self._mailbox = SharedMemoryMailbox(key)

    def wait_ready(self, *, timeout: float = 120.0, retry_interval: float = 0.05) -> None:
        self._mailbox.wait_ready(timeout=timeout, retry_interval=retry_interval)

    def send(self, message: Any) -> bool:
        self._mailbox.write(message)
        return True

    def clear(self) -> None:
        self._mailbox.clear()

    def close(self) -> None:
        self._mailbox.close()


DatagramReceiver = SharedMemoryReceiver
DatagramSender = SharedMemorySender
