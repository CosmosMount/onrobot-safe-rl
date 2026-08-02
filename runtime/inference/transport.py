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

_QUEUE_MAGIC = 0x47525153
_QUEUE_VERSION = 1
_QUEUE_HEADER = struct.Struct("<IIQQQQ")
_QUEUE_SLOT_HEADER = struct.Struct("<QI")
_QUEUE_WRITE_OFFSET = struct.calcsize("<IIQQ")
_QUEUE_READ_OFFSET = struct.calcsize("<IIQQQ")
_OWNED_SHM_NAMES: set[str] = set()


def _open_existing(name: str) -> shared_memory.SharedMemory:
    """Attach without claiming ownership of another process's mailbox."""
    shm = shared_memory.SharedMemory(name, create=False)
    # Python <=3.12 registers every attachment for process-exit unlinking.
    # Only the process that created the mailbox may own that lifecycle.
    # The tracker is process-global.  Do not unregister a second attachment
    # when this same process also created (and therefore owns) the segment.
    if shm._name not in _OWNED_SHM_NAMES:
        resource_tracker.unregister(shm._name, "shared_memory")
    return shm


def _name(key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]
    return f"go2_rl_{digest}"


def _queue_name(key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]
    return f"go2_rlq_{digest}"


class SharedMemoryQueueFull(BufferError):
    """Raised instead of silently overwriting an unread transition."""


class SharedMemoryRingQueue:
    """Bounded single-producer/single-consumer ordered shared-memory queue.

    Unlike :class:`SharedMemoryMailbox`, every item remains readable until the
    consumer advances the read cursor.  A full queue raises explicitly; it
    never drops or overwrites an unread terminal transition.

    The queue is intentionally SPSC.  The runtime is the sole producer and a
    collector is the sole consumer.  Producer and consumer publish separate,
    naturally aligned uint64 cursors after their slot data is complete.
    """

    def __init__(self, key: str, *, capacity: int = 2048,
                 slot_size: int = 16 * 1024):
        if capacity <= 0 or slot_size <= 0:
            raise ValueError("capacity and slot_size must be positive")
        self.key = key
        self.name = _queue_name(key)
        self.capacity = int(capacity)
        self.slot_size = int(slot_size)
        self.size = (
            _QUEUE_HEADER.size
            + self.capacity * (_QUEUE_SLOT_HEADER.size + self.slot_size)
        )
        self._shm: shared_memory.SharedMemory | None = None
        self._owner = False

    @property
    def socket_path(self) -> str:
        return self.key

    @property
    def owner(self) -> bool:
        return self._owner

    @classmethod
    def unlink_existing(cls, key: str) -> bool:
        """Unlink a stale queue after its producer has been stopped."""
        name = _queue_name(key)
        try:
            shm = shared_memory.SharedMemory(name, create=False)
        except FileNotFoundError:
            return False
        try:
            shm.unlink()
        finally:
            shm.close()
        return True

    def recv(self, *, timeout: float | None = None) -> Any:
        """Wait for and consume the next ordered item."""
        self.wait_ready(timeout=120.0 if timeout is None else timeout)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            message = self.read()
            if message is not None:
                return message
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for shared memory queue: {self.name}")
            time.sleep(0.001)

    def create(self) -> None:
        if self._shm is not None:
            return
        try:
            self._shm = shared_memory.SharedMemory(
                self.name, create=True, size=self.size)
            self._owner = True
            _OWNED_SHM_NAMES.add(self._shm._name)
            _QUEUE_HEADER.pack_into(
                self._shm.buf, 0, _QUEUE_MAGIC, _QUEUE_VERSION,
                self.capacity, self.slot_size, 0, 0)
            for index in range(self.capacity):
                _QUEUE_SLOT_HEADER.pack_into(
                    self._shm.buf, self._slot_offset(index), 0, 0)
        except FileExistsError:
            self._shm = _open_existing(self.name)
            self._validate_header()

    def open(self) -> None:
        if self._shm is not None:
            return
        self._shm = _open_existing(self.name)
        self._validate_header()

    def wait_ready(self, *, timeout: float = 120.0,
                   retry_interval: float = 0.05) -> None:
        deadline = time.time() + timeout
        while True:
            try:
                self.open()
                return
            except FileNotFoundError:
                if time.time() >= deadline:
                    raise TimeoutError(
                        f"Shared memory queue not found: {self.name}")
                time.sleep(retry_interval)

    def _validate_header(self) -> tuple[int, int]:
        assert self._shm is not None
        magic, version, capacity, slot_size, write_seq, read_seq = (
            _QUEUE_HEADER.unpack_from(self._shm.buf, 0))
        if magic != _QUEUE_MAGIC or version != _QUEUE_VERSION:
            raise RuntimeError(f"Invalid shared memory queue header: {self.name}")
        if capacity != self.capacity or slot_size != self.slot_size:
            raise ValueError(
                "Shared memory queue geometry mismatch: "
                f"existing=({capacity}, {slot_size}) "
                f"requested=({self.capacity}, {self.slot_size})")
        if write_seq < read_seq or write_seq - read_seq > capacity:
            raise RuntimeError(
                f"Corrupt shared memory queue cursors: {write_seq}, {read_seq}")
        return int(write_seq), int(read_seq)

    def _slot_offset(self, index: int) -> int:
        return (
            _QUEUE_HEADER.size
            + index * (_QUEUE_SLOT_HEADER.size + self.slot_size)
        )

    def write(self, message: Any) -> None:
        if self._shm is None:
            self.create()
        assert self._shm is not None
        payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload) > self.slot_size:
            raise ValueError(
                f"Shared memory queue item too large: {len(payload)} bytes")
        write_seq, read_seq = self._validate_header()
        if write_seq - read_seq >= self.capacity:
            raise SharedMemoryQueueFull(
                f"Shared memory queue is full: {self.name}")
        sequence = write_seq + 1
        offset = self._slot_offset(write_seq % self.capacity)
        _QUEUE_SLOT_HEADER.pack_into(self._shm.buf, offset, 0, 0)
        payload_offset = offset + _QUEUE_SLOT_HEADER.size
        self._shm.buf[payload_offset:payload_offset + len(payload)] = payload
        _QUEUE_SLOT_HEADER.pack_into(
            self._shm.buf, offset, sequence, len(payload))
        struct.pack_into("<Q", self._shm.buf, _QUEUE_WRITE_OFFSET, sequence)

    def read(self) -> Any | None:
        if self._shm is None:
            self.open()
        assert self._shm is not None
        write_seq, read_seq = self._validate_header()
        if read_seq >= write_seq:
            return None
        expected = read_seq + 1
        offset = self._slot_offset(read_seq % self.capacity)
        for _ in range(3):
            sequence1, length1 = _QUEUE_SLOT_HEADER.unpack_from(
                self._shm.buf, offset)
            if sequence1 != expected or length1 > self.slot_size:
                time.sleep(0.001)
                continue
            payload_offset = offset + _QUEUE_SLOT_HEADER.size
            payload = bytes(
                self._shm.buf[payload_offset:payload_offset + length1])
            sequence2, length2 = _QUEUE_SLOT_HEADER.unpack_from(
                self._shm.buf, offset)
            if sequence1 == sequence2 and length1 == length2:
                struct.pack_into(
                    "<Q", self._shm.buf, _QUEUE_READ_OFFSET, expected)
                return pickle.loads(payload)
        raise RuntimeError(
            f"Could not read stable shared memory queue slot: {self.name}")

    def depth(self) -> int:
        if self._shm is None:
            self.open()
        write_seq, read_seq = self._validate_header()
        return write_seq - read_seq

    def clear(self) -> None:
        if self._shm is None:
            self.create()
        assert self._shm is not None
        struct.pack_into("<Q", self._shm.buf, _QUEUE_WRITE_OFFSET, 0)
        struct.pack_into("<Q", self._shm.buf, _QUEUE_READ_OFFSET, 0)
        for index in range(self.capacity):
            _QUEUE_SLOT_HEADER.pack_into(
                self._shm.buf, self._slot_offset(index), 0, 0)

    def close(self, *, unlink: bool = False) -> None:
        if self._shm is None:
            return
        self._shm.close()
        if unlink:
            if not self._owner:
                raise RuntimeError("Only the queue owner may unlink it")
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
            _OWNED_SHM_NAMES.discard(self._shm._name)
        self._shm = None


class SharedMemoryMailbox:
    def __init__(self, key: str, *, size: int = _DEFAULT_SIZE):
        self.key = key
        self.name = _name(key)
        self.size = size
        self._shm: shared_memory.SharedMemory | None = None
        self._seq = 0
        self._last_read_seq = 0
        self._owner = False

    @property
    def socket_path(self) -> str:
        return self.key

    def create(self) -> None:
        if self._shm is not None:
            return
        try:
            self._shm = shared_memory.SharedMemory(self.name, create=True, size=self.size)
            self._owner = True
            _OWNED_SHM_NAMES.add(self._shm._name)
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

    @property
    def owner(self) -> bool:
        return self._owner

    def close(self, *, unlink: bool = False) -> None:
        if self._shm is not None:
            self._shm.close()
            if unlink:
                if not self._owner:
                    raise RuntimeError("Only the mailbox owner may unlink it")
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass
                _OWNED_SHM_NAMES.discard(self._shm._name)
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

    def close(self, *, unlink: bool = False) -> None:
        self._mailbox.close(unlink=unlink)


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

    def close(self, *, unlink: bool = False) -> None:
        self._mailbox.close(unlink=unlink)


DatagramReceiver = SharedMemoryReceiver
DatagramSender = SharedMemorySender
