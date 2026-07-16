"""IPC client for controller/go2_control."""

from __future__ import annotations

import socket
import struct
import time
from typing import Optional

import numpy as np

POLICY_SOF = 0xA5
POLICY_FLAG_STAND_UP = 0x01
POLICY_FLAG_RECOVERY = 0x02
POLICY_PACKET = struct.Struct('<BBd12f')


class PolicyClient:

    def __init__(self, socket_path: str = '/tmp/go2_policy.sock'):
        self.socket_path = socket_path
        self._sock: Optional[socket.socket] = None

    def connect(self,
                timeout: float = 120.0,
                retry_interval: float = 0.5) -> None:
        if self._sock is not None:
            return

        deadline = time.time() + timeout
        last_log = 0.0
        while time.time() < deadline:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(self.socket_path)
                self._sock = sock
                return
            except ConnectionRefusedError:
                sock.close()
                now = time.time()
                if now - last_log >= 5.0:
                    print(f'[train] waiting for controller at {self.socket_path}...',
                          flush=True)
                    last_log = now
                time.sleep(retry_interval)

        raise TimeoutError(
            f'Could not connect to controller at {self.socket_path} within '
            f'{timeout:.0f}s. Start go2_control first.')

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _send(self,
              q_target: np.ndarray,
              flags: int,
              timestamp: Optional[float] = None) -> None:
        ts = time.time() if timestamp is None else timestamp
        payload = POLICY_PACKET.pack(
            POLICY_SOF,
            flags,
            ts,
            *q_target.astype(np.float32),
        )
        for attempt in range(2):
            if self._sock is None:
                self.connect()
            assert self._sock is not None
            try:
                self._sock.sendall(payload)
                return
            except (BrokenPipeError, ConnectionResetError, OSError):
                self.close()
                if attempt == 0:
                    continue
                raise

    def send_target(self,
                    q_target: np.ndarray,
                    timestamp: Optional[float] = None) -> None:
        self._send(q_target, 0, timestamp)

    def send_standup(self,
                     *,
                     with_recovery: bool = False,
                     q_target: np.ndarray | None = None,
                     timestamp: Optional[float] = None) -> None:
        """Request controller standup_fsm. with_recovery runs recovery→standup."""
        if q_target is None:
            q_target = np.zeros(12, dtype=np.float32)
        flags = POLICY_FLAG_RECOVERY if with_recovery else POLICY_FLAG_STAND_UP
        self._send(q_target, flags, timestamp)
