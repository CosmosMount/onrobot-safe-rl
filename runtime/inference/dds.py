"""Controller state stream reader."""

from __future__ import annotations

import os
import socket
import struct
import threading
import time
from dataclasses import dataclass

import numpy as np

from runtime.inference.state import NUM_JOINTS, RobotState
from runtime.inference.velocity import quat_world_to_body


@dataclass(frozen=True)
class DdsConfig:
    domain_id: int
    interface: str


MUJOCO = DdsConfig(domain_id=1, interface='lo')
REAL = DdsConfig(domain_id=0, interface='eth0')

STATE_SOF = 0x5A
STATE_PACKET = struct.Struct('<BBdII12f12f4f3f3f3f3f12f')


class StateReader:
    """Thread-safe cache of state packets forwarded by go2_control."""

    def __init__(
        self,
        *,
        socket_path: str = '/tmp/go2_policy.sock.state',
        sport_velocity_world_frame: bool = True,
    ):
        self.socket_path = socket_path
        self.sport_velocity_world_frame = sport_velocity_world_frame
        self._lock = threading.Lock()
        self._state = RobotState()
        self._initialized = False
        self._running = False
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None
        self.phase = 0
        self.low_message_count = 0
        self.sport_message_count = 0

    def connect(self) -> None:
        if self._initialized:
            return
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.bind(self.socket_path)
        self._sock = sock
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._initialized = True

    def _loop(self) -> None:
        assert self._sock is not None
        while self._running:
            try:
                data = self._sock.recv(STATE_PACKET.size)
            except OSError:
                break
            if len(data) != STATE_PACKET.size:
                continue
            unpacked = STATE_PACKET.unpack(data)
            if unpacked[0] != STATE_SOF:
                continue
            self._apply_packet(unpacked)

    def _apply_packet(self, packet: tuple) -> None:
        phase = int(packet[1])
        timestamp = float(packet[2])
        low_count = int(packet[3])
        sport_count = int(packet[4])
        i = 5
        joint_q = np.asarray(packet[i:i + NUM_JOINTS], dtype=np.float32)
        i += NUM_JOINTS
        joint_dq = np.asarray(packet[i:i + NUM_JOINTS], dtype=np.float32)
        i += NUM_JOINTS
        imu_quat = np.asarray(packet[i:i + 4], dtype=np.float32)
        i += 4
        imu_gyro = np.asarray(packet[i:i + 3], dtype=np.float32)
        i += 3
        imu_accel = np.asarray(packet[i:i + 3], dtype=np.float32)
        i += 3
        velocity = np.asarray(packet[i:i + 3], dtype=np.float32)
        i += 3
        world_position = np.asarray(packet[i:i + 3], dtype=np.float32)
        now = time.time()
        body_velocity = (
            quat_world_to_body(velocity, imu_quat)
            if self.sport_velocity_world_frame
            else velocity
        )
        with self._lock:
            self.phase = phase
            self.low_message_count = low_count
            self.sport_message_count = sport_count
            self._state = RobotState(
                joint_q=joint_q,
                joint_dq=joint_dq,
                imu_quat=imu_quat,
                imu_gyro=imu_gyro,
                imu_accel=imu_accel,
                body_velocity=body_velocity,
                world_position=world_position,
                timestamp=timestamp,
                low_state_timestamp=now,
                sport_state_timestamp=now if sport_count else 0.0,
                low_state_count=low_count,
                sport_state_count=sport_count,
                phase=phase,
            )

    def get_state(self) -> RobotState:
        with self._lock:
            return RobotState(
                joint_q=self._state.joint_q.copy(),
                joint_dq=self._state.joint_dq.copy(),
                imu_quat=self._state.imu_quat.copy(),
                imu_gyro=self._state.imu_gyro.copy(),
                imu_accel=self._state.imu_accel.copy(),
                body_velocity=self._state.body_velocity.copy(),
                world_position=self._state.world_position.copy(),
                timestamp=self._state.timestamp,
                low_state_timestamp=self._state.low_state_timestamp,
                sport_state_timestamp=self._state.sport_state_timestamp,
                low_state_count=self._state.low_state_count,
                sport_state_count=self._state.sport_state_count,
                phase=self._state.phase,
            )

    def low_state_age(self, now: float | None = None) -> float:
        now = time.time() if now is None else now
        with self._lock:
            return float('inf') if self._state.low_state_timestamp <= 0 else now - self._state.low_state_timestamp

    def sport_state_age(self, now: float | None = None) -> float:
        now = time.time() if now is None else now
        with self._lock:
            return float('inf') if self._state.sport_state_timestamp <= 0 else now - self._state.sport_state_timestamp

    def require_fresh_sport_state(self, max_age_s: float) -> None:
        age = self.sport_state_age()
        if not np.isfinite(age) or age > max_age_s:
            raise RuntimeError(
                'Controller state stream velocity is missing or stale: '
                f'age={age:.3f}s max_age={max_age_s:.3f}s.')

    def wait_for_state(self, timeout: float = 5.0) -> RobotState:
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = self.get_state()
            if state.low_state_timestamp > 0:
                return state
            time.sleep(0.01)
        raise TimeoutError(f'Timed out waiting for controller state stream at {self.socket_path}')

    def close(self) -> None:
        self._running = False
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass
