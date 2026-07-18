"""DDS state reader and connection presets (SI units in RobotState)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from train.types import NUM_JOINTS, RobotState
from train.velocity import quat_world_to_body


@dataclass(frozen=True)
class DdsConfig:
    domain_id: int
    interface: str


MUJOCO = DdsConfig(domain_id=1, interface='lo')
REAL = DdsConfig(domain_id=0, interface='eth0')


class StateReader:
    """Thread-safe cache of the latest robot state from DDS."""

    def __init__(self, *, sport_velocity_world_frame: bool = True):
        self.sport_velocity_world_frame = sport_velocity_world_frame
        self._lock = threading.Lock()
        self._state = RobotState()
        self._initialized = False
        self._low_state_sub = None
        self._sport_state_sub = None
        self._prev_sport_pos: np.ndarray | None = None
        self._prev_sport_time = 0.0
        self.low_message_count = 0
        self.sport_message_count = 0

    def init_dds(self, domain_id: int, interface: str) -> None:
        if self._initialized:
            return

        from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                                 ChannelSubscriber)
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import (LowState_,
                                                            SportModeState_)

        ChannelFactoryInitialize(domain_id, interface)

        self._low_state_sub = ChannelSubscriber('rt/lowstate', LowState_)
        self._low_state_sub.Init(self._on_low_state, 10)

        self._sport_state_sub = ChannelSubscriber('rt/sportmodestate',
                                                  SportModeState_)
        self._sport_state_sub.Init(self._on_sport_state, 10)

        self._initialized = True

    def _on_low_state(self, msg) -> None:
        with self._lock:
            now = time.time()
            self.low_message_count += 1
            for i in range(NUM_JOINTS):
                self._state.joint_q[i] = msg.motor_state[i].q
                self._state.joint_dq[i] = msg.motor_state[i].dq
            self._state.imu_quat[:] = np.asarray(msg.imu_state.quaternion[:4],
                                                 dtype=np.float32)
            self._state.imu_gyro[:] = np.asarray(msg.imu_state.gyroscope[:3],
                                                  dtype=np.float32)
            self._state.imu_accel[:] = np.asarray(
                msg.imu_state.accelerometer[:3], dtype=np.float32)
            self._state.timestamp = now
            self._state.low_state_timestamp = now
            self._state.low_state_count = self.low_message_count

    def _on_sport_state(self, msg) -> None:
        with self._lock:
            now = time.time()
            self.sport_message_count += 1
            pos = np.asarray(msg.position[:3], dtype=np.float32)
            raw_vel = np.asarray(msg.velocity[:3], dtype=np.float32)
            vel = raw_vel.copy()
            used_pos_diff = False

            if self._prev_sport_pos is not None:
                dt = now - self._prev_sport_time
                if dt > 1e-3 and float(np.linalg.norm(vel)) < 1e-5:
                    vel = (pos - self._prev_sport_pos) / dt
                    used_pos_diff = True

            if self.sport_velocity_world_frame:
                vel_body = quat_world_to_body(vel, self._state.imu_quat)
            else:
                vel_body = vel

            self._state.world_position[:] = pos
            self._state.body_velocity[:] = vel_body
            self._state.sport_state_timestamp = now
            self._state.sport_state_count = self.sport_message_count
            self._prev_sport_pos = pos.copy()
            self._prev_sport_time = now

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
            )

    def low_state_age(self, now: float | None = None) -> float:
        now = time.time() if now is None else now
        with self._lock:
            if self._state.low_state_timestamp <= 0:
                return float('inf')
            return now - self._state.low_state_timestamp

    def sport_state_age(self, now: float | None = None) -> float:
        now = time.time() if now is None else now
        with self._lock:
            if self._state.sport_state_timestamp <= 0:
                return float('inf')
            return now - self._state.sport_state_timestamp

    def require_fresh_sport_state(self, max_age_s: float) -> None:
        age = self.sport_state_age()
        if not np.isfinite(age) or age > max_age_s:
            raise RuntimeError(
                'SportModeState velocity is missing or stale: '
                f'age={age:.3f}s max_age={max_age_s:.3f}s. '
                'Refusing to train with implicit zero/stale body velocity.')

    def wait_for_state(self, timeout: float = 5.0) -> RobotState:
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = self.get_state()
            if state.timestamp > 0:
                return state
            time.sleep(0.01)
        raise TimeoutError('Timed out waiting for LowState')
