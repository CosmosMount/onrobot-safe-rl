from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, TypeVar

import gymnasium as gym
import numpy as np

from rl.agents.base.agent import BaseAgent
from rl.utils.types import NDArray, Tensor

Config = TypeVar("Config")


class RobotSafetyState(Protocol):
    imu_quat: np.ndarray
    imu_accel: np.ndarray


class SafeRawMode(str, Enum):
    POLICY = "policy"
    RECOVERY = "recovery"


@dataclass
class SafeRawStatus:
    mode: SafeRawMode
    terminated: bool = False
    replay_enabled: bool = True
    restart_required: bool = False
    recovery_requested: bool = False
    inverted: bool = False
    fallen: bool = False
    roll: float = 0.0
    pitch: float = 0.0
    acc_z: float = 0.0
    reason: str = "policy"

    @property
    def policy_enabled(self) -> bool:
        return self.mode == SafeRawMode.POLICY


@dataclass
class SafeRawSupervisor:
    """Low-level safety gate for raw policy actions."""

    inverted_acc_z_threshold: float = -3.0
    fallen_roll_pitch_limit_rad: float = 0.523599
    trigger_steps: int = 5
    stable_steps: int = 10
    timeout_steps: int = 200

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.mode = SafeRawMode.POLICY
        self._inverted_count = 0
        self._stable_count = 0
        self._recovery_steps = 0
        self._recovery_requested = False

    def _enter_motion(
        self,
        *,
        inverted: bool,
        fallen: bool,
        recovery_requested: bool,
        roll: float,
        pitch: float,
        acc_z: float,
        reason: str,
    ) -> SafeRawStatus:
        self.mode = SafeRawMode.RECOVERY
        self._recovery_steps = 0
        self._stable_count = 0
        self._recovery_requested = recovery_requested
        return SafeRawStatus(
            mode=self.mode,
            terminated=True,
            replay_enabled=False,
            restart_required=True,
            recovery_requested=recovery_requested,
            inverted=inverted,
            fallen=fallen,
            roll=roll,
            pitch=pitch,
            acc_z=acc_z,
            reason=reason,
        )

    def update(self, state: RobotSafetyState) -> SafeRawStatus:
        roll, pitch = quat_to_roll_pitch(state.imu_quat)
        acc_z = float(np.asarray(state.imu_accel, dtype=np.float32)[2])
        inverted = acc_z < self.inverted_acc_z_threshold
        fallen = abs(roll) > self.fallen_roll_pitch_limit_rad or abs(pitch) > self.fallen_roll_pitch_limit_rad

        if self.mode == SafeRawMode.POLICY:
            self._inverted_count = self._inverted_count + 1 if inverted else 0
            if self._inverted_count >= self.trigger_steps:
                return self._enter_motion(
                    inverted=inverted,
                    fallen=fallen,
                    recovery_requested=True,
                    roll=roll,
                    pitch=pitch,
                    acc_z=acc_z,
                    reason="inverted",
                )
            if fallen:
                return self._enter_motion(
                    inverted=inverted,
                    fallen=fallen,
                    recovery_requested=False,
                    roll=roll,
                    pitch=pitch,
                    acc_z=acc_z,
                    reason="fallen",
                )
            return SafeRawStatus(
                mode=self.mode,
                inverted=inverted,
                fallen=fallen,
                roll=roll,
                pitch=pitch,
                acc_z=acc_z,
                reason="policy",
            )

        self._recovery_steps += 1
        if inverted:
            self._recovery_requested = True
        if not inverted and not fallen:
            self._recovery_requested = False
            self._stable_count += 1
            if self._stable_count >= self.stable_steps:
                self.reset()
                return SafeRawStatus(
                    mode=SafeRawMode.POLICY,
                    roll=roll,
                    pitch=pitch,
                    acc_z=acc_z,
                    reason="recovered",
                )
        else:
            self._stable_count = 0

        timed_out = self._recovery_steps >= self.timeout_steps
        return SafeRawStatus(
            mode=SafeRawMode.RECOVERY,
            terminated=False,
            replay_enabled=False,
            restart_required=False,
            recovery_requested=self._recovery_requested,
            inverted=inverted,
            fallen=fallen,
            roll=roll,
            pitch=pitch,
            acc_z=acc_z,
            reason="recovery_timeout" if timed_out else ("inverted" if inverted else "recovering"),
        )


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / norm).astype(np.float32)


def quat_to_roll_pitch(quat: np.ndarray) -> tuple[float, float]:
    w, x, y, z = normalize_quat(quat)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = np.copysign(np.pi / 2, sinp) if abs(sinp) >= 1.0 else np.arcsin(sinp)
    return float(roll), float(pitch)


def is_inverted(state: RobotSafetyState, acc_z_threshold: float) -> bool:
    return float(np.asarray(state.imu_accel, dtype=np.float32)[2]) < acc_z_threshold


def is_fallen(state: RobotSafetyState, limit_rad: float) -> bool:
    roll, pitch = quat_to_roll_pitch(state.imu_quat)
    return abs(roll) > limit_rad or abs(pitch) > limit_rad


class RandomAgent(BaseAgent[Config]):
    def __init__(
        self,
        observation_space: gym.spaces.Space[NDArray],
        action_space: gym.spaces.Space[NDArray],
        env_info: dict[str, Any],
        cfg: Config,
    ):
        super().__init__(observation_space, action_space, env_info, cfg)

    def sample_actions(
        self,
        interaction_step: int,
        prev_transition: Mapping[str, Tensor],
        training: bool,
    ) -> Tensor:
        num_envs = prev_transition["next_observation"].shape[0]
        return np.stack([self._action_space.sample() for _ in range(num_envs)])

    def process_transition(self, transition: Mapping[str, Tensor]) -> None:
        pass

    def can_start_training(self) -> bool:
        return False

    def update(self) -> dict[str, Any]:
        return {}

    def get_metrics(self) -> dict[str, Any]:
        return {}

    def save(self, path: str) -> None:
        pass

    def save_replay_buffer(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

    def load_replay_buffer(self, path: str) -> None:
        pass
