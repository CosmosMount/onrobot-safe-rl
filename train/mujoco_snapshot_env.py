"""Small in-process MuJoCo backend for exact-state policy branch evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from runtime.inference.actions import action_to_qpos
from runtime.inference.observations import build_observation, quat_to_euler_xyz
from runtime.inference.state import RobotState


JOINT_NAMES = tuple(
    f"{leg}_{joint}"
    for leg in ("FR", "FL", "RR", "RL")
    for joint in ("hip", "thigh", "calf")
)


@dataclass(frozen=True)
class RolloutMeasurement:
    failure: bool
    near_failure: bool
    tilt_rad: float
    height_m: float
    contact_count: int


class MujocoSnapshotEnv:
    """Go2 simulation with lossless ``mjSTATE_INTEGRATION`` save/restore."""

    def __init__(
        self,
        model_path: str | Path,
        cfg,
        *,
        policy_frequency: float = 50.0,
        kp: float = 60.0,
        kd: float = 5.0,
    ):
        import mujoco

        self.mujoco = mujoco
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(
            str(Path(model_path).expanduser().resolve()))
        self.data = mujoco.MjData(self.model)
        self.policy_dt = 1.0 / float(policy_frequency)
        self.substeps = max(
            1, int(round(self.policy_dt / float(self.model.opt.timestep))))
        self.kp = float(kp)
        self.kd = float(kd)
        self._state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._state_size = mujoco.mj_stateSize(
            self.model, self._state_spec)
        self.base_body_id = self._name_id(
            mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.actuator_ids = np.asarray([
            self._name_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in JOINT_NAMES
        ], dtype=np.int32)
        joint_ids = self.model.actuator_trnid[self.actuator_ids, 0]
        self.qpos_addresses = self.model.jnt_qposadr[joint_ids]
        self.qvel_addresses = self.model.jnt_dofadr[joint_ids]
        self.ctrl_low = self.model.actuator_ctrlrange[
            self.actuator_ids, 0]
        self.ctrl_high = self.model.actuator_ctrlrange[
            self.actuator_ids, 1]

    def _name_id(self, kind, name: str) -> int:
        value = int(self.mujoco.mj_name2id(self.model, kind, name))
        if value < 0:
            raise ValueError(f"MuJoCo model has no {name!r}")
        return value

    def reset_standing(
        self,
        *,
        settle_seconds: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = np.asarray([0.0, 0.0, 0.445])
        self.data.qpos[3:7] = np.asarray([1.0, 0.0, 0.0, 0.0])
        self.data.qpos[self.qpos_addresses] = self.cfg.init_qpos
        self.data.qvel[:] = 0.0
        if rng is not None:
            self.data.qpos[self.qpos_addresses] += rng.normal(
                0.0, 0.008, size=len(self.qpos_addresses))
            self.data.qvel[self.qvel_addresses] += rng.normal(
                0.0, 0.015, size=len(self.qvel_addresses))
        self.mujoco.mj_forward(self.model, self.data)
        neutral = np.zeros(self.cfg.num_joints, dtype=np.float32)
        for _ in range(max(
                0, int(round(settle_seconds / self.policy_dt)))):
            self.step(neutral)

    def capture(self) -> np.ndarray:
        value = np.empty(self._state_size, dtype=np.float64)
        self.mujoco.mj_getState(
            self.model, self.data, value, self._state_spec)
        return value

    def apply_base_velocity_impulse(
        self,
        *,
        linear_velocity_delta: np.ndarray,
        angular_velocity_delta: np.ndarray,
    ) -> None:
        """Apply a reproducible velocity impulse before capturing a snapshot."""
        # The model's free joint occupies the first six dofs: translation then
        # rotation. Both counterfactual branches restore the resulting state.
        self.data.qvel[:3] += np.asarray(
            linear_velocity_delta, dtype=np.float64)
        self.data.qvel[3:6] += np.asarray(
            angular_velocity_delta, dtype=np.float64)
        self.mujoco.mj_forward(self.model, self.data)

    def restore(self, value: np.ndarray) -> None:
        state = np.asarray(value, dtype=np.float64)
        if state.shape != (self._state_size,):
            raise ValueError(
                f"snapshot shape {state.shape}, expected {(self._state_size,)}")
        self.mujoco.mj_setState(
            self.model, self.data, state, self._state_spec)
        self.mujoco.mj_forward(self.model, self.data)

    def robot_state(self) -> RobotState:
        velocity = np.zeros(6, dtype=np.float64)
        self.mujoco.mj_objectVelocity(
            self.model, self.data,
            self.mujoco.mjtObj.mjOBJ_BODY, self.base_body_id,
            velocity, 1)
        return RobotState(
            joint_q=np.asarray(
                self.data.qpos[self.qpos_addresses],
                dtype=np.float32).copy(),
            joint_dq=np.asarray(
                self.data.qvel[self.qvel_addresses],
                dtype=np.float32).copy(),
            imu_quat=np.asarray(
                self.data.xquat[self.base_body_id],
                dtype=np.float32).copy(),
            imu_gyro=np.asarray(velocity[:3], dtype=np.float32),
            imu_accel=np.zeros(3, dtype=np.float32),
            body_velocity=np.asarray(velocity[3:], dtype=np.float32),
            world_position=np.asarray(
                self.data.xpos[self.base_body_id],
                dtype=np.float32).copy(),
            timestamp=float(self.data.time),
            low_state_timestamp=float(self.data.time),
            sport_state_timestamp=float(self.data.time),
        )

    def observation(self, previous_action_q_target: np.ndarray) -> np.ndarray:
        return build_observation(
            self.robot_state(),
            np.asarray(previous_action_q_target, dtype=np.float32),
            self.cfg)

    def measurement(self) -> RolloutMeasurement:
        state = self.robot_state()
        roll, pitch, _ = quat_to_euler_xyz(state.imu_quat)
        tilt = max(abs(roll), abs(pitch))
        height = float(state.world_position[2])
        failure = bool(
            tilt >= float(self.cfg.fallen_orientation_rad)
            or height < 0.18)
        near_failure = bool(
            not failure
            and (tilt >= float(self.cfg.fallen_risk_rad)
                 or height < 0.25))
        return RolloutMeasurement(
            failure=failure,
            near_failure=near_failure,
            tilt_rad=float(tilt),
            height_m=height,
            contact_count=int(self.data.ncon),
        )

    def step(self, action: np.ndarray) -> RolloutMeasurement:
        target = action_to_qpos(
            action,
            init_qpos=self.cfg.init_qpos,
            action_offset=self.cfg.action_offset,
            joint_min=self.cfg.joint_min,
            joint_max=self.cfg.joint_max)
        for _ in range(self.substeps):
            q = self.data.qpos[self.qpos_addresses]
            dq = self.data.qvel[self.qvel_addresses]
            torque = self.kp * (target - q) - self.kd * dq
            self.data.ctrl[self.actuator_ids] = np.clip(
                torque, self.ctrl_low, self.ctrl_high)
            self.mujoco.mj_step(self.model, self.data)
        return self.measurement()
