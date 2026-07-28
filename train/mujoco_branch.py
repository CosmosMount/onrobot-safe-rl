"""Exact-state MuJoCo backend for counterfactual action branch rollouts.

This module is intentionally separate from ``Go2Env``.  The online environment
continues to communicate with the external simulator/controller over DDS.  The
backend here is an offline data-generation tool and imports the optional
``mujoco`` Python package lazily.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from learner.counterfactual_dataset import (
    BranchMeasurement,
    BranchSnapshot,
)
from train.config import Go2Config
from train.env import action_to_qpos
from train.obs import (
    build_observation,
    is_belly_up,
    is_fallen,
    tilt_from_upright,
)
from train.safety import safety_signals
from train.types import RobotState


JOINT_ORDER = tuple(
    f'{leg}_{joint}'
    for leg in ('FR', 'FL', 'RR', 'RL')
    for joint in ('hip', 'thigh', 'calf')
)
FOOT_GEOM_NAMES = frozenset(('FR', 'FL', 'RR', 'RL'))


class MujocoBranchBackend:
    """In-process Go2 simulation supporting exact integration-state restore."""

    def __init__(
            self,
            model_path: str | Path,
            cfg: Go2Config,
            *,
            policy_frequency: float = 20.0,
            kp: float = 60.0,
            kd: float = 5.0):
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError(
                'MuJoCo branch collection requires the optional Python '
                'package `mujoco` in the active environment.') from exc
        self.mujoco = mujoco
        self.model_path = Path(model_path).expanduser().resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.cfg = cfg
        self.policy_frequency = float(policy_frequency)
        self.policy_dt = 1.0 / self.policy_frequency
        self.kp = float(kp)
        self.kd = float(kd)
        self.substeps = max(
            1, int(round(self.policy_dt / float(self.model.opt.timestep))))
        self._state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._state_size = mujoco.mj_stateSize(
            self.model, self._state_spec)
        self.base_body_id = self._optional_name_id(
            mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        if self.base_body_id < 0:
            self.base_body_id = self._name_id(
                mujoco.mjtObj.mjOBJ_BODY, 'base')
        self.floor_geom_id = self._optional_name_id(
            mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.foot_geom_ids = {
            value for name in FOOT_GEOM_NAMES
            if (value := self._optional_name_id(
                mujoco.mjtObj.mjOBJ_GEOM, name)) >= 0
        }

        self.actuator_ids = np.asarray([
            self._name_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in JOINT_ORDER
        ], dtype=np.int32)
        joint_ids = self.model.actuator_trnid[self.actuator_ids, 0]
        self.qpos_addresses = self.model.jnt_qposadr[joint_ids].astype(
            np.int32)
        self.qvel_addresses = self.model.jnt_dofadr[joint_ids].astype(
            np.int32)
        self.ctrl_low = self.model.actuator_ctrlrange[
            self.actuator_ids, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[
            self.actuator_ids, 1].copy()

    def _name_id(self, object_type, name: str) -> int:
        value = int(self.mujoco.mj_name2id(
            self.model, object_type, name))
        if value < 0:
            raise ValueError(
                f'{self.model_path} has no required MuJoCo object {name!r}')
        return value

    def _optional_name_id(self, object_type, name: str) -> int:
        return int(self.mujoco.mj_name2id(
            self.model, object_type, name))

    def reset_standing(self, *, settle_seconds: float = 1.0) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        # The Go2 model has one free joint followed by 12 hinge joints.
        self.data.qpos[:3] = np.asarray([0.0, 0.0, 0.445])
        self.data.qpos[3:7] = np.asarray([1.0, 0.0, 0.0, 0.0])
        self.data.qpos[self.qpos_addresses] = self.cfg.init_qpos
        self.data.qvel[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        settle_steps = max(
            0, int(round(float(settle_seconds) / self.policy_dt)))
        neutral = np.zeros(self.cfg.num_joints, dtype=np.float32)
        for _ in range(settle_steps):
            self.step_action(neutral)

    def capture_state(self) -> np.ndarray:
        state = np.empty(self._state_size, dtype=np.float64)
        self.mujoco.mj_getState(
            self.model, self.data, state, self._state_spec)
        return state

    def restore_state(self, state: np.ndarray) -> None:
        value = np.asarray(state, dtype=np.float64)
        if value.shape != (self._state_size,):
            raise ValueError(
                f'MuJoCo integration state has shape {value.shape}; '
                f'expected {(self._state_size,)}')
        self.mujoco.mj_setState(
            self.model, self.data, value, self._state_spec)
        self.mujoco.mj_forward(self.model, self.data)

    def robot_state(self) -> RobotState:
        spatial_velocity = np.zeros(6, dtype=np.float64)
        self.mujoco.mj_objectVelocity(
            self.model, self.data,
            self.mujoco.mjtObj.mjOBJ_BODY, self.base_body_id,
            spatial_velocity, 1)
        # MuJoCo spatial velocity stores rotation followed by translation.
        angular_velocity = spatial_velocity[:3]
        linear_velocity = spatial_velocity[3:]
        actuator_force = np.asarray(
            self.data.actuator_force[self.actuator_ids], dtype=np.float32)
        return RobotState(
            joint_q=np.asarray(
                self.data.qpos[self.qpos_addresses], dtype=np.float32).copy(),
            joint_dq=np.asarray(
                self.data.qvel[self.qvel_addresses], dtype=np.float32).copy(),
            joint_tau=actuator_force.copy(),
            imu_quat=np.asarray(
                self.data.xquat[self.base_body_id],
                dtype=np.float32).copy(),
            imu_gyro=np.asarray(angular_velocity, dtype=np.float32),
            # safety_signals falls back to quaternion/contact-derived labels;
            # acceleration is not needed for exact state restoration.
            imu_accel=np.zeros(3, dtype=np.float32),
            body_velocity=np.asarray(linear_velocity, dtype=np.float32),
            world_position=np.asarray(
                self.data.xpos[self.base_body_id],
                dtype=np.float32).copy(),
            timestamp=float(self.data.time),
            low_state_timestamp=float(self.data.time),
            sport_state_timestamp=float(self.data.time),
        )

    def observation(self, previous_action: np.ndarray,
                    previous_executed_action: np.ndarray,
                    command_speed: float) -> np.ndarray:
        if abs(float(command_speed) - self.cfg.move_speed) > 1e-9:
            import dataclasses
            cfg = dataclasses.replace(
                self.cfg, move_speed=float(command_speed))
        else:
            cfg = self.cfg
        return build_observation(
            self.robot_state(),
            np.asarray(previous_action, dtype=np.float32),
            cfg,
            np.asarray(previous_executed_action, dtype=np.float32))

    def snapshot(self, *, previous_action: np.ndarray,
                 previous_executed_action: np.ndarray,
                 command_speed: float, episode_id: int = 0,
                 policy_step: int = 0) -> BranchSnapshot:
        return BranchSnapshot(
            simulator_state=self.capture_state(),
            observation=self.observation(
                previous_action, previous_executed_action, command_speed),
            previous_action=np.asarray(
                previous_action, dtype=np.float32).copy(),
            previous_executed_action=np.asarray(
                previous_executed_action, dtype=np.float32).copy(),
            command_speed=float(command_speed),
            episode_id=int(episode_id),
            policy_step=int(policy_step),
        )

    def _contact_metrics(self) -> tuple[int, int, float]:
        contact_count = int(self.data.ncon)
        undesired = 0
        max_force = 0.0
        force = np.zeros(6, dtype=np.float64)
        for index in range(contact_count):
            contact = self.data.contact[index]
            geom_ids = (int(contact.geom1), int(contact.geom2))
            bodies = {
                int(self.model.geom_bodyid[geom])
                for geom in geom_ids
            }
            touches_floor = (
                self.floor_geom_id < 0
                or self.floor_geom_id in geom_ids)
            foot_contact = any(
                geom in self.foot_geom_ids for geom in geom_ids)
            if (self.base_body_id in bodies and touches_floor) or (
                    touches_floor and not foot_contact
                    and any(body != 0 for body in bodies)):
                undesired += 1
            self.mujoco.mj_contactForce(
                self.model, self.data, index, force)
            max_force = max(max_force, float(np.linalg.norm(force[:3])))
        return contact_count, undesired, max_force

    def step_action(self, action: np.ndarray) -> BranchMeasurement:
        target = action_to_qpos(
            np.asarray(action, dtype=np.float32), self.cfg)
        for _ in range(self.substeps):
            q = self.data.qpos[self.qpos_addresses]
            dq = self.data.qvel[self.qvel_addresses]
            torque = self.kp * (target - q) - self.kd * dq
            self.data.ctrl[self.actuator_ids] = np.clip(
                torque, self.ctrl_low, self.ctrl_high)
            self.mujoco.mj_step(self.model, self.data)
        state = self.robot_state()
        contacts, undesired, max_force = self._contact_metrics()
        signals = safety_signals(
            state, self.cfg,
            terminated=is_fallen(state, self.cfg),
            recovering=False, intervention_mask=False)
        failure = bool(
            is_fallen(state, self.cfg)
            or is_belly_up(state, self.cfg)
            or state.world_position[2] < self.cfg.safety_base_height_m)
        return BranchMeasurement(
            failure=failure,
            near_failure=bool(
                signals['near_failure_label'] or undesired > 0),
            base_tilt_rad=tilt_from_upright(state.imu_quat),
            base_height_m=float(state.world_position[2]),
            contact_count=contacts,
            undesired_contact_count=undesired,
            max_contact_force=max_force,
        )
