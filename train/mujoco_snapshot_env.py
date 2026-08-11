"""In-process MuJoCo reference backend for exact policy branch evaluation.

The reference backend deliberately snapshots both MuJoCo integration state and
the causal application state that lives outside MuJoCo: requested/executed
actions, the absolute joint target used by the corrected observation, the
five-frame observation history, and optional action-filter history.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from safety_data.paths import (
    assert_development_path,
    require_v3_audit_consumed_or_safe_input,
)

from runtime.inference.actions import (
    ActionApplier,
    ActionFilterButter,
    ActionFilterState,
    qpos_to_action,
)
from runtime.inference.observations import (
    build_observation,
    normalize_quat,
    quat_to_euler_xyz,
)
from runtime.inference.state import RobotState
from runtime.inference.velocity import quat_world_to_body


JOINT_NAMES = tuple(
    f"{leg}_{joint}"
    for leg in ("FR", "FL", "RR", "RL")
    for joint in ("hip", "thigh", "calf")
)
OBSERVATION_HISTORY_FRAMES = 5
FAILURE_HEIGHT_REFERENCE = "base_link_body_origin_world_z"
FAILURE_MEASUREMENT_CADENCE = "post_policy_step_after_all_low_level_substeps"


def _frozen_array(value: np.ndarray, *, dtype: Any) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError("snapshot arrays must be finite")
    result.setflags(write=False)
    return result


def _joint_vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 0:
        result = np.full(size, float(result), dtype=np.float64)
    result = result.reshape(-1)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite scalar or {size}-vector")
    return result


@dataclass(frozen=True)
class RolloutMeasurement:
    failure: bool
    near_failure: bool
    tilt_rad: float
    height_m: float
    contact_count: int


@dataclass(frozen=True)
class ActionApplication:
    """One policy action and the exact joint target applied by the backend."""

    action_requested: np.ndarray
    action_executed: np.ndarray
    action_q_target: np.ndarray

    def __post_init__(self) -> None:
        requested = _frozen_array(self.action_requested, dtype=np.float32).reshape(-1)
        executed = _frozen_array(self.action_executed, dtype=np.float32).reshape(-1)
        q_target = _frozen_array(self.action_q_target, dtype=np.float32).reshape(-1)
        if requested.shape != executed.shape or q_target.shape != requested.shape:
            raise ValueError("requested, executed and q-target actions must align")
        object.__setattr__(self, "action_requested", requested)
        object.__setattr__(self, "action_executed", executed)
        object.__setattr__(self, "action_q_target", q_target)


@dataclass(frozen=True)
class ApplicationState:
    """Causal state outside ``mjSTATE_INTEGRATION`` required by a branch."""

    previous_action_requested: np.ndarray
    previous_action_executed: np.ndarray
    previous_action_q_target: np.ndarray
    observation_history: np.ndarray
    action_filter_state: ActionFilterState | None

    def __post_init__(self) -> None:
        requested = _frozen_array(
            self.previous_action_requested, dtype=np.float32).reshape(-1)
        executed = _frozen_array(
            self.previous_action_executed, dtype=np.float32).reshape(-1)
        q_target = _frozen_array(
            self.previous_action_q_target, dtype=np.float32).reshape(-1)
        history = _frozen_array(self.observation_history, dtype=np.float32)
        if requested.shape != executed.shape or q_target.shape != requested.shape:
            raise ValueError("application action vectors must have equal shapes")
        if history.ndim != 2 or history.shape[0] > OBSERVATION_HISTORY_FRAMES:
            raise ValueError(
                "observation history must have shape [T, O] with "
                f"T <= {OBSERVATION_HISTORY_FRAMES}")
        object.__setattr__(self, "previous_action_requested", requested)
        object.__setattr__(self, "previous_action_executed", executed)
        object.__setattr__(self, "previous_action_q_target", q_target)
        object.__setattr__(self, "observation_history", history)


@dataclass(frozen=True)
class BranchSnapshot:
    """Lossless native branch snapshot (physics plus application state)."""

    integration_state: np.ndarray
    application_state: ApplicationState

    def __post_init__(self) -> None:
        object.__setattr__(self, "integration_state", _frozen_array(
            self.integration_state, dtype=np.float64).reshape(-1))

    def compound_sha256(self) -> str:
        """Hash physics and every environment-side causal application field."""
        digest = hashlib.sha256(b"sha256_compound_snapshot_v1\0")

        def update(name: str, value: np.ndarray) -> None:
            array = np.ascontiguousarray(value)
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())

        update("integration_state", self.integration_state)
        application = self.application_state
        update("previous_action_requested", application.previous_action_requested)
        update("previous_action_executed", application.previous_action_executed)
        update("previous_action_q_target", application.previous_action_q_target)
        update("observation_history", application.observation_history)
        if application.action_filter_state is None:
            digest.update(b"action_filter_state\0none")
        else:
            update("action_filter_x_history", application.action_filter_state.x_history)
            update("action_filter_y_history", application.action_filter_state.y_history)
        return digest.hexdigest()


@dataclass(frozen=True)
class RolloutStepResult:
    """Step output with compatibility properties for legacy evaluators."""

    application: ActionApplication
    measurement: RolloutMeasurement

    @property
    def failure(self) -> bool:
        return self.measurement.failure

    @property
    def near_failure(self) -> bool:
        return self.measurement.near_failure

    @property
    def tilt_rad(self) -> float:
        return self.measurement.tilt_rad

    @property
    def height_m(self) -> float:
        return self.measurement.height_m

    @property
    def contact_count(self) -> int:
        return self.measurement.contact_count


class MujocoSnapshotEnv:
    """Go2 simulation with compound, reproducible same-state snapshots."""

    def __init__(
        self,
        model_path: str | Path,
        cfg: Any,
        *,
        policy_frequency: float = 50.0,
        kp: float | np.ndarray | None = None,
        kd: float | np.ndarray | None = None,
        max_joint_delta: float | np.ndarray | None = None,
        use_action_filter: bool = False,
        action_filter_highcut: float | None = None,
    ):
        import mujoco

        self.mujoco = mujoco
        self.cfg = cfg
        self.model_path = assert_development_path(
            require_v3_audit_consumed_or_safe_input(model_path))
        dependency_hash_before = self._xml_dependency_hash(self.model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        dependency_hash_after = self._xml_dependency_hash(self.model_path)
        if dependency_hash_after != dependency_hash_before:
            raise RuntimeError(
                "MJCF dependency bytes changed while the MuJoCo model loaded")
        # Fingerprints must describe the bytes that built this in-memory model,
        # not a later re-read of external files after collection has started.
        self._mjcf_dependency_sha256 = dependency_hash_before
        self.data = mujoco.MjData(self.model)
        self.policy_frequency = float(policy_frequency)
        if not np.isfinite(self.policy_frequency) or self.policy_frequency <= 0.0:
            raise ValueError("policy_frequency must be finite and positive")
        self.policy_dt = 1.0 / self.policy_frequency
        self.substeps = max(
            1, int(round(self.policy_dt / float(self.model.opt.timestep))))
        self.kp = _joint_vector(
            cfg.kp if kp is None else kp, cfg.num_joints, "kp")
        self.kd = _joint_vector(
            cfg.kd if kd is None else kd, cfg.num_joints, "kd")
        self._state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._state_size = mujoco.mj_stateSize(self.model, self._state_spec)
        self.base_body_id = self._name_id(
            mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.actuator_ids = np.asarray([
            self._name_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in JOINT_NAMES
        ], dtype=np.int32)
        joint_ids = self.model.actuator_trnid[self.actuator_ids, 0]
        self.qpos_addresses = self.model.jnt_qposadr[joint_ids]
        self.qvel_addresses = self.model.jnt_dofadr[joint_ids]
        self.ctrl_low = self.model.actuator_ctrlrange[self.actuator_ids, 0]
        self.ctrl_high = self.model.actuator_ctrlrange[self.actuator_ids, 1]
        # These exact sensors feed the Unitree SDK bridge.  Reading a body
        # velocity directly is not equivalent when the IMU has a site offset.
        self.sensor_ids = {
            name: self._name_id(mujoco.mjtObj.mjOBJ_SENSOR, name)
            for name in (
                "imu_quat", "imu_gyro", "imu_acc", "frame_pos", "frame_vel")
        }
        action_filter = None
        if use_action_filter:
            action_filter = ActionFilterButter(
                cfg.num_joints,
                sampling_rate=self.policy_frequency,
                highcut=(
                    float(cfg.action_filter_highcut)
                    if action_filter_highcut is None
                    else float(action_filter_highcut)),
            )
        self.action_applier = ActionApplier(
            init_qpos=np.asarray(cfg.init_qpos, dtype=np.float32),
            action_offset=np.asarray(cfg.action_offset, dtype=np.float32),
            joint_min=np.asarray(cfg.joint_min, dtype=np.float32),
            joint_max=np.asarray(cfg.joint_max, dtype=np.float32),
            max_joint_delta=max_joint_delta,
            action_filter=action_filter,
        )
        self._observation_history: list[np.ndarray] = []
        self._reset_application_state()

    def _name_id(self, kind: Any, name: str) -> int:
        value = int(self.mujoco.mj_name2id(self.model, kind, name))
        if value < 0:
            raise ValueError(f"MuJoCo model has no {name!r}")
        return value

    def _reset_application_state(self) -> None:
        self.action_applier.reset_filter()
        zeros = np.zeros(self.cfg.num_joints, dtype=np.float32)
        self._previous_action_requested = zeros.copy()
        self._previous_action_executed = zeros.copy()
        self._previous_action_q_target = np.asarray(
            self.cfg.init_qpos, dtype=np.float32).copy()
        self._observation_history = []

    @property
    def previous_action_requested(self) -> np.ndarray:
        return self._previous_action_requested.copy()

    @property
    def previous_action_executed(self) -> np.ndarray:
        return self._previous_action_executed.copy()

    @property
    def previous_action_q_target(self) -> np.ndarray:
        return self._previous_action_q_target.copy()

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
        self._reset_application_state()
        neutral = np.zeros(self.cfg.num_joints, dtype=np.float32)
        for _ in range(max(0, int(round(settle_seconds / self.policy_dt)))):
            self.step(neutral)
        # Settling is simulator initialization, not deployable policy history.
        self._observation_history = []

    def _application_state(self) -> ApplicationState:
        history = np.asarray(self._observation_history, dtype=np.float32)
        if not self._observation_history:
            history = np.empty((0, self.cfg.obs_dim), dtype=np.float32)
        filter_state = (
            None if self.action_applier.action_filter is None
            else self.action_applier.action_filter.capture_state())
        return ApplicationState(
            previous_action_requested=self._previous_action_requested,
            previous_action_executed=self._previous_action_executed,
            previous_action_q_target=self._previous_action_q_target,
            observation_history=history,
            action_filter_state=filter_state,
        )

    def capture(self) -> BranchSnapshot:
        # mj_step leaves some derived sensor fields at the pre-forward stage;
        # synchronize before comparing raw observations across a restore.
        self.mujoco.mj_forward(self.model, self.data)
        value = np.empty(self._state_size, dtype=np.float64)
        self.mujoco.mj_getState(
            self.model, self.data, value, self._state_spec)
        return BranchSnapshot(value, self._application_state())

    def apply_base_velocity_impulse(
        self,
        *,
        linear_velocity_delta: np.ndarray,
        angular_velocity_delta: np.ndarray,
    ) -> None:
        """Apply a reproducible velocity impulse before capturing a snapshot."""
        self.data.qvel[:3] += np.asarray(
            linear_velocity_delta, dtype=np.float64)
        self.data.qvel[3:6] += np.asarray(
            angular_velocity_delta, dtype=np.float64)
        self.mujoco.mj_forward(self.model, self.data)

    def restore(self, snapshot: BranchSnapshot) -> None:
        if not isinstance(snapshot, BranchSnapshot):
            raise TypeError(
                "restore requires BranchSnapshot; physics-only arrays omit "
                "q_send/filter/history application state")
        state = np.asarray(snapshot.integration_state, dtype=np.float64)
        if state.shape != (self._state_size,):
            raise ValueError(
                f"snapshot shape {state.shape}, expected {(self._state_size,)}")
        application = snapshot.application_state
        if application.previous_action_q_target.shape != (self.cfg.num_joints,):
            raise ValueError("snapshot action dimension does not match environment")
        if application.observation_history.shape[1:] != (self.cfg.obs_dim,):
            raise ValueError("snapshot observation dimension does not match environment")
        self.mujoco.mj_setState(
            self.model, self.data, state, self._state_spec)
        self.mujoco.mj_forward(self.model, self.data)
        self._previous_action_requested = application.previous_action_requested.copy()
        self._previous_action_executed = application.previous_action_executed.copy()
        self._previous_action_q_target = application.previous_action_q_target.copy()
        self._observation_history = [
            row.copy() for row in application.observation_history]
        if self.action_applier.action_filter is None:
            if application.action_filter_state is not None:
                raise ValueError("snapshot has filter state but environment has no filter")
        else:
            if application.action_filter_state is None:
                raise ValueError("filtered environment requires filter state in snapshot")
            self.action_applier.action_filter.restore_state(
                application.action_filter_state)

    def _sensor(self, name: str) -> np.ndarray:
        sensor_id = self.sensor_ids[name]
        address = int(self.model.sensor_adr[sensor_id])
        dimension = int(self.model.sensor_dim[sensor_id])
        return np.asarray(
            self.data.sensordata[address:address + dimension],
            dtype=np.float32,
        ).copy()

    def robot_state(self) -> RobotState:
        self.mujoco.mj_forward(self.model, self.data)
        quat = normalize_quat(self._sensor("imu_quat"))
        world_velocity = self._sensor("frame_vel")
        return RobotState(
            joint_q=np.asarray(
                self.data.qpos[self.qpos_addresses],
                dtype=np.float32).copy(),
            joint_dq=np.asarray(
                self.data.qvel[self.qvel_addresses],
                dtype=np.float32).copy(),
            imu_quat=quat,
            imu_gyro=self._sensor("imu_gyro"),
            imu_accel=self._sensor("imu_acc"),
            body_velocity=quat_world_to_body(world_velocity, quat),
            world_position=self._sensor("frame_pos"),
            timestamp=float(self.data.time),
            low_state_timestamp=float(self.data.time),
            sport_state_timestamp=float(self.data.time),
        )

    def _validated_q_target(self, value: np.ndarray) -> np.ndarray:
        target = np.asarray(value, dtype=np.float32).reshape(-1)
        if target.shape != (self.cfg.num_joints,) or not np.all(np.isfinite(target)):
            raise ValueError("previous_action_q_target must be a finite joint vector")
        lower = np.maximum(self.cfg.joint_min, self.cfg.init_qpos - self.cfg.action_offset)
        upper = np.minimum(self.cfg.joint_max, self.cfg.init_qpos + self.cfg.action_offset)
        if np.any(target < lower - 1e-6) or np.any(target > upper + 1e-6):
            raise ValueError(
                "previous_action_q_target is outside absolute q_send bounds; "
                "a normalized policy action was likely supplied")
        return target

    def observation(
        self, previous_action_q_target: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build corrected 46D observation using an absolute prior q target."""
        target = (
            self._previous_action_q_target
            if previous_action_q_target is None
            else self._validated_q_target(previous_action_q_target))
        return build_observation(self.robot_state(), target, self.cfg)

    def record_observation(self) -> np.ndarray:
        """Record one policy-time observation and return padded five-frame history."""
        observation = self.observation()
        self._observation_history.append(observation.copy())
        if len(self._observation_history) > OBSERVATION_HISTORY_FRAMES:
            self._observation_history.pop(0)
        padding = [self._observation_history[0]] * (
            OBSERVATION_HISTORY_FRAMES - len(self._observation_history))
        return np.stack(padding + self._observation_history, axis=0).astype(np.float32)

    def observation_history(self) -> np.ndarray:
        """Return a five-frame view without implicitly recording a new frame."""
        if not self._observation_history:
            raise RuntimeError("record_observation must be called before history access")
        padding = [self._observation_history[0]] * (
            OBSERVATION_HISTORY_FRAMES - len(self._observation_history))
        return np.stack(padding + self._observation_history, axis=0).astype(np.float32)

    def measurement(self) -> RolloutMeasurement:
        state = self.robot_state()
        roll, pitch, _ = quat_to_euler_xyz(state.imu_quat)
        tilt = max(abs(roll), abs(pitch))
        # ``state.world_position`` is the MJCF ``frame_pos`` sensor attached
        # to the IMU site.  On Go2 that site is offset above ``base_link``, so
        # using it here silently shifts both the cohort pre-screen and the
        # fall threshold.  The v3 contract says base height: bind that label to
        # the base body origin explicitly.
        height = float(self.data.xpos[self.base_body_id, 2])
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

    def step(self, action: np.ndarray) -> RolloutStepResult:
        projection = self.action_applier.project(
            action, self.data.qpos[self.qpos_addresses])
        target = projection.action_q_target.astype(np.float64)
        return self._step_absolute_target(
            target,
            kp=self.kp,
            kd=self.kd,
            low_level_steps=self.substeps,
            application=ActionApplication(
                projection.action_requested,
                projection.action_executed,
                projection.action_q_target,
            ),
        )

    def _step_absolute_target(
        self,
        target: np.ndarray,
        *,
        kp: np.ndarray,
        kd: np.ndarray,
        low_level_steps: int,
        application: ActionApplication,
    ) -> RolloutStepResult:
        if isinstance(low_level_steps, (bool, np.bool_)) or not isinstance(
                low_level_steps, (int, np.integer)) or int(low_level_steps) <= 0:
            raise ValueError("low_level_steps must be a positive integer")
        for _ in range(int(low_level_steps)):
            q = self.data.qpos[self.qpos_addresses]
            dq = self.data.qvel[self.qvel_addresses]
            torque = kp * (target - q) - kd * dq
            self.data.ctrl[self.actuator_ids] = np.clip(
                torque, self.ctrl_low, self.ctrl_high)
            self.mujoco.mj_step(self.model, self.data)
        self.mujoco.mj_forward(self.model, self.data)
        self._previous_action_requested = application.action_requested.copy()
        self._previous_action_executed = application.action_executed.copy()
        self._previous_action_q_target = application.action_q_target.copy()
        return RolloutStepResult(application, self.measurement())

    def step_recovery_target(
        self,
        q_target: np.ndarray,
        *,
        kp: float | np.ndarray = 100.0,
        kd: float | np.ndarray = 8.0,
    ) -> RolloutStepResult:
        """Apply one 500-Hz absolute target from the fixed recovery motion.

        Unlike :meth:`step`, this method intentionally bypasses the task
        policy's narrow normalized action range.  The C++ recovery controller
        owns the low-level command and uses the robot joint limits plus its own
        gains.  One call advances exactly one MuJoCo/500-Hz control tick.
        """
        target = np.asarray(q_target, dtype=np.float32).reshape(-1)
        if target.shape != (self.cfg.num_joints,) or not np.all(np.isfinite(target)):
            raise ValueError("recovery q_target must be a finite joint vector")
        lower = np.asarray(self.cfg.joint_min, dtype=np.float32)
        upper = np.asarray(self.cfg.joint_max, dtype=np.float32)
        if np.any(target < lower) or np.any(target > upper):
            raise ValueError("recovery q_target is outside controller joint limits")
        recovery_kp = _joint_vector(kp, self.cfg.num_joints, "recovery kp")
        recovery_kd = _joint_vector(kd, self.cfg.num_joints, "recovery kd")
        normalized = qpos_to_action(
            target,
            init_qpos=np.asarray(self.cfg.init_qpos, dtype=np.float32),
            action_offset=np.asarray(self.cfg.action_offset, dtype=np.float32),
        )
        application = ActionApplication(normalized, normalized, target)
        return self._step_absolute_target(
            target.astype(np.float64),
            kp=recovery_kp,
            kd=recovery_kd,
            low_level_steps=1,
            application=application,
        )

    def simulator_fingerprint(self) -> dict[str, Any]:
        """Return claim-bearing simulator/controller fields for a manifest."""
        return {
            "backend": "mujoco",
            "mujoco_version": str(self.mujoco.__version__),
            "model_path": str(self.model_path),
            "mjcf_xml_sha256": self._mjcf_dependency_sha256,
            "timestep_s": float(self.model.opt.timestep),
            "policy_frequency_hz": self.policy_frequency,
            "substeps": self.substeps,
            "failure_measurement": {
                "height_reference": FAILURE_HEIGHT_REFERENCE,
                "cadence": FAILURE_MEASUREMENT_CADENCE,
                "low_level_substeps_per_policy_step": self.substeps,
            },
            "kp": self.kp.tolist(),
            "kd": self.kd.tolist(),
            "actuator_ctrl_low": self.ctrl_low.astype(float).tolist(),
            "actuator_ctrl_high": self.ctrl_high.astype(float).tolist(),
            "action_filter": (
                None if self.action_applier.action_filter is None
                else self.action_applier.action_filter.fingerprint()),
            "max_joint_delta": (
                None if self.action_applier.max_joint_delta is None
                else np.asarray(
                    self.action_applier.max_joint_delta, dtype=float).tolist()),
        }

    @staticmethod
    def _xml_dependency_hash(root_path: Path) -> str:
        """Hash MJCF, recursive includes, and referenced mesh/texture assets."""
        digest = hashlib.sha256()
        root_path = assert_development_path(
            require_v3_audit_consumed_or_safe_input(root_path))
        hash_root = root_path.parent
        pending: list[tuple[Path, bool]] = [(root_path, True)]
        seen: set[Path] = set()
        while pending:
            path, parse_xml = pending.pop()
            if path in seen:
                continue
            seen.add(path)
            content = path.read_bytes()
            try:
                label = str(path.relative_to(hash_root))
            except ValueError:
                label = path.name
            digest.update(label.encode("utf-8"))
            digest.update(b"\0")
            digest.update(content)
            if not parse_xml:
                continue
            try:
                tree = ET.fromstring(content)
            except ET.ParseError:
                continue
            compiler = tree.find("compiler")
            asset_dir = "" if compiler is None else compiler.attrib.get("assetdir", "")
            mesh_dir = (
                asset_dir if compiler is None
                else compiler.attrib.get("meshdir", asset_dir))
            texture_dir = (
                asset_dir if compiler is None
                else compiler.attrib.get("texturedir", asset_dir))
            for element in tree.iter():
                filename = element.attrib.get("file")
                if not filename:
                    continue
                if element.tag == "include":
                    dependency = path.parent / filename
                    pending.append((assert_development_path(
                        require_v3_audit_consumed_or_safe_input(dependency)), True))
                elif element.tag == "mesh":
                    dependency = path.parent / mesh_dir / filename
                    pending.append((assert_development_path(
                        require_v3_audit_consumed_or_safe_input(dependency)), False))
                elif element.tag in {"texture", "hfield", "skin"}:
                    dependency = path.parent / texture_dir / filename
                    pending.append((assert_development_path(
                        require_v3_audit_consumed_or_safe_input(dependency)), False))
        return digest.hexdigest()
