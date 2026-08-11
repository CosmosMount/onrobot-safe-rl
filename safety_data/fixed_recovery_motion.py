"""Exact Python model of the Go2 controller's fixed recovery motion.

This is deliberately not a learned policy.  It mirrors
``runtime/control/go2/motions/src/recovery.cpp``: at 500 Hz it executes the
configured Fold -> Above -> SwingDown -> Push joint-target sequence.  Keeping
the motion in one small state machine lets native MuJoCo evidence use the same
controller semantics as deployment without importing Unitree DDS bindings.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


JOINT_COUNT = 12
LEG_COUNT = 4
JOINTS_PER_LEG = 3


def _vector(value: Any, *, name: str, dtype: Any = np.float32) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != (JOINT_COUNT,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 12-vector")
    frozen = result.copy()
    frozen.setflags(write=False)
    return frozen


def _mask(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != (LEG_COUNT,) or result.dtype.kind != "b":
        raise ValueError(f"{name} must be a four-element boolean vector")
    frozen = result.astype(bool, copy=True)
    frozen.setflags(write=False)
    return frozen


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seconds_to_ticks(seconds: float, control_hz: float) -> int:
    """Match positive-input C++ ``std::lround`` followed by ``max(1, ...)``."""
    if not np.isfinite(seconds) or seconds < 0.0:
        raise ValueError("recovery durations must be finite and nonnegative")
    if not np.isfinite(control_hz) or control_hz <= 0.0:
        raise ValueError("control_hz must be finite and positive")
    return max(1, int(np.floor(seconds * control_hz + 0.5)))


@dataclass(frozen=True)
class FixedRecoveryConfig:
    fold_jpos: np.ndarray
    above_jpos: np.ndarray
    swing_down_jpos: np.ndarray
    push_jpos: np.ndarray
    swing_legs: np.ndarray
    push_legs: np.ndarray
    fold_ramp_s: float
    fold_settle_s: float
    above_ramp_s: float
    above_settle_s: float
    swing_down_ramp_s: float
    swing_down_settle_s: float
    push_ramp_s: float
    push_settle_s: float
    joint_reach_tol: float
    kp: float
    kd: float

    def __post_init__(self) -> None:
        for name in ("fold_jpos", "above_jpos", "swing_down_jpos", "push_jpos"):
            object.__setattr__(self, name, _vector(getattr(self, name), name=name))
        for name in ("swing_legs", "push_legs"):
            object.__setattr__(self, name, _mask(getattr(self, name), name=name))
        for name in (
            "fold_ramp_s", "fold_settle_s", "above_ramp_s", "above_settle_s",
            "swing_down_ramp_s", "swing_down_settle_s", "push_ramp_s",
            "push_settle_s",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        for name in ("joint_reach_tol", "kp", "kd"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)

    @classmethod
    def from_controller_yaml(cls, path: str | Path) -> "FixedRecoveryConfig":
        source = Path(path)
        root = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(root, dict) or not isinstance(root.get("recovery"), dict):
            raise ValueError("controller YAML must contain a recovery mapping")
        value = root["recovery"]
        return cls(
            fold_jpos=value["fold_jpos"],
            above_jpos=value["above_jpos"],
            swing_down_jpos=value["swing_down_jpos"],
            push_jpos=value["push_jpos"],
            swing_legs=value.get("swing_legs", [True] * LEG_COUNT),
            push_legs=value.get("push_legs", [False, False, True, True]),
            fold_ramp_s=value.get("fold_ramp_s", 0.45),
            fold_settle_s=value.get("fold_settle_s", 0.50),
            above_ramp_s=value.get("above_ramp_s", value.get("extend_ramp_s", 0.45)),
            above_settle_s=value.get("above_settle_s", value.get("extend_settle_s", 0.35)),
            swing_down_ramp_s=value.get("swing_down_ramp_s", 0.55),
            swing_down_settle_s=value.get("swing_down_settle_s", 0.45),
            push_ramp_s=value.get("push_ramp_s", 0.30),
            push_settle_s=value.get("push_settle_s", 0.25),
            joint_reach_tol=value.get("joint_reach_tol", 0.12),
            kp=value.get("kp", 100.0),
            kd=value.get("kd", 8.0),
        )

    def manifest(
        self,
        *,
        control_hz: float,
        controller_yaml: str | Path | None = None,
        controller_source: str | Path | None = None,
    ) -> dict[str, Any]:
        payload = {
            "schema_version": "qsafe.original_go2_fixed_recovery.v1",
            "implementation": "Fold->Above->SwingDown->Push",
            "learned_policy_used": False,
            "control_hz": float(control_hz),
            "poses": {
                name: np.asarray(getattr(self, name), dtype=float).tolist()
                for name in ("fold_jpos", "above_jpos", "swing_down_jpos", "push_jpos")
            },
            "swing_legs": self.swing_legs.tolist(),
            "push_legs": self.push_legs.tolist(),
            "durations_s": {
                name: float(getattr(self, name))
                for name in (
                    "fold_ramp_s", "fold_settle_s", "above_ramp_s",
                    "above_settle_s", "swing_down_ramp_s",
                    "swing_down_settle_s", "push_ramp_s", "push_settle_s",
                )
            },
            "joint_reach_tol": self.joint_reach_tol,
            "kp": self.kp,
            "kd": self.kd,
        }
        if (controller_yaml is None) != (controller_source is None):
            raise ValueError(
                "controller_yaml and controller_source must be supplied together")
        if controller_yaml is not None and controller_source is not None:
            yaml_path = Path(controller_yaml)
            source_path = Path(controller_source)
            payload["authoritative_files"] = {
                "controller_yaml": str(yaml_path),
                "controller_yaml_sha256": _file_sha256(yaml_path),
                "recovery_cpp": str(source_path),
                "recovery_cpp_sha256": _file_sha256(source_path),
            }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        return payload | {"contract_sha256": hashlib.sha256(encoded).hexdigest()}


class RecoveryStage(str, Enum):
    FOLD = "fold"
    ABOVE = "above"
    SWING_DOWN = "swing_down"
    PUSH = "push"


@dataclass(frozen=True)
class RecoveryMotionState:
    tick: int
    segment_start_tick: int
    stage: RecoveryStage
    initial_jpos: np.ndarray


@dataclass(frozen=True)
class RecoveryTick:
    q_target: np.ndarray
    done: bool
    stage_executed: RecoveryStage
    tick: int


@dataclass(frozen=True)
class RecoveryExecutionTick:
    motion: RecoveryTick
    rollout: Any


class FixedRecoveryMotion:
    """Stateful 500-Hz replica of the original controller recovery primitive."""

    def __init__(self, config: FixedRecoveryConfig, *, control_hz: float = 500.0):
        if not isinstance(config, FixedRecoveryConfig):
            raise TypeError("config must be FixedRecoveryConfig")
        self.config = config
        self.control_hz = float(control_hz)
        self._ticks = {
            name: _seconds_to_ticks(float(getattr(config, name)), self.control_hz)
            for name in (
                "fold_ramp_s", "fold_settle_s", "above_ramp_s", "above_settle_s",
                "swing_down_ramp_s", "swing_down_settle_s", "push_ramp_s",
                "push_settle_s",
            )
        }
        self._state: RecoveryMotionState | None = None

    def reset(self, joint_q: np.ndarray) -> None:
        initial = _vector(joint_q, name="joint_q")
        self._state = RecoveryMotionState(0, 0, RecoveryStage.FOLD, initial)

    def capture_state(self) -> RecoveryMotionState:
        if self._state is None:
            raise RuntimeError("reset must be called before capture_state")
        state = self._state
        return RecoveryMotionState(
            state.tick, state.segment_start_tick, state.stage,
            state.initial_jpos.copy(),
        )

    def restore_state(self, state: RecoveryMotionState) -> None:
        if not isinstance(state, RecoveryMotionState):
            raise TypeError("state must be RecoveryMotionState")
        self._state = RecoveryMotionState(
            int(state.tick), int(state.segment_start_tick), RecoveryStage(state.stage),
            _vector(state.initial_jpos, name="initial_jpos"),
        )

    @staticmethod
    def _interpolate(start: np.ndarray, finish: np.ndarray, tick: int, ticks: int) -> np.ndarray:
        blend = 1.0 if tick > ticks or ticks <= 0 else float(tick) / float(ticks)
        return ((1.0 - blend) * start + blend * finish).astype(np.float32)

    def _leg_pose(self, active_pose: np.ndarray) -> np.ndarray:
        q_target = self.config.fold_jpos.copy()
        for leg in range(LEG_COUNT):
            if self.config.swing_legs[leg]:
                sl = slice(leg * JOINTS_PER_LEG, (leg + 1) * JOINTS_PER_LEG)
                q_target[sl] = active_pose[sl]
        return q_target

    def _begin_segment(self, q_target: np.ndarray, *, tick: int, stage: RecoveryStage) -> None:
        self._state = RecoveryMotionState(
            tick=tick,
            segment_start_tick=tick,
            stage=stage,
            initial_jpos=q_target.copy(),
        )

    def update(self, joint_q: np.ndarray, *, state_received: bool = True) -> RecoveryTick:
        if self._state is None:
            raise RuntimeError("reset must be called before update")
        measured = _vector(joint_q, name="joint_q")
        state = self._state
        tick = state.tick
        if not state_received:
            return RecoveryTick(self.config.fold_jpos.copy(), False, state.stage, tick)

        local = tick - state.segment_start_tick
        stage = state.stage
        done = False
        if stage is RecoveryStage.FOLD:
            q_target = self._interpolate(
                state.initial_jpos, self.config.fold_jpos, local,
                self._ticks["fold_ramp_s"],
            )
            if local >= self._ticks["fold_ramp_s"] + self._ticks["fold_settle_s"]:
                self._begin_segment(
                    self.config.fold_jpos, tick=tick + 1, stage=RecoveryStage.ABOVE)
        elif stage is RecoveryStage.ABOVE:
            target = self._leg_pose(self.config.above_jpos)
            q_target = self.config.fold_jpos.copy()
            for leg in range(LEG_COUNT):
                if self.config.swing_legs[leg]:
                    sl = slice(leg * JOINTS_PER_LEG, (leg + 1) * JOINTS_PER_LEG)
                    q_target[sl] = self._interpolate(
                        state.initial_jpos[sl], target[sl], local,
                        self._ticks["above_ramp_s"],
                    )
            if local >= self._ticks["above_ramp_s"] + self._ticks["above_settle_s"]:
                q_target = self._leg_pose(self.config.above_jpos)
                self._begin_segment(q_target, tick=tick + 1, stage=RecoveryStage.SWING_DOWN)
        elif stage is RecoveryStage.SWING_DOWN:
            target = self._leg_pose(self.config.swing_down_jpos)
            q_target = self.config.fold_jpos.copy()
            for leg in range(LEG_COUNT):
                if self.config.swing_legs[leg]:
                    sl = slice(leg * JOINTS_PER_LEG, (leg + 1) * JOINTS_PER_LEG)
                    q_target[sl] = self._interpolate(
                        state.initial_jpos[sl], target[sl], local,
                        self._ticks["swing_down_ramp_s"],
                    )
            reached = bool(np.all(
                np.abs(measured.reshape(LEG_COUNT, JOINTS_PER_LEG)[self.config.swing_legs]
                       - self.config.swing_down_jpos.reshape(
                           LEG_COUNT, JOINTS_PER_LEG)[self.config.swing_legs])
                <= self.config.joint_reach_tol
            ))
            if (local >= self._ticks["swing_down_ramp_s"]
                    + self._ticks["swing_down_settle_s"] and reached):
                q_target = target
                self._begin_segment(q_target, tick=tick + 1, stage=RecoveryStage.PUSH)
        else:
            q_target = self._leg_pose(self.config.swing_down_jpos)
            for leg in range(LEG_COUNT):
                if self.config.push_legs[leg]:
                    calf = leg * JOINTS_PER_LEG + 2
                    q_target[calf] = self._interpolate(
                        state.initial_jpos[calf:calf + 1],
                        self.config.push_jpos[calf:calf + 1], local,
                        self._ticks["push_ramp_s"],
                    )[0]
            done = local >= self._ticks["push_ramp_s"] + self._ticks["push_settle_s"]

        # C++ increments tick after every received-state update.  A stage
        # transition already installed its next segment at tick+1.
        current = self._state
        assert current is not None
        if current.tick == tick:
            self._state = RecoveryMotionState(
                tick + 1, current.segment_start_tick, current.stage,
                current.initial_jpos.copy(),
            )
        return RecoveryTick(q_target.copy(), done, stage, tick)


class FixedRecoveryExecutor:
    """Drive the fixed motion through a native 500-Hz MuJoCo environment."""

    def __init__(self, motion: FixedRecoveryMotion):
        if not isinstance(motion, FixedRecoveryMotion):
            raise TypeError("motion must be FixedRecoveryMotion")
        self.motion = motion
        self._started = False
        self._done = False

    @property
    def done(self) -> bool:
        return self._done

    def start(self, env: Any) -> None:
        timestep = float(env.model.opt.timestep)
        if abs(timestep - 1.0 / self.motion.control_hz) > 1e-12:
            raise ValueError(
                "fixed recovery requires one physics step per 500-Hz controller tick")
        self.motion.reset(np.asarray(env.robot_state().joint_q, dtype=np.float32))
        self._started = True
        self._done = False

    def tick(self, env: Any) -> RecoveryExecutionTick:
        if not self._started:
            raise RuntimeError("start must be called before tick")
        if self._done:
            raise RuntimeError("fixed recovery sequence is already complete")
        motion_tick = self.motion.update(
            np.asarray(env.robot_state().joint_q, dtype=np.float32))
        rollout = env.step_recovery_target(
            motion_tick.q_target,
            kp=self.motion.config.kp,
            kd=self.motion.config.kd,
        )
        self._done = motion_tick.done
        return RecoveryExecutionTick(motion_tick, rollout)

    def policy_interval(self, env: Any) -> tuple[RecoveryExecutionTick, ...]:
        """Advance at most one 50-Hz policy interval (ten recovery ticks)."""
        if not self._started:
            raise RuntimeError("start must be called before policy_interval")
        ticks_per_policy = int(round(self.motion.control_hz / env.policy_frequency))
        if ticks_per_policy <= 0 or abs(
                ticks_per_policy * env.policy_frequency - self.motion.control_hz) > 1e-9:
            raise ValueError("policy and recovery frequencies must divide exactly")
        results: list[RecoveryExecutionTick] = []
        for _ in range(ticks_per_policy):
            if self._done:
                break
            results.append(self.tick(env))
        return tuple(results)


__all__ = [
    "FixedRecoveryConfig", "FixedRecoveryExecutor", "FixedRecoveryMotion",
    "RecoveryExecutionTick", "RecoveryMotionState", "RecoveryStage",
    "RecoveryTick",
]
