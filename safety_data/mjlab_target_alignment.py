"""Target-aligned MjLab Go2 configuration for natural PPO collection.

The upstream ``Unitree-Go2-Flat`` task is a useful locomotion implementation,
but its default posture, action range, PD gains and MuJoCo integration settings
do not match the native SAC task in this repository.  Natural PPO states are
only useful as direct Q_safe supervision after those application semantics are
made explicit and aligned.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from typing import Any


TARGET_COMMAND = (0.30, 0.0, 0.0)
TARGET_POLICY_FREQUENCY_HZ = 50.0
TARGET_PHYSICS_TIMESTEP_S = 0.002
TARGET_DECIMATION = 10
TARGET_EPISODE_POLICY_STEPS = 500
TARGET_INIT_HEIGHT_M = 0.445
TARGET_INIT_JOINT = {
    ".*hip_joint": 0.05,
    ".*thigh_joint": 0.70,
    ".*calf_joint": -1.40,
}
TARGET_ACTION_SCALE = {
    ".*hip_joint": 0.20,
    ".*thigh_joint": 0.40,
    ".*calf_joint": 0.40,
}
TARGET_KP = 60.0
TARGET_KD = 5.0
TARGET_PASSIVE_JOINT_DAMPING = 0.1
TARGET_FRICTIONLOSS = 0.2
TARGET_ARMATURE = 0.01
TARGET_HIP_THIGH_EFFORT = 23.7
TARGET_CALF_EFFORT = 45.43


def target_alignment_manifest() -> dict[str, Any]:
    payload = {
        "schema_version": "qsafe.mjlab_target_alignment.v1",
        "command": {
            "vx": TARGET_COMMAND[0],
            "vy": TARGET_COMMAND[1],
            "yaw_rate": TARGET_COMMAND[2],
        },
        "policy_frequency_hz": TARGET_POLICY_FREQUENCY_HZ,
        "physics_timestep_s": TARGET_PHYSICS_TIMESTEP_S,
        "physics_substeps_per_policy_step": TARGET_DECIMATION,
        "episode_policy_steps": TARGET_EPISODE_POLICY_STEPS,
        "initial_base_height_m": TARGET_INIT_HEIGHT_M,
        "initial_joint_position": TARGET_INIT_JOINT,
        "normalized_action_scale": TARGET_ACTION_SCALE,
        "normalized_action_offset": "initial_joint_position",
        "pd": {"kp": TARGET_KP, "kd": TARGET_KD},
        "passive_joint_damping": TARGET_PASSIVE_JOINT_DAMPING,
        "joint_frictionloss": TARGET_FRICTIONLOSS,
        "joint_armature": TARGET_ARMATURE,
        "effort_limit": {
            "hip_and_thigh": TARGET_HIP_THIGH_EFFORT,
            "calf": TARGET_CALF_EFFORT,
        },
        "mujoco_option": {
            "integrator": "euler",
            "cone": "elliptic",
            # Dense and sparse are algebraically equivalent here.  Dense is
            # required because MuJoCo-Warp's sparse Newton path accumulates
            # materially larger native/Warp drift on this 18-DoF model.
            "jacobian": "dense",
            "solver": "newton",
            "iterations": 100,
            "tolerance": 1e-8,
            "impratio": 100.0,
            "ls_iterations": 50,
            "ls_tolerance": 0.01,
            "ccd_iterations": 35,
        },
        "fall_predicate": {
            "minimum_base_height_m": 0.18,
            "maximum_abs_roll_or_pitch_rad": 1.047198,
        },
        "external_push": False,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return payload | {
        "contract_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def target_fall_termination(env: Any):
    """Exact native height/roll/pitch terminal predicate for MjLab."""
    from safety_data.mjlab_natural_falls import target_fall_predicate

    return target_fall_predicate(env.sim.data.qpos)


def configure_target_aligned_go2(cfg: Any) -> Any:
    """Mutate one fresh ``Unitree-Go2-Flat`` config to target SAC semantics."""
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.managers import TerminationTermCfg

    cfg.sim.mujoco.timestep = TARGET_PHYSICS_TIMESTEP_S
    cfg.sim.mujoco.integrator = "euler"
    cfg.sim.mujoco.cone = "elliptic"
    cfg.sim.mujoco.jacobian = "dense"
    cfg.sim.mujoco.solver = "newton"
    cfg.sim.mujoco.iterations = 100
    cfg.sim.mujoco.tolerance = 1e-8
    cfg.sim.mujoco.impratio = 100.0
    cfg.sim.mujoco.ls_iterations = 50
    cfg.sim.mujoco.ls_tolerance = 0.01
    cfg.sim.mujoco.ccd_iterations = 35
    cfg.decimation = TARGET_DECIMATION
    cfg.episode_length_s = (
        TARGET_EPISODE_POLICY_STEPS / TARGET_POLICY_FREQUENCY_HZ)

    robot = cfg.scene.entities["robot"]
    robot.init_state.pos = (0.0, 0.0, TARGET_INIT_HEIGHT_M)
    robot.init_state.joint_pos = dict(TARGET_INIT_JOINT)
    robot.init_state.joint_vel = {".*": 0.0}

    upstream_spec_fn = robot.spec_fn

    def target_spec_fn():
        spec = upstream_spec_fn()
        for joint in spec.joints:
            if joint.name and joint.name.endswith("_joint"):
                joint.damping = TARGET_PASSIVE_JOINT_DAMPING
        return spec

    robot.spec_fn = target_spec_fn

    if robot.articulation is None:
        raise ValueError("Go2 target alignment requires articulated actuators")
    aligned_actuators = []
    for actuator in robot.articulation.actuators:
        if not isinstance(actuator, BuiltinPositionActuatorCfg):
            raise TypeError("Go2 target alignment requires position actuators")
        names = tuple(actuator.target_names_expr)
        calf = any("calf" in name for name in names)
        aligned_actuators.append(replace(
            actuator,
            stiffness=TARGET_KP,
            damping=TARGET_KD,
            effort_limit=(
                TARGET_CALF_EFFORT if calf else TARGET_HIP_THIGH_EFFORT),
            armature=TARGET_ARMATURE,
            frictionloss=TARGET_FRICTIONLOSS,
        ))
    robot.articulation.actuators = tuple(aligned_actuators)

    action = cfg.actions["joint_pos"]
    action.scale = dict(TARGET_ACTION_SCALE)
    action.offset = 0.0
    action.use_default_offset = True

    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    twist = cfg.commands["twist"]
    twist.rel_standing_envs = 0.0
    twist.heading_command = False
    twist.ranges.lin_vel_x = (TARGET_COMMAND[0], TARGET_COMMAND[0])
    twist.ranges.lin_vel_y = (TARGET_COMMAND[1], TARGET_COMMAND[1])
    twist.ranges.ang_vel_z = (TARGET_COMMAND[2], TARGET_COMMAND[2])
    twist.ranges.heading = None

    time_out = cfg.terminations["time_out"]
    cfg.terminations = {
        "time_out": time_out,
        "target_fall": TerminationTermCfg(
            func=target_fall_termination, params={}, time_out=False),
    }
    return cfg


def validate_target_aligned_go2(cfg: Any) -> None:
    """Fail closed if a runner silently drifts from the target contract."""
    if float(cfg.sim.mujoco.timestep) != TARGET_PHYSICS_TIMESTEP_S or (
            int(cfg.decimation) != TARGET_DECIMATION):
        raise ValueError("target-aligned MjLab timing drifted")
    if abs(float(cfg.episode_length_s) * TARGET_POLICY_FREQUENCY_HZ
           - TARGET_EPISODE_POLICY_STEPS) > 1e-9:
        raise ValueError("target-aligned episode duration drifted")
    if "push_robot" in cfg.events or set(cfg.terminations) != {
            "time_out", "target_fall"}:
        raise ValueError("target-aligned force or termination contract drifted")
    twist = cfg.commands["twist"]
    realized = (
        tuple(twist.ranges.lin_vel_x), tuple(twist.ranges.lin_vel_y),
        tuple(twist.ranges.ang_vel_z),
    )
    expected = tuple((value, value) for value in TARGET_COMMAND)
    if realized != expected or float(twist.rel_standing_envs) != 0.0:
        raise ValueError("target-aligned command distribution drifted")
    robot = cfg.scene.entities["robot"]
    if robot.init_state.joint_pos != TARGET_INIT_JOINT or tuple(
            robot.init_state.pos) != (0.0, 0.0, TARGET_INIT_HEIGHT_M):
        raise ValueError("target-aligned initial pose drifted")
    action = cfg.actions["joint_pos"]
    if action.scale != TARGET_ACTION_SCALE or not action.use_default_offset:
        raise ValueError("target-aligned action application drifted")
