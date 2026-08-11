"""Structural contract between target-aligned MjLab and native SAC MuJoCo.

The Warp/native trajectory parity test compares two simulators using one
compiled MjLab model.  This module closes the other half of the gate: it proves
that the compiled proposal model has the same robot mechanics and collision
model as the native MuJoCo model used by SAC.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from safety_data.mjlab_natural_falls import MJLAB_TO_TARGET_JOINT
from safety_data.mjlab_target_alignment import TARGET_KD, TARGET_KP


def _name(mujoco: Any, model: Any, kind: Any, index: int) -> str | None:
    return mujoco.mj_id2name(model, kind, index)


def _plain_name(name: str | None) -> str | None:
    if name is None:
        return None
    return name.removeprefix("robot/")


def _exact(name: str, left: Any, right: Any, failures: list[str], *,
           atol: float = 1e-12) -> None:
    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape or not np.allclose(a, b, rtol=0.0, atol=atol):
        maximum = None
        if a.shape == b.shape and a.size:
            maximum = float(np.max(np.abs(a.astype(float) - b.astype(float))))
        failures.append(f"{name}: shape={a.shape}/{b.shape}, max_error={maximum}")


def _named_ids(mujoco: Any, model: Any, kind: Any, count: int) -> dict[str, int]:
    result: dict[str, int] = {}
    for index in range(count):
        name = _plain_name(_name(mujoco, model, kind, index))
        if name is not None:
            result[name] = index
    return result


def validate_compiled_target_model(
    mjlab_model: Any,
    target_model: Any,
    *,
    action_joint_names: Iterable[str],
) -> dict[str, Any]:
    """Return a fail-closed, JSON-ready structural alignment report."""
    import mujoco

    failures: list[str] = []
    dimensions = {key: [int(getattr(mjlab_model, key)),
                        int(getattr(target_model, key))]
                  for key in ("nq", "nv", "nu", "njnt", "ngeom")}
    for key, values in dimensions.items():
        if values[0] != values[1]:
            failures.append(f"dimension {key}: {values[0]}/{values[1]}")

    option_fields = (
        "timestep", "integrator", "cone", "solver", "iterations",
        "tolerance", "impratio", "ls_iterations", "ls_tolerance",
        "ccd_iterations",
    )
    for field in option_fields:
        _exact(f"option.{field}", getattr(mjlab_model.opt, field),
               getattr(target_model.opt, field), failures)
    # Dense versus sparse Jacobian storage changes representation, not the
    # equations.  MjLab requires dense for stable Warp Newton integration.
    jacobian_exception = {
        "mjlab": int(mjlab_model.opt.jacobian),
        "target": int(target_model.opt.jacobian),
        "accepted_reason": "dense_and_sparse_are_algebraically_equivalent",
    }

    joint_kind = mujoco.mjtObj.mjOBJ_JOINT
    mjlab_joints = _named_ids(mujoco, mjlab_model, joint_kind, mjlab_model.njnt)
    target_joints = _named_ids(mujoco, target_model, joint_kind, target_model.njnt)
    target_motor_joints = [
        _name(mujoco, target_model, joint_kind,
              int(target_model.actuator_trnid[index, 0]))
        for index in range(target_model.nu)
    ]
    if any(name is None for name in target_motor_joints):
        failures.append("target actuator references an unnamed joint")
    motor_names = [str(name) for name in target_motor_joints]
    if set(motor_names) != set(mjlab_joints) - {"floating_base_joint"}:
        failures.append("MjLab and target motor-joint name sets differ")

    joint_fields = ("jnt_type", "jnt_pos", "jnt_axis", "jnt_range")
    dof_fields = ("dof_damping", "dof_armature", "dof_frictionloss")
    for name in motor_names:
        left_id = mjlab_joints.get(name)
        right_id = target_joints.get(name)
        if left_id is None or right_id is None:
            failures.append(f"missing joint {name}")
            continue
        for field in joint_fields:
            _exact(f"joint.{name}.{field}", getattr(mjlab_model, field)[left_id],
                   getattr(target_model, field)[right_id], failures)
        left_dof = int(mjlab_model.jnt_dofadr[left_id])
        right_dof = int(target_model.jnt_dofadr[right_id])
        for field in dof_fields:
            _exact(f"joint.{name}.{field}", getattr(mjlab_model, field)[left_dof],
                   getattr(target_model, field)[right_dof], failures)
    _exact("free_joint.damping", mjlab_model.dof_damping[:6],
           target_model.dof_damping[:6], failures)

    body_kind = mujoco.mjtObj.mjOBJ_BODY
    mjlab_bodies = _named_ids(mujoco, mjlab_model, body_kind, mjlab_model.nbody)
    target_bodies = _named_ids(mujoco, target_model, body_kind, target_model.nbody)
    robot_body_names = sorted(
        name for name in mjlab_bodies
        if name not in {"world", "terrain"})
    body_fields = ("body_pos", "body_quat", "body_mass", "body_inertia",
                   "body_ipos", "body_iquat")
    for name in robot_body_names:
        if name not in target_bodies:
            failures.append(f"target model is missing robot body {name}")
            continue
        for field in body_fields:
            _exact(f"body.{name}.{field}",
                   getattr(mjlab_model, field)[mjlab_bodies[name]],
                   getattr(target_model, field)[target_bodies[name]], failures)

    # Both source MJCFs preserve geometry order.  Visual-only geoms are
    # deliberately ignored: they have contype=conaffinity=0 and cannot affect
    # safety outcomes.  Every active robot collision geom must match exactly.
    active_left = np.flatnonzero(
        (mjlab_model.geom_contype != 0) | (mjlab_model.geom_conaffinity != 0))
    active_right = np.flatnonzero(
        (target_model.geom_contype != 0) | (target_model.geom_conaffinity != 0))
    robot_left = active_left[active_left != 0]
    robot_right = active_right[active_right != 0]
    _exact("active_robot_geom_indices", robot_left, robot_right, failures)
    geom_fields = (
        "geom_type", "geom_size", "geom_pos", "geom_quat", "geom_condim",
        "geom_priority", "geom_friction", "geom_margin", "geom_contype",
        "geom_conaffinity",
    )
    if robot_left.shape == robot_right.shape:
        for field in geom_fields:
            _exact(f"active_collision.{field}",
                   getattr(mjlab_model, field)[robot_left],
                   getattr(target_model, field)[robot_right], failures)

    actuator_kind = mujoco.mjtObj.mjOBJ_ACTUATOR
    mjlab_actuator_by_joint: dict[str, int] = {}
    for index in range(mjlab_model.nu):
        joint_id = int(mjlab_model.actuator_trnid[index, 0])
        joint_name = _plain_name(_name(mujoco, mjlab_model, joint_kind, joint_id))
        if joint_name is not None:
            mjlab_actuator_by_joint[joint_name] = index
    for target_id, name in enumerate(motor_names):
        left_id = mjlab_actuator_by_joint.get(name)
        if left_id is None:
            failures.append(f"missing MjLab actuator for {name}")
            continue
        _exact(f"actuator.{name}.effort",
               mjlab_model.actuator_forcerange[left_id],
               target_model.actuator_ctrlrange[target_id], failures)
        _exact(f"actuator.{name}.gain_kp",
               mjlab_model.actuator_gainprm[left_id, 0], TARGET_KP, failures)
        _exact(f"actuator.{name}.bias_kp",
               mjlab_model.actuator_biasprm[left_id, 1], -TARGET_KP, failures)
        _exact(f"actuator.{name}.bias_kd",
               mjlab_model.actuator_biasprm[left_id, 2], -TARGET_KD, failures)
        if not bool(mjlab_model.actuator_forcelimited[left_id]):
            failures.append(f"actuator {name} is not effort limited")
        if not bool(target_model.actuator_ctrllimited[target_id]):
            failures.append(f"target motor {name} is not control limited")

    source_action_names = [_plain_name(name) for name in action_joint_names]
    if len(source_action_names) != 12:
        failures.append("MjLab action does not contain exactly 12 joints")
        mapped_action_names: list[str | None] = []
    else:
        mapped_action_names = [source_action_names[index]
                               for index in MJLAB_TO_TARGET_JOINT]
        if mapped_action_names != motor_names:
            failures.append("MjLab-to-target action permutation is incorrect")

    return {
        "schema_version": "qsafe.mjlab_target_model_contract.v1",
        "pass": not failures,
        "failures": failures,
        "dimensions": dimensions,
        "jacobian_storage_exception": jacobian_exception,
        "target_action_joint_order": motor_names,
        "mapped_mjlab_action_joint_order": mapped_action_names,
        "robot_body_count_compared": len(robot_body_names),
        "active_robot_collision_geoms_compared": int(len(robot_left)),
        "motor_joints_compared": len(motor_names),
        "visual_only_geometries_ignored": True,
    }
