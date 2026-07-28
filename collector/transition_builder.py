"""Build explicit transition records from environment step results."""

from __future__ import annotations

import numpy as np

from common.transition import COST_KEYS, TerminationReason, Transition, zero_costs


def reason_from_info(info: dict) -> TerminationReason:
    explicit = info.get('termination_reason')
    if explicit is not None:
        try:
            return TerminationReason(int(explicit))
        except (TypeError, ValueError):
            pass
    if info.get('standup_timed_out'):
        return TerminationReason.RECOVERY_FAILED
    if info.get('terminated') and info.get('is_belly_up'):
        return TerminationReason.EXCESSIVE_TILT
    if info.get('terminated'):
        return TerminationReason.EXCESSIVE_TILT
    if info.get('truncated'):
        return TerminationReason.TIME_LIMIT
    return TerminationReason.NONE


def costs_from_info(info: dict) -> dict[str, float]:
    costs = zero_costs()
    supplied = info.get('costs') or {}
    for key in COST_KEYS:
        value = float(supplied.get(key, info.get(key, 0.0)))
        costs[key] = value if np.isfinite(value) else 0.0
    return costs


def build_transition(observation: np.ndarray,
                     action: np.ndarray,
                     reward: float,
                     next_observation: np.ndarray,
                     done: bool,
                     info: dict,
                     *,
                     projected_action: np.ndarray | None = None,
                     executed_q_target: np.ndarray | None = None,
                     policy_version: int = 0,
                     episode_id: int = 0,
                     command_speed: float | None = None) -> Transition:
    action = np.asarray(action, dtype=np.float32)
    projected = action if projected_action is None else np.asarray(
        projected_action, dtype=np.float32)
    executed = np.zeros_like(action) if executed_q_target is None else np.asarray(
        executed_q_target, dtype=np.float32)
    reason = reason_from_info(info)
    unsafe_reasons = {
        TerminationReason.EXCESSIVE_TILT,
        TerminationReason.JOINT_LIMIT,
        TerminationReason.MOTOR_FAULT,
        TerminationReason.RECOVERY_FAILED,
    }
    unsafe = bool(info.get('unsafe_label', False)
                  or reason in unsafe_reasons
                  or info.get('is_belly_up')
                  or info.get('hard_fall'))
    if command_speed is None:
        command_speed = info.get(
            'cmd_speed',
            float(observation[-1]) if np.asarray(observation).size else 0.0)
    command_speed = float(command_speed)
    if not np.isfinite(command_speed):
        command_speed = 0.0
    return Transition(
        observation=np.asarray(observation, dtype=np.float32),
        requested_action=action,
        projected_action=projected,
        executed_q_target=executed,
        reward=float(reward),
        costs=costs_from_info(info),
        next_observation=np.asarray(next_observation, dtype=np.float32),
        terminated=bool(info.get('terminated', done and not info.get('truncated'))),
        truncated=bool(info.get('truncated', False)),
        termination_reason=reason,
        intervention_mask=bool(info.get('intervention_mask', False)),
        unsafe_label=unsafe,
        near_failure_label=bool(info.get('near_failure_label', False)
                                or unsafe),
        policy_version=int(policy_version),
        episode_id=int(episode_id),
        command_speed=command_speed,
    )
