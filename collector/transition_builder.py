"""Build explicit transition records from environment step results."""

from __future__ import annotations

import numpy as np

from common.transition import TerminationReason, Transition, zero_costs


def reason_from_info(info: dict) -> TerminationReason:
    if info.get('standup_timed_out'):
        return TerminationReason.RECOVERY_FAILED
    if info.get('terminated') and info.get('is_belly_up'):
        return TerminationReason.EXCESSIVE_TILT
    if info.get('terminated'):
        return TerminationReason.EXCESSIVE_TILT
    if info.get('truncated'):
        return TerminationReason.TIME_LIMIT
    return TerminationReason.NONE


def build_transition(observation: np.ndarray,
                     action: np.ndarray,
                     reward: float,
                     next_observation: np.ndarray,
                     done: bool,
                     info: dict,
                     *,
                     projected_action: np.ndarray | None = None,
                     executed_q_target: np.ndarray | None = None,
                     policy_version: int = 0) -> Transition:
    action = np.asarray(action, dtype=np.float32)
    projected = action if projected_action is None else np.asarray(
        projected_action, dtype=np.float32)
    executed = np.zeros_like(action) if executed_q_target is None else np.asarray(
        executed_q_target, dtype=np.float32)
    return Transition(
        observation=np.asarray(observation, dtype=np.float32),
        requested_action=action,
        projected_action=projected,
        executed_q_target=executed,
        reward=float(reward),
        costs=zero_costs(),
        next_observation=np.asarray(next_observation, dtype=np.float32),
        terminated=bool(info.get('terminated', done and not info.get('truncated'))),
        truncated=bool(info.get('truncated', False)),
        termination_reason=reason_from_info(info),
        intervention_mask=bool(info.get('intervention_mask', False)),
        policy_version=policy_version,
    )
