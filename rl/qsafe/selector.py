"""Fail-closed, ensemble-aware Q-safe candidate selection.

The selector deliberately has no "minimum predicted risk" fallback.  Candidate
zero is normally the task policy's nominal action.  It remains selected unless
the nominal state triggers intervention *and* a non-nominal candidate passes
every preregistered gate.

All uncertainty bounds use paired ensemble members.  In particular, safety
benefit is computed member by member as ``nominal_risk - candidate_risk``
before its lower bound is formed.  This preserves correlations between the two
predictions and matches same-state candidate comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CandidateBatch:
    """Candidate action representations aligned along their first dimension.

    ``mask`` is the externally measured candidate/support mask.  It does not
    bypass any selector gate.  ``requested`` is the actor/reward-Q action-space
    value, ``executed`` is the normalized action after runtime projection, and
    ``q_target`` is the absolute joint target actually sent to the controller.
    """

    requested: np.ndarray
    executed: np.ndarray
    q_target: np.ndarray
    reward_q: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class SelectorConfig:
    """Locked thresholds for :func:`select_candidate`.

    ``uncertainty_beta`` multiplies the sample standard deviation across
    ensemble members.  It is an uncertainty envelope, not a claim that the
    small ensemble is an iid sample suitable for a standard-error interval.
    """

    nominal_risk_lcb_trigger: float
    min_benefit_lcb: float
    max_risk_ucb: float
    max_epistemic_std: float
    max_action_delta_rms: float
    max_q_target_delta_rms: float
    reward_q_margin: float
    uncertainty_beta: float = 1.0

    def __post_init__(self) -> None:
        fields = {
            "nominal_risk_lcb_trigger": self.nominal_risk_lcb_trigger,
            "min_benefit_lcb": self.min_benefit_lcb,
            "max_risk_ucb": self.max_risk_ucb,
            "max_epistemic_std": self.max_epistemic_std,
            "max_action_delta_rms": self.max_action_delta_rms,
            "max_q_target_delta_rms": self.max_q_target_delta_rms,
            "reward_q_margin": self.reward_q_margin,
            "uncertainty_beta": self.uncertainty_beta,
        }
        for name, value in fields.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "nominal_risk_lcb_trigger",
            "min_benefit_lcb",
            "max_risk_ucb",
            "max_epistemic_std",
        ):
            if fields[name] > 1.0:
                raise ValueError(f"{name} must not exceed one")


@dataclass(frozen=True)
class SelectionResult:
    """Selection decision and per-candidate diagnostics.

    ``reason`` is one of ``selected``, ``state_below_trigger``,
    ``no_eligible``, ``nonfinite_input``, ``invalid_risk``,
    ``invalid_mask``, ``nominal_masked``, or ``insufficient_members``.
    """

    selected_index: int
    nominal_index: int
    intervened: bool
    reason: str
    nominal_risk_lcb: float
    risk_mean: np.ndarray
    risk_std: np.ndarray
    risk_ucb: np.ndarray
    benefit_mean: np.ndarray
    benefit_std: np.ndarray
    benefit_lcb: np.ndarray
    requested_delta_rms: np.ndarray
    executed_delta_rms: np.ndarray
    action_delta_rms: np.ndarray
    q_target_delta_rms: np.ndarray
    reward_q_drop: np.ndarray
    support_gate: np.ndarray
    benefit_gate: np.ndarray
    risk_gate: np.ndarray
    uncertainty_gate: np.ndarray
    action_delta_gate: np.ndarray
    q_target_delta_gate: np.ndarray
    reward_q_gate: np.ndarray
    eligible: np.ndarray


def _rms_delta(values: np.ndarray, nominal_index: int) -> np.ndarray:
    delta = values - values[nominal_index]
    axes = tuple(range(1, values.ndim))
    return np.sqrt(np.mean(np.square(delta), axis=axes))


def _empty_result(
    candidate_count: int,
    nominal_index: int,
    reason: str,
) -> SelectionResult:
    nan = np.full(candidate_count, np.nan, dtype=np.float64)
    false = np.zeros(candidate_count, dtype=bool)
    return SelectionResult(
        selected_index=nominal_index,
        nominal_index=nominal_index,
        intervened=False,
        reason=reason,
        nominal_risk_lcb=float("nan"),
        risk_mean=nan.copy(),
        risk_std=nan.copy(),
        risk_ucb=nan.copy(),
        benefit_mean=nan.copy(),
        benefit_std=nan.copy(),
        benefit_lcb=nan.copy(),
        requested_delta_rms=nan.copy(),
        executed_delta_rms=nan.copy(),
        action_delta_rms=nan.copy(),
        q_target_delta_rms=nan.copy(),
        reward_q_drop=nan.copy(),
        support_gate=false.copy(),
        benefit_gate=false.copy(),
        risk_gate=false.copy(),
        uncertainty_gate=false.copy(),
        action_delta_gate=false.copy(),
        q_target_delta_gate=false.copy(),
        reward_q_gate=false.copy(),
        eligible=false.copy(),
    )


def _as_candidate_array(
    name: str,
    value: np.ndarray,
    candidate_count: int,
    *,
    min_dimensions: int,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if array.ndim < min_dimensions or array.shape[0] != candidate_count:
        raise ValueError(
            f"{name} must have candidate dimension {candidate_count} first; "
            f"got shape {array.shape}")
    if any(size == 0 for size in array.shape[1:]):
        raise ValueError(f"{name} must not have an empty action dimension")
    return array


def _mask_array(value: np.ndarray, candidate_count: int) -> tuple[np.ndarray | None, str | None]:
    mask = np.asarray(value)
    if mask.shape != (candidate_count,):
        raise ValueError(
            f"mask must have shape ({candidate_count},); got {mask.shape}")
    if mask.dtype == np.bool_:
        return mask.astype(bool, copy=True), None
    try:
        numeric = np.asarray(mask, dtype=np.float64)
    except (TypeError, ValueError):
        return None, "invalid_mask"
    if not np.all(np.isfinite(numeric)):
        return None, "nonfinite_input"
    if not np.all((numeric == 0.0) | (numeric == 1.0)):
        return None, "invalid_mask"
    return numeric.astype(bool), None


def select_candidate(
    member_risk: np.ndarray,
    candidates: CandidateBatch,
    config: SelectorConfig,
    *,
    nominal_index: int = 0,
) -> SelectionResult:
    """Select a candidate only if every safety and performance gate passes.

    Args:
        member_risk: Array shaped ``(ensemble_members, candidates)`` containing
            fall probabilities.  At least two members are required so an
            epistemic spread can be measured.
        candidates: Aligned requested, executed, q-target, reward-Q, and
            support-mask arrays.
        config: Preregistered gate thresholds.
        nominal_index: Index of the task policy's nominal action.

    Returns:
        A fail-closed decision.  Numeric invalidity, a low nominal state-risk
        trigger, or the absence of a fully eligible alternative all return the
        nominal index.  The selector never falls back to the minimum-risk
        ineligible candidate.

    Raises:
        ValueError: If array ranks or candidate dimensions violate the API.
    """
    try:
        risk = np.asarray(member_risk, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("member_risk must be numeric") from exc
    if risk.ndim != 2 or risk.shape[1] == 0:
        raise ValueError(
            "member_risk must have shape (ensemble_members, candidates)")
    member_count, candidate_count = risk.shape
    if not 0 <= nominal_index < candidate_count:
        raise ValueError("nominal_index is outside the candidate dimension")

    requested = _as_candidate_array(
        "requested", candidates.requested, candidate_count, min_dimensions=2)
    executed = _as_candidate_array(
        "executed", candidates.executed, candidate_count, min_dimensions=2)
    q_target = _as_candidate_array(
        "q_target", candidates.q_target, candidate_count, min_dimensions=2)
    reward_q = _as_candidate_array(
        "reward_q", candidates.reward_q, candidate_count, min_dimensions=1)
    if reward_q.ndim != 1:
        raise ValueError(
            f"reward_q must have shape ({candidate_count},); got {reward_q.shape}")
    if requested.shape != executed.shape:
        raise ValueError(
            "requested and executed must have identical action shapes; got "
            f"{requested.shape} and {executed.shape}")

    mask, mask_error = _mask_array(candidates.mask, candidate_count)
    if mask_error is not None:
        return _empty_result(candidate_count, nominal_index, mask_error)
    assert mask is not None
    if member_count < 2:
        return _empty_result(
            candidate_count, nominal_index, "insufficient_members")
    if not all(np.all(np.isfinite(array)) for array in (
        risk, requested, executed, q_target, reward_q,
    )):
        return _empty_result(candidate_count, nominal_index, "nonfinite_input")
    if np.any((risk < 0.0) | (risk > 1.0)):
        return _empty_result(candidate_count, nominal_index, "invalid_risk")
    if not bool(mask[nominal_index]):
        return _empty_result(candidate_count, nominal_index, "nominal_masked")

    risk_mean = np.mean(risk, axis=0)
    risk_std = np.std(risk, axis=0, ddof=1)
    risk_ucb = risk_mean + config.uncertainty_beta * risk_std
    nominal_risk_lcb = float(
        risk_mean[nominal_index]
        - config.uncertainty_beta * risk_std[nominal_index])

    benefit_members = risk[:, nominal_index, None] - risk
    benefit_mean = np.mean(benefit_members, axis=0)
    benefit_std = np.std(benefit_members, axis=0, ddof=1)
    benefit_lcb = benefit_mean - config.uncertainty_beta * benefit_std

    requested_delta_rms = _rms_delta(requested, nominal_index)
    executed_delta_rms = _rms_delta(executed, nominal_index)
    # Both representations must be local.  Taking the maximum avoids accepting
    # a large proposal merely because the runtime projection contracted it.
    action_delta_rms = np.maximum(
        requested_delta_rms, executed_delta_rms)
    q_target_delta_rms = _rms_delta(q_target, nominal_index)
    reward_q_drop = reward_q[nominal_index] - reward_q

    support_gate = mask.copy()
    benefit_gate = benefit_lcb > config.min_benefit_lcb
    risk_gate = risk_ucb <= config.max_risk_ucb
    uncertainty_gate = risk_std <= config.max_epistemic_std
    action_delta_gate = action_delta_rms <= config.max_action_delta_rms
    q_target_delta_gate = (
        q_target_delta_rms <= config.max_q_target_delta_rms)
    reward_q_gate = reward_q_drop <= config.reward_q_margin
    eligible = (
        support_gate
        & benefit_gate
        & risk_gate
        & uncertainty_gate
        & action_delta_gate
        & q_target_delta_gate
        & reward_q_gate
    )
    eligible[nominal_index] = False

    result_fields = dict(
        nominal_index=nominal_index,
        nominal_risk_lcb=nominal_risk_lcb,
        risk_mean=risk_mean,
        risk_std=risk_std,
        risk_ucb=risk_ucb,
        benefit_mean=benefit_mean,
        benefit_std=benefit_std,
        benefit_lcb=benefit_lcb,
        requested_delta_rms=requested_delta_rms,
        executed_delta_rms=executed_delta_rms,
        action_delta_rms=action_delta_rms,
        q_target_delta_rms=q_target_delta_rms,
        reward_q_drop=reward_q_drop,
        support_gate=support_gate,
        benefit_gate=benefit_gate,
        risk_gate=risk_gate,
        uncertainty_gate=uncertainty_gate,
        action_delta_gate=action_delta_gate,
        q_target_delta_gate=q_target_delta_gate,
        reward_q_gate=reward_q_gate,
        eligible=eligible,
    )
    if nominal_risk_lcb < config.nominal_risk_lcb_trigger:
        return SelectionResult(
            selected_index=nominal_index,
            intervened=False,
            reason="state_below_trigger",
            **result_fields,
        )

    indices = np.flatnonzero(eligible)
    if indices.size == 0:
        return SelectionResult(
            selected_index=nominal_index,
            intervened=False,
            reason="no_eligible",
            **result_fields,
        )

    # Primary objective: smallest conservative risk upper bound.  Remaining
    # keys make all ties deterministic without relying on candidate order in a
    # platform-dependent reduction: larger benefit LCB, larger reward-Q, then
    # the smaller explicit candidate index.
    order = np.lexsort((
        indices,
        -reward_q[indices],
        -benefit_lcb[indices],
        risk_ucb[indices],
    ))
    selected_index = int(indices[order[0]])
    return SelectionResult(
        selected_index=selected_index,
        intervened=True,
        reason="selected",
        **result_fields,
    )


__all__ = [
    "CandidateBatch",
    "SelectionResult",
    "SelectorConfig",
    "select_candidate",
]
