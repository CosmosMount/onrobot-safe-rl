"""Quota-controlled snapshot selection for active counterfactual collection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


ACTIVE_REASONS = (
    'near_failure',
    'disagreement',
    'risk_boundary',
    'support_boundary',
    'selector_decision',
    'normal',
)


@dataclass(frozen=True)
class ActiveSnapshotSignals:
    """Cheap probe diagnostics computed before an expensive branch rollout."""

    near_failure: bool = False
    max_disagreement: float = 0.0
    min_risk_boundary_distance: float = float('inf')
    min_support_boundary_distance: float = float('inf')
    would_replace: bool = False
    would_abstain: bool = False
    stable_normal: bool = False


def build_active_snapshot_signals(
        selector_risks,
        *,
        validator_risks=None,
        supported=None,
        behavior_log_prob_per_dim=None,
        action_distances=None,
        epsilon: float = 0.2,
        min_behavior_log_prob_per_dim: float = -4.0,
        max_nominal_action_distance: float = 1.0,
        improvement_margin: float = 0.02,
        normal_risk_max: float = 0.1,
        near_failure: bool = False,
) -> ActiveSnapshotSignals:
    """Summarize candidate probes using the same A-select/B-validate policy."""
    selector = np.asarray(selector_risks, dtype=np.float64).reshape(-1)
    if not len(selector):
        raise ValueError('active snapshot probe needs candidate risks')
    validator = (
        selector if validator_risks is None
        else np.asarray(validator_risks, dtype=np.float64).reshape(-1))
    if len(validator) != len(selector):
        raise ValueError('validator risks must align with selector risks')
    support = (
        np.ones(len(selector), dtype=bool)
        if supported is None
        else np.asarray(supported, dtype=bool).reshape(-1))
    if len(support) != len(selector):
        raise ValueError('support must align with candidate risks')

    relevant = support if np.any(support) else np.ones_like(support)
    boundary_distance = float(min(
        np.min(np.abs(selector[relevant] - epsilon)),
        np.min(np.abs(validator[relevant] - epsilon)),
    ))
    disagreement = float(np.max(
        np.abs(selector[relevant] - validator[relevant])))

    support_boundary_distance = float('inf')
    if (behavior_log_prob_per_dim is not None
            and action_distances is not None):
        log_probability = np.asarray(
            behavior_log_prob_per_dim, dtype=np.float64).reshape(-1)
        distances = np.asarray(
            action_distances, dtype=np.float64).reshape(-1)
        if len(log_probability) != len(selector):
            raise ValueError('behavior log probabilities must align')
        if len(distances) != len(selector):
            raise ValueError('action distances must align')
        log_scale = max(abs(min_behavior_log_prob_per_dim), 1.0)
        distance_scale = max(max_nominal_action_distance, 1e-6)
        support_boundary_distance = float(min(
            np.min(np.abs(
                log_probability - min_behavior_log_prob_per_dim)
                / log_scale),
            np.min(np.abs(
                distances - max_nominal_action_distance)
                / distance_scale),
        ))

    nominal_safe = bool(
        support[0]
        and selector[0] <= epsilon
        and validator[0] <= epsilon)
    would_replace = False
    would_abstain = False
    if not nominal_safe:
        eligible = np.flatnonzero(
            support & (selector <= epsilon)
            & (np.arange(len(selector)) != 0))
        if not len(eligible):
            would_abstain = True
        else:
            selected = int(eligible[np.argmin(selector[eligible])])
            would_replace = bool(
                validator[selected] <= epsilon
                and selector[0] - selector[selected] >= improvement_margin
                and validator[0] - validator[selected]
                >= improvement_margin)
            would_abstain = not would_replace
    stable_normal = bool(
        not near_failure
        and support[0]
        and selector[0] <= normal_risk_max
        and validator[0] <= normal_risk_max)
    return ActiveSnapshotSignals(
        near_failure=bool(near_failure),
        max_disagreement=disagreement,
        min_risk_boundary_distance=boundary_distance,
        min_support_boundary_distance=support_boundary_distance,
        would_replace=would_replace,
        would_abstain=would_abstain,
        stable_normal=stable_normal,
    )


@dataclass
class ActiveBranchSampler:
    """Choose informative snapshots without starving normal-state coverage."""

    quota_per_reason: int = 40
    normal_quota: int = 80
    risk_boundary_width: float = 0.05
    disagreement_threshold: float = 0.15
    support_boundary_width: float = 0.10
    min_snapshot_gap: int = 2
    normal_interval: int = 10
    counts: dict[str, int] = field(default_factory=dict)
    last_selected_step: int = field(default=-1_000_000, init=False)

    def __post_init__(self):
        if self.quota_per_reason < 0 or self.normal_quota < 0:
            raise ValueError('active snapshot quotas must be non-negative')
        if self.min_snapshot_gap < 0:
            raise ValueError('min_snapshot_gap must be non-negative')
        if self.normal_interval <= 0:
            raise ValueError('normal_interval must be positive')
        for reason in ACTIVE_REASONS:
            self.counts.setdefault(reason, 0)

    def quota(self, reason: str) -> int:
        if reason not in ACTIVE_REASONS:
            raise ValueError(f'unknown active snapshot reason: {reason}')
        return (
            self.normal_quota if reason == 'normal'
            else self.quota_per_reason)

    def triggered_reasons(
            self, step: int,
            signals: ActiveSnapshotSignals) -> list[str]:
        reasons = []
        if signals.near_failure:
            reasons.append('near_failure')
        if signals.max_disagreement >= self.disagreement_threshold:
            reasons.append('disagreement')
        if (signals.min_risk_boundary_distance
                <= self.risk_boundary_width):
            reasons.append('risk_boundary')
        if (signals.min_support_boundary_distance
                <= self.support_boundary_width):
            reasons.append('support_boundary')
        if signals.would_replace or signals.would_abstain:
            reasons.append('selector_decision')
        if (signals.stable_normal
                and step % self.normal_interval == 0):
            reasons.append('normal')
        return reasons

    def consider(
            self, step: int,
            signals: ActiveSnapshotSignals,
    ) -> tuple[str | None, list[str]]:
        """Return the highest-priority available reason and all triggers."""
        triggered = self.triggered_reasons(step, signals)
        if step - self.last_selected_step < self.min_snapshot_gap:
            return None, triggered
        selected = next((
            reason for reason in ACTIVE_REASONS
            if reason in triggered
            and self.counts[reason] < self.quota(reason)
        ), None)
        if selected is not None:
            self.counts[selected] += 1
            self.last_selected_step = int(step)
        return selected, triggered

    @property
    def full(self) -> bool:
        return all(
            self.counts[reason] >= self.quota(reason)
            for reason in ACTIVE_REASONS
            if self.quota(reason) > 0)

    def summary(self) -> dict[str, object]:
        return {
            'counts': dict(self.counts),
            'quotas': {
                reason: self.quota(reason) for reason in ACTIVE_REASONS
            },
            'full': self.full,
        }
