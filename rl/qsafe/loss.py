"""Replica-aware, group-macro objectives for Selective Advantage Q_safe."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from rl.qsafe.network import QSafeOutput


@dataclass(frozen=True)
class QSafeLossConfig:
    absolute_risk_weight: float = 1.0
    state_risk_weight: float = 0.5
    relative_risk_weight: float = 1.0
    ranking_weight: float = 0.5
    ttf_weight: float = 0.1
    max_tilt_weight: float = 0.1
    min_height_weight: float = 0.1

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or (
                    not math.isfinite(value)) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class QSafeLossResult:
    total: torch.Tensor
    absolute_risk: torch.Tensor
    state_risk: torch.Tensor
    relative_risk: torch.Tensor
    ranking: torch.Tensor
    ttf: torch.Tensor
    max_tilt: torch.Tensor
    min_height: torch.Tensor
    ranked_groups: int
    ranked_pairs: int


def _weighted_group_mean(
    values: torch.Tensor,
    group_weight: torch.Tensor,
    valid_group: torch.Tensor | None = None,
) -> torch.Tensor:
    if values.ndim != 1 or group_weight.shape != values.shape:
        raise ValueError("group values and weights must be vectors with equal shape")
    if not bool(torch.all(torch.isfinite(group_weight))) or not bool(
            torch.all(group_weight > 0.0)):
        raise ValueError("group weights must be finite and positive")
    valid = torch.ones_like(values, dtype=torch.bool)
    if valid_group is not None:
        if valid_group.shape != values.shape:
            raise ValueError("valid_group must match group values")
        valid = valid_group.to(dtype=torch.bool)
    if not bool(torch.any(valid)):
        return torch.where(torch.isfinite(values), values, torch.zeros_like(values)).sum() * 0.0
    if not bool(torch.all(torch.isfinite(values[valid]))):
        raise ValueError("non-finite loss in a valid group")
    weight = group_weight[valid]
    return torch.sum(values[valid] * weight) / torch.sum(weight)


def _masked_candidate_group_mean(
    values: torch.Tensor,
    candidate_mask: torch.Tensor,
    group_weight: torch.Tensor,
) -> torch.Tensor:
    masked = torch.where(candidate_mask, values, torch.zeros_like(values))
    count = candidate_mask.sum(dim=1).clamp_min(1)
    per_group = masked.sum(dim=1) / count
    return _weighted_group_mean(per_group, group_weight)


def _ranking_loss(
    risk_logits: torch.Tensor,
    empirical_risk: torch.Tensor,
    candidate_mask: torch.Tensor,
    group_weight: torch.Tensor,
    minimum_gap: float,
) -> tuple[torch.Tensor, int, int]:
    candidates = risk_logits.shape[1]
    pair_indices = torch.triu_indices(
        candidates,
        candidates,
        offset=1,
        device=risk_logits.device,
    )
    left, right = pair_indices.unbind(dim=0)
    target_delta = empirical_risk[:, left] - empirical_risk[:, right]
    pair_mask = (
        candidate_mask[:, left].to(dtype=torch.bool)
        & candidate_mask[:, right].to(dtype=torch.bool)
        & (torch.abs(target_delta) >= minimum_gap)
    )
    predicted_delta = risk_logits[:, left] - risk_logits[:, right]
    safe_target_delta = torch.where(
        pair_mask, target_delta, torch.zeros_like(target_delta))
    safe_predicted_delta = torch.where(
        pair_mask, predicted_delta, torch.zeros_like(predicted_delta))
    pair_losses = F.softplus(
        -torch.sign(safe_target_delta) * safe_predicted_delta
    )
    pair_count_per_group = pair_mask.sum(dim=1)
    valid_groups = pair_count_per_group > 0
    group_losses = torch.where(
        pair_mask,
        pair_losses,
        torch.zeros_like(pair_losses),
    ).sum(dim=1) / pair_count_per_group.clamp_min(1)
    ranked_groups, pair_count = torch.stack((
        valid_groups.sum(), pair_count_per_group.sum(),
    )).tolist()
    if pair_count == 0:
        # Match the loop implementation's detached scalar for an empty ranking
        # objective and retain its weight validation behavior.
        empty_group_losses = risk_logits.new_zeros(risk_logits.shape[0])
        return (
            _weighted_group_mean(
                empty_group_losses, group_weight, valid_groups),
            0,
            0,
        )
    return (
        _weighted_group_mean(group_losses, group_weight, valid_groups),
        int(ranked_groups),
        int(pair_count),
    )


def qsafe_group_loss(
    output: QSafeOutput,
    *,
    fall: torch.Tensor,
    first_failure_step: torch.Tensor,
    max_tilt_rad: torch.Tensor,
    min_height_m: torch.Tensor,
    candidate_mask: torch.Tensor,
    horizon_steps: int,
    group_weight: torch.Tensor | None = None,
    config: QSafeLossConfig | None = None,
) -> QSafeLossResult:
    """Compute losses with candidate and replica reductions inside each group."""
    config = QSafeLossConfig() if config is None else config
    if fall.ndim != 3 or fall.shape[:2] != output.risk_logits.shape:
        raise ValueError("fall must have shape [B,K,R] matching model logits")
    if fall.shape[0] == 0 or fall.shape[2] == 0:
        raise ValueError("fall must contain at least one group and replica")
    if candidate_mask.shape != output.risk_logits.shape:
        raise ValueError("candidate_mask must match model logits")
    for name, value in (
        ("first_failure_step", first_failure_step),
        ("max_tilt_rad", max_tilt_rad),
        ("min_height_m", min_height_m),
    ):
        if value.shape != fall.shape:
            raise ValueError(f"{name} must match fall shape")
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    mask = candidate_mask.to(dtype=torch.bool)
    if not bool(torch.all(mask[:, 0])) or not bool(torch.all(mask.sum(dim=1) >= 2)):
        raise ValueError("every group must have valid nominal and non-nominal candidates")
    valid_outcome = mask[..., None].expand_as(fall)
    raw_target = fall.to(dtype=output.risk_logits.dtype)
    if not bool(torch.all(torch.isfinite(raw_target[valid_outcome]))) or not bool(
            torch.all((raw_target[valid_outcome] == 0.0) | (
                raw_target[valid_outcome] == 1.0))):
        raise ValueError("valid fall labels must be finite and binary")
    target = torch.where(valid_outcome, raw_target, torch.zeros_like(raw_target))
    batch = fall.shape[0]
    if group_weight is None:
        weight = torch.ones(
            batch, device=output.risk_logits.device,
            dtype=output.risk_logits.dtype)
    else:
        weight = group_weight.to(
            device=output.risk_logits.device,
            dtype=output.risk_logits.dtype).reshape(-1)
    if weight.shape != (batch,) or not bool(torch.all(torch.isfinite(weight))) or (
            not bool(torch.all(weight > 0.0))):
        raise ValueError("group_weight must be a finite positive [B] vector")

    logits_replica = output.risk_logits[..., None].expand_as(target)
    absolute_per_candidate = F.binary_cross_entropy_with_logits(
        logits_replica, target, reduction="none").mean(dim=2)
    absolute_risk = _masked_candidate_group_mean(
        absolute_per_candidate, mask, weight)

    nominal_target = target[:, 0]
    state_per_group = F.binary_cross_entropy_with_logits(
        output.state_risk_logit[:, None].expand_as(nominal_target),
        nominal_target,
        reduction="none",
    ).mean(dim=1)
    state_risk = _weighted_group_mean(state_per_group, weight)

    empirical_risk = target.mean(dim=2)
    relative_target = empirical_risk - empirical_risk[:, :1]
    non_nominal_mask = mask.clone()
    non_nominal_mask[:, 0] = False
    relative_per_candidate = F.smooth_l1_loss(
        output.relative_risk,
        relative_target,
        reduction="none",
    )
    relative_risk = _masked_candidate_group_mean(
        relative_per_candidate, non_nominal_mask, weight)

    minimum_gap = 1.0 / fall.shape[2]
    ranking, ranked_groups, ranked_pairs = _ranking_loss(
        output.risk_logits,
        empirical_risk,
        mask,
        weight,
        minimum_gap,
    )

    ttf_target = torch.clamp(
        (first_failure_step[:, 0].to(output.risk.dtype) - 1.0)
        / float(horizon_steps),
        0.0,
        1.0,
    ).mean(dim=1)
    ttf = _weighted_group_mean(
        F.smooth_l1_loss(
            output.ttf_fraction, ttf_target, reduction="none"),
        weight,
    )
    tilt_target = max_tilt_rad[:, 0].to(output.risk.dtype).mean(dim=1)
    max_tilt = _weighted_group_mean(
        F.smooth_l1_loss(
            output.max_tilt_rad, tilt_target, reduction="none"),
        weight,
    )
    height_target = min_height_m[:, 0].to(output.risk.dtype).mean(dim=1)
    min_height = _weighted_group_mean(
        F.smooth_l1_loss(
            output.min_height_m, height_target, reduction="none"),
        weight,
    )
    total = (
        config.absolute_risk_weight * absolute_risk
        + config.state_risk_weight * state_risk
        + config.relative_risk_weight * relative_risk
        + config.ranking_weight * ranking
        + config.ttf_weight * ttf
        + config.max_tilt_weight * max_tilt
        + config.min_height_weight * min_height
    )
    return QSafeLossResult(
        total=total,
        absolute_risk=absolute_risk,
        state_risk=state_risk,
        relative_risk=relative_risk,
        ranking=ranking,
        ttf=ttf,
        max_tilt=max_tilt,
        min_height=min_height,
        ranked_groups=ranked_groups,
        ranked_pairs=ranked_pairs,
    )
