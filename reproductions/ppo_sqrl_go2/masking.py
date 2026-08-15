"""Vectorized finite-candidate SQRL rejection sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class MaskBatch:
    policy_action: torch.Tensor
    critic_action: torch.Tensor
    risk: torch.Tensor
    accepted: torch.Tensor
    no_safe: torch.Tensor
    attempts: torch.Tensor
    candidate_safe_fraction: torch.Tensor
    candidate_risk_mean: torch.Tensor
    candidate_risk_p50: torch.Tensor
    candidate_risk_p90: torch.Tensor


def select_masked_actions(
    observation: torch.Tensor,
    *,
    sample_policy_actions: Callable[[torch.Tensor, int], torch.Tensor],
    project_for_critic: Callable[[torch.Tensor], torch.Tensor],
    critic: torch.nn.Module,
    epsilon: float,
    candidates: int,
) -> MaskBatch:
    if observation.ndim != 2 or candidates <= 0:
        raise ValueError("masking requires [B,D] observations and positive K")
    batch = observation.shape[0]
    policy_actions = sample_policy_actions(observation, candidates)
    if policy_actions.ndim != 3 or policy_actions.shape[:2] != (batch, candidates):
        raise ValueError("policy sampler must return [B,K,A]")
    critic_actions = project_for_critic(policy_actions)
    if critic_actions.shape != policy_actions.shape:
        raise ValueError("projected critic actions must preserve [B,K,A]")
    expanded = observation[:, None, :].expand(-1, candidates, -1).reshape(
        batch * candidates, -1)
    risk = critic(expanded, critic_actions.reshape(batch * candidates, -1)).reshape(
        batch, candidates)
    safe = risk <= float(epsilon)
    any_safe = safe.any(dim=1)
    first_safe = safe.to(torch.int64).argmax(dim=1)
    minimum = risk.argmin(dim=1)
    chosen = torch.where(any_safe, first_safe, minimum)
    row = torch.arange(batch, device=observation.device)
    return MaskBatch(
        policy_action=policy_actions[row, chosen],
        critic_action=critic_actions[row, chosen],
        risk=risk[row, chosen],
        accepted=any_safe,
        no_safe=~any_safe,
        attempts=torch.where(any_safe, first_safe + 1,
                             torch.full_like(first_safe, candidates)),
        candidate_safe_fraction=safe.float().mean(dim=1),
        candidate_risk_mean=risk.mean(dim=1),
        candidate_risk_p50=torch.quantile(risk, 0.5, dim=1),
        candidate_risk_p90=torch.quantile(risk, 0.9, dim=1),
    )


@torch.no_grad()
def constrained_next_actions(
    observation: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    return select_masked_actions(observation, **kwargs).critic_action
