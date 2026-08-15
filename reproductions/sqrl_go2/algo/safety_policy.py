"""Finite-sample approximation to the SQRL projected policy bar-pi."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


@dataclass(frozen=True)
class MaskResult:
    requested_action: np.ndarray
    critic_action: np.ndarray
    q_target: np.ndarray
    risk: float
    accepted: bool
    candidate_count: int
    no_safe_candidate: bool
    risk_mean: float
    risk_p50: float
    risk_p90: float


class SafetyPolicy:
    def __init__(self, actor: torch.nn.Module, safety_critic: torch.nn.Module,
                 epsilon: float, max_candidates: int, device: torch.device):
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        self.actor = actor
        self.safety_critic = safety_critic
        self.epsilon = float(epsilon)
        self.max_candidates = int(max_candidates)
        self.device = device

    @torch.no_grad()
    def select(self, observation: np.ndarray, preview: Callable[[np.ndarray], object],
               deterministic: bool = False) -> MaskResult:
        state = torch.as_tensor(observation, dtype=torch.float32, device=self.device).reshape(1, -1)
        states = state.expand(self.max_candidates, -1)
        candidates, _ = self.actor.sample(states, deterministic=deterministic)
        projected = preview(candidates.cpu().numpy().astype(np.float32))
        critic_actions = torch.as_tensor(
            projected.critic_actions, dtype=torch.float32, device=self.device)
        risks = self.safety_critic(states, critic_actions)
        safe = torch.nonzero(risks <= self.epsilon).flatten()
        if safe.numel():
            index = int(safe[0])
            accepted = True
            candidate_count = index + 1
        else:
            index = int(torch.argmin(risks))
            accepted = False
            candidate_count = self.max_candidates
        return MaskResult(
            requested_action=projected.requested[index].copy(),
            critic_action=projected.critic_actions[index].copy(),
            q_target=projected.q_targets[index].copy(),
            risk=float(risks[index]),
            accepted=accepted,
            candidate_count=candidate_count,
            no_safe_candidate=not accepted,
            risk_mean=float(risks.mean()),
            risk_p50=float(torch.quantile(risks, 0.5)),
            risk_p90=float(torch.quantile(risks, 0.9)),
        )

    @torch.no_grad()
    def sample_tensor(self, observations: torch.Tensor) -> torch.Tensor:
        """Batch bar-pi approximation for the Bellman next-action target.

        The replay action is already the executed normalized action. Under the
        frozen Go2 reproduction config (no action filter and no slew limit),
        requested and executed normalized actions are identical, so no runtime
        stateful preview is needed inside this batched learner operation.
        """
        batch = observations.shape[0]
        expanded = observations[:, None, :].expand(
            batch, self.max_candidates, observations.shape[-1]).reshape(
                batch * self.max_candidates, -1)
        candidates, _ = self.actor.sample(expanded)
        risks = self.safety_critic(expanded, candidates).reshape(batch, self.max_candidates)
        candidates = candidates.reshape(batch, self.max_candidates, -1)
        safe = risks <= self.epsilon
        first_safe = safe.to(torch.int64).argmax(dim=1)
        minimum = risks.argmin(dim=1)
        indices = torch.where(safe.any(dim=1), first_safe, minimum)
        return candidates[torch.arange(batch, device=observations.device), indices]
