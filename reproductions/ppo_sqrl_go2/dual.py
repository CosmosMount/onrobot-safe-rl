"""Frozen-Q_safe Lagrangian helpers for PPO target adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal


@dataclass
class ProjectedDual:
    learning_rate: float = 3e-4
    initial_value: float = 0.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.initial_value < 0:
            raise ValueError("invalid projected dual configuration")
        self.value = float(self.initial_value)

    def update(self, mean_violation: float) -> float:
        if not torch.isfinite(torch.tensor(mean_violation)):
            raise ValueError("dual violation must be finite")
        self.value = max(0.0, self.value + self.learning_rate * float(mean_violation))
        return self.value


def reparameterized_gaussian_action(mean: torch.Tensor,
                                    std: torch.Tensor) -> torch.Tensor:
    if mean.shape != std.shape or torch.any(std <= 0):
        raise ValueError("Gaussian mean/std must be aligned and std positive")
    return Normal(mean, std).rsample()


def frozen_qsafe_penalty(
    critic: nn.Module,
    qsafe_observation: torch.Tensor,
    critic_action: torch.Tensor,
    *, epsilon: float,
    dual_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if any(parameter.requires_grad for parameter in critic.parameters()):
        raise ValueError("target Q_safe must be frozen")
    risk = critic(qsafe_observation, critic_action)
    violation = risk - float(epsilon)
    return float(dual_value) * violation.mean(), violation
