"""Target-task SQRL branch semantics and dual update."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .sac import VanillaSAC
from .safety_critic import SafetyCriticLearner


TARGET_BRANCHES = ("sac_transfer", "sqrl_mask", "sqrl_full")


@dataclass
class SafetyLagrange:
    initial_value: float
    lr: float
    device: torch.device

    def __post_init__(self) -> None:
        if self.initial_value < 0 or self.lr <= 0:
            raise ValueError("invalid safety Lagrange parameters")
        self.value = torch.tensor(
            self.initial_value, dtype=torch.float32, device=self.device,
            requires_grad=True)
        self.optimizer = torch.optim.Adam([self.value], lr=self.lr)

    def update(self, violation: torch.Tensor) -> dict[str, float]:
        # Minimize -nu * violation: positive violation causes nu to increase.
        loss = -(self.value * violation.detach().mean())
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            self.value.clamp_(min=0.0)
        return {"sqrl/nu": float(self.value.detach()), "sqrl/nu_loss": float(loss.detach())}


def initialize_target_branch(
    sac: VanillaSAC,
    safety: SafetyCriticLearner | None,
    branch: str,
) -> None:
    if branch not in TARGET_BRANCHES:
        raise ValueError(f"unknown target branch: {branch}")
    sac.reinitialize_task_critics_and_alpha()
    if branch == "sac_transfer":
        return
    if safety is None:
        raise ValueError(f"{branch} requires pretrained Q_safe")
    safety.freeze()
