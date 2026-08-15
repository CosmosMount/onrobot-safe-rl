"""SQRL Bellman safety critic with the formal Go2 network semantics."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F


def _mlp(input_dim: int, hidden: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden:
        linear = nn.Linear(previous, width)
        nn.init.orthogonal_(linear.weight, gain=2 ** 0.5)
        nn.init.zeros_(linear.bias)
        layers.extend((linear, nn.ReLU()))
        previous = width
    output = nn.Linear(previous, output_dim)
    nn.init.orthogonal_(output.weight, gain=1.0)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class SafetyCriticConfig:
    observation_dim: int = 230
    action_dim: int = 12
    hidden_dims: tuple[int, ...] = (256, 256)
    gamma: float = 0.70
    learning_rate: float = 3e-4
    tau: float = 0.005


class SafetyQNetwork(nn.Module):
    def __init__(self, cfg: SafetyCriticConfig):
        super().__init__()
        self.cfg = cfg
        self.network = _mlp(
            cfg.observation_dim + cfg.action_dim, cfg.hidden_dims, 1)

    def forward(self, observation: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[1] != self.cfg.observation_dim:
            raise ValueError("Q_safe observation must have shape [B,230]")
        if action.shape != (observation.shape[0], self.cfg.action_dim):
            raise ValueError("Q_safe action must have shape [B,12]")
        return self.network(torch.cat((observation, action), dim=-1)).squeeze(-1)


def safety_bellman_target(cost: torch.Tensor, terminated: torch.Tensor,
                          truncated: torch.Tensor, next_q: torch.Tensor,
                          gamma: float) -> torch.Tensor:
    values = (cost, terminated, truncated, next_q)
    if any(value.ndim != 1 for value in values) or len({len(x) for x in values}) != 1:
        raise ValueError("Bellman target tensors must be aligned vectors")
    if torch.any((cost != 0) & (cost != 1)):
        raise ValueError("safety cost must be binary")
    if torch.any(cost.to(torch.bool) != terminated.to(torch.bool)):
        raise ValueError("cost must equal first-fall termination")
    # Match the frozen formal SQRL recurrence exactly. Episode boundaries do
    # not introduce an additional terminal factor; first-fall cost already
    # makes the continuation term zero when c=1.
    return cost.to(next_q.dtype) + (
        (1.0 - cost.to(next_q.dtype)) * float(gamma) * next_q)


class SafetyCriticLearner:
    def __init__(self, cfg: SafetyCriticConfig,
                 device: str | torch.device = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.critic = SafetyQNetwork(cfg).to(self.device)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.learning_rate)
        self.updates = 0

    def update(self, batch: dict[str, torch.Tensor],
               constrained_next_action: Callable[
                   [torch.Tensor, torch.Tensor], torch.Tensor],
               ) -> dict[str, float]:
        with torch.no_grad():
            action = constrained_next_action(
                batch["next_observation"], batch["next_policy_observation"])
            target_q = self.target(batch["next_observation"], action)
            target = safety_bellman_target(
                batch["cost"], batch["terminated"], batch["truncated"],
                target_q, self.cfg.gamma)
        prediction = self.critic(batch["observation"], batch["action"])
        loss = 0.5 * F.mse_loss(prediction, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            for source, destination in zip(
                    self.critic.parameters(), self.target.parameters(), strict=True):
                destination.lerp_(source, self.cfg.tau)
        self.updates += 1
        return {
            "safety/loss": float(loss.detach()),
            "safety/q_mean": float(prediction.mean().detach()),
            "safety/q_p10": float(torch.quantile(prediction.detach(), 0.1)),
            "safety/q_p50": float(torch.quantile(prediction.detach(), 0.5)),
            "safety/q_p90": float(torch.quantile(prediction.detach(), 0.9)),
            "safety/target_mean": float(target.mean()),
            "safety/calibration_brier_immediate_cost": float(
                (prediction.detach().clamp(0.0, 1.0) - batch["cost"]).square().mean()),
            "safety/updates": float(self.updates),
        }

    def freeze(self) -> nn.Module:
        self.critic.requires_grad_(False)
        self.critic.eval()
        return self.critic

    def state_dict(self) -> dict[str, object]:
        return {
            "config": asdict(self.cfg),
            "critic": self.critic.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, value: dict[str, object], *, optimizer: bool = True) -> None:
        self.critic.load_state_dict(value["critic"])  # type: ignore[arg-type]
        self.target.load_state_dict(value["target"])  # type: ignore[arg-type]
        if optimizer:
            self.optimizer.load_state_dict(value["optimizer"])  # type: ignore[arg-type]
        self.updates = int(value.get("updates", 0))
