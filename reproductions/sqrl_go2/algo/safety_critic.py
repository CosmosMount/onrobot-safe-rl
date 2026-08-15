"""SQRL dynamic-programming safety critic."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .networks import QNetwork


@dataclass(frozen=True)
class SafetyCriticConfig:
    observation_dim: int = 230
    action_dim: int = 12
    hidden_dims: tuple[int, ...] = (256, 256)
    gamma: float = 0.7
    lr: float = 3e-4
    tau: float = 0.005


def safety_bellman_target(cost: torch.Tensor, next_q: torch.Tensor,
                          gamma: float) -> torch.Tensor:
    if cost.shape != next_q.shape:
        raise ValueError("cost and next_q shapes must match")
    if torch.any((cost != 0) & (cost != 1)):
        raise ValueError("SQRL cost must be binary")
    return cost + (1.0 - cost) * float(gamma) * next_q


class SafetyCriticLearner:
    def __init__(self, cfg: SafetyCriticConfig,
                 device: str | torch.device = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.critic = QNetwork(cfg.observation_dim, cfg.action_dim, cfg.hidden_dims).to(self.device)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

    def update(
        self,
        batch: dict[str, torch.Tensor],
        sample_constrained_next_action: Callable[[torch.Tensor], torch.Tensor],
    ) -> dict[str, float]:
        with torch.no_grad():
            next_action = sample_constrained_next_action(batch["next_observation"])
            next_q = self.target(batch["next_observation"], next_action)
            target = safety_bellman_target(batch["cost"], next_q, self.cfg.gamma)
        prediction = self.critic(batch["observation"], batch["action"])
        loss = 0.5 * F.mse_loss(prediction, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self._soft_update()
        return {
            "safety/loss": float(loss.detach()),
            "safety/q_mean": float(prediction.mean().detach()),
            "safety/target_mean": float(target.mean().detach()),
        }

    def freeze(self) -> None:
        self.critic.requires_grad_(False)
        self.target.requires_grad_(False)
        self.critic.eval()
        self.target.eval()

    @torch.no_grad()
    def _soft_update(self) -> None:
        for source, target in zip(self.critic.parameters(), self.target.parameters()):
            target.lerp_(source, self.cfg.tau)

    def checkpoint(self) -> dict[str, Any]:
        return {
            "config": asdict(self.cfg),
            "critic": self.critic.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint(), path)

    def load_checkpoint(self, payload: dict[str, Any], *, load_optimizer: bool = True) -> None:
        self.critic.load_state_dict(payload["critic"])
        self.target.load_state_dict(payload["target"])
        if load_optimizer:
            self.optimizer.load_state_dict(payload["optimizer"])
