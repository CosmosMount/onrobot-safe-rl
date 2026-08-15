"""A deliberately plain Soft Actor-Critic reference implementation."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .networks import QNetwork, TanhGaussianActor


@dataclass(frozen=True)
class SACConfig:
    observation_dim: int = 230
    action_dim: int = 12
    hidden_dims: tuple[int, ...] = (256, 256)
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    initial_alpha: float = 0.1
    target_entropy: float | None = None


class VanillaSAC:
    """Twin scalar Q, target Q, tanh-Gaussian actor, and learned alpha."""

    def __init__(self, cfg: SACConfig, device: str | torch.device = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = TanhGaussianActor(
            cfg.observation_dim, cfg.action_dim, cfg.hidden_dims).to(self.device)
        self.q1 = QNetwork(cfg.observation_dim, cfg.action_dim, cfg.hidden_dims).to(self.device)
        self.q2 = QNetwork(cfg.observation_dim, cfg.action_dim, cfg.hidden_dims).to(self.device)
        self.target_q1 = copy.deepcopy(self.q1).requires_grad_(False)
        self.target_q2 = copy.deepcopy(self.q2).requires_grad_(False)
        self.log_alpha = torch.tensor(
            np.log(cfg.initial_alpha), dtype=torch.float32, device=self.device,
            requires_grad=True)
        self.target_entropy = float(
            -cfg.action_dim if cfg.target_entropy is None else cfg.target_entropy)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            (*self.q1.parameters(), *self.q2.parameters()), lr=cfg.critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, observation: np.ndarray, deterministic: bool = False,
            count: int = 1) -> np.ndarray:
        value = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        value = value.reshape(1, -1).expand(int(count), -1)
        action, _ = self.actor.sample(value, deterministic=deterministic)
        return action.cpu().numpy().astype(np.float32)

    def update_critic(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(batch["next_observation"])
            target_q = torch.minimum(
                self.target_q1(batch["next_observation"], next_action),
                self.target_q2(batch["next_observation"], next_action),
            ) - self.alpha.detach() * next_log_prob
            # True falls terminate the MDP; time limits bootstrap.
            target = batch["reward"] + self.cfg.gamma * (
                1.0 - batch["terminated"]) * target_q
        q1 = self.q1(batch["observation"], batch["action"])
        q2 = self.q2(batch["observation"], batch["action"])
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_optimizer.step()
        self._soft_update_targets()
        return {"sac/critic_loss": float(loss.detach()), "sac/target_q": float(target.mean())}

    def update_actor_and_alpha(
        self,
        observation: torch.Tensor,
        *,
        safety_critic: torch.nn.Module | None = None,
        safety_epsilon: float = 0.0,
        safety_lagrange: torch.Tensor | None = None,
    ) -> tuple[dict[str, float], torch.Tensor | None]:
        action, log_prob = self.actor.sample(observation)
        q = torch.minimum(self.q1(observation, action), self.q2(observation, action))
        risk = None
        safety_term = torch.zeros_like(q)
        if safety_critic is not None:
            if safety_lagrange is None:
                raise ValueError("safety_lagrange is required with safety_critic")
            risk = safety_critic(observation, action)
            safety_term = safety_lagrange.detach() * (risk - safety_epsilon)
        actor_loss = (self.alpha.detach() * log_prob - q + safety_term).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        metrics = {
            "sac/actor_loss": float(actor_loss.detach()),
            "sac/alpha": float(self.alpha.detach()),
            "sac/alpha_loss": float(alpha_loss.detach()),
            "sac/entropy": float(-log_prob.mean().detach()),
        }
        if risk is not None:
            metrics["sqrl/actor_risk"] = float(risk.mean().detach())
            metrics["sqrl/actor_violation"] = float((risk - safety_epsilon).mean().detach())
        return metrics, risk

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        metrics = self.update_critic(batch)
        actor_metrics, _ = self.update_actor_and_alpha(batch["observation"])
        metrics.update(actor_metrics)
        return metrics

    def reinitialize_task_critics_and_alpha(self) -> None:
        """Target-task convention: preserve actor, reset Qs and temperature."""
        fresh = VanillaSAC(self.cfg, self.device)
        self.q1.load_state_dict(fresh.q1.state_dict())
        self.q2.load_state_dict(fresh.q2.state_dict())
        self.target_q1.load_state_dict(fresh.target_q1.state_dict())
        self.target_q2.load_state_dict(fresh.target_q2.state_dict())
        self.log_alpha.data.copy_(fresh.log_alpha.data)
        self.critic_optimizer = torch.optim.Adam(
            (*self.q1.parameters(), *self.q2.parameters()), lr=self.cfg.critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.cfg.alpha_lr)

    @torch.no_grad()
    def _soft_update_targets(self) -> None:
        for source, target in ((self.q1, self.target_q1), (self.q2, self.target_q2)):
            for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
                target_parameter.lerp_(source_parameter, self.cfg.tau)

    def checkpoint(self) -> dict[str, Any]:
        return {
            "config": asdict(self.cfg),
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint(), path)

    def load_checkpoint(self, payload: dict[str, Any], *, actor_only: bool = False) -> None:
        self.actor.load_state_dict(payload["actor"])
        if actor_only:
            return
        for name in ("q1", "q2", "target_q1", "target_q2"):
            getattr(self, name).load_state_dict(payload[name])
        self.log_alpha.data.copy_(payload["log_alpha"].to(self.device))
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
