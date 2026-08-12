"""Action-conditioned safety critic used by the PPO data-source study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


CriticMode = Literal["action", "state_only"]


@dataclass(frozen=True)
class PpoSqrlCriticConfig:
    observation_dim: int = 46
    history_frames: int = 5
    action_dim: int = 12
    hidden_dim: int = 256
    mode: CriticMode = "action"


class PpoSqrlSafetyCritic(nn.Module):
    """Predict discounted failure cost from deployable history and PD target."""

    def __init__(self, config: PpoSqrlCriticConfig) -> None:
        super().__init__()
        self.config = config
        self.frame = nn.Sequential(
            nn.LayerNorm(config.observation_dim),
            nn.Linear(config.observation_dim, config.hidden_dim),
            nn.ELU(),
        )
        self.temporal = nn.GRU(
            config.hidden_dim, config.hidden_dim, batch_first=True)
        action_dim = 0 if config.mode == "state_only" else config.action_dim
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim + action_dim, config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ELU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, history: torch.Tensor, critic_action: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if history.ndim != 3 or history.shape[1:] != (
                cfg.history_frames, cfg.observation_dim):
            raise ValueError("history must have shape [B,5,46]")
        if critic_action.shape != (history.shape[0], cfg.action_dim):
            raise ValueError("critic_action must have shape [B,12]")
        _, hidden = self.temporal(self.frame(history))
        state = hidden[-1]
        features = state if cfg.mode == "state_only" else torch.cat(
            [state, critic_action], dim=-1)
        return torch.sigmoid(self.head(features).reshape(-1))


def sqrl_bellman_target(
    cost_t_plus_1: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    next_q: torch.Tensor,
    *,
    gamma_safe: float,
) -> torch.Tensor:
    """SQRL target with c[t+1] and no bootstrap across either boundary."""
    tensors = [cost_t_plus_1, terminated, truncated, next_q]
    if any(value.ndim != 1 for value in tensors) or len({len(v) for v in tensors}) != 1:
        raise ValueError("Bellman target inputs must be matching vectors")
    if not 0.0 < gamma_safe < 1.0:
        raise ValueError("gamma_safe must be in (0,1)")
    cost = cost_t_plus_1.to(next_q.dtype)
    done = terminated.to(torch.bool) | truncated.to(torch.bool)
    if torch.any(cost.to(torch.bool) != terminated.to(torch.bool)):
        raise ValueError("c[t+1] must equal first-fall termination")
    return cost + (1.0 - cost) * (~done).to(next_q.dtype) * gamma_safe * next_q
