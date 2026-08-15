"""Small reference networks for vanilla SAC and Q_safe."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Normal


def mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        linear = nn.Linear(previous, int(width))
        nn.init.orthogonal_(linear.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(linear.bias)
        layers.extend((linear, nn.ReLU()))
        previous = int(width)
    output = nn.Linear(previous, output_dim)
    nn.init.orthogonal_(output.weight, gain=1.0)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


class TanhGaussianActor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int,
                 hidden_dims: Sequence[int], log_std_min: float = -5.0,
                 log_std_max: float = 2.0):
        super().__init__()
        self.action_dim = int(action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.network = mlp(observation_dim, hidden_dims, 2 * action_dim)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_log_std = self.network(observation).chunk(2, dim=-1)
        log_std = torch.tanh(raw_log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min) * (log_std + 1.0)
        return mean, log_std

    def sample(self, observation: torch.Tensor,
               deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(observation)
        normal = Normal(mean, log_std.exp())
        raw = mean if deterministic else normal.rsample()
        action = torch.tanh(raw)
        if deterministic:
            return action, torch.zeros(action.shape[0], device=action.device)
        log_prob = normal.log_prob(raw) - torch.log(
            torch.clamp(1.0 - action.square(), min=1e-6))
        return action, log_prob.sum(dim=-1)


class QNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int,
                 hidden_dims: Sequence[int]):
        super().__init__()
        self.network = mlp(observation_dim + action_dim, hidden_dims, 1)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((observation, action), dim=-1)).squeeze(-1)
