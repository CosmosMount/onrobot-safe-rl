from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.agents.livesac.categorical import expected_q, make_support


class LiveSACCriticBlock(nn.Module):
    def __init__(self, hidden_dim: int = 256, expansion: int = 2) -> None:
        super().__init__()
        self.linear_expand = nn.Linear(hidden_dim, hidden_dim * expansion)
        self.norm_expand = nn.LayerNorm(hidden_dim * expansion)
        self.linear_project = nn.Linear(hidden_dim * expansion, hidden_dim)
        self.norm_project = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.norm_expand(self.linear_expand(x)))
        x = F.relu(self.norm_project(self.linear_project(x)))
        return x + residual


class LiveSACCriticHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, expansion: int = 2,
                 num_blocks: int = 1, num_bins: int = 101, min_v: float = -5.0,
                 max_v: float = 5.0) -> None:
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.embed_norm = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList([LiveSACCriticBlock(hidden_dim, expansion) for _ in range(num_blocks)])
        self.rms_norm = nn.RMSNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, num_bins)
        self.register_buffer("bin_values", make_support(num_bins, min_v, max_v))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.embed_norm(self.embed(x)))
        for block in self.blocks:
            x = block(x)
        return F.log_softmax(self.output(self.rms_norm(x)), dim=-1)


class LiveSACDoubleCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 256,
                 expansion: int = 2, num_blocks: int = 1, num_qs: int = 2,
                 num_bins: int = 101, min_v: float = -5.0, max_v: float = 5.0) -> None:
        super().__init__()
        self.critics = nn.ModuleList([LiveSACCriticHead(observation_dim + action_dim, hidden_dim, expansion,
                                                         num_blocks, num_bins, min_v, max_v) for _ in range(num_qs)])

    def forward(self, observations: torch.Tensor, actions: torch.Tensor, training: bool = True):
        x = torch.cat([observations, actions], dim=-1)
        log_probs = torch.stack([critic(x) for critic in self.critics], dim=0)
        qs = expected_q(log_probs, self.critics[0].bin_values)
        return qs, {"log_prob": log_probs, "prob": log_probs.exp()}
