from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.agents.droq.network import DroQMLP


class CategoricalDroQCriticHead(nn.Module):
    """DroQ critic backbone with only the scalar head replaced by logits."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], num_bins: int,
                 *, dropout_rate: float, use_layer_norm: bool):
        super().__init__()
        self.base = DroQMLP(input_dim, hidden_dims, activate_final=True,
                            dropout_rate=dropout_rate,
                            use_layer_norm=use_layer_norm)
        self.logits = nn.Linear(self.base.output_dim, num_bins)
        nn.init.xavier_uniform_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        return self.logits(self.base(x, training=training))


class CategoricalDroQEnsembleCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int,
                 hidden_dims: Sequence[int], num_qs: int, num_bins: int,
                 min_v: float, max_v: float, *, dropout_rate: float,
                 use_layer_norm: bool):
        super().__init__()
        if num_qs <= 0 or num_bins < 2 or min_v >= max_v:
            raise ValueError("invalid categorical DroQ critic dimensions/support")
        self.num_qs = num_qs
        self.num_bins = num_bins
        self.register_buffer("support", torch.linspace(min_v, max_v, num_bins))
        input_dim = observation_dim + action_dim
        self.qs = nn.ModuleList([
            CategoricalDroQCriticHead(
                input_dim, hidden_dims, num_bins,
                dropout_rate=dropout_rate, use_layer_norm=use_layer_norm)
            for _ in range(num_qs)
        ])

    def forward(self, observations: torch.Tensor, actions: torch.Tensor,
                training: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = torch.cat([observations, actions], dim=-1)
        logits = torch.stack([q(x, training=training) for q in self.qs], dim=0)
        log_prob = F.log_softmax(logits, dim=-1)
        values = (log_prob.exp() * self.support.view(1, 1, -1)).sum(dim=-1)
        return values, {"log_prob": log_prob}
