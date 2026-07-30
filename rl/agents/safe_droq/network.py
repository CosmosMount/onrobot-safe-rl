from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from rl.agents.droq.network import DroQMLP


class SafetyCritic(nn.Module):
    """Failure-probability critic over an observation/action pair."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
    ):
        super().__init__()
        self.base = DroQMLP(
            observation_dim + action_dim,
            hidden_dims,
            activate_final=True,
        )
        self.logit = nn.Linear(self.base.output_dim, 1)
        nn.init.xavier_uniform_(self.logit.weight)
        nn.init.zeros_(self.logit.bias)

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        training: bool,
    ) -> torch.Tensor:
        features = torch.cat([observations, actions], dim=-1)
        return self.logit(
            self.base(features, training=training)).squeeze(-1)
