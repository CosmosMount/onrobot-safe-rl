from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.agents.droq.network import DroQTemperature
from rl.utils.normalizations import (
    EnsembleCategoricalValue,
    EnsembleUnitLinear,
    EnsembleUnitRMSNorm,
    NormalTanhPolicy,
    UnitLinear,
    UnitRMSNorm,
)


class LiveSACEmbedder(nn.Module):
    """FlashSAC-style input projection with LayerNorm instead of BatchNorm."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.w = UnitLinear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        del training
        return self.w(self.norm(x))


class LiveSACBlock(nn.Module):
    """FlashSAC residual MLP block with per-sample LayerNorm."""

    def __init__(self, hidden_dim: int, expansion: int = 4):
        super().__init__()
        self.w1 = UnitLinear(hidden_dim, hidden_dim * expansion)
        self.w2 = UnitLinear(hidden_dim * expansion, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim * expansion)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        del training
        residual = x
        x = F.relu(self.norm1(self.w1(x)))
        x = F.relu(self.norm2(self.w2(x)))
        return x + residual


class LiveSACActor(nn.Module):
    """FlashSAC actor topology with LayerNorm and final RMSNorm."""

    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.embedder = LiveSACEmbedder(input_dim, hidden_dim)
        self.encoder = nn.ModuleList([LiveSACBlock(hidden_dim) for _ in range(num_blocks)])
        self.post_norm = UnitRMSNorm(hidden_dim)
        self.predictor = NormalTanhPolicy(hidden_dim, action_dim)

    def get_mean_and_std(self, observations: torch.Tensor, training: bool):
        x = self.embedder(observations, training)
        for block in self.encoder:
            x = block(x, training)
        return self.predictor.get_mean_and_std(self.post_norm(x), training)

    def forward(self, observations: torch.Tensor, training: bool, sample: bool = True):
        x = self.embedder(observations, training)
        for block in self.encoder:
            x = block(x, training)
        return self.predictor(self.post_norm(x), training, sample=sample)


class LiveSACEnsembleBlock(nn.Module):
    """Ensemble FlashSAC block with LayerNorm over the feature dimension."""

    def __init__(self, num_qs: int, hidden_dim: int, expansion: int = 4):
        super().__init__()
        self.w1 = EnsembleUnitLinear(num_qs, hidden_dim, hidden_dim * expansion)
        self.w2 = EnsembleUnitLinear(num_qs, hidden_dim * expansion, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim * expansion)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        del training
        residual = x
        x = F.relu(self.norm1(self.w1(x)))
        x = F.relu(self.norm2(self.w2(x)))
        return x + residual


class LiveSACDoubleCritic(nn.Module):
    """FlashSAC-style ensemble backbone with a categorical Q head."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        expansion: int = 4,
        num_blocks: int = 1,
        num_qs: int = 2,
        num_bins: int = 101,
        min_v: float = -5.0,
        max_v: float = 5.0,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        del dropout_rate  # FlashSAC backbone has no dropout layers.
        if num_qs <= 0:
            raise ValueError("LiveSAC ensemble must contain at least one critic")
        self.num_qs = num_qs
        input_dim = observation_dim + action_dim
        self.embedder_norm = nn.LayerNorm(input_dim)
        self.embedder = EnsembleUnitLinear(num_qs, input_dim, hidden_dim)
        self.encoder = nn.ModuleList([
            LiveSACEnsembleBlock(num_qs, hidden_dim, expansion)
            for _ in range(num_blocks)
        ])
        self.post_norm = EnsembleUnitRMSNorm(num_qs, hidden_dim)
        self.predictor = EnsembleCategoricalValue(
            num_ensemble=num_qs,
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
        )

    @property
    def bin_values(self) -> torch.Tensor:
        return self.predictor.bin_values.reshape(-1)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor, training: bool = True):
        x = torch.cat((observations, actions), dim=-1)
        x = self.embedder(self.embedder_norm(x).unsqueeze(0).expand(self.num_qs, -1, -1))
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        return self.predictor(x, training)


# Kept as a public symbol for checkpoint/config code that imports the
# temperature module from the LiveSAC network module.
LiveSACTemperature = DroQTemperature
