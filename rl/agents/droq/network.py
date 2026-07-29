import math
from typing import Sequence

import torch
import torch.nn as nn

from rl.utils.normalizations import NormalTanhPolicy


class DroQMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        *,
        activate_final: bool = True,
        dropout_rate: float = 0.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        if not activate_final:
            while layers and isinstance(layers[-1], (nn.ReLU, nn.Dropout)):
                layers.pop()
        self.net = nn.Sequential(*layers)
        self.output_dim = prev_dim

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        self.train(training)
        return self.net(x)


class DroQActor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.base = DroQMLP(observation_dim, hidden_dims, activate_final=True)
        self.predictor = NormalTanhPolicy(
            hidden_dim=self.base.output_dim,
            action_dim=action_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )

    def get_mean_and_log_std(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.base(observations, training=False)
        mean, std = self.predictor.get_mean_and_std(x, training=False)
        return mean, torch.log(std)

    def forward(
        self,
        observations: torch.Tensor,
        training: bool,
        sample: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.base(observations, training=training)
        return self.predictor(x, training=training, sample=sample)


class DroQEnsembleCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        num_qs: int,
        dropout_rate: float = 0.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        input_dim = observation_dim + action_dim
        self.qs = nn.ModuleList(
            [
                DroQCriticHead(
                    input_dim,
                    hidden_dims,
                    dropout_rate=dropout_rate,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(num_qs)
            ]
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = torch.cat([observations, actions], dim=-1)
        values = []
        for q in self.qs:
            values.append(q(x, training=training))
        return torch.stack(values, dim=0), {}


class DroQCriticHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        *,
        dropout_rate: float = 0.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.base = DroQMLP(
            input_dim,
            hidden_dims,
            activate_final=True,
            dropout_rate=dropout_rate,
            use_layer_norm=use_layer_norm,
        )
        self.value = nn.Linear(self.base.output_dim, 1)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.zeros_(self.value.bias)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        x = self.base(x, training=training)
        return self.value(x).squeeze(-1)


class DroQTemperature(nn.Module):
    def __init__(self, initial_value: float = 1.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(math.log(initial_value), dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return torch.exp(self.log_temp)
