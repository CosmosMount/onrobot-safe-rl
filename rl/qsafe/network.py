"""Five-frame, nominal-centered safety-risk models."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal, Sequence

import torch
from torch import nn
from torch.nn import functional as F


ActionMode = Literal["selective_advantage", "pointwise", "state_only"]


@dataclass(frozen=True)
class QSafeNetworkConfig:
    observation_dim: int = 46
    action_dim: int = 12
    history_frames: int = 5
    frame_hidden_dim: int = 128
    state_hidden_dim: int = 128
    action_hidden_dim: int = 128
    privileged_dim: int = 0
    action_mode: ActionMode = "selective_advantage"

    def __post_init__(self) -> None:
        for name in (
            "observation_dim", "action_dim", "history_frames",
            "frame_hidden_dim", "state_hidden_dim", "action_hidden_dim",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.privileged_dim, bool) or not isinstance(
                self.privileged_dim, Integral) or self.privileged_dim < 0:
            raise ValueError("privileged_dim must be a nonnegative integer")
        if self.action_mode not in {
            "selective_advantage", "pointwise", "state_only",
        }:
            raise ValueError(f"unknown action_mode={self.action_mode!r}")


@dataclass(frozen=True)
class QSafeOutput:
    risk_logits: torch.Tensor
    risk: torch.Tensor
    state_risk_logit: torch.Tensor
    state_risk: torch.Tensor
    advantage_logit: torch.Tensor
    relative_risk: torch.Tensor
    ttf_fraction: torch.Tensor
    max_tilt_rad: torch.Tensor
    min_height_m: torch.Tensor


class SelectiveAdvantageQSafe(nn.Module):
    """Predict finite-horizon risk with an exactly centered nominal action."""

    def __init__(self, config: QSafeNetworkConfig):
        super().__init__()
        self.config = config
        self.frame_encoder = nn.Sequential(
            nn.LayerNorm(config.observation_dim),
            nn.Linear(config.observation_dim, config.frame_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.frame_hidden_dim, config.state_hidden_dim),
            nn.SiLU(),
        )
        self.temporal_encoder = nn.GRU(
            config.state_hidden_dim,
            config.state_hidden_dim,
            batch_first=True,
        )
        if config.privileged_dim:
            self.privileged_encoder: nn.Module | None = nn.Sequential(
                nn.LayerNorm(config.privileged_dim),
                nn.Linear(config.privileged_dim, config.state_hidden_dim),
                nn.SiLU(),
            )
            self.state_fusion: nn.Module = nn.Sequential(
                nn.Linear(2 * config.state_hidden_dim, config.state_hidden_dim),
                nn.SiLU(),
            )
        else:
            self.privileged_encoder = None
            self.state_fusion = nn.Identity()
        self.state_risk_head = nn.Sequential(
            nn.Linear(config.state_hidden_dim, config.state_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.state_hidden_dim, 1),
        )
        self.auxiliary_head = nn.Sequential(
            nn.Linear(config.state_hidden_dim, config.state_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.state_hidden_dim, 3),
        )
        action_input_dim = config.state_hidden_dim + 3 * config.action_dim
        self.action_head = nn.Sequential(
            nn.LayerNorm(action_input_dim),
            nn.Linear(action_input_dim, config.action_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.action_hidden_dim, config.action_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.action_hidden_dim, 1),
        )

    def encode_state(
        self,
        observation_history: torch.Tensor,
        privileged_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        config = self.config
        if observation_history.ndim != 3 or observation_history.shape[1:] != (
                config.history_frames, config.observation_dim):
            raise ValueError(
                "observation_history must have shape "
                f"[B,{config.history_frames},{config.observation_dim}]")
        encoded_frames = self.frame_encoder(observation_history)
        _, hidden = self.temporal_encoder(encoded_frames)
        state = hidden[-1]
        if self.privileged_encoder is None:
            if privileged_state is not None:
                raise ValueError("deployable model must not receive privileged state")
            return self.state_fusion(state)
        if privileged_state is None or privileged_state.ndim != 2 or (
                privileged_state.shape != (
                    observation_history.shape[0], config.privileged_dim)):
            raise ValueError(
                f"privileged_state must have shape [B,{config.privileged_dim}]")
        privileged = self.privileged_encoder(privileged_state)
        return self.state_fusion(torch.cat([state, privileged], dim=-1))

    def _raw_action_score(
        self,
        state: torch.Tensor,
        nominal_action: torch.Tensor,
        candidate_action: torch.Tensor,
    ) -> torch.Tensor:
        batch, candidates, _ = candidate_action.shape
        state_expanded = state[:, None, :].expand(-1, candidates, -1)
        nominal_expanded = nominal_action[:, None, :].expand(-1, candidates, -1)
        action_input = torch.cat([
            state_expanded,
            nominal_expanded,
            candidate_action,
            candidate_action - nominal_expanded,
        ], dim=-1)
        return self.action_head(action_input).reshape(batch, candidates)

    def forward(
        self,
        observation_history: torch.Tensor,
        nominal_action: torch.Tensor,
        candidate_action: torch.Tensor,
        privileged_state: torch.Tensor | None = None,
    ) -> QSafeOutput:
        config = self.config
        if nominal_action.ndim != 2 or nominal_action.shape[1] != config.action_dim:
            raise ValueError(
                f"nominal_action must have shape [B,{config.action_dim}]")
        if candidate_action.ndim != 3 or candidate_action.shape[0] != (
                nominal_action.shape[0]) or candidate_action.shape[2] != (
                    config.action_dim):
            raise ValueError(
                f"candidate_action must have shape [B,K,{config.action_dim}]")
        if observation_history.shape[0] != nominal_action.shape[0]:
            raise ValueError("observation and action batch sizes differ")
        state = self.encode_state(observation_history, privileged_state)
        state_risk_logit = self.state_risk_head(state).reshape(-1)
        raw_candidate = self._raw_action_score(
            state, nominal_action, candidate_action)
        if config.action_mode == "selective_advantage":
            raw_nominal = self._raw_action_score(
                state, nominal_action, nominal_action[:, None, :]).reshape(-1)
            advantage_logit = raw_candidate - raw_nominal[:, None]
            risk_logits = state_risk_logit[:, None] + advantage_logit
        elif config.action_mode == "pointwise":
            advantage_logit = raw_candidate
            risk_logits = state_risk_logit[:, None] + raw_candidate
        else:
            advantage_logit = torch.zeros_like(raw_candidate)
            risk_logits = state_risk_logit[:, None].expand_as(raw_candidate)
        risk = torch.sigmoid(risk_logits)
        state_risk = torch.sigmoid(state_risk_logit)
        auxiliary = self.auxiliary_head(state)
        return QSafeOutput(
            risk_logits=risk_logits,
            risk=risk,
            state_risk_logit=state_risk_logit,
            state_risk=state_risk,
            advantage_logit=advantage_logit,
            relative_risk=risk - state_risk[:, None],
            ttf_fraction=torch.sigmoid(auxiliary[:, 0]),
            max_tilt_rad=F.softplus(auxiliary[:, 1]),
            min_height_m=F.softplus(auxiliary[:, 2]),
        )


@dataclass(frozen=True)
class EnsemblePrediction:
    member_risk: torch.Tensor
    risk_mean: torch.Tensor
    risk_std: torch.Tensor
    member_benefit: torch.Tensor
    benefit_mean: torch.Tensor
    benefit_std: torch.Tensor
    member_state_risk: torch.Tensor


class QSafeEnsemble(nn.Module):
    """An independently trained ensemble with per-member calibration."""

    def __init__(
        self,
        members: Sequence[SelectiveAdvantageQSafe],
        temperatures: Sequence[float] | torch.Tensor | None = None,
    ):
        super().__init__()
        if not members:
            raise ValueError("Q_safe ensemble requires at least one member")
        if any(member.config != members[0].config for member in members[1:]):
            raise ValueError("Q_safe ensemble members must share one configuration")
        self.members = nn.ModuleList(members)
        if temperatures is None:
            values = torch.ones(len(members), dtype=torch.float32)
        else:
            values = torch.as_tensor(temperatures, dtype=torch.float32).reshape(-1)
        if values.shape != (len(members),) or not torch.all(
                torch.isfinite(values)) or torch.any(values <= 0.0):
            raise ValueError("temperatures must be finite positive values per member")
        self.register_buffer("temperatures", values.clone())

    def predict(
        self,
        observation_history: torch.Tensor,
        nominal_action: torch.Tensor,
        candidate_action: torch.Tensor,
        privileged_state: torch.Tensor | None = None,
    ) -> EnsemblePrediction:
        logits = []
        state_logits = []
        for member in self.members:
            output = member(
                observation_history,
                nominal_action,
                candidate_action,
                privileged_state,
            )
            logits.append(output.risk_logits)
            state_logits.append(output.state_risk_logit)
        member_logits = torch.stack(logits, dim=0)
        member_state_logits = torch.stack(state_logits, dim=0)
        temperature = self.temperatures[:, None, None].to(member_logits)
        member_risk = torch.sigmoid(member_logits / temperature)
        member_state_risk = torch.sigmoid(
            member_state_logits / self.temperatures[:, None].to(member_state_logits))
        member_benefit = member_risk[..., :1] - member_risk
        return EnsemblePrediction(
            member_risk=member_risk,
            risk_mean=member_risk.mean(dim=0),
            risk_std=member_risk.std(dim=0, unbiased=False),
            member_benefit=member_benefit,
            benefit_mean=member_benefit.mean(dim=0),
            benefit_std=member_benefit.std(dim=0, unbiased=False),
            member_state_risk=member_state_risk,
        )
