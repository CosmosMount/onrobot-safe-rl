"""Trajectory-bootstrap training and calibration for grouped Q_safe data."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from rl.qsafe.data import (
    NormalizationStats,
    TorchGroupedView,
    trajectory_bootstrap_indices,
)
from rl.qsafe.loss import QSafeLossConfig, qsafe_group_loss
from rl.qsafe.network import (
    QSafeEnsemble,
    QSafeNetworkConfig,
    SelectiveAdvantageQSafe,
)
from safety_data.schema import audit_split_disjointness


@dataclass(frozen=True)
class QSafeTrainingConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 5.0
    ensemble_members: int = 5
    seed: int = 20260809
    device: str = "cpu"
    calibration_steps: int = 100

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "ensemble_members", "seed",
                     "calibration_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        if self.epochs <= 0 or self.batch_size <= 0 or self.ensemble_members <= 0:
            raise ValueError("epochs, batch_size and ensemble_members must be positive")
        if self.seed < 0 or self.calibration_steps < 0:
            raise ValueError("seed and calibration_steps must be nonnegative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0 or (
                not math.isfinite(self.weight_decay)) or self.weight_decay < 0.0:
            raise ValueError("invalid optimizer configuration")
        if not math.isfinite(self.gradient_clip_norm) or (
                self.gradient_clip_norm <= 0.0):
            raise ValueError("invalid gradient clip or calibration step count")


@dataclass
class TrainedQSafeMember:
    model: SelectiveAdvantageQSafe
    seed: int
    bootstrap_trajectories: list[str]
    epoch_loss: list[float]
    temperature: float = 1.0


@dataclass
class TrainedQSafeEnsemble:
    ensemble: QSafeEnsemble
    members: list[TrainedQSafeMember]
    normalization: NormalizationStats | None = None
    command_vx: float | None = None
    privileged_dim: int | None = None
    train_split: str | None = None
    action_view: str | None = None
    action_dim: int | None = None


def _forward_loss(
    model: SelectiveAdvantageQSafe,
    batch,
    *,
    horizon_steps: int,
    loss_config: QSafeLossConfig,
):
    output = model(
        batch.observation_history,
        batch.nominal_action,
        batch.candidate_action,
        batch.privileged_state,
    )
    return qsafe_group_loss(
        output,
        fall=batch.fall,
        first_failure_step=batch.first_failure_step,
        max_tilt_rad=batch.max_tilt_rad,
        min_height_m=batch.min_height_m,
        candidate_mask=batch.candidate_mask,
        horizon_steps=horizon_steps,
        group_weight=batch.group_weight,
        config=loss_config,
    )


def train_qsafe_member(
    view: TorchGroupedView,
    network_config: QSafeNetworkConfig,
    training_config: QSafeTrainingConfig,
    loss_config: QSafeLossConfig,
    *,
    seed: int,
    bootstrap: bool = True,
) -> TrainedQSafeMember:
    if network_config.privileged_dim != view.privileged_dim:
        raise ValueError("network privileged_dim does not match dataset view")
    if network_config.action_dim != view.action_dim:
        raise ValueError("network action_dim does not match dataset action_view")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(training_config.device)
    model = SelectiveAdvantageQSafe(network_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    if bootstrap:
        training_indices, sampled_trajectories = trajectory_bootstrap_indices(
            view.trajectory_id, seed=seed)
    else:
        training_indices = view.all_indices()
        sampled_trajectories = np.unique(view.trajectory_id).tolist()
    rng = np.random.default_rng(seed)
    history: list[float] = []
    model.train()
    for _ in range(training_config.epochs):
        shuffled = training_indices.copy()
        rng.shuffle(shuffled)
        total_loss = 0.0
        batches = 0
        for start in range(0, len(shuffled), training_config.batch_size):
            indices = shuffled[start:start + training_config.batch_size]
            batch = view.batch(indices, device)
            result = _forward_loss(
                model,
                batch,
                horizon_steps=view.dataset.horizon_steps,
                loss_config=loss_config,
            )
            optimizer.zero_grad(set_to_none=True)
            result.total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), training_config.gradient_clip_norm,
                error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(result.total.detach().cpu())
            batches += 1
        history.append(total_loss / max(batches, 1))
    model.eval()
    return TrainedQSafeMember(
        model=model,
        seed=seed,
        bootstrap_trajectories=sampled_trajectories,
        epoch_loss=history,
    )


def _temperature_loss(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    empirical_fall_risk: torch.Tensor,
    candidate_mask: torch.Tensor,
    group_weight: torch.Tensor,
) -> torch.Tensor:
    scaled = logits / temperature
    per_candidate = F.binary_cross_entropy_with_logits(
        scaled, empirical_fall_risk, reduction="none")
    mask = candidate_mask.to(per_candidate.dtype)
    per_group = (per_candidate * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return torch.sum(per_group * group_weight) / torch.sum(group_weight)


def fit_temperature(
    model: SelectiveAdvantageQSafe,
    calibration: TorchGroupedView,
    *,
    device: torch.device | str,
    steps: int,
    batch_size: int = 256,
) -> float:
    if steps <= 0:
        return 1.0
    if batch_size <= 0:
        raise ValueError("calibration batch_size must be positive")
    model.eval()
    all_indices = calibration.all_indices()
    logits_parts = []
    target_parts = []
    mask_parts = []
    weight_parts = []
    # Cache only [G,K] calibration sufficient statistics.  A full 100k-group
    # observation forward does not fit typical GPU memory, while replica-level
    # Bernoulli NLL is exactly BCE against each candidate's empirical risk.
    with torch.no_grad():
        for start in range(0, len(all_indices), batch_size):
            batch = calibration.batch(
                all_indices[start:start + batch_size], device)
            logits_parts.append(model(
                batch.observation_history,
                batch.nominal_action,
                batch.candidate_action,
                batch.privileged_state,
            ).risk_logits.detach().cpu())
            target_parts.append(batch.fall.mean(dim=2).detach().cpu())
            mask_parts.append(batch.candidate_mask.detach().cpu())
            weight_parts.append(batch.group_weight.detach().cpu())
    target_device = torch.device(device)
    logits = torch.cat(logits_parts).to(target_device)
    empirical_fall_risk = torch.cat(target_parts).to(target_device)
    candidate_mask = torch.cat(mask_parts).to(target_device)
    group_weight = torch.cat(weight_parts).to(target_device)
    if not bool(torch.all(torch.isfinite(logits))):
        raise ValueError("model produced non-finite calibration logits")
    log_temperature = torch.zeros((), dtype=logits.dtype, device=logits.device,
                                  requires_grad=True)
    optimizer = torch.optim.Adam([log_temperature], lr=0.05)
    for _ in range(steps):
        temperature = torch.exp(torch.clamp(log_temperature, -4.0, 4.0))
        loss = _temperature_loss(
            logits,
            temperature,
            empirical_fall_risk,
            candidate_mask,
            group_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return float(torch.exp(torch.clamp(
        log_temperature.detach(), -4.0, 4.0)).cpu())


def train_qsafe_ensemble(
    train: TorchGroupedView,
    network_config: QSafeNetworkConfig,
    training_config: QSafeTrainingConfig,
    loss_config: QSafeLossConfig | None = None,
    calibration: TorchGroupedView | None = None,
) -> TrainedQSafeEnsemble:
    loss_config = QSafeLossConfig() if loss_config is None else loss_config
    if network_config.privileged_dim != train.privileged_dim:
        raise ValueError(
            "network privileged_dim must explicitly match the training view")
    if network_config.action_dim != train.action_dim:
        raise ValueError(
            "network action_dim must explicitly match the training action_view")
    if calibration is not None:
        audit_split_disjointness([train.dataset, calibration.dataset])
        if not calibration.normalization.equivalent_to(train.normalization):
            raise ValueError("calibration must use train-fitted normalization")
        if calibration.privileged_dim != train.privileged_dim:
            raise ValueError("train/calibration feature views differ")
        if calibration.action_view != train.action_view or (
                calibration.action_dim != train.action_dim):
            raise ValueError("train/calibration action views differ")
        if abs(calibration.command_vx - train.command_vx) > 1e-6:
            raise ValueError(
                "train/calibration command speeds differ for an unconditioned model")
    config = network_config
    members = []
    for member_index in range(training_config.ensemble_members):
        seed = training_config.seed + 1009 * member_index
        trained = train_qsafe_member(
            train,
            config,
            training_config,
            loss_config,
            seed=seed,
            bootstrap=True,
        )
        if calibration is not None:
            trained.temperature = fit_temperature(
                trained.model,
                calibration,
                device=training_config.device,
                steps=training_config.calibration_steps,
                batch_size=training_config.batch_size,
            )
        members.append(trained)
    ensemble = QSafeEnsemble(
        [member.model for member in members],
        temperatures=[member.temperature for member in members],
    )
    return TrainedQSafeEnsemble(
        ensemble=ensemble,
        members=members,
        normalization=train.normalization,
        command_vx=train.command_vx,
        privileged_dim=train.privileged_dim,
        train_split=str(train.dataset.manifest["split"]),
        action_view=train.action_view,
        action_dim=train.action_dim,
    )


@torch.no_grad()
def predict_qsafe_ensemble(
    trained: TrainedQSafeEnsemble,
    view: TorchGroupedView,
    *,
    device: torch.device | str,
    batch_size: int = 256,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if any(value is None for value in (
            trained.normalization,
            trained.command_vx,
            trained.privileged_dim,
            trained.action_view,
            trained.action_dim,
    )):
        raise ValueError("trained ensemble is missing preprocessing provenance")
    if trained.normalization is not None and not (
            view.normalization.equivalent_to(trained.normalization)):
        raise ValueError("prediction view must use the train-fitted normalization")
    if trained.privileged_dim is not None and (
            view.privileged_dim != trained.privileged_dim):
        raise ValueError("prediction and trained feature views differ")
    if trained.action_view is not None and view.action_view != trained.action_view:
        raise ValueError("prediction and trained action views differ")
    if trained.action_dim is not None and view.action_dim != trained.action_dim:
        raise ValueError("prediction and trained action dimensions differ")
    if trained.command_vx is not None and abs(
            view.command_vx - trained.command_vx) > 1e-6:
        raise ValueError(
            "prediction command speed differs for an unconditioned Q_safe")
    ensemble = trained.ensemble.to(device).eval()
    predictions = np.full(
        (view.group_count, view.dataset.candidate_count),
        np.nan,
        dtype=np.float32,
    )
    indices = view.all_indices()
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        batch = view.batch(selected, device)
        output = ensemble.predict(
            batch.observation_history,
            batch.nominal_action,
            batch.candidate_action,
            batch.privileged_state,
        )
        value = output.risk_mean.detach().cpu().numpy().astype(np.float32)
        if not np.all(np.isfinite(value[batch.candidate_mask.cpu().numpy()])):
            raise ValueError("ensemble produced non-finite valid predictions")
        predictions[selected] = value
    predictions[~view.mask] = np.nan
    return predictions
