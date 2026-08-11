from __future__ import annotations

from typing import Any, Optional

import torch
from torch.amp.grad_scaler import GradScaler

from rl.agents.base.network import Network


def project_distribution(values: torch.Tensor, probabilities: torch.Tensor, *,
                         num_bins: int, min_v: float, max_v: float) -> torch.Tensor:
    values = values.float().clamp(min_v, max_v)
    delta = (max_v - min_v) / (num_bins - 1)
    b = (values - min_v) / delta
    lower = b.floor().long().clamp(0, num_bins - 1)
    upper = b.ceil().long().clamp(0, num_bins - 1)
    target = torch.zeros_like(probabilities, device=values.device)
    lo_weight = (upper.float() - b).where(lower != upper, torch.zeros_like(b))
    hi_weight = (b - lower.float()).where(lower != upper, torch.zeros_like(b))
    target.scatter_add_(-1, lower, probabilities * lo_weight)
    target.scatter_add_(-1, upper, probabilities * hi_weight)
    target.scatter_add_(-1, lower, probabilities * (lower == upper).float())
    return target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def boundary_mass(probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return probabilities[..., 0].mean(), probabilities[..., -1].mean()


def _step(loss: torch.Tensor, network: Network, *, use_amp: bool,
          grad_scaler: Optional[GradScaler]) -> None:
    assert network.optimizer is not None
    network.optimizer.zero_grad(set_to_none=True)
    if use_amp:
        assert grad_scaler is not None
        grad_scaler.scale(loss).backward()
        grad_scaler.step(network.optimizer)
        grad_scaler.update()
    else:
        loss.backward()
        network.optimizer.step()
    if network.scheduler is not None:
        network.scheduler.step()


def update_critic(actor: Network, critic: Network, target_critic: Network,
                  temperature: Network, batch: dict[str, torch.Tensor],
                  support: torch.Tensor, device: torch.device, *,
                  num_min_qs: Optional[int], sampled_backup: bool,
                  target_q_min: Optional[float], target_q_max: Optional[float],
                  use_amp: bool, grad_scaler: Optional[GradScaler]) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        next_actions, next_info = actor(batch["actor_next_observation"], training=False, sample=True)
        next_qs, next_info_critic = target_critic(
            batch["next_observation"], next_actions, training=False)
        next_log_prob = next_info_critic["log_prob"]
        if num_min_qs is not None and num_min_qs < next_qs.shape[0]:
            indices = torch.randperm(next_qs.shape[0], device=device)[:num_min_qs]
            next_qs = next_qs.index_select(0, indices)
            next_log_prob = next_log_prob.index_select(0, indices)
        selected = next_qs.argmin(dim=0)
        batch_index = torch.arange(next_qs.shape[1], device=device)
        selected_log_prob = next_log_prob[selected, batch_index]
        entropy_cost = (temperature() * next_info["log_prob"]
                        if sampled_backup else torch.zeros_like(next_info["log_prob"]))
        discount = batch["discount"].float().unsqueeze(-1)
        tz = batch["reward"].float().unsqueeze(-1) + discount * support - discount * entropy_cost.unsqueeze(-1)
        tz = tz.clamp(min=-torch.inf if target_q_min is None else target_q_min,
                      max=torch.inf if target_q_max is None else target_q_max)
        target_probs = project_distribution(
            tz, selected_log_prob.exp(), num_bins=support.numel(),
            min_v=float(support[0]), max_v=float(support[-1]))

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        pred_qs, pred_info = critic(batch["observation"], batch["action"], training=True)
        pred_log_prob = pred_info["log_prob"]
        loss = -(target_probs.detach().unsqueeze(0) * pred_log_prob).sum(dim=-1).mean()
    _step(loss, critic, use_amp=use_amp, grad_scaler=grad_scaler)
    target_critic.ema_update_parameters()
    pred_probs = pred_log_prob.exp()
    low, high = boundary_mass(pred_probs)
    tlow, thigh = boundary_mass(target_probs)
    expected = pred_probs @ support
    target_expected = target_probs @ support
    return {"critic/loss": loss.detach(), "critic/q_mean": expected.mean().detach(),
            "critic/q_min": expected.min().detach(), "critic/q_max": expected.max().detach(),
            "critic/target_q_mean": target_expected.mean().detach(),
            "critic/target_q_min": target_expected.min().detach(),
            "critic/target_q_max": target_expected.max().detach(),
            "critic/pred_boundary_mass_low": low.detach(),
            "critic/pred_boundary_mass_high": high.detach(),
            "critic/target_boundary_mass_low": tlow.detach(),
            "critic/target_boundary_mass_high": thigh.detach()}


def update_actor(actor: Network, critic: Network, temperature: Network,
                 batch: dict[str, torch.Tensor], actor_q_reduction: str,
                 device: torch.device, use_amp: bool,
                 grad_scaler: Optional[GradScaler]) -> dict[str, torch.Tensor]:
    flags = [p.requires_grad for p in critic.network.parameters()]
    actor.optimizer.zero_grad(set_to_none=True)  # type: ignore[union-attr]
    try:
        for p in critic.network.parameters(): p.requires_grad_(False)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            actions, info = actor(batch["actor_observation"], training=True, sample=True)
            qs, _ = critic(batch["observation"], actions, training=True)
            q = qs.min(dim=0).values if actor_q_reduction == "min" else qs.mean(dim=0)
            loss = (temperature().detach() * info["log_prob"] - q).mean()
            if use_amp:
                assert grad_scaler is not None
                grad_scaler.scale(loss).backward()
            else:
                loss.backward()
    finally:
        for p, flag in zip(critic.network.parameters(), flags): p.requires_grad_(flag)
    if use_amp:
        assert grad_scaler is not None
        grad_scaler.step(actor.optimizer)  # type: ignore[arg-type]
        grad_scaler.update()
    else:
        actor.optimizer.step()  # type: ignore[union-attr]
    return {"actor/loss": loss.detach(), "actor/entropy": -info["log_prob"].mean().detach(),
            "actor/q_mean": q.mean().detach(), "actor/action_mean": actions.mean().detach(),
            "actor/action_std": actions.std(unbiased=False).detach(),
            "actor/action_saturation": (actions.abs() >= .99).float().mean().detach(),
            "actor/log_std_mean": info["log_std"].mean().detach(),
            "actor/log_std_min": info["log_std"].min().detach(),
            "actor/log_std_max": info["log_std"].max().detach()}


def update_temperature(temperature: Network, entropy: torch.Tensor,
                       target_entropy: float) -> dict[str, torch.Tensor]:
    value = temperature()
    loss = value * (entropy.detach() - target_entropy)
    _step(loss, temperature, use_amp=False, grad_scaler=None)
    return {"temperature/value": value.detach(), "temperature/loss": loss.detach()}
