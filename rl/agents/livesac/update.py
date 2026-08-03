from __future__ import annotations

from typing import Any
import torch
import torch.nn.functional as F

from rl.agents.base.network import Network
from rl.agents.livesac.categorical import boundary_mass, project_distribution, select_min_distribution


def _step(loss: torch.Tensor, network: Network) -> None:
    assert network.optimizer is not None
    network.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    network.optimizer.step()


def update_critic(actor: Network, critic: Network, target_critic: Network, temperature: Network,
                  batch: dict[str, torch.Tensor], support: torch.Tensor, device: torch.device,
                  min_v: float, max_v: float) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        next_actions, next_info = actor(batch["actor_next_observation"], training=False, sample=True)
        next_qs, next_critic_info = target_critic(batch["next_observation"], next_actions, training=False)
        selected_log_probs, _ = select_min_distribution(next_qs, next_critic_info["log_prob"])
        entropy_cost = temperature() * next_info["log_prob"]
        tz = batch["reward"].float().unsqueeze(-1) + batch["discount"].float().unsqueeze(-1) * (support.to(device) - entropy_cost.unsqueeze(-1))
        target_probs = project_distribution(tz, selected_log_probs.exp(), num_bins=support.numel(), min_v=min_v, max_v=max_v)
    _, info = critic(batch["observation"], batch["action"], training=True)
    pred = info["log_prob"]
    loss = -(target_probs.detach().unsqueeze(0) * pred).sum(dim=-1).mean()
    _step(loss, critic)
    target_critic.ema_update_parameters()
    pred_probs = pred.exp()
    low, high = boundary_mass(pred_probs)
    tlow, thigh = boundary_mass(target_probs)
    return {"critic/loss": loss.detach(), "critic/q_mean": (pred_probs * support).sum(-1).mean().detach(),
            "critic/q_min": (pred_probs * support).sum(-1).min().detach(), "critic/q_max": (pred_probs * support).sum(-1).max().detach(),
            "critic/target_q_mean": (target_probs * support).sum(-1).mean().detach(), "critic/target_q_min": (target_probs * support).sum(-1).min().detach(),
            "critic/target_q_max": (target_probs * support).sum(-1).max().detach(), "critic/pred_boundary_mass_low": low.detach(),
            "critic/pred_boundary_mass_high": high.detach(), "critic/target_boundary_mass_low": tlow.detach(),
            "critic/target_boundary_mass_high": thigh.detach()}


def update_actor(actor: Network, critic: Network, temperature: Network, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    assert actor.optimizer is not None
    actor.optimizer.zero_grad(set_to_none=True)
    flags = [p.requires_grad for p in critic.network.parameters()]
    try:
        for p in critic.network.parameters(): p.requires_grad_(False)
        actions, info = actor(batch["actor_observation"], training=True, sample=True)
        qs, _ = critic(batch["observation"], actions, training=True)
        q = qs.min(dim=0).values
        log_prob = info["log_prob"]
        loss = (temperature().detach() * log_prob - q).mean()
        loss.backward()
    finally:
        for p, flag in zip(critic.network.parameters(), flags): p.requires_grad_(flag)
    actor.optimizer.step()
    return {"actor/loss": loss.detach(), "actor/entropy": -log_prob.mean().detach(), "actor/q_mean": q.mean().detach(),
            "actor/action_mean": actions.mean().detach(), "actor/action_std": actions.std(unbiased=False).detach(),
            "actor/action_saturation": (actions.abs() >= .99).float().mean().detach(), "actor/log_std_mean": info["log_std"].mean().detach(),
            "actor/log_std_min": info["log_std"].min().detach(), "actor/log_std_max": info["log_std"].max().detach()}


def update_temperature(temperature: Network, entropy: torch.Tensor, target_entropy: float) -> dict[str, torch.Tensor]:
    value = temperature()
    loss = value * (entropy.detach() - target_entropy)
    _step(loss, temperature)
    return {"temperature/value": value.detach(), "temperature/loss": loss.detach()}
