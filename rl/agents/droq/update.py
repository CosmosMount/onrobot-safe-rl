from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler

from rl.agents.base.network import Network
from rl.buffers.buffer import Batch


def add_prefix_to_keys(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}/{k}": v for k, v in d.items()}


def _optimizer_step(
    loss: torch.Tensor,
    network: Network,
    *,
    use_amp: bool,
    grad_scaler: Optional[GradScaler],
) -> None:
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


def update_actor(
    actor: Network,
    critic: Network,
    temperature: Network,
    batch: Batch,
    actor_q_reduction: str,
    device: torch.device,
    use_amp: bool,
    grad_scaler: Optional[GradScaler],
) -> dict[str, torch.Tensor]:
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        actions, info = actor(
            observations=batch["actor_observation"],
            training=True,
            sample=True,
        )
        log_probs = info["log_prob"]

        critic.network.requires_grad_(False)
        qs, _ = critic(
            observations=batch["observation"],
            actions=actions,
            training=True,
        )
        critic.network.requires_grad_(True)

        if actor_q_reduction == "min":
            q = qs.min(dim=0).values
        else:
            q = qs.mean(dim=0)
        temp_value = temperature().detach()
        actor_loss = (temp_value * log_probs - q).mean()
        entropy = -log_probs.mean()

    _optimizer_step(actor_loss, actor, use_amp=use_amp, grad_scaler=grad_scaler)

    return add_prefix_to_keys(
        {
            "loss": actor_loss.detach(),
            "entropy": entropy.detach(),
            "q": q.mean().detach(),
            "action_mean": actions.mean().detach(),
            "action_std": actions.std(unbiased=False).detach(),
            "action_saturation": (actions.abs() >= 0.99).float().mean().detach(),
            "mean_abs": info["mean"].abs().mean().detach(),
            "log_std_mean": info["log_std"].mean().detach(),
            "log_std_min": info["log_std"].min().detach(),
            "log_std_max": info["log_std"].max().detach(),
        },
        "actor",
    )


def update_critic(
    actor: Network,
    critic: Network,
    target_critic: Network,
    temperature: Network,
    batch: Batch,
    num_min_qs: Optional[int],
    sampled_backup: bool,
    target_q_min: Optional[float],
    target_q_max: Optional[float],
    device: torch.device,
    use_amp: bool,
    grad_scaler: Optional[GradScaler],
) -> dict[str, torch.Tensor]:
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        with torch.no_grad():
            next_actions, next_info = actor(
                observations=batch["actor_next_observation"],
                training=False,
                sample=sampled_backup,
            )
            next_qs, _ = target_critic(
                observations=batch["next_observation"],
                actions=next_actions,
                training=True,
            )
            if num_min_qs is not None and num_min_qs < next_qs.shape[0]:
                idx = torch.randperm(next_qs.shape[0], device=next_qs.device)[:num_min_qs]
                next_qs = next_qs.index_select(0, idx)
            next_q = next_qs.min(dim=0).values

            target_q = batch["reward"] + batch["discount"] * next_q
            if sampled_backup:
                target_q = target_q - batch["discount"] * temperature() * next_info["log_prob"]
            if target_q_min is not None or target_q_max is not None:
                target_q = torch.clamp(
                    target_q,
                    min=-torch.inf if target_q_min is None else target_q_min,
                    max=torch.inf if target_q_max is None else target_q_max,
                )

        pred_qs, _ = critic(
            observations=batch["observation"],
            actions=batch["action"],
            training=True,
        )
        critic_loss = F.mse_loss(pred_qs, target_q.unsqueeze(0).expand_as(pred_qs))

    _optimizer_step(critic_loss, critic, use_amp=use_amp, grad_scaler=grad_scaler)
    target_critic.ema_update_parameters()

    return add_prefix_to_keys(
        {
            "loss": critic_loss.detach(),
            "q": pred_qs.mean().detach(),
            "q_min": pred_qs.min().detach(),
            "q_max": pred_qs.max().detach(),
            "target_q": target_q.mean().detach(),
            "target_q_min": target_q.min().detach(),
            "target_q_max": target_q.max().detach(),
        },
        "critic",
    )


def update_temperature(
    temperature: Network,
    entropy: torch.Tensor,
    target_entropy: float,
) -> dict[str, torch.Tensor]:
    temperature_value = temperature()
    temperature_loss = temperature_value * (entropy.detach() - target_entropy).mean()

    assert temperature.optimizer is not None
    temperature.optimizer.zero_grad(set_to_none=True)
    temperature_loss.backward()
    temperature.optimizer.step()
    if temperature.scheduler is not None:
        temperature.scheduler.step()

    return add_prefix_to_keys(
        {
            "value": temperature_value.detach(),
            "loss": temperature_loss.detach(),
        },
        "temperature",
    )
