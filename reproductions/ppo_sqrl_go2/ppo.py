"""Clipped PPO matching the registered MjLab/RSL-RL training geometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.modules import EmpiricalNormalization, GaussianDistribution, MLP

from .buffers import TaskRollout
from .dual import ProjectedDual, frozen_qsafe_penalty


@dataclass(frozen=True)
class PpoConfig:
    actor_observation_dim: int = 47
    critic_observation_dim: int = 74
    action_dim: int = 12
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    epochs: int = 5
    mini_batches: int = 4
    learning_rate: float = 1e-3
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coefficient: float = 0.01
    value_coefficient: float = 1.0
    clip: float = 0.2
    max_gradient_norm: float = 1.0
    desired_kl: float = 0.01


class GaussianActor(nn.Module):
    def __init__(self, cfg: PpoConfig):
        super().__init__()
        self.cfg = cfg
        self.normalizer = EmpiricalNormalization(cfg.actor_observation_dim)
        self.network = MLP(
            cfg.actor_observation_dim, cfg.action_dim, cfg.hidden_dims, "elu")
        self.output_distribution = GaussianDistribution(
            cfg.action_dim, init_std=1.0, std_type="scalar")

    def distribution(self, observation: torch.Tensor) -> Normal:
        mean = self.network(self.normalizer(observation))
        self.output_distribution.update(mean)
        return self.output_distribution._distribution

    def sample(self, observation: torch.Tensor, *, reparameterized: bool = False,
               deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        action = (distribution.mean if deterministic else
                  distribution.rsample() if reparameterized else distribution.sample())
        return (action, distribution.log_prob(action).sum(-1),
                distribution.mean, distribution.stddev)


class ValueNetwork(nn.Module):
    def __init__(self, cfg: PpoConfig):
        super().__init__()
        self.normalizer = EmpiricalNormalization(cfg.critic_observation_dim)
        self.network = MLP(
            cfg.critic_observation_dim, 1, cfg.hidden_dims, "elu")

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(self.normalizer(observation)).squeeze(-1)


class PpoLearner:
    def __init__(self, cfg: PpoConfig, *, device: str | torch.device,
                 seed: int):
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = GaussianActor(cfg).to(self.device)
        self.value = ValueNetwork(cfg).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.value.parameters()),
            lr=cfg.learning_rate)
        self.learning_rate = cfg.learning_rate
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.updates = 0

    @torch.no_grad()
    def act(self, observation: torch.Tensor, critic_observation: torch.Tensor,
            *, deterministic: bool = False):
        action, log_probability, mean, std = self.actor.sample(
            observation, deterministic=deterministic)
        return action, log_probability, self.value(critic_observation), mean, std

    @torch.no_grad()
    def update_normalizers(self, actor_observation: torch.Tensor,
                           critic_observation: torch.Tensor) -> None:
        self.actor.normalizer.update(actor_observation)
        self.value.normalizer.update(critic_observation)

    def update(
        self,
        rollout: TaskRollout,
        *,
        safety_critic: nn.Module | None = None,
        dual: ProjectedDual | None = None,
        epsilon_safe: float = 0.1,
        to_critic_action=lambda value: value,
    ) -> dict[str, float]:
        safe = safety_critic is not None or dual is not None
        if safe and (safety_critic is None or dual is None):
            raise ValueError("safe PPO requires both frozen Q_safe and dual")
        totals = {name: 0.0 for name in (
            "surrogate", "value", "entropy", "safety_penalty", "violation", "kl")}
        count = 0
        for batch in rollout.batches(
                self.cfg.mini_batches, self.cfg.epochs, self.generator):
            distribution = self.actor.distribution(batch["actor_observation"])
            log_probability = distribution.log_prob(batch["action"]).sum(-1)
            entropy = distribution.entropy().sum(-1).mean()
            ratio = torch.exp(log_probability - batch["old_log_probability"])
            surrogate = -batch["advantage"] * ratio
            clipped = -batch["advantage"] * ratio.clamp(
                1.0 - self.cfg.clip, 1.0 + self.cfg.clip)
            surrogate_loss = torch.maximum(surrogate, clipped).mean()
            value = self.value(batch["critic_observation"])
            clipped_value = batch["old_value"] + (
                value - batch["old_value"]).clamp(-self.cfg.clip, self.cfg.clip)
            value_loss = torch.maximum(
                (value - batch["return"]).square(),
                (clipped_value - batch["return"]).square()).mean()
            loss = (surrogate_loss + self.cfg.value_coefficient * value_loss
                    - self.cfg.entropy_coefficient * entropy)
            penalty = torch.zeros((), device=self.device)
            violation = torch.zeros((), device=self.device)
            if safe:
                candidate = distribution.rsample()
                penalty, values = frozen_qsafe_penalty(
                    safety_critic, batch["qsafe_observation"],
                    to_critic_action(candidate), epsilon=epsilon_safe,
                    dual_value=dual.value)
                violation = values.mean()
                loss = loss + penalty
            with torch.no_grad():
                old = Normal(batch["old_mean"], batch["old_std"])
                kl = torch.distributions.kl_divergence(old, distribution).sum(-1).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.value.parameters()),
                self.cfg.max_gradient_norm)
            self.optimizer.step()
            totals["surrogate"] += float(surrogate_loss.detach())
            totals["value"] += float(value_loss.detach())
            totals["entropy"] += float(entropy.detach())
            totals["safety_penalty"] += float(penalty.detach())
            totals["violation"] += float(violation.detach())
            totals["kl"] += float(kl)
            count += 1
        mean_kl = totals["kl"] / count
        if mean_kl > 2 * self.cfg.desired_kl:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        elif 0 < mean_kl < self.cfg.desired_kl / 2:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate
        self.updates += 1
        return {f"ppo/{name}": value / count for name, value in totals.items()} | {
            "ppo/learning_rate": self.learning_rate,
            "ppo/updates": float(self.updates),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "config": asdict(self.cfg),
            "actor": self.actor.state_dict(),
            "value": self.value.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "learning_rate": self.learning_rate,
            "updates": self.updates,
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, value: dict[str, object], *, optimizer: bool = True) -> None:
        self.actor.load_state_dict(value["actor"])  # type: ignore[arg-type]
        self.value.load_state_dict(value["value"])  # type: ignore[arg-type]
        if optimizer:
            self.optimizer.load_state_dict(value["optimizer"])  # type: ignore[arg-type]
        self.learning_rate = float(value.get("learning_rate", self.cfg.learning_rate))
        self.updates = int(value.get("updates", 0))
        if "generator_state" in value:
            self.generator.set_state(value["generator_state"])  # type: ignore[arg-type]
