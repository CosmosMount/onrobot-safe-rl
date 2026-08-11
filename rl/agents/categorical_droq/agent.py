from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, MutableMapping, cast

import gymnasium as gym
import torch
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler

from rl.agents.base.agent import BaseAgent
from rl.agents.base.inference_snapshot import inference_snapshot
from rl.agents.base.network import Network
from rl.agents.base.update import PolicyUpdateRequest, UpdateCounters
from rl.agents.droq.agent import DroQConfig, _resolve_device
from rl.agents.droq.network import DroQActor, DroQTemperature
from rl.agents.categorical_droq.network import CategoricalDroQEnsembleCritic
from rl.agents.categorical_droq.update import update_actor, update_critic, update_temperature
from rl.buffers.torch_buffer import TorchUniformBuffer
from rl.utils.types import NDArray, Tensor


@dataclass
class CategoricalDroQConfig(DroQConfig):
    critic_num_bins: int
    critic_min_v: float
    critic_max_v: float


class CategoricalDroQAgent(BaseAgent[CategoricalDroQConfig]):
    def __init__(self, observation_space: gym.spaces.Space[NDArray],
                 action_space: gym.spaces.Space[NDArray], env_info: dict[str, Any],
                 cfg: CategoricalDroQConfig):
        self._critic_observation_dim = int(observation_space.shape[-1])  # type: ignore[union-attr]
        self._action_dim = int(action_space.shape[-1])  # type: ignore[union-attr]
        self._actor_observation_dim = (
            int(env_info["actor_observation_size"][-1])
            if cfg.asymmetric_observation else self._critic_observation_dim)
        super().__init__(observation_space, action_space, env_info, cfg)
        self._device = _resolve_device(cfg.device_type)
        if cfg.num_min_qs is not None and not 1 <= cfg.num_min_qs <= cfg.num_qs:
            raise ValueError("num_min_qs must be in [1, num_qs]")
        if cfg.actor_q_reduction not in {"mean", "min"}:
            raise ValueError("actor_q_reduction must be mean or min")
        torch.manual_seed(cfg.seed)
        fused = self._device.type == "cuda" and torch.cuda.is_available()
        actor_net = DroQActor(self._actor_observation_dim, self._action_dim, cfg.hidden_dims).to(self._device)
        self._actor = Network(actor_net, optim.Adam(actor_net.parameters(), lr=cfg.actor_lr, fused=fused))
        critic_net = CategoricalDroQEnsembleCritic(
            self._critic_observation_dim, self._action_dim, cfg.hidden_dims,
            cfg.num_qs, cfg.critic_num_bins, cfg.critic_min_v, cfg.critic_max_v,
            dropout_rate=cfg.critic_dropout_rate, use_layer_norm=cfg.critic_layer_norm).to(self._device)
        self._critic = Network(critic_net, optim.Adam(critic_net.parameters(), lr=cfg.critic_lr, fused=fused))
        target_net = CategoricalDroQEnsembleCritic(
            self._critic_observation_dim, self._action_dim, cfg.hidden_dims,
            cfg.num_qs, cfg.critic_num_bins, cfg.critic_min_v, cfg.critic_max_v,
            dropout_rate=cfg.critic_dropout_rate, use_layer_norm=cfg.critic_layer_norm).to(self._device)
        target_net.load_state_dict(critic_net.state_dict())
        self._target_critic = Network(target_net, ema_source=self._critic, ema_tau=cfg.critic_target_update_tau)
        temp_net = DroQTemperature(cfg.temp_initial_value).to(self._device)
        self._temperature = Network(temp_net, optim.Adam(temp_net.parameters(), lr=cfg.temp_lr, fused=fused))
        self._target_entropy = -0.5 * self._action_dim if cfg.target_entropy is None else float(cfg.target_entropy)
        self._grad_scaler = GradScaler(device=self._device.type, enabled=cfg.use_amp)
        self._replay_buffer = TorchUniformBuffer(
            observation_space, action_space, cfg.n_step, cfg.gamma, cfg.buffer_max_length,
            cfg.buffer_min_length, cfg.sample_batch_size, cfg.buffer_device_type)
        self._update_step = 0
        self._update_counters = UpdateCounters()

    def _actor_observations(self, observations: Tensor) -> torch.Tensor:
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self._device)
        return obs[:, :self._actor_observation_dim] if self._cfg.asymmetric_observation else obs

    def sample_actions(self, interaction_step: int, prev_transition: MutableMapping[str, Tensor], training: bool) -> Tensor:
        with torch.no_grad():
            actions, _ = self._actor(self._actor_observations(prev_transition["next_observation"]), training=False, sample=training)
        return actions.cpu().numpy()

    def get_update_step(self) -> int: return self._update_step
    def get_policy_update_step(self) -> int: return int(self._update_counters.actor_steps)
    def get_update_counters(self) -> dict[str, int]:
        return {k: int(v) for k, v in self._update_counters.state_dict().items() if k != "legacy_counters_inferred"}

    def _batch(self) -> dict[str, torch.Tensor]:
        batch = cast(dict[str, torch.Tensor], self._replay_buffer.sample())
        batch = {k: v.to(self._device, non_blocking=True) for k, v in batch.items()}
        if self._cfg.asymmetric_observation:
            batch["actor_observation"] = batch["observation"][:, :self._actor_observation_dim]
            batch["actor_next_observation"] = batch["next_observation"][:, :self._actor_observation_dim]
        else:
            batch["actor_observation"] = batch["observation"]
            batch["actor_next_observation"] = batch["next_observation"]
        return batch

    def update_policy_steps(self, request: PolicyUpdateRequest) -> dict[str, Any]:
        if request.critic_updates <= 0:
            raise ValueError("critic_updates must be positive")
        metrics: dict[str, Any] = {}
        last_batch = None
        actor_calls = 0
        for _ in range(request.critic_updates):
            last_batch = self._batch()
            info = update_critic(self._actor, self._critic, self._target_critic, self._temperature,
                                 last_batch, self._critic.network.support, self._device,
                                 num_min_qs=self._cfg.num_min_qs, sampled_backup=self._cfg.sampled_backup,
                                 target_q_min=self._cfg.target_q_min, target_q_max=self._cfg.target_q_max,
                                 use_amp=self._cfg.use_amp, grad_scaler=self._grad_scaler)
            metrics.update({k: float(v.item()) for k, v in info.items()})
            self._update_counters.critic_steps += 1
            self._update_counters.target_steps += 1
        policy_before = self._update_counters.policy_steps
        self._update_counters.policy_steps += request.policy_steps
        if last_batch is not None:
            first = policy_before // self._cfg.actor_update_interval + 1
            last = self._update_counters.policy_steps // self._cfg.actor_update_interval
            for _ in range(max(0, last - first + 1)):
                info = update_actor(self._actor, self._critic, self._temperature, last_batch,
                                    self._cfg.actor_q_reduction, self._device, self._cfg.use_amp, self._grad_scaler)
                metrics.update({k: float(v.item()) for k, v in info.items()})
                temp = update_temperature(self._temperature, info["actor/entropy"], self._target_entropy)
                metrics.update({k: float(v.item()) for k, v in temp.items()})
                self._update_counters.actor_steps += 1
                self._update_counters.temperature_steps += 1
                actor_calls += 1
        self._update_step = self._update_counters.critic_steps
        metrics.update({"updates/call_policy_steps": float(request.policy_steps),
                        "updates/call_critic_steps": float(request.critic_updates),
                        "updates/call_actor_steps": float(actor_calls),
                        "updates/total_policy_steps": float(self._update_counters.policy_steps),
                        "updates/total_critic_steps": float(self._update_counters.critic_steps),
                        "updates/total_actor_steps": float(self._update_counters.actor_steps),
                        "updates/total_temperature_steps": float(self._update_counters.temperature_steps),
                        "updates/total_target_steps": float(self._update_counters.target_steps)})
        return metrics

    def process_transition(self, transition: MutableMapping[str, Tensor]) -> None: self._replay_buffer.add_batch(transition)
    def can_start_training(self) -> bool: return self._replay_buffer.can_sample()
    def update(self) -> dict[str, Any]: return self.update_policy_steps(PolicyUpdateRequest(1, 1))
    def export_inference_snapshot(self, *, snapshot_version: int) -> dict[str, Any]:
        return inference_snapshot(agent_type="categorical_droq", snapshot_version=snapshot_version,
                                  actor=self._actor.network, counters=self.get_update_counters())

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self._actor.save(os.path.join(path, "actor.pt")); self._critic.save(os.path.join(path, "critic.pt"))
        self._target_critic.save(os.path.join(path, "target_critic.pt")); self._temperature.save(os.path.join(path, "temperature.pt"))
        torch.save({"update_step": self._update_step, "update_counters": self._update_counters.state_dict(),
                    "grad_scaler_state_dict": self._grad_scaler.state_dict()}, os.path.join(path, "agent_state.pt"))
    def save_replay_buffer(self, path: str) -> None: self._replay_buffer.save(os.path.join(path, "replay_buffer.pt"))
    def load(self, path: str) -> None:
        load = self._cfg.load_optimizer
        self._actor.load(os.path.join(path, "actor.pt"), load_optimizer=load); self._critic.load(os.path.join(path, "critic.pt"), load_optimizer=load)
        self._target_critic.load(os.path.join(path, "target_critic.pt"), load_optimizer=False); self._temperature.load(os.path.join(path, "temperature.pt"), load_optimizer=load)
        if load:
            state = torch.load(os.path.join(path, "agent_state.pt"), map_location=self._device)
            self._update_step = int(state["update_step"]); self._update_counters.load_state_dict(state["update_counters"])
            self._grad_scaler.load_state_dict(state.get("grad_scaler_state_dict", {}))
    def load_replay_buffer(self, path: str) -> None: self._replay_buffer.load(os.path.join(path, "replay_buffer.pt"))
    def get_metrics(self) -> dict[str, Any]: return {}
