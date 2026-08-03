from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, MutableMapping, Sequence, cast, Optional

import gymnasium as gym
import torch
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler

from rl.agents.base.agent import BaseAgent
from rl.agents.base.inference_snapshot import inference_snapshot
from rl.agents.base.network import Network
from rl.agents.base.update import PolicyUpdateRequest, UpdateCounters
from rl.agents.droq.network import DroQActor, DroQTemperature
from rl.agents.flashsac.reward_normalization import RewardNormalizer
from rl.agents.livesac.constants import LIVESAC_UTD_RATIO
from rl.agents.livesac.network import LiveSACDoubleCritic
from rl.agents.livesac.update import update_actor, update_critic, update_temperature
from rl.buffers.torch_buffer import TorchUniformBuffer
from rl.utils.types import NDArray, Tensor


@dataclass
class LiveSACConfig:
    seed: int; device_type: str; buffer_device_type: str; buffer_max_length: int; buffer_min_length: int; sample_batch_size: int
    actor_lr: float; critic_lr: float; temp_lr: float; actor_hidden_dims: Sequence[int]
    critic_hidden_dim: int; critic_expansion: int; critic_num_blocks: int; critic_num_qs: int; critic_num_bins: int
    critic_min_v: float; critic_max_v: float; critic_target_update_tau: float
    normalize_reward: bool; normalized_G_max: float; gamma: float; n_step: int
    target_entropy: Optional[float]; temp_initial_value: float; asymmetric_observation: bool
    actor_update_interval: int; actor_update_unit: str; use_compile: bool; compile_mode: str; use_amp: bool
    load_optimizer: bool; load_reward_normalizer: bool


def _device(name: str) -> torch.device:
    return torch.device(name if name.startswith("cuda:") else ("cuda:0" if name.startswith("cuda") else "cpu"))


class LiveSACAgent(BaseAgent[LiveSACConfig]):
    def __init__(self, observation_space: gym.spaces.Space[NDArray], action_space: gym.spaces.Space[NDArray],
                 env_info: dict[str, Any], cfg: LiveSACConfig):
        self._validate_config(cfg)
        super().__init__(observation_space, action_space, env_info, cfg)
        self._device = _device(cfg.device_type)
        self._critic_observation_dim = int(observation_space.shape[-1])  # type: ignore[union-attr]
        self._action_dim = int(action_space.shape[-1])  # type: ignore[union-attr]
        self._actor_observation_dim = int(env_info["actor_observation_size"][-1]) if cfg.asymmetric_observation else self._critic_observation_dim
        fused = self._device.type == "cuda" and torch.cuda.is_available()
        torch.manual_seed(cfg.seed)
        actor_net = DroQActor(self._actor_observation_dim, self._action_dim, cfg.actor_hidden_dims).to(self._device)
        self._actor = Network(actor_net, optim.Adam(actor_net.parameters(), lr=cfg.actor_lr, fused=fused))
        critic_net = LiveSACDoubleCritic(self._critic_observation_dim, self._action_dim, cfg.critic_hidden_dim, cfg.critic_expansion,
                                         cfg.critic_num_blocks, cfg.critic_num_qs, cfg.critic_num_bins, cfg.critic_min_v, cfg.critic_max_v).to(self._device)
        self._critic = Network(critic_net, optim.Adam(critic_net.parameters(), lr=cfg.critic_lr, fused=fused), use_weight_normalization=False)
        target_net = LiveSACDoubleCritic(self._critic_observation_dim, self._action_dim, cfg.critic_hidden_dim, cfg.critic_expansion,
                                         cfg.critic_num_blocks, cfg.critic_num_qs, cfg.critic_num_bins, cfg.critic_min_v, cfg.critic_max_v).to(self._device)
        target_net.load_state_dict(critic_net.state_dict())
        self._target_critic = Network(target_net, use_weight_normalization=False, ema_source=self._critic, ema_tau=cfg.critic_target_update_tau)
        temp_net = DroQTemperature(cfg.temp_initial_value).to(self._device)
        self._temperature = Network(temp_net, optim.Adam(temp_net.parameters(), lr=cfg.temp_lr, fused=fused))
        self._target_entropy = -0.5 * self._action_dim if cfg.target_entropy is None else float(cfg.target_entropy)
        self._support = critic_net.critics[0].bin_values
        self._reward_normalizer = RewardNormalizer(cfg.gamma, cfg.normalized_G_max, cfg.load_reward_normalizer, self._device)
        self._grad_scaler = GradScaler(device=self._device.type, enabled=cfg.use_amp)
        self._replay_buffer = TorchUniformBuffer(observation_space, action_space, cfg.n_step, cfg.gamma, cfg.buffer_max_length,
                                                  cfg.buffer_min_length, cfg.sample_batch_size, cfg.buffer_device_type)
        self._update_counters = UpdateCounters(); self._update_step = 0; self._metrics: dict[str, Any] = {}

    @staticmethod
    def _validate_config(cfg: LiveSACConfig) -> None:
        fixed = ((list(cfg.actor_hidden_dims), [256, 256], "actor_hidden_dims"), (cfg.critic_hidden_dim, 256, "critic_hidden_dim"),
                 (cfg.critic_expansion, 2, "critic_expansion"), (cfg.critic_num_blocks, 1, "critic_num_blocks"), (cfg.critic_num_qs, 2, "critic_num_qs"),
                 (cfg.critic_num_bins, 101, "critic_num_bins"), (cfg.critic_min_v, -5.0, "critic_min_v"), (cfg.critic_max_v, 5.0, "critic_max_v"),
                 (cfg.normalize_reward, True, "normalize_reward"), (cfg.normalized_G_max, 5.0, "normalized_G_max"), (cfg.actor_update_interval, 1, "actor_update_interval"), (cfg.actor_update_unit, "policy_step", "actor_update_unit"))
        for actual, expected, name in fixed:
            if actual != expected: raise ValueError(f"LiveSAC v1.0 requires {name}={expected!r}")
        if not (0 < cfg.gamma <= 1) or cfg.n_step <= 0 or any(x <= 0 for x in (cfg.actor_lr, cfg.critic_lr, cfg.temp_lr)) or cfg.buffer_min_length <= 0 or cfg.sample_batch_size <= 0:
            raise ValueError("LiveSAC v1.0 requires valid gamma, n_step, learning rates, buffer_min_length, and sample_batch_size")
        if not (0 < cfg.critic_target_update_tau <= 1) or cfg.critic_min_v >= cfg.critic_max_v or cfg.temp_initial_value <= 0:
            raise ValueError("LiveSAC v1.0 requires valid critic support, target tau, and temperature")

    def _actor_obs(self, value: Tensor) -> torch.Tensor:
        obs = torch.as_tensor(value, dtype=torch.float32, device=self._device)
        return obs[:, :self._actor_observation_dim] if self._cfg.asymmetric_observation else obs

    def sample_actions(self, interaction_step: int, prev_transition: MutableMapping[str, Tensor], training: bool) -> Tensor:
        with torch.no_grad():
            action, _ = self._actor(self._actor_obs(prev_transition["next_observation"]), training=False, sample=training)
        return action.cpu().numpy()

    def process_transition(self, transition: MutableMapping[str, Tensor]) -> None:
        repeat = transition.get("replay_repeat_index")
        reward = torch.as_tensor(transition["reward"], device=self._device, dtype=torch.float32)
        terminated = torch.as_tensor(transition["terminated"], device=self._device).bool()
        truncated = torch.as_tensor(transition["truncated"], device=self._device).bool()
        if repeat is None:
            self._reward_normalizer.update_reward_stats(reward, terminated, truncated)
        else:
            first = torch.as_tensor(repeat, device=self._device).reshape(-1) == 0
            if first.any(): self._reward_normalizer.update_reward_stats(reward.reshape(-1)[first], terminated.reshape(-1)[first], truncated.reshape(-1)[first])
        self._replay_buffer.add_batch(transition)

    def can_start_training(self) -> bool: return self._replay_buffer.can_sample()
    def get_update_step(self) -> int: return self._update_step
    def get_policy_update_step(self) -> int: return self._update_counters.actor_steps
    def get_inference_observation_dim(self) -> int: return self._actor_observation_dim
    def get_update_counters(self) -> dict[str, int]: return {k: int(v) for k, v in self._update_counters.state_dict().items() if k != "legacy_counters_inferred"}

    def _batch(self) -> dict[str, torch.Tensor]:
        batch = cast(dict[str, torch.Tensor], self._replay_buffer.sample())
        batch = {k: v.to(self._device, non_blocking=True) for k, v in batch.items()}
        batch["reward"] = self._reward_normalizer.normalize_rewards(batch["reward"])
        batch["actor_observation"] = self._actor_obs(batch["observation"]); batch["actor_next_observation"] = self._actor_obs(batch["next_observation"])
        return batch

    def _attach(self, metrics: dict[str, Any], call: dict[str, int]) -> dict[str, Any]:
        c = self._update_counters
        for name, value in call.items(): metrics[f"updates/call_{name}"] = float(value)
        for name in ("policy_steps", "critic_steps", "target_steps", "actor_steps", "temperature_steps", "auxiliary_steps"):
            metrics[f"updates/total_{name}"] = float(getattr(c, name))
        return metrics

    def update_policy_steps(self, request: PolicyUpdateRequest) -> dict[str, Any]:
        if request.policy_steps != 1 or request.critic_updates_per_policy_step != LIVESAC_UTD_RATIO:
            raise ValueError("LiveSAC v1.0 requires exactly one policy step and a fixed critic_updates_per_policy_step=5")
        metrics: dict[str, Any] = {}; last_batch = None
        for _ in range(LIVESAC_UTD_RATIO):
            last_batch = self._batch()
            info = update_critic(self._actor, self._critic, self._target_critic, self._temperature, last_batch, self._support, self._device, self._cfg.critic_min_v, self._cfg.critic_max_v)
            metrics.update({k: float(v.item()) for k, v in info.items()}); self._update_counters.critic_steps += 1; self._update_counters.target_steps += 1
        assert last_batch is not None
        actor_info = update_actor(self._actor, self._critic, self._temperature, last_batch)
        metrics.update({k: float(v.item()) for k, v in actor_info.items()}); self._update_counters.actor_steps += 1
        temp_info = update_temperature(self._temperature, actor_info["actor/entropy"], self._target_entropy)
        metrics.update({k: float(v.item()) for k, v in temp_info.items()}); self._update_counters.temperature_steps += 1
        self._update_counters.policy_steps += 1; self._update_step = self._update_counters.critic_steps
        metrics = self._attach(metrics, {"policy_steps": 1, "critic_steps": 5, "target_steps": 5, "actor_steps": 1, "temperature_steps": 1, "auxiliary_steps": 0})
        self._metrics = dict(metrics)
        return metrics

    def update(self) -> dict[str, Any]:
        return self.update_policy_steps(PolicyUpdateRequest(1, LIVESAC_UTD_RATIO))

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self._actor.save(os.path.join(path, "actor.pt")); self._critic.save(os.path.join(path, "critic.pt")); self._target_critic.save(os.path.join(path, "target_critic.pt")); self._temperature.save(os.path.join(path, "temperature.pt"))
        self._reward_normalizer.save(os.path.join(path, "reward_normalizer.pt"))
        torch.save({"update_step": self._update_step, "update_counters": self._update_counters.state_dict(), "grad_scaler_state_dict": self._grad_scaler.state_dict()}, os.path.join(path, "agent_state.pt"))

    def save_replay_buffer(self, path: str) -> None: self._replay_buffer.save(os.path.join(path, "replay_buffer.pt"))
    def load(self, path: str) -> None:
        load = self._cfg.load_optimizer
        self._actor.load(os.path.join(path, "actor.pt"), load_optimizer=load); self._critic.load(os.path.join(path, "critic.pt"), load_optimizer=load); self._target_critic.load(os.path.join(path, "target_critic.pt"), load_optimizer=False); self._temperature.load(os.path.join(path, "temperature.pt"), load_optimizer=load)
        if self._cfg.load_reward_normalizer and os.path.exists(os.path.join(path, "reward_normalizer.pt")): self._reward_normalizer.load(os.path.join(path, "reward_normalizer.pt"))
        if load:
            state = torch.load(os.path.join(path, "agent_state.pt"), map_location=self._device); self._update_step = int(state["update_step"]); self._update_counters.load_state_dict(state["update_counters"]); self._grad_scaler.load_state_dict(state.get("grad_scaler_state_dict", {}))
    def load_replay_buffer(self, path: str) -> None: self._replay_buffer.load(os.path.join(path, "replay_buffer.pt"))
    def get_metrics(self) -> dict[str, Any]: return dict(self._metrics)
    def export_inference_snapshot(self, *, snapshot_version: int) -> dict[str, Any]: return inference_snapshot(agent_type="livesac", snapshot_version=snapshot_version, actor=self._actor.network, counters=self.get_update_counters())
