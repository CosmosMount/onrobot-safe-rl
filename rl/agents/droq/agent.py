import os
from dataclasses import dataclass
from typing import Any, MutableMapping, Optional, Sequence, cast

import gymnasium as gym
import torch
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler

from rl.agents.base.agent import BaseAgent
from rl.agents.base.network import Network
from rl.agents.droq.network import DroQActor, DroQEnsembleCritic, DroQTemperature
from rl.agents.droq.update import update_actor, update_critic, update_temperature
from rl.buffers.torch_buffer import TorchUniformBuffer
from rl.utils.types import NDArray, Tensor


@dataclass
class DroQConfig:
    seed: int
    device_type: str

    buffer_max_length: int
    buffer_min_length: int
    buffer_device_type: str
    sample_batch_size: int

    actor_lr: float
    critic_lr: float
    temp_lr: float
    hidden_dims: Sequence[int]

    gamma: float
    n_step: int
    critic_target_update_tau: float
    num_qs: int
    num_min_qs: Optional[int]
    critic_dropout_rate: float
    critic_layer_norm: bool
    sampled_backup: bool
    target_entropy: Optional[float]
    temp_initial_value: float

    asymmetric_observation: bool
    actor_update_period: int
    use_compile: bool
    compile_mode: str
    use_amp: bool
    load_optimizer: bool


def _resolve_device(device_type: str) -> torch.device:
    if device_type.startswith("cuda"):
        return torch.device(device_type if ":" in device_type else "cuda:0")
    return torch.device("cpu")


def _init_droq_networks(
    actor_observation_dim: int,
    critic_observation_dim: int,
    action_dim: int,
    cfg: DroQConfig,
    device: torch.device,
) -> tuple[Network, Network, Network, Network]:
    torch.manual_seed(cfg.seed)
    use_fused = device.type == "cuda" and torch.cuda.is_available()

    actor_net = DroQActor(
        observation_dim=actor_observation_dim,
        action_dim=action_dim,
        hidden_dims=cfg.hidden_dims,
    ).to(device)
    actor = Network(
        network=actor_net,
        optimizer=optim.Adam(actor_net.parameters(), lr=cfg.actor_lr, fused=use_fused),
        compile_network=cfg.use_compile,
        compile_mode=cfg.compile_mode,
    )

    critic_net = DroQEnsembleCritic(
        observation_dim=critic_observation_dim,
        action_dim=action_dim,
        hidden_dims=cfg.hidden_dims,
        num_qs=cfg.num_qs,
        dropout_rate=cfg.critic_dropout_rate,
        use_layer_norm=cfg.critic_layer_norm,
    ).to(device)
    critic = Network(
        network=critic_net,
        optimizer=optim.Adam(critic_net.parameters(), lr=cfg.critic_lr, fused=use_fused),
        compile_network=cfg.use_compile,
        compile_mode=cfg.compile_mode,
    )

    target_critic_net = DroQEnsembleCritic(
        observation_dim=critic_observation_dim,
        action_dim=action_dim,
        hidden_dims=cfg.hidden_dims,
        num_qs=cfg.num_qs,
        dropout_rate=cfg.critic_dropout_rate,
        use_layer_norm=cfg.critic_layer_norm,
    ).to(device)
    target_critic_net.load_state_dict(critic_net.state_dict())
    target_critic = Network(
        network=target_critic_net,
        compile_network=cfg.use_compile,
        compile_mode=cfg.compile_mode,
        ema_source=critic,
        ema_tau=cfg.critic_target_update_tau,
    )

    temp_net = DroQTemperature(cfg.temp_initial_value).to(device)
    temperature = Network(
        network=temp_net,
        optimizer=optim.Adam(temp_net.parameters(), lr=cfg.temp_lr, fused=use_fused),
        compile_network=cfg.use_compile,
        compile_mode=cfg.compile_mode,
    )

    return actor, critic, target_critic, temperature


class DroQAgent(BaseAgent[DroQConfig]):
    def __init__(
        self,
        observation_space: gym.spaces.Space[NDArray],
        action_space: gym.spaces.Space[NDArray],
        env_info: dict[str, Any],
        cfg: DroQConfig,
    ):
        self._critic_observation_dim: int = observation_space.shape[-1]  # type: ignore[union-attr]
        self._action_dim: int = action_space.shape[-1]  # type: ignore[union-attr]
        if cfg.asymmetric_observation:
            self._actor_observation_dim = env_info["actor_observation_size"][-1]
        else:
            self._actor_observation_dim = self._critic_observation_dim

        super().__init__(observation_space, action_space, env_info, cfg)
        self._device = _resolve_device(cfg.device_type)
        self._target_entropy = cfg.target_entropy
        if self._target_entropy is None:
            self._target_entropy = -0.5 * self._action_dim

        self._actor, self._critic, self._target_critic, self._temperature = _init_droq_networks(
            actor_observation_dim=self._actor_observation_dim,
            critic_observation_dim=self._critic_observation_dim,
            action_dim=self._action_dim,
            cfg=cfg,
            device=self._device,
        )
        self._update_step = 0
        self._grad_scaler = GradScaler(device=self._device.type, enabled=cfg.use_amp)

        self._replay_buffer = TorchUniformBuffer(
            observation_space=observation_space,
            action_space=action_space,
            n_step=cfg.n_step,
            gamma=cfg.gamma,
            max_length=cfg.buffer_max_length,
            min_length=cfg.buffer_min_length,
            sample_batch_size=cfg.sample_batch_size,
            device_type=cfg.buffer_device_type,
        )

    def _actor_observations(self, observations: Tensor) -> torch.Tensor:
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self._device)
        if self._cfg.asymmetric_observation:
            obs = obs[:, : self._actor_observation_dim]
        return obs

    def sample_actions(
        self,
        interaction_step: int,
        prev_transition: MutableMapping[str, Tensor],
        training: bool,
    ) -> Tensor:
        observations = self._actor_observations(prev_transition["next_observation"])
        with torch.no_grad():
            actions, _ = self._actor(
                observations=observations,
                training=False,
                sample=training,
            )
        return actions.cpu().numpy()

    def process_transition(self, transition: MutableMapping[str, Tensor]) -> None:
        self._replay_buffer.add_batch(transition)

    def can_start_training(self) -> bool:
        return self._replay_buffer.can_sample()

    def update(self) -> dict[str, Any]:
        batch = cast(dict[str, torch.Tensor], self._replay_buffer.sample())
        for key, value in batch.items():
            batch[key] = value.to(self._device, non_blocking=True)

        if self._cfg.asymmetric_observation:
            batch["actor_observation"] = batch["observation"][:, : self._actor_observation_dim]
            batch["actor_next_observation"] = batch["next_observation"][:, : self._actor_observation_dim]
        else:
            batch["actor_observation"] = batch["observation"]
            batch["actor_next_observation"] = batch["next_observation"]

        update_info: dict[str, torch.Tensor] = {}
        critic_info = update_critic(
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            temperature=self._temperature,
            batch=batch,
            num_min_qs=self._cfg.num_min_qs,
            sampled_backup=self._cfg.sampled_backup,
            device=self._device,
            use_amp=self._cfg.use_amp,
            grad_scaler=self._grad_scaler,
        )
        update_info.update(critic_info)

        if self._update_step % self._cfg.actor_update_period == 0:
            actor_info = update_actor(
                actor=self._actor,
                critic=self._critic,
                temperature=self._temperature,
                batch=batch,
                device=self._device,
                use_amp=self._cfg.use_amp,
                grad_scaler=self._grad_scaler,
            )
            update_info.update(actor_info)
            temp_info = update_temperature(
                temperature=self._temperature,
                entropy=actor_info["actor/entropy"],
                target_entropy=float(self._target_entropy),
            )
            update_info.update(temp_info)

        self._update_step += 1

        return {
            key: value.item() if isinstance(value, torch.Tensor) else float(value)
            for key, value in update_info.items()
        }

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self._actor.save(os.path.join(path, "actor.pt"))
        self._critic.save(os.path.join(path, "critic.pt"))
        self._target_critic.save(os.path.join(path, "target_critic.pt"))
        self._temperature.save(os.path.join(path, "temperature.pt"))
        torch.save(
            {
                "update_step": self._update_step,
                "grad_scaler_state_dict": self._grad_scaler.state_dict(),
            },
            os.path.join(path, "agent_state.pt"),
        )
        print(f"\033[32m[DroQ]\033[0m Successfully saved checkpoint {self._update_step} at {path}.")

    def save_replay_buffer(self, path: str) -> None:
        self._replay_buffer.save(os.path.join(path, "replay_buffer.pt"))
        print(f"\033[32m[DroQ]\033[0m Successfully saved replay buffer at {path}.")

    def load(self, path: str) -> None:
        load_optimizer = self._cfg.load_optimizer
        self._actor.load(os.path.join(path, "actor.pt"), load_optimizer=load_optimizer)
        self._critic.load(os.path.join(path, "critic.pt"), load_optimizer=load_optimizer)
        self._target_critic.load(os.path.join(path, "target_critic.pt"), load_optimizer=False)
        self._temperature.load(os.path.join(path, "temperature.pt"), load_optimizer=load_optimizer)

        if load_optimizer:
            agent_state = torch.load(os.path.join(path, "agent_state.pt"), map_location=self._device)
            self._update_step = int(agent_state["update_step"])
            self._grad_scaler.load_state_dict(agent_state["grad_scaler_state_dict"])

        print(f"\033[32m[DroQ]\033[0m Successfully loaded checkpoint from {path}.")

    def load_replay_buffer(self, path: str) -> None:
        self._replay_buffer.load(os.path.join(path, "replay_buffer.pt"))
        print(f"\033[32m[DroQ]\033[0m Successfully loaded replay buffer from {path}.")

    def get_metrics(self) -> dict[str, Any]:
        return {}
