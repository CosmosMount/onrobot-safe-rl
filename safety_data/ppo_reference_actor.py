"""Frozen RSL-RL PPO actor inference without constructing a simulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from safety_data.mjlab_natural_falls import MJLAB_TO_TARGET_JOINT


TARGET_SCALE = torch.tensor([0.2, 0.4, 0.4] * 4, dtype=torch.float32)
TARGET_OFFSET = torch.tensor([0.05, 0.7, -1.4] * 4, dtype=torch.float32)


class FrozenPpoReferenceActor(nn.Module):
    def __init__(self, checkpoint: str | Path) -> None:
        super().__init__()
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = loaded["actor_state_dict"]
        self.register_buffer("mean", state["obs_normalizer._mean"].reshape(-1))
        self.register_buffer("std", state["obs_normalizer._std"].reshape(-1))
        layers = []
        for index in (0, 2, 4, 6):
            weight = state[f"mlp.{index}.weight"]
            bias = state[f"mlp.{index}.bias"]
            linear = nn.Linear(weight.shape[1], weight.shape[0])
            linear.weight.data.copy_(weight)
            linear.bias.data.copy_(bias)
            layers.append(linear)
            if index != 6:
                layers.append(nn.ELU())
        self.network = nn.Sequential(*layers)
        self.register_buffer("action_std", state["distribution.std_param"].reshape(-1))
        self.register_buffer(
            "permutation", torch.tensor(MJLAB_TO_TARGET_JOINT, dtype=torch.long))
        self.register_buffer("target_scale", TARGET_SCALE)
        self.register_buffer("target_offset", TARGET_OFFSET)

    def requested_action(
        self, policy_observation: torch.Tensor, *, generator: torch.Generator,
    ) -> torch.Tensor:
        normalized = (policy_observation - self.mean) / self.std.clamp_min(1e-6)
        mean = self.network(normalized)
        noise = torch.randn(
            mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
        return mean + self.action_std * noise

    def critic_action(
        self,
        policy_observation: torch.Tensor,
        encoder_bias_target_order: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        raw = self.requested_action(policy_observation, generator=generator)
        target_order = raw[:, self.permutation]
        return (target_order * self.target_scale + self.target_offset
                - encoder_bias_target_order)


def sac_observation_to_ppo_actor_observation(
    observation: np.ndarray, *, episode_step: np.ndarray,
) -> np.ndarray:
    """Construct the frozen PPO actor's 47D inputs from corrected SAC state."""
    observation = np.asarray(observation, np.float32)
    episode_step = np.asarray(episode_step, np.int64).reshape(-1)
    if observation.ndim != 2 or observation.shape[1] != 46 or len(
            observation) != len(episode_step):
        raise ValueError("SAC corrected observations must have shape [N,46]")
    q = observation[:, :12]
    dq = observation[:, 12:24]
    angular = observation[:, 24:27]
    quat = observation[:, 30:34]
    w, x, y, z = quat.T
    gravity = np.stack([
        2 * (x * z - w * y),
        2 * (y * z + w * x),
        1 - 2 * (x * x + y * y),
    ], axis=1).astype(np.float32)
    phase = (episode_step.astype(np.float32) * 0.02 % 0.6) / 0.6
    phase_features = np.stack([
        np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)], axis=1)
    previous_action = (
        observation[:, 34:46] - TARGET_OFFSET.numpy()
    ) / TARGET_SCALE.numpy()
    return np.concatenate([
        angular, gravity, np.broadcast_to([0.3, 0.0, 0.0], (len(q), 3)),
        phase_features, q - TARGET_OFFSET.numpy(), dq, previous_action,
    ], axis=1).astype(np.float32)
