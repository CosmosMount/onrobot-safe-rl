from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rl.agents.base.inference import ActionDecision
from rl.agents.flashsac.network import FlashSACActor
from rl.agents.flashsac.noise import (
    build_truncated_zeta_cdf,
    sample_integer_from_cdf,
)


class FlashSACInferencePolicy:
    def __init__(self, observation_dim: int, action_dim: int, cfg: Any):
        self.cfg = cfg
        self.action_dim = int(action_dim)
        self.actor_observation_dim = int(getattr(cfg, "actor_observation_dim", observation_dim))
        self.device = torch.device(str(cfg.device_type) if ":" in str(cfg.device_type)
                                   else ("cuda:0" if str(cfg.device_type).startswith("cuda") else "cpu"))
        self.actor = FlashSACActor(cfg.actor_num_blocks, self.actor_observation_dim,
                                   cfg.actor_hidden_dim, action_dim).to(self.device).eval()
        self.zeta_cdf = build_truncated_zeta_cdf(
            mu=cfg.actor_noise_zeta_mu,
            max_n=cfg.actor_noise_zeta_max,
            device=self.device,
        )
        self.cur_noise_repeat_n = torch.tensor(1, dtype=torch.int32, device=self.device)
        self.cur_noise_repeat_count = torch.tensor(0, dtype=torch.int32, device=self.device)
        self.cached_noise = torch.randn((action_dim,), device=self.device)
        self.snapshot_version = -1
        self.actor_steps = 0
        self.auxiliary_steps = 0

    @torch.no_grad()
    def load_snapshot(self, snapshot: dict[str, Any]) -> None:
        version = int(snapshot["snapshot_version"])
        if version < self.snapshot_version:
            raise ValueError("inference snapshot version moved backwards")
        if version == self.snapshot_version:
            return
        self.actor.load_state_dict(snapshot["actor_state_dict"])
        self.snapshot_version = version
        self.actor_steps = int(snapshot.get("actor_steps", 0))
        self.auxiliary_steps = int(snapshot.get("auxiliary_steps", 0))

    @torch.no_grad()
    def decide(self, observation: np.ndarray, *, training: bool,
               action_nominal: np.ndarray | None = None) -> ActionDecision:
        if action_nominal is not None:
            nominal = np.asarray(action_nominal, dtype=np.float32).reshape(-1).copy()
        else:
            obs = torch.as_tensor(np.asarray(observation, dtype=np.float32).reshape(1, -1)[:, :self.actor_observation_dim], device=self.device)
            temperature = 1.0 if training else 0.0
            mean, std = self.actor.get_mean_and_std(obs, training=False)
            if temperature == 0.0:
                actions = torch.tanh(mean)
            else:
                reinit = ((self.cur_noise_repeat_count == 0)
                          | (self.cur_noise_repeat_count >= self.cur_noise_repeat_n))
                new_noise = torch.randn_like(mean)
                new_n = sample_integer_from_cdf(self.zeta_cdf)
                self.cached_noise = torch.where(reinit, new_noise, self.cached_noise)
                self.cur_noise_repeat_n = torch.where(reinit, new_n, self.cur_noise_repeat_n)
                self.cur_noise_repeat_count = torch.where(reinit, torch.zeros_like(self.cur_noise_repeat_count), self.cur_noise_repeat_count)
                actions = torch.tanh(mean + std * self.cached_noise.reshape(1, -1) * temperature)
                self.cur_noise_repeat_count = self.cur_noise_repeat_count + 1
            nominal = actions[0].cpu().numpy().astype(np.float32)
        return ActionDecision(nominal, nominal, {})

    def observe_transition(self, **_: Any) -> dict[str, bool]:
        return {}

    def transition_fields(self, decision: ActionDecision) -> dict[str, np.ndarray]:
        return {}
