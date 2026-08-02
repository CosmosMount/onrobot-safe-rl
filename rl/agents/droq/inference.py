from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rl.agents.base.inference import ActionDecision
from rl.agents.droq.network import DroQActor


class DroQInferencePolicy:
    def __init__(self, observation_dim: int, action_dim: int, cfg: Any):
        self.cfg = cfg
        self.action_dim = int(action_dim)
        self.actor_observation_dim = int(getattr(cfg, "actor_observation_dim", observation_dim))
        self.device = torch.device(str(cfg.device_type) if ":" in str(cfg.device_type)
                                   else ("cuda:0" if str(cfg.device_type).startswith("cuda") else "cpu"))
        self.actor = DroQActor(self.actor_observation_dim, action_dim, cfg.hidden_dims).to(self.device).eval()
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

    def _obs(self, observation: np.ndarray) -> torch.Tensor:
        value = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        return torch.as_tensor(value[:, :self.actor_observation_dim], device=self.device)

    @torch.no_grad()
    def decide(self, observation: np.ndarray, *, training: bool,
               action_nominal: np.ndarray | None = None) -> ActionDecision:
        if action_nominal is None:
            action, _ = self.actor(self._obs(observation), training=False, sample=training)
            nominal = action[0].cpu().numpy().astype(np.float32)
        else:
            nominal = np.asarray(action_nominal, dtype=np.float32).reshape(-1).copy()
        return ActionDecision(nominal, nominal, {})

    def observe_transition(self, **_: Any) -> dict[str, bool]:
        return {}

    def transition_fields(self, decision: ActionDecision) -> dict[str, np.ndarray]:
        return {}
