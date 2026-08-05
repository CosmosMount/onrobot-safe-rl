from __future__ import annotations
from typing import Any
import numpy as np
import torch
from rl.agents.base.inference import ActionDecision
from rl.agents.livesac.network import LiveSACActor

class LiveSACInferencePolicy:
    def __init__(self, observation_dim: int, action_dim: int, cfg: Any):
        self.action_dim = int(action_dim)
        self.actor_observation_dim = int(getattr(cfg, "actor_observation_dim", observation_dim))
        name = str(cfg.device_type)
        self.device = torch.device(name if ":" in name else ("cuda:0" if name.startswith("cuda") else "cpu"))
        hidden_dim = int(cfg.actor_hidden_dims[-1])
        self.actor = LiveSACActor(
            num_blocks=max(1, len(cfg.actor_hidden_dims)),
            input_dim=self.actor_observation_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
        ).to(self.device).eval()
        self.snapshot_version = -1; self.actor_steps = 0; self.auxiliary_steps = 0
    @torch.no_grad()
    def load_snapshot(self, snapshot: dict[str, Any]) -> None:
        version = int(snapshot["snapshot_version"])
        if version < self.snapshot_version: raise ValueError("inference snapshot version moved backwards")
        if version == self.snapshot_version: return
        self.actor.load_state_dict(snapshot["actor_state_dict"]); self.snapshot_version = version
        self.actor_steps = int(snapshot.get("actor_steps", 0)); self.auxiliary_steps = int(snapshot.get("auxiliary_steps", 0))
    @torch.no_grad()
    def decide(self, observation: np.ndarray, *, training: bool, action_nominal: np.ndarray | None = None) -> ActionDecision:
        if action_nominal is None:
            obs = torch.as_tensor(np.asarray(observation, dtype=np.float32).reshape(1, -1)[:, :self.actor_observation_dim], device=self.device)
            action, _ = self.actor(obs, training=False, sample=training); value = action[0].cpu().numpy()
        else: value = np.asarray(action_nominal, dtype=np.float32).reshape(-1)
        return ActionDecision(value, value, {})
    def observe_transition(self, **_: Any) -> dict[str, bool]: return {}
    def transition_fields(self, decision: ActionDecision) -> dict[str, np.ndarray]: return {}
