"""PaperSQRL-specific, optimizer-free collection policy."""
from __future__ import annotations

from typing import Any
import numpy as np
import torch

from rl.agents.base.inference import ActionDecision
from rl.agents.droq.network import DroQActor
from rl.agents.safe_droq.network import SafetyCritic


class PaperSQRLInferencePolicy:
    def __init__(self, observation_dim: int, action_dim: int, cfg: Any):
        self.cfg = cfg
        self.actor_observation_dim = int(getattr(cfg, "actor_observation_dim", observation_dim))
        self.device = torch.device(str(cfg.device_type) if ":" in str(cfg.device_type)
                                   else ("cuda:0" if str(cfg.device_type).startswith("cuda") else "cpu"))
        self.actor = DroQActor(self.actor_observation_dim, action_dim, cfg.hidden_dims).to(self.device).eval()
        self.safety_critic = SafetyCritic(observation_dim, action_dim, cfg.safety_hidden_dims).to(self.device).eval()
        self.phase = "task" if cfg.sqrl_phase == "pretrain" else "target"
        self.task_steps = 0
        self.safety_episodes = 0
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
        if snapshot.get("safety_critic_state_dict") is not None:
            self.safety_critic.load_state_dict(snapshot["safety_critic_state_dict"])
        state = snapshot.get("algorithm_state", {})
        if self.snapshot_version < 0:
            self.phase = str(state.get("collection_phase", self.phase))
            self.task_steps = int(state.get("task_steps", 0))
            self.safety_episodes = int(state.get("safety_episodes", 0))
        self.snapshot_version = version
        self.actor_steps = int(snapshot.get("actor_steps", 0))
        self.auxiliary_steps = int(snapshot.get("auxiliary_steps", 0))

    @torch.no_grad()
    def decide(self, observation: np.ndarray, *, training: bool,
               action_nominal: np.ndarray | None = None) -> ActionDecision:
        obs = torch.as_tensor(np.asarray(observation, dtype=np.float32).reshape(1, -1), device=self.device)
        constrained = self.cfg.sqrl_phase == "finetune" or self.phase == "safety"
        if action_nominal is None:
            actor_obs = obs[:, :self.actor_observation_dim]
            action, _ = self.actor(actor_obs, training=False, sample=training)
            original = action[0].cpu().numpy().astype(np.float32)
        else:
            original = np.asarray(action_nominal, dtype=np.float32).reshape(-1).copy()
        if not constrained:
            return ActionDecision(original, original, {"phase": self.phase, "active": False})
        multiplier = int(self.cfg.safety_boundary_pool_multiplier) if self.phase == "safety" else 1
        count = int(self.cfg.safety_num_candidates) * multiplier - 1
        actor_actions, actor_info = self.actor(obs[:, :self.actor_observation_dim].repeat(count, 1), training=False, sample=training)
        candidates = torch.cat([torch.as_tensor(original, device=self.device).reshape(1, -1), actor_actions], dim=0)
        log_probs = torch.cat([torch.tensor([-torch.inf], device=self.device), actor_info["log_prob"].reshape(-1)])
        risks = torch.sigmoid(self.safety_critic(obs.repeat(candidates.shape[0], 1), candidates, training=False))
        safe_indices = torch.nonzero(risks <= float(self.cfg.safety_epsilon), as_tuple=False).reshape(-1)
        no_safe = safe_indices.numel() == 0
        if no_safe:
            selected_index = int(torch.argmin(risks).item())
        elif self.phase == "safety":
            selected_index = int(safe_indices[torch.argmax(risks[safe_indices])].item())
        else:
            weights = torch.softmax(log_probs[safe_indices], dim=0)
            if not bool(torch.all(torch.isfinite(weights))):
                weights = torch.ones_like(weights) / len(weights)
            selected_index = int(safe_indices[torch.multinomial(weights, 1)].item())
        requested = candidates[selected_index].cpu().numpy().astype(np.float32)
        return ActionDecision(original, requested, {
            "phase": self.phase, "active": True,
            "replaced": selected_index != 0, "no_safe": no_safe,
            "safe_rate": float((risks <= float(self.cfg.safety_epsilon)).float().mean().item()),
            "nominal_risk": float(risks[0].item()),
            "selected_risk": float(risks[selected_index].item()),
        })

    def observe_transition(self, *, policy_step: bool, terminated: bool,
                           truncated: bool) -> dict[str, bool]:
        events = {"safety_trajectory_complete": False}
        if not policy_step or self.cfg.sqrl_phase != "pretrain":
            return events
        done = bool(terminated or truncated)
        if self.phase == "task":
            self.task_steps += 1
            if done and self.task_steps >= int(self.cfg.pretrain_task_steps_per_cycle):
                self.phase, self.safety_episodes = "safety", 0
        elif done:
            self.safety_episodes += 1
            events["safety_trajectory_complete"] = True
            if self.safety_episodes >= int(self.cfg.pretrain_safety_episodes_per_cycle):
                self.phase, self.task_steps = "task", 0
        return events

    def transition_fields(self, decision: ActionDecision) -> dict[str, np.ndarray]:
        m = decision.metadata
        out = {"sqrl_collection_phase": np.asarray([1 if m.get("phase") == "safety" else 0], dtype=np.int8),
               "sqrl_selector_active": np.asarray([m.get("active", False)], dtype=np.float32),
               "sqrl_selector_replaced": np.asarray([m.get("replaced", False)], dtype=np.float32),
               "sqrl_selector_no_safe": np.asarray([m.get("no_safe", False)], dtype=np.float32),
               "sqrl_selector_safe_rate": np.asarray([m.get("safe_rate", 0.0)], dtype=np.float32)}
        for source, target in (("nominal_risk", "sqrl_nominal_risk"), ("selected_risk", "sqrl_selected_risk")):
            if source in m:
                out[target] = np.asarray([m[source]], dtype=np.float32)
        return out
