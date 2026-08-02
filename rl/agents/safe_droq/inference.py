from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rl.agents.base.inference import ActionDecision
from rl.agents.droq.network import DroQActor
from rl.agents.droq.network import DroQEnsembleCritic
from rl.agents.safe_droq.network import SafetyCritic


class SafeDroQInferencePolicy:
    def __init__(self, observation_dim: int, action_dim: int, cfg: Any):
        self.cfg = cfg
        self.action_dim = int(action_dim)
        self.actor_observation_dim = int(getattr(cfg, "actor_observation_dim", observation_dim))
        self.device = torch.device(str(cfg.device_type) if ":" in str(cfg.device_type)
                                   else ("cuda:0" if str(cfg.device_type).startswith("cuda") else "cpu"))
        self.actor = DroQActor(self.actor_observation_dim, action_dim, cfg.hidden_dims).to(self.device).eval()
        self.safety_critic = SafetyCritic(observation_dim, action_dim, cfg.safety_hidden_dims).to(self.device).eval()
        self.critic = DroQEnsembleCritic(observation_dim, action_dim, cfg.hidden_dims,
                                         cfg.num_qs, cfg.critic_dropout_rate,
                                         cfg.critic_layer_norm).to(self.device).eval()
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
        safety = snapshot.get("safety_critic_state_dict")
        if safety is not None:
            self.safety_critic.load_state_dict(safety)
        if snapshot.get("critic_state_dict") is not None:
            self.critic.load_state_dict(snapshot["critic_state_dict"])
        self.snapshot_version = version
        self.actor_steps = int(snapshot.get("actor_steps", 0))
        self.auxiliary_steps = int(snapshot.get("auxiliary_steps", 0))
        state = snapshot.get("algorithm_state", {})
        self.safety_ready = bool(state.get("safety_ready", False))

    def _risk(self, observation: np.ndarray, actions: torch.Tensor) -> torch.Tensor:
        obs = torch.as_tensor(np.asarray(observation, dtype=np.float32).reshape(1, -1), device=self.device)
        return torch.sigmoid(self.safety_critic(obs.repeat(actions.shape[0], 1), actions, training=False))

    @torch.no_grad()
    def decide(self, observation: np.ndarray, *, training: bool,
               action_nominal: np.ndarray | None = None) -> ActionDecision:
        if action_nominal is None:
            obs = torch.as_tensor(np.asarray(observation, dtype=np.float32).reshape(1, -1)[:, :self.actor_observation_dim], device=self.device)
            action, _ = self.actor(obs, training=False, sample=training)
            original_nominal = action[0].cpu().numpy().astype(np.float32)
        else:
            original_nominal = np.asarray(action_nominal, dtype=np.float32).reshape(-1).copy()
        nominal = original_nominal.copy()
        metadata: dict[str, Any] = {
            "safety_active": False, "safety_replaced": False,
            "safety_no_safe_candidate": False, "safety_safe_rate": 0.0,
        }
        ready = bool(getattr(self, "safety_ready", False))
        active = (str(self.cfg.safety_mode) == "masking" and ready
                  and self.actor_steps >= int(self.cfg.safety_activation_step))
        if ready:
            obs = torch.as_tensor(np.asarray(observation, dtype=np.float32).reshape(1, -1)[:, :self.actor_observation_dim], device=self.device)
            alternatives, _ = self.actor(obs.repeat(int(self.cfg.safety_num_candidates) - 1, 1), training=False, sample=training)
            candidates = torch.cat([torch.as_tensor(nominal, device=self.device).reshape(1, -1), alternatives], dim=0)
            progress_steps = max(int(self.actor_steps) - int(self.cfg.safety_activation_step) + 1, 0)
            ramp = int(getattr(self.cfg, "safety_masking_ramp_steps", 0))
            progress = 1.0 if ramp == 0 else min(progress_steps / ramp, 1.0)
            if bool(getattr(self.cfg, "safety_contract_candidates", False)):
                delta = candidates - candidates[:1]
                rms = torch.sqrt(torch.mean(torch.square(delta), dim=-1, keepdim=True))
                max_rms = float(self.cfg.safety_max_action_rms) * progress
                scale = torch.clamp(max_rms / torch.clamp(rms, min=1e-8), max=1.0)
                candidates = torch.clamp(candidates[:1] + scale * delta, -1.0, 1.0)
            risks = self._risk(observation, candidates)
            safe = risks <= float(self.cfg.safety_epsilon)
            metadata.update({"safety_active": active, "safety_safe_rate": float(safe.float().mean().item()),
                             "safety_nominal_risk": float(risks[0].item())})
            selected = 0
            if active and not bool(safe[0]):
                nominal_risk = risks[0]
                delta_rms = torch.sqrt(torch.mean(torch.square(candidates - candidates[:1]), dim=-1))
                supported = delta_rms <= float(self.cfg.safety_max_action_rms) * progress
                improved = risks <= nominal_risk - float(self.cfg.safety_min_risk_improvement)
                if getattr(self.cfg, "safety_reward_q_margin", None) is None:
                    performance = torch.ones_like(safe, dtype=torch.bool)
                else:
                    q_values, _ = self.critic(torch.as_tensor(np.asarray(observation, dtype=np.float32).reshape(1, -1), device=self.device).repeat(candidates.shape[0], 1), candidates, training=False)
                    values = q_values.min(dim=0).values.reshape(-1)
                    performance = values >= values[0] - float(self.cfg.safety_reward_q_margin)
                eligible = safe & supported & improved & performance
                eligible[0] = False
                indices = torch.nonzero(eligible, as_tuple=False).reshape(-1)
                if indices.numel():
                    selected = int(indices[torch.argmin(risks[indices])].item())
                    metadata["safety_replaced"] = True
                else:
                    metadata["safety_no_safe_candidate"] = True
            nominal_risk = float(risks[0].item())
            metadata["safety_selected_risk"] = float(risks[selected].item())
            nominal = candidates[selected].cpu().numpy().astype(np.float32)
        return ActionDecision(original_nominal, nominal, metadata)

    def observe_transition(self, **_: Any) -> dict[str, bool]:
        return {}

    def transition_fields(self, decision: ActionDecision) -> dict[str, np.ndarray]:
        m = decision.metadata
        out = {
            "safety_selector_active": np.asarray([m.get("safety_active", False)], dtype=np.float32),
            "safety_selector_replaced": np.asarray([m.get("safety_replaced", False)], dtype=np.float32),
            "safety_selector_no_safe": np.asarray([m.get("safety_no_safe_candidate", False)], dtype=np.float32),
            "safety_selector_safe_rate": np.asarray([m.get("safety_safe_rate", 0.0)], dtype=np.float32),
        }
        for source, target in (("safety_nominal_risk", "safety_nominal_risk"), ("safety_selected_risk", "safety_selected_risk")):
            if source in m:
                out[target] = np.asarray([m[source]], dtype=np.float32)
        return out
