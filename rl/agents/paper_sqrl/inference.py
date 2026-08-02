"""Inference-only SQRL policy for the fixed-rate collector process."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from rl.agents.droq.agent import DroQConfig
from rl.agents.droq.network import DroQActor
from rl.agents.paper_sqrl.agent import PaperSQRLConfig
from rl.agents.safe_droq.network import SafetyCritic


@dataclass(frozen=True)
class SQRLActionDecision:
    action: np.ndarray
    phase: str
    active: bool
    replaced: bool
    no_safe: bool
    safe_rate: float
    nominal_risk: float | None
    selected_risk: float | None


class PaperSQRLInferencePolicy:
    """A replay/optimizer-free copy of the paper SQRL action policy.

    The collector process is the sole owner of this object. Training publishes
    immutable CPU state dicts, applied only between action decisions. This
    prevents an optimizer write from racing a 50 Hz policy forward pass.
    """

    def __init__(self, observation_dim: int, action_dim: int,
                 cfg: PaperSQRLConfig):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device_type if ":" in cfg.device_type
            else ("cuda:0" if cfg.device_type.startswith("cuda") else "cpu"))
        self.actor_observation_dim = observation_dim
        self.actor = DroQActor(
            observation_dim, action_dim, cfg.hidden_dims).to(self.device).eval()
        self.safety_critic = SafetyCritic(
            observation_dim, action_dim,
            cfg.safety_hidden_dims).to(self.device).eval()
        self.phase = "task" if cfg.sqrl_phase == "pretrain" else "target"
        self.task_steps = 0
        self.safety_episodes = 0
        self.weight_version = -1

    @torch.no_grad()
    def load_weights(self, payload: dict[str, Any]) -> None:
        version = int(payload["version"])
        if version < self.weight_version:
            raise ValueError("inference weight version moved backwards")
        first_load = self.weight_version < 0
        self.actor.load_state_dict(payload["actor"])
        self.safety_critic.load_state_dict(payload["safety_critic"])
        if first_load and "collection_phase" in payload:
            self.phase = str(payload["collection_phase"])
            self.task_steps = int(payload.get("task_steps_in_cycle", 0))
            self.safety_episodes = int(payload.get(
                "safety_episodes_in_cycle", 0))
        self.weight_version = version

    def _actor_obs(self, observation: np.ndarray) -> torch.Tensor:
        observation = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        return torch.as_tensor(
            observation[:, :self.actor_observation_dim], device=self.device)

    @torch.no_grad()
    def decide(self, observation: np.ndarray, *, training: bool,
               nominal: np.ndarray | None = None) -> SQRLActionDecision:
        obs = self._actor_obs(observation)
        constrain = self.cfg.sqrl_phase == "finetune" or self.phase == "safety"
        if not constrain:
            if nominal is None:
                action, _ = self.actor(
                    observations=obs, training=False, sample=training)
                selected = action[0].cpu().numpy().astype(np.float32)
            else:
                selected = np.asarray(nominal, dtype=np.float32).reshape(-1)
            return SQRLActionDecision(
                selected, self.phase, False, False, False, 0.0, None, None)

        multiplier = (
            self.cfg.safety_boundary_pool_multiplier
            if self.phase == "safety" else 1)
        count = self.cfg.safety_num_candidates * multiplier
        nominal_tensor: torch.Tensor | None = None
        if nominal is not None:
            nominal_tensor = torch.as_tensor(
                np.asarray(nominal, dtype=np.float32).reshape(1, -1),
                device=self.device)
            count -= 1
        repeated = obs.repeat(count, 1)
        policy_actions, actor_info = self.actor(
            observations=repeated, training=False, sample=training)
        policy_log_probs = actor_info["log_prob"].reshape(-1)
        if nominal_tensor is None:
            candidates = policy_actions
            log_probs = policy_log_probs
        else:
            candidates = torch.cat([nominal_tensor, policy_actions], dim=0)
            log_probs = torch.cat([
                torch.full((1,), -torch.inf, device=self.device),
                policy_log_probs])
        risks = torch.sigmoid(self.safety_critic(
            observations=torch.as_tensor(
                np.asarray(observation, dtype=np.float32).reshape(1, -1),
                device=self.device).repeat(candidates.shape[0], 1),
            actions=candidates,
            training=False))
        safe_indices = torch.nonzero(
            risks <= self.cfg.safety_epsilon, as_tuple=False).reshape(-1)
        no_safe = safe_indices.numel() == 0
        if no_safe:
            selected_index = int(torch.argmin(risks).item())
        elif self.phase == "safety":
            selected_index = int(safe_indices[
                torch.argmax(risks[safe_indices])].item())
        else:
            weights = torch.softmax(log_probs[safe_indices], dim=0)
            if not bool(torch.all(torch.isfinite(weights))):
                weights = torch.ones_like(weights) / len(weights)
            selected_index = int(safe_indices[
                torch.multinomial(weights, 1)].item())
        nominal_index = 0
        selected = candidates[selected_index].cpu().numpy().astype(np.float32)
        return SQRLActionDecision(
            action=selected,
            phase=self.phase,
            active=True,
            replaced=selected_index != nominal_index,
            no_safe=no_safe,
            safe_rate=float((risks <= self.cfg.safety_epsilon).float().mean().item()),
            nominal_risk=float(risks[nominal_index].item()),
            selected_risk=float(risks[selected_index].item()),
        )

    def observe_transition(self, *, policy_step: bool, terminated: bool,
                           truncated: bool) -> dict[str, bool]:
        """Advance Algorithm 1's collection schedule in collector time."""
        events = {"safety_trajectory_complete": False}
        if not policy_step or self.cfg.sqrl_phase != "pretrain":
            return events
        done = bool(terminated or truncated)
        if self.phase == "task":
            self.task_steps += 1
            if done and self.task_steps >= self.cfg.pretrain_task_steps_per_cycle:
                self.phase = "safety"
                self.safety_episodes = 0
        elif self.phase == "safety" and done:
            self.safety_episodes += 1
            events["safety_trajectory_complete"] = True
            if self.safety_episodes >= self.cfg.pretrain_safety_episodes_per_cycle:
                self.phase = "task"
                self.task_steps = 0
        return events


class SACInferencePolicy:
    """Optimizer/replay-free actor for the asynchronous SAC baseline."""

    def __init__(self, observation_dim: int, action_dim: int,
                 cfg: DroQConfig):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device_type if ":" in cfg.device_type
            else ("cuda:0" if cfg.device_type.startswith("cuda") else "cpu"))
        self.actor = DroQActor(
            observation_dim, action_dim, cfg.hidden_dims).to(self.device).eval()
        self.phase = "task"
        self.weight_version = -1

    @torch.no_grad()
    def load_weights(self, payload: dict[str, Any]) -> None:
        version = int(payload["version"])
        if version < self.weight_version:
            raise ValueError("inference weight version moved backwards")
        self.actor.load_state_dict(payload["actor"])
        self.weight_version = version

    @torch.no_grad()
    def decide(self, observation: np.ndarray, *, training: bool,
               nominal: np.ndarray | None = None) -> SQRLActionDecision:
        if nominal is None:
            obs = torch.as_tensor(
                np.asarray(observation, dtype=np.float32).reshape(1, -1),
                device=self.device)
            selected, _ = self.actor(
                observations=obs, training=False, sample=training)
            action = selected[0].cpu().numpy().astype(np.float32)
        else:
            action = np.asarray(nominal, dtype=np.float32).reshape(-1)
        return SQRLActionDecision(
            action, "task", False, False, False, 0.0, None, None)

    def observe_transition(self, **_: Any) -> dict[str, bool]:
        return {"safety_trajectory_complete": False}


def build_inference_policy(observation_dim: int, action_dim: int,
                           cfg: Any) -> Any:
    agent_type = str(getattr(cfg, "agent_type", ""))
    if agent_type == "paper_sqrl":
        return PaperSQRLInferencePolicy(observation_dim, action_dim, cfg)
    if agent_type == "droq":
        return SACInferencePolicy(observation_dim, action_dim, cfg)
    raise ValueError(
        f"async collector does not support agent_type={agent_type!r}")


def export_inference_weights(agent: Any, *, version: int) -> dict[str, Any]:
    """Create an immutable CPU payload for a collector process."""
    def cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in module.state_dict().items()
        }

    payload = {
        "version": int(version),
        "actor": cpu_state(agent._actor.network),
        "snapshot_version": int(version),
        "actor_steps": int(agent.get_update_counters().get("actor_steps", 0)),
        "critic_steps": int(agent.get_update_counters().get("critic_steps", 0)),
        "temperature_steps": int(agent.get_update_counters().get("temperature_steps", 0)),
        "auxiliary_steps": int(agent.get_update_counters().get("auxiliary_steps", 0)),
    }
    safety_critic = getattr(agent, "_safety_critic", None)
    if safety_critic is not None:
        payload["safety_critic"] = cpu_state(safety_critic.network)
        payload.update({
            "collection_phase": str(agent._collection_phase),
            "task_steps_in_cycle": int(agent._task_steps_in_cycle),
            "safety_episodes_in_cycle": int(
                agent._safety_episodes_in_cycle),
            "safety_update_step": int(agent.get_update_counters().get(
                "auxiliary_steps", 0)),
        })
    return payload
