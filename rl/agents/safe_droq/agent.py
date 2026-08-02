from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, MutableMapping, Optional, Sequence

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from rl.agents.base.network import Network
from rl.agents.droq.agent import DroQAgent, DroQConfig
from rl.agents.base.update import PolicyUpdateRequest
from rl.agents.safe_droq.network import SafetyCritic
from rl.agents.safe_droq.replay import SafetyReplay
from rl.utils.types import NDArray, Tensor


@dataclass
class SafeDroQConfig(DroQConfig):
    safety_mode: str
    safety_hidden_dims: Sequence[int]
    safety_lr: float
    safety_gamma: float
    safety_target_update_tau: float
    safety_buffer_max_length: int
    safety_buffer_min_length: int
    safety_batch_size: int
    safety_failure_horizon: int
    safety_update_period: int
    safety_update_interval: int
    safety_update_unit: str
    safety_updates_per_event: int
    safety_future_loss_weight: float
    safety_num_candidates: int
    safety_epsilon: float
    safety_activation_step: int
    safety_masking_ramp_steps: int
    safety_min_risk_improvement: float
    safety_max_action_rms: float
    safety_contract_candidates: bool
    safety_reward_q_margin: Optional[float]
    safety_pretrained_path: Optional[str]
    freeze_safety_critic: bool


class SafeDroQAgent(DroQAgent):
    """DroQ plus an auxiliary failure critic and optional action masking.

    ``logging`` predicts only the risk of the exact DroQ action and therefore
    keeps the control path cheap. ``masking`` evaluates a single batched set of
    policy candidates and replaces the nominal action only when Q_safe is
    trained, the nominal is unsafe, and a safe alternative exists.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space[NDArray],
        action_space: gym.spaces.Space[NDArray],
        env_info: dict[str, Any],
        cfg: SafeDroQConfig,
    ):
        if cfg.safety_mode not in {"logging", "masking"}:
            raise ValueError("safety_mode must be 'logging' or 'masking'")
        if cfg.safety_num_candidates < 2:
            raise ValueError("safety_num_candidates must be at least 2")
        if not 0.0 <= cfg.safety_epsilon <= 1.0:
            raise ValueError("safety_epsilon must be in [0, 1]")
        if cfg.safety_masking_ramp_steps < 0:
            raise ValueError("safety_masking_ramp_steps must be non-negative")
        if cfg.safety_min_risk_improvement < 0.0:
            raise ValueError("safety_min_risk_improvement must be non-negative")
        if cfg.safety_max_action_rms <= 0.0:
            raise ValueError("safety_max_action_rms must be positive")
        if (
            cfg.safety_reward_q_margin is not None
            and cfg.safety_reward_q_margin < 0.0
        ):
            raise ValueError("safety_reward_q_margin must be non-negative")
        if cfg.safety_update_interval <= 0:
            raise ValueError("safety_update_interval must be positive")
        if cfg.safety_update_unit not in {"policy_step", "critic_step"}:
            raise ValueError("safety_update_unit must be policy_step or critic_step")
        if cfg.safety_updates_per_event < 0:
            raise ValueError("safety_updates_per_event must be non-negative")
        super().__init__(observation_space, action_space, env_info, cfg)
        self._cfg = cfg
        observation_dim = int(observation_space.shape[-1])  # type: ignore[union-attr]
        action_dim = int(action_space.shape[-1])  # type: ignore[union-attr]
        critic_net = SafetyCritic(
            observation_dim, action_dim, cfg.safety_hidden_dims).to(
                self._device)
        use_fused = self._device.type == "cuda" and torch.cuda.is_available()
        self._safety_critic = Network(
            network=critic_net,
            optimizer=optim.Adam(
                critic_net.parameters(), lr=cfg.safety_lr, fused=use_fused),
            compile_network=cfg.use_compile,
            compile_mode=cfg.compile_mode)
        target_net = SafetyCritic(
            observation_dim, action_dim, cfg.safety_hidden_dims).to(
                self._device)
        target_net.load_state_dict(critic_net.state_dict())
        self._target_safety_critic = Network(
            network=target_net,
            compile_network=cfg.use_compile,
            compile_mode=cfg.compile_mode,
            ema_source=self._safety_critic,
            ema_tau=cfg.safety_target_update_tau)
        self._safety_replay = SafetyReplay(
            capacity=cfg.safety_buffer_max_length,
            min_length=cfg.safety_buffer_min_length,
            batch_size=cfg.safety_batch_size,
            failure_horizon=cfg.safety_failure_horizon,
            device=self._device,
            seed=cfg.seed + 10_000)
        self._safety_updates = 0
        self._safety_ready = False
        self._safety_action_steps = 0
        self._safety_active_steps = 0
        self._safety_replacements = 0
        self._safety_no_safe = 0
        self._latest_safety_metrics: dict[str, float] = {
            "safety/ready": 0.0,
            "safety/active": 0.0,
            "safety/replaced": 0.0,
            "safety/no_safe_candidate": 0.0,
            "safety/sample_ms": 0.0,
        }
        if cfg.safety_pretrained_path:
            self._safety_critic.load(
                cfg.safety_pretrained_path, load_optimizer=False)
            self._target_safety_critic.network.load_state_dict(
                self._safety_critic.network.state_dict())

    @torch.no_grad()
    def _risks(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        logits = self._safety_critic(
            observations=observations,
            actions=actions,
            training=False)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def _reward_values(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        values, _ = self._critic(
            observations=observations,
            actions=actions,
            training=False)
        # Match the conservative actor objective used by this DroQ config.
        return values.min(dim=0).values.reshape(-1)

    def _masking_progress(self, interaction_step: int) -> float:
        if interaction_step < self._cfg.safety_activation_step:
            return 0.0
        ramp_steps = int(self._cfg.safety_masking_ramp_steps)
        if ramp_steps == 0:
            return 1.0
        return float(np.clip(
            (interaction_step - self._cfg.safety_activation_step + 1)
            / ramp_steps,
            0.0,
            1.0))

    def sample_actions(
        self,
        interaction_step: int,
        prev_transition: MutableMapping[str, Tensor],
        training: bool,
    ) -> Tensor:
        nominal_np = np.asarray(
            super().sample_actions(
                interaction_step, prev_transition, training),
            dtype=np.float32)
        return self.filter_nominal_action(
            interaction_step,
            prev_transition,
            nominal_np,
            training=training,
        )

    def filter_nominal_action(
        self,
        interaction_step: int,
        prev_transition: MutableMapping[str, Tensor],
        nominal_action: Tensor,
        *,
        training: bool,
    ) -> Tensor:
        """Apply Q_safe to a caller-provided nominal action.

        This keeps the first ``start_training`` steps comparable with plain
        DroQ: both receive the same seeded uniform action before the safety
        agent optionally replaces it.
        """
        started = time.perf_counter()
        nominal_np = np.asarray(nominal_action, dtype=np.float32)
        if nominal_np.ndim == 1:
            nominal_np = nominal_np[None, ...]
        observations = torch.as_tensor(
            prev_transition["next_observation"],
            dtype=torch.float32,
            device=self._device)
        nominal = torch.as_tensor(
            nominal_np, dtype=torch.float32, device=self._device)
        ready = self._safety_replay.can_sample() or bool(
            self._cfg.safety_pretrained_path)
        self._safety_ready = ready
        active = bool(
            self._cfg.safety_mode == "masking"
            and ready
            and interaction_step >= self._cfg.safety_activation_step)
        masking_progress = self._masking_progress(interaction_step)
        selected = nominal
        replaced = False
        no_safe = False

        if ready:
            if active:
                actor_obs = self._actor_observations(observations)
                repeated_actor_obs = actor_obs.repeat(
                    self._cfg.safety_num_candidates - 1, 1)
                with torch.no_grad():
                    alternatives, _ = self._actor(
                        observations=repeated_actor_obs,
                        training=False,
                        sample=training)
                candidates = torch.cat([nominal, alternatives], dim=0)
                if self._cfg.safety_contract_candidates:
                    # Keep counterfactual actions in a local neighborhood of
                    # the nominal action.  Raw policy samples are often far
                    # outside the action support where Q_safe was trained.
                    delta = candidates - nominal
                    delta_rms = torch.sqrt(torch.mean(
                        torch.square(delta), dim=-1, keepdim=True))
                    max_rms = (
                        self._cfg.safety_max_action_rms
                        * masking_progress)
                    contraction = torch.clamp(
                        max_rms / torch.clamp(delta_rms, min=1e-8),
                        max=1.0)
                    candidates = torch.clamp(
                        nominal + contraction * delta, -1.0, 1.0)
                critic_observations = observations.repeat(
                    self._cfg.safety_num_candidates, 1)
                risks = self._risks(critic_observations, candidates)
                risk_safe = risks <= self._cfg.safety_epsilon
                nominal_risk = risks[0]
                action_rms = torch.sqrt(torch.mean(
                    torch.square(candidates - nominal), dim=-1))
                effective_max_rms = (
                    self._cfg.safety_max_action_rms * masking_progress)
                supported = action_rms <= effective_max_rms
                improved = (
                    risks
                    <= nominal_risk
                    - self._cfg.safety_min_risk_improvement)
                reward_values = self._reward_values(
                    critic_observations, candidates)
                if self._cfg.safety_reward_q_margin is None:
                    performance_ok = torch.ones_like(
                        risk_safe, dtype=torch.bool)
                else:
                    performance_ok = (
                        reward_values
                        >= reward_values[0]
                        - self._cfg.safety_reward_q_margin)
                eligible = risk_safe & supported & improved & performance_ok
                eligible[0] = False
                if bool(risk_safe[0]):
                    selected = candidates[:1]
                elif bool(torch.any(eligible)):
                    safe_risks = torch.where(
                        eligible, risks, torch.full_like(risks, torch.inf))
                    selected_index = int(torch.argmin(safe_risks).item())
                    selected = candidates[selected_index:selected_index + 1]
                    replaced = selected_index != 0
                else:
                    # Abstain instead of trusting the minimum over an
                    # unsupported/unsafe candidate set.
                    no_safe = True
                    selected = candidates[:1]
                selected_risk = float(self._risks(
                    observations, selected).item())
                self._latest_safety_metrics.update({
                    "safety/nominal_risk": float(risks[0].item()),
                    "safety/selected_risk": selected_risk,
                    "safety/candidate_risk_min": float(risks.min().item()),
                    "safety/candidate_risk_max": float(risks.max().item()),
                    "safety/safe_candidate_rate": float(
                        risk_safe.float().mean().item()),
                    "safety/eligible_candidate_rate": float(
                        eligible.float().mean().item()),
                    "safety/support_candidate_rate": float(
                        supported.float().mean().item()),
                    "safety/performance_candidate_rate": float(
                        performance_ok.float().mean().item()),
                    "safety/masking_progress": masking_progress,
                    "safety/effective_max_action_rms": effective_max_rms,
                    "safety/nominal_reward_q": float(
                        reward_values[0].item()),
                    "safety/selected_reward_q": float(
                        self._reward_values(
                            observations, selected).item()),
                })
            else:
                risk = float(self._risks(observations, nominal).item())
                self._latest_safety_metrics.update({
                    "safety/nominal_risk": risk,
                    "safety/selected_risk": risk,
                })

        self._latest_safety_metrics.update({
            "safety/ready": float(ready),
            "safety/active": float(active),
            "safety/replaced": float(replaced),
            "safety/no_safe_candidate": float(no_safe),
            "safety/masking_progress": masking_progress,
            "safety/sample_ms": (
                time.perf_counter() - started) * 1000.0,
            "safety/replay_size": float(len(self._safety_replay)),
            "safety/failure_samples": float(
                self._safety_replay.positive_count),
        })
        self._safety_action_steps += 1
        self._safety_active_steps += int(active)
        self._safety_replacements += int(replaced)
        self._safety_no_safe += int(no_safe)
        action_nominal_copy = nominal_np[0].copy()
        action_selected_copy = selected.detach().cpu().numpy()[0].copy()
        self._last_action_trace = {
            "action_nominal": action_nominal_copy,
            "action_requested": action_selected_copy,
            "action_safety_replaced": bool(replaced),
            "action_safety_no_safe_candidate": bool(no_safe),
            "action_safety_intervened": bool(replaced),
            "action_safety_intervention_norm": float(np.linalg.norm(
                action_selected_copy - action_nominal_copy)),
        }
        self._latest_safety_metrics.update({
            "safety/action_steps": float(self._safety_action_steps),
            "safety/active_steps": float(self._safety_active_steps),
            "safety/replacement_count": float(self._safety_replacements),
            "safety/no_safe_count": float(self._safety_no_safe),
            "safety/replacement_rate": float(
                self._safety_replacements
                / max(self._safety_active_steps, 1)),
            "safety/no_safe_rate": float(
                self._safety_no_safe
                / max(self._safety_active_steps, 1)),
        })
        return selected.cpu().numpy()

    def get_last_action_trace(self) -> dict[str, Any]:
        trace = getattr(self, "_last_action_trace", {})
        result = dict(trace)
        for key in ("action_nominal", "action_requested"):
            if key in result:
                result[key] = np.asarray(result[key], dtype=np.float32).copy()
        return result

    def process_transition(
        self,
        transition: MutableMapping[str, Tensor],
    ) -> None:
        super().process_transition(transition)
        if not self._cfg.freeze_safety_critic:
            self._safety_replay.add_batch(transition)

    def _update_safety(self) -> dict[str, float]:
        if self._cfg.freeze_safety_critic:
            return {}
        if not self._safety_replay.can_sample():
            return {}
        batch = self._safety_replay.sample()
        with torch.no_grad():
            actor_next = self._actor_observations(
                batch["next_observation"])
            next_actions, _ = self._actor(
                observations=actor_next, training=False, sample=True)
            next_risk = torch.sigmoid(self._target_safety_critic(
                observations=batch["next_observation"],
                actions=next_actions,
                training=False))
            bellman_target = (
                batch["unsafe"]
                + (1.0 - batch["unsafe"])
                * (1.0 - batch["done"])
                * self._cfg.safety_gamma
                * next_risk)
            bellman_target = bellman_target.clamp(0.0, 1.0)

        logits = self._safety_critic(
            observations=batch["observation"],
            actions=batch["action"],
            training=True)
        predictions = torch.sigmoid(logits)
        bellman_loss = F.mse_loss(predictions, bellman_target)
        future_loss = F.binary_cross_entropy_with_logits(
            logits, batch["future_failure"])
        loss = (
            bellman_loss
            + self._cfg.safety_future_loss_weight * future_loss)
        assert self._safety_critic.optimizer is not None
        self._safety_critic.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._safety_critic.optimizer.step()
        self._target_safety_critic.ema_update_parameters()
        self._safety_updates += 1
        self._update_counters.auxiliary_steps += 1
        labels = batch["future_failure"]
        with torch.no_grad():
            positive = labels >= 0.5
            negative = ~positive
            brier = torch.mean(torch.square(predictions - labels))
            order = torch.argsort(predictions)
            ranks = torch.empty_like(
                order, dtype=torch.float32)
            ranks[order] = torch.arange(
                1, len(order) + 1, device=self._device,
                dtype=torch.float32)
            positives = positive.sum()
            negatives = negative.sum()
            auroc = (
                (
                    ranks[positive].sum()
                    - positives * (positives + 1) / 2
                ) / (positives * negatives)
                if int(positives) > 0 and int(negatives) > 0
                else torch.tensor(float("nan"), device=self._device)
            )
        return {
            "safety/loss": float(loss.detach().item()),
            "safety/bellman_loss": float(bellman_loss.detach().item()),
            "safety/future_loss": float(future_loss.detach().item()),
            "safety/mean_q": float(predictions.detach().mean().item()),
            "safety/target_mean": float(
                bellman_target.detach().mean().item()),
            "safety/auroc": float(auroc.item()),
            "safety/brier": float(brier.item()),
            "safety/positive_risk": float(
                predictions[positive].mean().item()),
            "safety/negative_risk": float(
                predictions[negative].mean().item()),
            "safety/update_steps": float(self._safety_updates),
        }

    def update(self) -> dict[str, Any]:
        info = super().update()
        if (
            self._cfg.safety_update_period > 0
            and self._update_step % self._cfg.safety_update_period == 0
        ):
            safety_info = self._update_safety()
            info.update(safety_info)
            self._latest_safety_metrics.update(safety_info)
        return info

    def update_policy_steps(self, request: PolicyUpdateRequest) -> dict[str, Any]:
        info = super().update_policy_steps(request)
        if self._cfg.freeze_safety_critic:
            return info
        if self._cfg.safety_update_unit == "policy_step":
            before = self._update_counters.policy_steps - request.policy_steps
            after = self._update_counters.policy_steps
        else:
            before = self._update_counters.critic_steps - request.critic_updates
            after = self._update_counters.critic_steps
        first = before // self._cfg.safety_update_interval + 1
        last = after // self._cfg.safety_update_interval
        call_auxiliary = 0
        for _ in range(max(0, last - first + 1)):
            for _ in range(self._cfg.safety_updates_per_event):
                safety_info = self._update_safety()
                if safety_info:
                    info.update(safety_info)
                    self._latest_safety_metrics.update(safety_info)
                    call_auxiliary += 1
        info["updates/call_auxiliary_steps"] = float(call_auxiliary)
        info["updates/total_auxiliary_steps"] = float(
            self._update_counters.auxiliary_steps)
        info["updates/auxiliary_per_policy_step"] = (
            self._update_counters.auxiliary_steps
            / max(self._update_counters.policy_steps, 1))
        return info

    def save(self, path: str) -> None:
        super().save(path)
        self._safety_critic.save(os.path.join(path, "safety_critic.pt"))
        self._target_safety_critic.save(
            os.path.join(path, "target_safety_critic.pt"))
        torch.save(
            {
                "safety_updates": self._safety_updates,
                "safety_action_steps": self._safety_action_steps,
                "safety_active_steps": self._safety_active_steps,
                "safety_replacements": self._safety_replacements,
                "safety_no_safe": self._safety_no_safe,
            },
            os.path.join(path, "safety_agent_state.pt"))

    def save_replay_buffer(self, path: str) -> None:
        super().save_replay_buffer(path)
        self._safety_replay.save(
            os.path.join(path, "safety_replay.pt"))

    def load(self, path: str) -> None:
        super().load(path)
        self._safety_critic.load(
            os.path.join(path, "safety_critic.pt"),
            load_optimizer=self._cfg.load_optimizer)
        self._target_safety_critic.load(
            os.path.join(path, "target_safety_critic.pt"),
            load_optimizer=False)
        state = torch.load(
            os.path.join(path, "safety_agent_state.pt"),
            map_location=self._device)
        self._safety_updates = int(state["safety_updates"])
        self._safety_action_steps = int(
            state.get("safety_action_steps", 0))
        self._safety_active_steps = int(
            state.get("safety_active_steps", 0))
        self._safety_replacements = int(
            state.get("safety_replacements", 0))
        self._safety_no_safe = int(state.get("safety_no_safe", 0))

    def load_replay_buffer(self, path: str) -> None:
        super().load_replay_buffer(path)
        self._safety_replay.load(
            os.path.join(path, "safety_replay.pt"))

    def get_metrics(self) -> dict[str, Any]:
        return dict(self._latest_safety_metrics)
