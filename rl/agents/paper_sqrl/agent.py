from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, MutableMapping, Sequence, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from rl.agents.base.network import Network
from rl.agents.droq.agent import DroQAgent, DroQConfig
from rl.agents.base.update import PolicyUpdateRequest
from rl.agents.droq.update import update_critic, update_temperature
from rl.agents.safe_droq.network import SafetyCritic
from rl.agents.paper_sqrl.replay import RecentTrajectoryReplay
from rl.utils.types import NDArray, Tensor


@dataclass
class PaperSQRLConfig(DroQConfig):
    sqrl_phase: str
    safety_hidden_dims: Sequence[int]
    safety_lr: float
    safety_gamma: float
    safety_target_update_tau: float
    safety_replay_trajectories: int
    safety_replay_min_transitions: int
    safety_batch_size: int
    safety_update_period: int
    safety_updates_per_cycle: int
    safety_num_candidates: int
    safety_boundary_pool_multiplier: int
    safety_epsilon: float
    pretrain_task_steps_per_cycle: int
    pretrain_safety_episodes_per_cycle: int
    safety_lagrange_lr: float
    safety_lagrange_initial: float
    safety_lagrange_max: float
    finetune_update_safety_critic: bool


class PaperSQRLAgent(DroQAgent):
    """SQRL as specified by Srinivasan et al. (arXiv:2010.14603).

    The implementation intentionally excludes later project extensions such
    as H-step binary relabeling, balanced failure batches, support gates,
    action contraction, double critics, and reward-Q candidate ranking.
    """

    def __init__(self, observation_space: gym.spaces.Space[NDArray],
                 action_space: gym.spaces.Space[NDArray],
                 env_info: dict[str, Any], cfg: PaperSQRLConfig):
        if cfg.sqrl_phase not in {"pretrain", "finetune"}:
            raise ValueError("sqrl_phase must be 'pretrain' or 'finetune'")
        if cfg.safety_num_candidates < 2:
            raise ValueError("safety_num_candidates must be at least 2")
        if not 0.0 < cfg.safety_epsilon < 1.0:
            raise ValueError("safety_epsilon must be in (0, 1)")
        if cfg.pretrain_task_steps_per_cycle <= 0:
            raise ValueError("pretrain_task_steps_per_cycle must be positive")
        if cfg.pretrain_safety_episodes_per_cycle <= 0:
            raise ValueError("pretrain_safety_episodes_per_cycle must be positive")
        super().__init__(observation_space, action_space, env_info, cfg)
        self._cfg = cfg
        observation_dim = int(observation_space.shape[-1])  # type: ignore
        action_dim = int(action_space.shape[-1])  # type: ignore
        use_fused = self._device.type == "cuda" and torch.cuda.is_available()

        torch.manual_seed(cfg.seed + 1)
        critic_net = SafetyCritic(
            observation_dim, action_dim, cfg.safety_hidden_dims).to(self._device)
        self._safety_critic = Network(
            critic_net,
            optimizer=optim.Adam(
                critic_net.parameters(), lr=cfg.safety_lr, fused=use_fused),
            compile_network=cfg.use_compile,
            compile_mode=cfg.compile_mode)
        target_net = SafetyCritic(
            observation_dim, action_dim, cfg.safety_hidden_dims).to(self._device)
        target_net.load_state_dict(critic_net.state_dict())
        self._target_safety_critic = Network(
            target_net,
            compile_network=cfg.use_compile,
            compile_mode=cfg.compile_mode,
            ema_source=self._safety_critic,
            ema_tau=cfg.safety_target_update_tau)
        self._safety_replay = RecentTrajectoryReplay(
            max_trajectories=cfg.safety_replay_trajectories,
            min_transitions=cfg.safety_replay_min_transitions,
            batch_size=cfg.safety_batch_size,
            device=self._device,
            seed=cfg.seed + 10_000)

        # Pre-training alternates ordinary SAC data collection/updates with k
        # complete rollouts from the current Q_safe-constrained policy.
        self._collection_phase = (
            "task" if cfg.sqrl_phase == "pretrain" else "target")
        self._task_steps_in_cycle = 0
        self._safety_episodes_in_cycle = 0
        self._pending_safety_updates = 0
        self._safety_updates = 0
        self._action_steps = 0
        self._active_steps = 0
        self._replacement_count = 0
        self._no_safe_count = 0
        self._pending_selector: dict[str, float] = {}
        initial = max(float(cfg.safety_lagrange_initial), 1e-8)
        self._log_safety_lagrange = torch.tensor(
            np.log(initial), dtype=torch.float32, device=self._device,
            requires_grad=True)
        self._safety_lagrange_optimizer = optim.Adam(
            [self._log_safety_lagrange], lr=cfg.safety_lagrange_lr)
        self._latest_metrics: dict[str, float] = {}

    @torch.no_grad()
    def _risk(self, observations: torch.Tensor,
              actions: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self._safety_critic(
            observations=observations, actions=actions, training=False))

    def _should_constrain(self) -> bool:
        return (
            self._cfg.sqrl_phase == "finetune"
            or self._collection_phase == "safety")

    def sample_actions(self, interaction_step: int,
                       prev_transition: MutableMapping[str, Tensor],
                       training: bool) -> Tensor:
        observations = torch.as_tensor(
            prev_transition["next_observation"], dtype=torch.float32,
            device=self._device)
        if not self._should_constrain():
            action = super().sample_actions(
                interaction_step, prev_transition, training)
            self._record_selector(False, False, None, None, 0.0)
            return action
        return self._sample_constrained(observations, training=training)

    def filter_nominal_action(self, interaction_step: int,
                              prev_transition: MutableMapping[str, Tensor],
                              nominal_action: Tensor, *, training: bool) -> Tensor:
        # Uniform exploration belongs to the unconstrained task block during
        # pre-training. During target fine-tuning, SQRL constrains it too.
        if not self._should_constrain():
            self._record_selector(False, False, None, None, 0.0)
            return np.asarray(nominal_action, dtype=np.float32)[None, ...]
        observations = torch.as_tensor(
            prev_transition["next_observation"], dtype=torch.float32,
            device=self._device)
        nominal = torch.as_tensor(
            np.asarray(nominal_action, dtype=np.float32).reshape(1, -1),
            device=self._device)
        return self._sample_constrained(
            observations, training=training, nominal=nominal)

    @torch.no_grad()
    def _sample_constrained(self, observations: torch.Tensor, *,
                            training: bool,
                            nominal: torch.Tensor | None = None) -> Tensor:
        started = time.perf_counter()
        actor_obs = self._actor_observations(observations)
        pool_multiplier = (
            self._cfg.safety_boundary_pool_multiplier
            if self._collection_phase == "safety" else 1)
        policy_count = self._cfg.safety_num_candidates * pool_multiplier
        if nominal is not None:
            policy_count -= 1
        repeated = actor_obs.repeat(policy_count, 1)
        policy_actions, policy_info = self._actor(
            observations=repeated, training=False, sample=training)
        policy_log_probs = policy_info["log_prob"].reshape(-1)
        if nominal is None:
            candidates = policy_actions
            log_probs = policy_log_probs
            nominal_index = 0
        else:
            candidates = torch.cat([nominal, policy_actions], dim=0)
            # The externally sampled uniform nominal has no actor density;
            # retain it as the reference but do not preferentially resample it.
            log_probs = torch.cat([
                torch.full((1,), -torch.inf, device=self._device),
                policy_log_probs])
            nominal_index = 0
        critic_obs = observations.repeat(candidates.shape[0], 1)
        risks = self._risk(critic_obs, candidates)
        safe = risks <= self._cfg.safety_epsilon
        safe_indices = torch.nonzero(safe, as_tuple=False).reshape(-1)
        no_safe = safe_indices.numel() == 0

        if no_safe:
            selected_index = int(torch.argmin(risks).item())
        elif self._collection_phase == "safety":
            # Paper pre-training deliberately samples safe actions immediately
            # below epsilon to identify the safety boundary.
            selected_index = int(safe_indices[
                torch.argmax(risks[safe_indices])].item())
        else:
            # Eq. 3: the constrained distribution is the actor distribution
            # renormalized over actions accepted by Q_safe.
            weights = torch.softmax(log_probs[safe_indices], dim=0)
            if not bool(torch.all(torch.isfinite(weights))):
                weights = torch.ones_like(weights) / len(weights)
            selected_index = int(safe_indices[
                torch.multinomial(weights, 1)].item())

        selected = candidates[selected_index:selected_index + 1]
        nominal_risk = float(risks[nominal_index].item())
        selected_risk = float(risks[selected_index].item())
        replaced = selected_index != nominal_index
        self._record_selector(
            True, replaced, nominal_risk, selected_risk,
            (time.perf_counter() - started) * 1000.0,
            no_safe=no_safe,
            safe_rate=float(safe.float().mean().item()))
        return selected.cpu().numpy()

    def _record_selector(self, active: bool, replaced: bool,
                         nominal_risk: float | None,
                         selected_risk: float | None, sample_ms: float,
                         *, no_safe: bool = False,
                         safe_rate: float = 0.0) -> None:
        # Selection can also be requested while the C++ supervisor is in
        # stand-up/recovery. Commit counters only when process_transition
        # confirms this was an actual policy transition.
        self._pending_selector = {
            "safety/active": float(active),
            "safety/replaced": float(replaced),
            "safety/no_safe_candidate": float(no_safe),
            "safety/safe_candidate_rate": safe_rate,
            "safety/sample_ms": sample_ms,
            "sqrl/collection_is_safety": float(
                self._collection_phase == "safety"),
        }
        if nominal_risk is not None:
            self._pending_selector["safety/nominal_risk"] = nominal_risk
        if selected_risk is not None:
            self._pending_selector["safety/selected_risk"] = selected_risk
        self._latest_metrics.update(self._pending_selector)

    def _commit_selector_metrics(
            self, transition: MutableMapping[str, Tensor]) -> None:
        repeat = int(np.asarray(
            transition.get("replay_repeat_index", [0])).reshape(-1)[0])
        if repeat:
            return
        if "sqrl_selector_active" in transition:
            def scalar(key: str, default: float = 0.0) -> float:
                return float(np.asarray(
                    transition.get(key, [default])).reshape(-1)[0])

            self._pending_selector = {
                "safety/active": scalar("sqrl_selector_active"),
                "safety/replaced": scalar("sqrl_selector_replaced"),
                "safety/no_safe_candidate": scalar(
                    "sqrl_selector_no_safe"),
                "safety/safe_candidate_rate": scalar(
                    "sqrl_selector_safe_rate"),
                "safety/sample_ms": scalar("sqrl_selector_sample_ms"),
                "sqrl/collection_is_safety": scalar(
                    "sqrl_collection_phase"),
            }
            if "sqrl_nominal_risk" in transition:
                self._pending_selector["safety/nominal_risk"] = scalar(
                    "sqrl_nominal_risk")
            if "sqrl_selected_risk" in transition:
                self._pending_selector["safety/selected_risk"] = scalar(
                    "sqrl_selected_risk")
            self._latest_metrics.update(self._pending_selector)
        if not self._pending_selector:
            return
        active = bool(self._pending_selector.get("safety/active", 0.0))
        replaced = bool(self._pending_selector.get("safety/replaced", 0.0))
        no_safe = bool(self._pending_selector.get(
            "safety/no_safe_candidate", 0.0))
        self._action_steps += 1
        self._active_steps += int(active)
        self._replacement_count += int(replaced)
        self._no_safe_count += int(no_safe)
        self._latest_metrics.update({
            "safety/action_steps": float(self._action_steps),
            "safety/active_steps": float(self._active_steps),
            "safety/replacement_count": float(self._replacement_count),
            "safety/no_safe_count": float(self._no_safe_count),
            "safety/replacement_rate": self._replacement_count / max(
                self._active_steps, 1),
            "safety/no_safe_rate": self._no_safe_count / max(
                self._active_steps, 1),
        })

    def process_transition(self, transition: MutableMapping[str, Tensor]) -> None:
        self._commit_selector_metrics(transition)
        if self._cfg.sqrl_phase == "finetune":
            super().process_transition(transition)
            if self._cfg.finetune_update_safety_critic:
                completed = self._safety_replay.add_batch(transition)
                if completed:
                    self._pending_safety_updates += (
                        self._cfg.safety_updates_per_cycle)
            return

        externally_scheduled = "sqrl_collection_phase" in transition
        if externally_scheduled:
            self._collection_phase = (
                "safety" if bool(np.asarray(
                    transition["sqrl_collection_phase"]).reshape(-1)[0])
                else "task")

        if self._collection_phase == "task":
            super().process_transition(transition)
            repeat = int(np.asarray(
                transition.get("replay_repeat_index", [0])).reshape(-1)[0])
            if repeat == 0:
                self._task_steps_in_cycle += 1
            episode_done = bool(np.asarray(
                transition["terminated"]).reshape(-1)[0]) or bool(np.asarray(
                    transition["truncated"]).reshape(-1)[0])
            if (episode_done and self._task_steps_in_cycle >=
                    self._cfg.pretrain_task_steps_per_cycle):
                self._collection_phase = "safety"
                self._safety_episodes_in_cycle = 0
        else:
            completed = self._safety_replay.add_batch(transition)
            if completed:
                self._safety_episodes_in_cycle += 1
                if self._safety_episodes_in_cycle >= (
                        self._cfg.pretrain_safety_episodes_per_cycle):
                    self._pending_safety_updates += (
                        self._cfg.safety_updates_per_cycle)
                    self._collection_phase = "task"
                    self._task_steps_in_cycle = 0

    def _update_safety(self) -> dict[str, float]:
        if not self._safety_replay.can_sample():
            return {}
        batch = self._safety_replay.sample()
        with torch.no_grad():
            # Appendix D of SQRL samples the Bellman target action from the
            # unconstrained actor, despite constrained rollout collection.
            next_actions, _ = self._actor(
                observations=self._actor_observations(
                    batch["next_observation"]),
                training=False, sample=True)
            next_risk = torch.sigmoid(self._target_safety_critic(
                observations=batch["next_observation"],
                actions=next_actions, training=False))
            target = batch["unsafe"] + (
                (1.0 - batch["unsafe"])
                * (1.0 - batch["done"])
                * self._cfg.safety_gamma * next_risk)
            target = target.clamp(0.0, 1.0)
        logits = self._safety_critic(
            observations=batch["observation"], actions=batch["action"],
            training=True)
        prediction = torch.sigmoid(logits)
        loss = F.mse_loss(prediction, target)
        assert self._safety_critic.optimizer is not None
        self._safety_critic.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._safety_critic.optimizer.step()
        self._target_safety_critic.ema_update_parameters()
        self._safety_updates += 1
        self._update_counters.auxiliary_steps += 1
        return {
            "safety/loss": float(loss.detach().item()),
            "safety/mean_q": float(prediction.detach().mean().item()),
            "safety/target_mean": float(target.mean().item()),
            "safety/update_steps": float(self._safety_updates),
            "safety/replay_size": float(len(self._safety_replay)),
            "safety/replay_trajectories": float(
                self._safety_replay.trajectory_count),
            "safety/failure_samples": float(
                self._safety_replay.failure_count),
        }

    def _update_finetune(self) -> dict[str, Any]:
        batch = cast(dict[str, torch.Tensor], self._replay_buffer.sample())
        for key, value in batch.items():
            batch[key] = value.to(self._device, non_blocking=True)
        batch["actor_observation"] = self._actor_observations(
            batch["observation"])
        batch["actor_next_observation"] = self._actor_observations(
            batch["next_observation"])
        info: dict[str, torch.Tensor] = {}
        info.update(update_critic(
            actor=self._actor, critic=self._critic,
            target_critic=self._target_critic,
            temperature=self._temperature, batch=batch,
            num_min_qs=self._cfg.num_min_qs,
            sampled_backup=self._cfg.sampled_backup,
            target_q_min=self._cfg.target_q_min,
            target_q_max=self._cfg.target_q_max,
            device=self._device, use_amp=self._cfg.use_amp,
            grad_scaler=self._grad_scaler))
        if self._update_step % self._cfg.actor_update_period == 0:
            actions, actor_info = self._actor(
                observations=batch["actor_observation"],
                training=True, sample=True)
            log_probs = actor_info["log_prob"]
            self._critic.network.requires_grad_(False)
            qs, _ = self._critic(
                observations=batch["observation"], actions=actions,
                training=True)
            self._critic.network.requires_grad_(True)
            q = (qs.min(dim=0).values if self._cfg.actor_q_reduction == "min"
                 else qs.mean(dim=0))
            self._safety_critic.network.requires_grad_(False)
            risk = torch.sigmoid(self._safety_critic(
                observations=batch["observation"], actions=actions,
                training=False))
            self._safety_critic.network.requires_grad_(True)
            lagrange = torch.exp(self._log_safety_lagrange).detach()
            constraint = risk - self._cfg.safety_epsilon
            actor_loss = (
                self._temperature().detach() * log_probs - q
                + lagrange * constraint).mean()
            assert self._actor.optimizer is not None
            self._actor.optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self._actor.optimizer.step()

            lagrange_value = torch.exp(self._log_safety_lagrange)
            lagrange_loss = -(lagrange_value * constraint.detach()).mean()
            self._safety_lagrange_optimizer.zero_grad(set_to_none=True)
            lagrange_loss.backward()
            self._safety_lagrange_optimizer.step()
            with torch.no_grad():
                self._log_safety_lagrange.clamp_(
                    max=float(np.log(self._cfg.safety_lagrange_max)))
            entropy = -log_probs.mean()
            info.update({
                "actor/loss": actor_loss.detach(),
                "actor/entropy": entropy.detach(),
                "actor/q": q.mean().detach(),
                "actor/safety_risk": risk.mean().detach(),
                "actor/safety_constraint": constraint.mean().detach(),
                "actor/safety_lagrange": lagrange.detach(),
                "actor/safety_lagrange_loss": lagrange_loss.detach(),
            })
            info.update(update_temperature(
                self._temperature, entropy, float(self._target_entropy)))
        self._update_step += 1
        self._update_counters.critic_steps += 1
        self._update_counters.target_steps += 1
        self._update_counters.policy_steps += 1
        if self._update_step % max(int(self._cfg.actor_update_period), 1) == 1:
            self._update_counters.actor_steps += 1
            self._update_counters.temperature_steps += 1
        return {key: float(value.item()) for key, value in info.items()}

    def update(self) -> dict[str, Any]:
        safety_info: dict[str, float] = {}
        should_update_safety = (
            self._pending_safety_updates > 0
            and self._update_step % max(
                self._cfg.safety_update_period, 1) == 0)
        if should_update_safety:
            safety_info = self._update_safety()
            self._pending_safety_updates -= int(bool(safety_info))
        if self._cfg.sqrl_phase == "finetune":
            info = self._update_finetune()
        elif self._collection_phase == "task":
            info = super().update()
        else:
            info = {"sqrl/collecting_safety": 1.0}
        info.update(safety_info)
        self._latest_metrics.update({k: float(v) for k, v in info.items()})
        return info

    def update_policy_steps(self, request: PolicyUpdateRequest) -> dict[str, Any]:
        """Adapt the existing SQRL phase machine to the shared entrypoint."""
        policy_before = self._update_counters.policy_steps
        critic_before = self._update_counters.critic_steps
        actor_before = self._update_counters.actor_steps
        temperature_before = self._update_counters.temperature_steps
        target_before = self._update_counters.target_steps
        auxiliary_before = self._update_counters.auxiliary_steps
        metrics: dict[str, Any] = {}
        for _ in range(request.critic_updates):
            metrics.update(self.update())
        self._update_counters.policy_steps = policy_before + request.policy_steps
        metrics.update({
            "updates/call_policy_steps": float(request.policy_steps),
            "updates/call_critic_steps": float(self._update_counters.critic_steps - critic_before),
            "updates/call_actor_steps": float(self._update_counters.actor_steps - actor_before),
            "updates/call_temperature_steps": float(self._update_counters.temperature_steps - temperature_before),
            "updates/call_target_steps": float(self._update_counters.target_steps - target_before),
            "updates/call_auxiliary_steps": float(self._update_counters.auxiliary_steps - auxiliary_before),
            "updates/total_policy_steps": float(
                self._update_counters.policy_steps),
            "updates/total_critic_steps": float(
                self._update_counters.critic_steps),
            "updates/total_actor_steps": float(
                self._update_counters.actor_steps),
            "updates/total_temperature_steps": float(
                self._update_counters.temperature_steps),
            "updates/total_target_steps": float(
                self._update_counters.target_steps),
            "updates/total_auxiliary_steps": float(
                self._update_counters.auxiliary_steps),
        })
        return metrics

    def save(self, path: str) -> None:
        super().save(path)
        self._safety_critic.save(os.path.join(path, "safety_critic.pt"))
        self._target_safety_critic.save(
            os.path.join(path, "target_safety_critic.pt"))
        torch.save({
            "collection_phase": self._collection_phase,
            "task_steps_in_cycle": self._task_steps_in_cycle,
            "safety_episodes_in_cycle": self._safety_episodes_in_cycle,
            "pending_safety_updates": self._pending_safety_updates,
            "safety_updates": self._safety_updates,
            "action_steps": self._action_steps,
            "active_steps": self._active_steps,
            "replacement_count": self._replacement_count,
            "no_safe_count": self._no_safe_count,
            "log_safety_lagrange": self._log_safety_lagrange.detach().cpu(),
            "safety_lagrange_optimizer": (
                self._safety_lagrange_optimizer.state_dict()),
        }, os.path.join(path, "sqrl_state.pt"))

    def save_replay_buffer(self, path: str) -> None:
        super().save_replay_buffer(path)
        self._safety_replay.save(os.path.join(path, "sqrl_safety_replay.pt"))

    def load(self, path: str) -> None:
        super().load(path)
        self._safety_critic.load(
            os.path.join(path, "safety_critic.pt"),
            load_optimizer=self._cfg.load_optimizer)
        self._target_safety_critic.load(
            os.path.join(path, "target_safety_critic.pt"),
            load_optimizer=False)
        state = torch.load(os.path.join(path, "sqrl_state.pt"),
                           map_location=self._device)
        if self._cfg.sqrl_phase == "pretrain":
            self._collection_phase = str(state.get(
                "collection_phase", self._collection_phase))
            self._task_steps_in_cycle = int(state.get(
                "task_steps_in_cycle", 0))
            self._safety_episodes_in_cycle = int(state.get(
                "safety_episodes_in_cycle", 0))
            self._pending_safety_updates = int(state.get(
                "pending_safety_updates", 0))
        self._safety_updates = int(state.get("safety_updates", 0))
        self._action_steps = int(state.get("action_steps", 0))
        self._active_steps = int(state.get("active_steps", 0))
        self._replacement_count = int(state.get("replacement_count", 0))
        self._no_safe_count = int(state.get("no_safe_count", 0))
        with torch.no_grad():
            self._log_safety_lagrange.copy_(
                state["log_safety_lagrange"].to(self._device))
        if self._cfg.load_optimizer:
            self._safety_lagrange_optimizer.load_state_dict(
                state["safety_lagrange_optimizer"])

    def load_replay_buffer(self, path: str) -> None:
        super().load_replay_buffer(path)
        self._safety_replay.load(
            os.path.join(path, "sqrl_safety_replay.pt"))

    def get_metrics(self) -> dict[str, Any]:
        return dict(self._latest_metrics)
