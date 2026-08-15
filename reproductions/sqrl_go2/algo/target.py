"""Algorithm 2 target fine-tuning with paired branch semantics."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .buffers import ReplayBuffer, Transition
from .finetune import SafetyLagrange, TARGET_BRANCHES
from .pretrain import ActionDecision
from .sac import VanillaSAC
from .safety_critic import SafetyCriticLearner
from .safety_policy import SafetyPolicy


class TargetTrainer:
    def __init__(self, sac: VanillaSAC, replay: ReplayBuffer, *, branch: str,
                 batch_size: int, minimum_transitions: int,
                 safety: SafetyCriticLearner | None = None,
                 policy: SafetyPolicy | None = None,
                 lagrange: SafetyLagrange | None = None,
                 epsilon: float = 0.1):
        if branch not in TARGET_BRANCHES:
            raise ValueError(f"unknown target branch: {branch}")
        if branch != "sac_transfer" and (safety is None or policy is None):
            raise ValueError(f"{branch} requires frozen Q_safe and masking")
        if branch == "sqrl_full" and lagrange is None:
            raise ValueError("sqrl_full requires a safety Lagrange multiplier")
        self.sac = sac
        self.replay = replay
        self.branch = branch
        self.batch_size = int(batch_size)
        self.minimum_transitions = int(minimum_transitions)
        self.safety = safety
        self.policy = policy
        self.lagrange = lagrange
        self.epsilon = float(epsilon)

    def decide(self, observation: np.ndarray,
               preview: Callable[[np.ndarray], object]) -> ActionDecision:
        if self.branch != "sac_transfer":
            assert self.policy is not None
            result = self.policy.select(observation, preview)
            return ActionDecision(
                result.requested_action, result.critic_action, result.q_target,
                constrained=True, mask=result)
        candidates = self.sac.act(observation, count=1)
        projected = preview(candidates)
        return ActionDecision(
            projected.requested[0].copy(), projected.critic_actions[0].copy(),
            projected.q_targets[0].copy(), constrained=False)

    def observe(self, transition: Transition) -> dict[str, float]:
        self.replay.add(transition)
        if len(self.replay) < self.minimum_transitions:
            return {}
        batch = self.replay.sample(self.batch_size, self.sac.device)
        metrics = self.sac.update_critic(batch)
        if self.branch == "sqrl_full":
            assert self.safety is not None and self.lagrange is not None
            actor_metrics, risk = self.sac.update_actor_and_alpha(
                batch["observation"], safety_critic=self.safety.critic,
                safety_epsilon=self.epsilon,
                safety_lagrange=self.lagrange.value)
            assert risk is not None
            actor_metrics.update(self.lagrange.update(risk - self.epsilon))
        else:
            actor_metrics, _ = self.sac.update_actor_and_alpha(batch["observation"])
        metrics.update(actor_metrics)
        return metrics
