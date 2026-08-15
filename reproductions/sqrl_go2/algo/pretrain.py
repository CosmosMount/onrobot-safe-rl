"""Algorithm 1 phase machine: task SAC and recent bar-pi safety rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .buffers import ReplayBuffer, SafetyReplayBuffer, Transition
from .sac import VanillaSAC
from .safety_critic import SafetyCriticLearner
from .safety_policy import MaskResult, SafetyPolicy


@dataclass(frozen=True)
class ActionDecision:
    requested_action: np.ndarray
    critic_action: np.ndarray
    q_target: np.ndarray
    constrained: bool
    mask: MaskResult | None = None


class SQRLPretrainer:
    def __init__(
        self,
        sac: VanillaSAC,
        safety: SafetyCriticLearner,
        task_replay: ReplayBuffer,
        safety_replay: SafetyReplayBuffer,
        policy: SafetyPolicy,
        *,
        batch_size: int,
        minimum_task_transitions: int,
        minimum_safety_transitions: int,
        task_steps_per_cycle: int,
        safety_trajectories_per_cycle: int,
        safety_updates_per_cycle: int,
    ):
        self.sac = sac
        self.safety = safety
        self.task_replay = task_replay
        self.safety_replay = safety_replay
        self.policy = policy
        self.batch_size = int(batch_size)
        self.minimum_task_transitions = int(minimum_task_transitions)
        self.minimum_safety_transitions = int(minimum_safety_transitions)
        self.task_steps_per_cycle = int(task_steps_per_cycle)
        self.safety_trajectories_per_cycle = int(safety_trajectories_per_cycle)
        self.safety_updates_per_cycle = int(safety_updates_per_cycle)
        self.phase = "task"
        self.task_steps = 0
        self.safety_trajectories = 0
        self.total_steps = 0

    def decide(self, observation: np.ndarray,
               preview: Callable[[np.ndarray], object]) -> ActionDecision:
        if self.phase == "safety":
            result = self.policy.select(observation, preview)
            return ActionDecision(
                result.requested_action, result.critic_action, result.q_target,
                constrained=True, mask=result)
        candidates = self.sac.act(observation, count=1)
        projected = preview(candidates)
        return ActionDecision(
            projected.requested[0].copy(), projected.critic_actions[0].copy(),
            projected.q_targets[0].copy(), constrained=False)

    def observe(self, transition: Transition, *,
                collection_phase: str | None = None) -> dict[str, float]:
        externally_scheduled = collection_phase is not None
        if collection_phase is not None:
            if collection_phase not in {"task", "safety"}:
                raise ValueError("collection_phase must be task or safety")
            self.phase = collection_phase
        metrics: dict[str, float] = {"sqrl/phase_safety": float(self.phase == "safety")}
        self.total_steps += 1
        if self.phase == "task":
            self.task_replay.add(transition)
            self.task_steps += 1
            if len(self.task_replay) >= self.minimum_task_transitions:
                metrics.update(self.sac.update(
                    self.task_replay.sample(self.batch_size, self.sac.device)))
            # Start a safety rollout only at an actual episode boundary.
            if not externally_scheduled and self.task_steps >= self.task_steps_per_cycle and (
                    transition.terminated or transition.truncated):
                self.phase = "safety"
                self.task_steps = 0
            return metrics

        completed = self.safety_replay.add(transition)
        if completed:
            self.safety_trajectories += 1
        if (
            completed
            and self.safety_trajectories >= self.safety_trajectories_per_cycle
            and len(self.safety_replay) >= self.minimum_safety_transitions
        ):
            for _ in range(self.safety_updates_per_cycle):
                metrics.update(self.safety.update(
                    self.safety_replay.sample(self.batch_size, self.safety.device),
                    self.policy.sample_tensor,
                ))
            if not externally_scheduled:
                self.phase = "task"
            self.safety_trajectories = 0
        metrics.update({
            "safety/replay_size": float(len(self.safety_replay)),
            "safety/replay_falls": float(self.safety_replay.fall_count),
        })
        return metrics
