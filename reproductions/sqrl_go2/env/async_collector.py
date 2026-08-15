"""Lossless 50 Hz inference collector separated from the learner."""

from __future__ import annotations

import queue
from dataclasses import dataclass, replace
from multiprocessing.queues import Queue
from typing import Any

import numpy as np
import torch

from runtime.inference.actions import ActionApplier
from train.config import load_app_config
from train.ordered_runtime import OrderedRuntimeChannel, RuntimeEnvelope

from ..algo.buffers import Transition
from ..algo.networks import QNetwork, TanhGaussianActor
from ..algo.safety_policy import MaskResult, SafetyPolicy
from ..config import ExperimentConfig, load_config
from .action_preview import ActionPreview
from .adapter import ObservationStack
from .failure import failure_cost


@dataclass
class PendingDecision:
    observation: np.ndarray
    requested_action: np.ndarray
    critic_action: np.ndarray
    phase: str
    mask: MaskResult | None


@dataclass(frozen=True)
class CollectedTransition:
    transition: Transition
    phase: str
    mask: MaskResult | None
    info: dict[str, Any]
    queue_depth: int


class AsyncInferencePolicy:
    def __init__(self, cfg: ExperimentConfig, branch: str, seed: int):
        torch.manual_seed(seed)
        self.device = torch.device("cpu")
        self.actor = TanhGaussianActor(
            cfg.stacked_observation_dim, cfg.environment.action_dim,
            cfg.training.hidden_dims).to(self.device)
        self.safety = QNetwork(
            cfg.stacked_observation_dim, cfg.environment.action_dim,
            cfg.training.hidden_dims).to(self.device)
        self.mask = SafetyPolicy(
            self.actor, self.safety, cfg.sqrl.epsilon_safe,
            cfg.sqrl.mask_candidates, self.device)
        self.branch = branch

    def load(self, snapshot: dict[str, Any]) -> None:
        self.actor.load_state_dict(snapshot["actor"])
        if "safety" in snapshot:
            self.safety.load_state_dict(snapshot["safety"])
        self.actor.eval()
        self.safety.eval()

    @torch.no_grad()
    def decide(self, observation: np.ndarray, phase: str,
               preview: ActionPreview) -> PendingDecision:
        constrained = phase == "safety" or self.branch in {"sqrl_mask", "sqrl_full"}
        callback = lambda candidates: preview.preview(candidates, observation)
        if constrained:
            result = self.mask.select(observation, callback)
            committed = preview.commit(result.requested_action, observation)
            if not np.allclose(
                    committed.action_executed, result.critic_action,
                    rtol=0.0, atol=1e-6):
                raise RuntimeError("committed constrained action differs from Q_safe action")
            return PendingDecision(
                observation.copy(), result.requested_action,
                result.critic_action, phase, result)
        state = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        action, _ = self.actor.sample(state)
        projected = callback(action.cpu().numpy())
        committed = preview.commit(projected.requested[0], observation)
        if not np.allclose(
                committed.action_executed, projected.critic_actions[0],
                rtol=0.0, atol=1e-6):
            raise RuntimeError("committed task action differs from critic action")
        return PendingDecision(
            observation.copy(), projected.requested[0],
            projected.critic_actions[0], phase, None)


def inference_snapshot(actor: torch.nn.Module,
                       safety: torch.nn.Module | None) -> dict[str, Any]:
    snapshot = {
        "actor": {key: value.detach().cpu().clone()
                  for key, value in actor.state_dict().items()}}
    if safety is not None:
        snapshot["safety"] = {
            key: value.detach().cpu().clone()
            for key, value in safety.state_dict().items()}
    return snapshot


def _latest(queue_: Queue) -> dict[str, Any] | None:
    value = None
    while True:
        try:
            value = queue_.get_nowait()
        except queue.Empty:
            return value


def run_async_collector(*, config_path: str, branch: str, seed: int,
                        weight_queue: Queue, transition_queue: Queue,
                        control_queue: Queue) -> None:
    torch.set_num_threads(1)
    cfg = load_config(config_path)
    robot, _, _ = load_app_config(path=config_path)
    robot = replace(robot, move_speed=cfg.move_speed,
                    reward_command_vx=cfg.move_speed)
    preview = ActionPreview(ActionApplier(
        init_qpos=robot.init_qpos, action_offset=robot.action_offset,
        joint_min=robot.joint_min, joint_max=robot.joint_max,
        max_joint_delta=None, action_filter=None))
    policy = AsyncInferencePolicy(cfg, branch, seed)
    policy.load(weight_queue.get(timeout=120.0))
    stack = ObservationStack(
        cfg.environment.observation_frames, cfg.environment.observation_dim)
    channel = OrderedRuntimeChannel(robot.runtime_action_shm, robot.runtime_state_shm)
    channel.connect(timeout=120.0)
    channel.clear_action()
    initial = channel.recv(timeout=10.0)
    observation = stack.reset(initial.observation)
    preview.reset(initial.observation[:12])
    phase = "task" if cfg.phase == "pretrain" else branch
    task_steps = 0
    safety_trajectories = 0
    current_episode = initial.episode_id
    pending: dict[int, PendingDecision] = {}

    def send() -> None:
        decision_phase = phase if cfg.phase == "pretrain" else branch
        decision = policy.decide(observation, decision_phase, preview)
        action_id = channel.send_action(decision.requested_action)
        pending[action_id] = decision

    send()
    try:
        while True:
            try:
                if control_queue.get_nowait() == "stop":
                    break
            except queue.Empty:
                pass
            snapshot = _latest(weight_queue)
            if snapshot is not None:
                policy.load(snapshot)
            message = channel.recv(timeout=10.0)
            if message.episode_id != current_episode:
                current_episode = message.episode_id
                observation = stack.reset(message.observation)
                preview.reset(message.observation[:12])
            if message.applied_action_id >= 0:
                decision = pending.get(message.applied_action_id)
                if decision is None:
                    raise RuntimeError("runtime applied an action missing from SQRL collector history")
                executed = np.asarray(message.info["action_executed"], np.float32)
                if not np.allclose(executed, decision.critic_action, rtol=0.0, atol=1e-5):
                    raise RuntimeError("Q_safe-scored action differs from runtime execution")
                next_observation = stack.append(message.observation)
                terminated = bool(message.info.get("terminated", message.done))
                truncated = bool(message.info.get("truncated", False))
                transition = Transition(
                    decision.observation.copy(), executed.copy(), message.reward,
                    next_observation.copy(),
                    failure_cost(message.info, terminated=terminated, truncated=truncated),
                    terminated, truncated)
                transition_queue.put(CollectedTransition(
                    transition, decision.phase, decision.mask,
                    dict(message.info), channel.queue_depth), timeout=10.0)
                observation = next_observation
                pending[message.applied_action_id] = PendingDecision(
                    observation.copy(), decision.requested_action,
                    decision.critic_action, decision.phase, decision.mask)
                for stale in [key for key in pending if key < message.applied_action_id]:
                    pending.pop(stale, None)
                if cfg.phase == "pretrain":
                    if phase == "task":
                        task_steps += 1
                        if task_steps >= cfg.sqrl.task_steps_per_cycle and (terminated or truncated):
                            phase = "safety"
                            task_steps = 0
                    elif terminated or truncated:
                        safety_trajectories += 1
                        if safety_trajectories >= cfg.sqrl.safety_trajectories_per_cycle:
                            phase = "task"
                            safety_trajectories = 0
            else:
                # Non-MDP recovery ticks must not contaminate history.
                observation = stack.reset(message.observation)
                preview.reset(message.observation[:12])
            send()
    finally:
        channel.stop()
        channel.close()
