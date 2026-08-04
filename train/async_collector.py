"""Algorithm-independent ordered runtime collector."""
from __future__ import annotations

import queue
import time
from collections import deque
from dataclasses import asdict, dataclass
from multiprocessing.queues import Queue
from typing import Any

import numpy as np

from rl.agents.base.inference import ActionDecision, InferencePolicy
from rl.agents.inference import build_inference_policy
from train.ordered_runtime import OrderedRuntimeChannel, RuntimeEnvelope


@dataclass(frozen=True)
class PendingAction:
    observation: np.ndarray
    decision: ActionDecision
    snapshot_version: int
    actor_steps: int
    auxiliary_steps: int


def _frozen_observation(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).copy()
    result.setflags(write=False)
    return result


def build_ordered_transition(*, pending: PendingAction, message: RuntimeEnvelope,
                             policy: InferencePolicy) -> dict[str, np.ndarray]:
    info = message.info
    terminated = bool(info.get("terminated", message.done))
    truncated = bool(info.get("truncated", False))
    transition: dict[str, np.ndarray] = {
        "observation": np.asarray(pending.observation, dtype=np.float32)[None, ...],
        "action": np.asarray(pending.decision.action_requested, dtype=np.float32)[None, ...],
        "action_nominal": np.asarray(pending.decision.action_nominal, dtype=np.float32)[None, ...],
        "action_requested": np.asarray(pending.decision.action_requested, dtype=np.float32)[None, ...],
        "reward": np.asarray([message.reward], dtype=np.float32),
        "terminated": np.asarray([terminated], dtype=np.float32),
        "truncated": np.asarray([truncated], dtype=np.float32),
        "next_observation": np.asarray(message.observation, dtype=np.float32)[None, ...],
        "unsafe_label": np.asarray([terminated or bool(info.get("fallen", False)) or bool(info.get("inverted", False))], dtype=np.float32),
        "near_failure_label": np.asarray([bool(info.get("near_failure", False))], dtype=np.float32),
        "replay_repeat_index": np.asarray([0], dtype=np.int32),
        "runtime_step_id": np.asarray([message.runtime_step_id], dtype=np.int64),
        "episode_id": np.asarray([message.episode_id], dtype=np.int64),
        "episode_step": np.asarray([message.episode_step], dtype=np.int32),
        "applied_action_id": np.asarray([message.applied_action_id], dtype=np.int64),
        "action_policy_snapshot_version": np.asarray([pending.snapshot_version], dtype=np.int64),
        "action_policy_update_step": np.asarray([pending.actor_steps], dtype=np.int64),
        "action_safety_update_step": np.asarray([pending.auxiliary_steps], dtype=np.int64),
    }
    for key, dtype in (("action_executed", np.float32), ("action_q_target", np.float32),
                       ("action_runtime_intervened", np.float32),
                       ("action_runtime_intervention_norm", np.float32),
                       ("action_age_ms", np.float32), ("action_repeated_steps", np.int32)):
        if key in info:
            transition[key] = np.asarray(info[key], dtype=dtype).reshape(1, -1) if np.asarray(info[key]).ndim else np.asarray([info[key]], dtype=dtype)
    transition.update(policy.transition_fields(pending.decision))
    return transition


class OrderedAsyncCollectorCore:
    def __init__(self, policy: InferencePolicy, action_space: Any, *,
                 start_training: int, explore_action_scale: float, seed: int,
                 initial_policy_steps: int = 0):
        self.policy = policy
        self.action_space = action_space
        self.action_space.seed(seed)
        self.start_training = int(start_training)
        self.explore_action_scale = float(explore_action_scale)
        self.policy_steps = int(initial_policy_steps)
        self.pending: dict[int, PendingAction] = {}
        self.last_applied_action_id: int | None = None
        self.repeated_action_steps = 0

    def accept_state(self, message: RuntimeEnvelope) -> dict[str, Any] | None:
        if message.applied_action_id < 0:
            return None
        pending = self.pending.get(message.applied_action_id)
        if pending is None:
            raise RuntimeError("runtime executed an action missing from collector history: "
                               f"action_id={message.applied_action_id}")
        repeated = self.last_applied_action_id == message.applied_action_id
        self.repeated_action_steps += int(repeated)
        result = build_ordered_transition(pending=pending, message=message, policy=self.policy)
        self.last_applied_action_id = message.applied_action_id
        self.pending[message.applied_action_id] = PendingAction(
            observation=_frozen_observation(message.observation),
            decision=pending.decision, snapshot_version=pending.snapshot_version,
            actor_steps=pending.actor_steps, auxiliary_steps=pending.auxiliary_steps)
        policy_step = bool(message.info.get("policy_step", True))
        self.policy.observe_transition(policy_step=policy_step,
                                        terminated=bool(result["terminated"][0]),
                                        truncated=bool(result["truncated"][0]))
        self.policy_steps += int(policy_step)
        for stale_id in [key for key in self.pending if key < message.applied_action_id]:
            self.pending.pop(stale_id, None)
        return {"transition": result, "info": message.info,
                "decision": asdict(pending.decision)}

    def next_decision(self, observation: np.ndarray) -> ActionDecision:
        nominal = None
        if self.policy_steps < self.start_training:
            nominal = self.action_space.sample().astype(np.float32) * self.explore_action_scale
        return self.policy.decide(observation, training=True, action_nominal=nominal)

    def remember_action(self, action_id: int, observation: np.ndarray,
                        decision: ActionDecision) -> None:
        self.pending[int(action_id)] = PendingAction(
            observation=_frozen_observation(observation),
            decision=decision, snapshot_version=int(self.policy.snapshot_version),
            actor_steps=int(self.policy.actor_steps),
            auxiliary_steps=int(self.policy.auxiliary_steps))


def _latest_nowait(weight_queue: Queue) -> dict[str, Any] | None:
    latest = None
    while True:
        try:
            latest = weight_queue.get_nowait()
        except queue.Empty:
            return latest


def run_async_collector(*, robot_cfg: Any, agent_cfg: Any, train_cfg: Any,
                        observation_dim: int, action_dim: int, action_space: Any,
                        transition_queue: Queue, weight_queue: Queue,
                        control_queue: Queue, initial_policy_steps: int = 0) -> None:
    policy = build_inference_policy(observation_dim, action_dim, agent_cfg)
    policy.load_snapshot(weight_queue.get(timeout=120.0))
    core = OrderedAsyncCollectorCore(policy, action_space,
        start_training=train_cfg.start_training,
        explore_action_scale=train_cfg.explore_action_scale, seed=train_cfg.seed,
        initial_policy_steps=initial_policy_steps)
    channel = OrderedRuntimeChannel(robot_cfg.runtime_action_shm, robot_cfg.runtime_state_shm)
    channel.connect(timeout=120.0)
    initial = channel.recv(timeout=10.0)
    initial_episode_id = initial.episode_id
    channel.clear_action()
    while True:
        initial = channel.recv(timeout=10.0)
        if initial.episode_id > initial_episode_id:
            break
    last_time = time.perf_counter()
    local_transitions: deque[dict[str, Any]] = deque()
    local_capacity = max(1, int(getattr(train_cfg, "async_transition_queue_capacity", 8192)))
    try:
        decision = core.next_decision(initial.observation)
        core.remember_action(channel.send_action(decision.action_requested), initial.observation, decision)
        while True:
            while local_transitions:
                try:
                    transition_queue.put_nowait(local_transitions[0])
                except queue.Full:
                    break
                local_transitions.popleft()
            try:
                if control_queue.get_nowait() == "stop":
                    break
            except queue.Empty:
                pass
            snapshot = _latest_nowait(weight_queue)
            if snapshot is not None:
                policy.load_snapshot(snapshot)
            message = channel.recv(timeout=10.0)
            now = time.perf_counter()
            collected = core.accept_state(message)
            if collected is not None:
                collected["collector_interval_ms"] = (now - last_time) * 1000.0
                collected["runtime_queue_depth"] = channel.queue_depth
                collected["inference_weight_version"] = policy.snapshot_version
                collected["repeated_action_steps"] = core.repeated_action_steps
                collected["repeated_action_rate"] = core.repeated_action_steps / max(core.policy_steps, 1)
                if len(local_transitions) >= local_capacity:
                    raise RuntimeError(
                        "learner transition queue is full and local FIFO buffer "
                        "reached its bounded capacity")
                local_transitions.append(collected)
                last_time = now
            decision = core.next_decision(message.observation)
            core.remember_action(channel.send_action(decision.action_requested), message.observation, decision)
    finally:
        # The learner owns the global step budget.  Tell the runtime to stop
        # publishing before closing the collector; otherwise the runtime can
        # continue writing into the ordered queue after this process exits.
        channel.stop()
        channel.close()
