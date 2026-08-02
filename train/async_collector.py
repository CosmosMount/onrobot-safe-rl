"""Dedicated fixed-rate collector process for paper SQRL."""

from __future__ import annotations

import queue
import time
from dataclasses import asdict
from multiprocessing.queues import Queue
from typing import Any

import numpy as np

from rl.agents.paper_sqrl.inference import (
    SQRLActionDecision,
    build_inference_policy,
)
from train.ordered_runtime import OrderedRuntimeChannel, RuntimeEnvelope


def _transition(observation: np.ndarray, decision: SQRLActionDecision,
                message: RuntimeEnvelope) -> dict[str, np.ndarray]:
    info = message.info
    return {
        "observation": np.asarray(observation, dtype=np.float32)[None, ...],
        "action": np.asarray(decision.action, dtype=np.float32)[None, ...],
        "reward": np.asarray([message.reward], dtype=np.float32),
        "terminated": np.asarray(
            [bool(info.get("terminated", False))], dtype=np.float32),
        "truncated": np.asarray(
            [bool(info.get("truncated", False))], dtype=np.float32),
        "next_observation": np.asarray(
            message.observation, dtype=np.float32)[None, ...],
        "unsafe_label": np.asarray([
            bool(info.get("terminated", False))
            or bool(info.get("fallen", False))
            or bool(info.get("inverted", False))
        ], dtype=np.float32),
        "near_failure_label": np.asarray(
            [bool(info.get("near_failure", False))], dtype=np.float32),
        "replay_repeat_index": np.asarray([0], dtype=np.int32),
        "sqrl_collection_phase": np.asarray(
            [1 if decision.phase == "safety" else 0], dtype=np.int8),
        "sqrl_selector_active": np.asarray([decision.active], dtype=np.float32),
        "sqrl_selector_replaced": np.asarray([decision.replaced], dtype=np.float32),
        "sqrl_selector_no_safe": np.asarray([decision.no_safe], dtype=np.float32),
        "sqrl_selector_safe_rate": np.asarray([decision.safe_rate], dtype=np.float32),
        "runtime_step_id": np.asarray([message.runtime_step_id], dtype=np.int64),
        "episode_id": np.asarray([message.episode_id], dtype=np.int64),
        "episode_step": np.asarray([message.episode_step], dtype=np.int32),
        "applied_action_id": np.asarray(
            [message.applied_action_id], dtype=np.int64),
        **({"sqrl_nominal_risk": np.asarray(
            [decision.nominal_risk], dtype=np.float32)}
           if decision.nominal_risk is not None else {}),
        **({"sqrl_selected_risk": np.asarray(
            [decision.selected_risk], dtype=np.float32)}
           if decision.selected_risk is not None else {}),
    }


class PaperSQRLCollectorCore:
    """Testable state/action attribution core; one instance per collector."""

    def __init__(self, policy: Any, action_space: Any,
                 *, start_training: int, explore_action_scale: float,
                 seed: int, initial_policy_steps: int = 0):
        self.policy = policy
        self.action_space = action_space
        self.action_space.seed(seed)
        self.start_training = int(start_training)
        self.explore_action_scale = float(explore_action_scale)
        self.policy_steps = int(initial_policy_steps)
        self.pending: dict[int, tuple[np.ndarray, SQRLActionDecision]] = {}
        self.last_applied_action_id: int | None = None
        self.repeated_action_steps = 0

    def accept_state(self, message: RuntimeEnvelope) -> dict[str, Any] | None:
        if message.applied_action_id < 0:
            return None
        pending = self.pending.get(message.applied_action_id)
        if pending is None:
            raise RuntimeError(
                "runtime executed an action missing from collector history: "
                f"action_id={message.applied_action_id}")
        observation, decision = pending
        self.repeated_action_steps += int(
            self.last_applied_action_id == message.applied_action_id)
        self.last_applied_action_id = message.applied_action_id
        result = _transition(observation, decision, message)
        # A runtime tick can legitimately repeat the latest command when
        # inference takes longer than one 20 ms period. Attribute each repeat
        # to the state at the start of that tick, not to the original state on
        # which the command was first produced.
        self.pending[message.applied_action_id] = (
            message.observation.copy(), decision)
        self.policy.observe_transition(
            policy_step=bool(message.info.get("policy_step", True)),
            terminated=bool(message.info.get("terminated", False)),
            truncated=bool(message.info.get("truncated", False)))
        self.policy_steps += int(bool(message.info.get("policy_step", True)))
        # Any older command was skipped by the latest-action mailbox and can
        # never become attributable to a future state.
        for stale_id in [key for key in self.pending
                         if key < message.applied_action_id]:
            self.pending.pop(stale_id, None)
        return {"transition": result, "info": message.info,
                "decision": asdict(decision)}

    def next_decision(self, observation: np.ndarray) -> SQRLActionDecision:
        nominal = None
        if self.policy_steps < self.start_training:
            nominal = (
                self.action_space.sample().astype(np.float32)
                * self.explore_action_scale)
        return self.policy.decide(
            observation, training=True, nominal=nominal)

    def remember_action(self, action_id: int, observation: np.ndarray,
                        decision: SQRLActionDecision) -> None:
        self.pending[int(action_id)] = (
            np.asarray(observation, dtype=np.float32).copy(), decision)


def _latest_nowait(weight_queue: Queue) -> dict[str, Any] | None:
    latest = None
    while True:
        try:
            latest = weight_queue.get_nowait()
        except queue.Empty:
            return latest


def run_async_collector(*, robot_cfg: Any, agent_cfg: Any,
                             train_cfg: Any, observation_dim: int,
                             action_dim: int, action_space: Any,
                             transition_queue: Queue, weight_queue: Queue,
                             control_queue: Queue,
                             initial_policy_steps: int = 0) -> None:
    """Process target: collect every ordered runtime tick independently."""
    policy = build_inference_policy(observation_dim, action_dim, agent_cfg)
    initial = weight_queue.get(timeout=120.0)
    policy.load_weights(initial)
    core = PaperSQRLCollectorCore(
        policy, action_space, start_training=train_cfg.start_training,
        explore_action_scale=train_cfg.explore_action_scale,
        seed=train_cfg.seed, initial_policy_steps=initial_policy_steps)
    channel = OrderedRuntimeChannel(
        robot_cfg.runtime_action_shm, robot_cfg.runtime_state_shm)
    channel.connect(timeout=120.0)
    initial = channel.recv(timeout=10.0)
    initial_episode_id = initial.episode_id
    channel.clear_action()
    # Do not overwrite the reset command with an action. Consume the ordered
    # stream until runtime acknowledges the new episode generation.
    while True:
        initial = channel.recv(timeout=10.0)
        if initial.episode_id > initial_episode_id:
            break
    last_time = time.perf_counter()
    try:
        decision = core.next_decision(initial.observation)
        action_id = channel.send_action(decision.action)
        core.remember_action(action_id, initial.observation, decision)
        while True:
            try:
                command = control_queue.get_nowait()
                if command == "stop":
                    break
            except queue.Empty:
                pass
            weights = _latest_nowait(weight_queue)
            if weights is not None:
                policy.load_weights(weights)
            message = channel.recv(timeout=10.0)
            now = time.perf_counter()
            collected = core.accept_state(message)
            if collected is not None:
                collected["collector_interval_ms"] = (now - last_time) * 1000.0
                collected["runtime_queue_depth"] = channel.queue_depth
                collected["inference_weight_version"] = policy.weight_version
                collected["repeated_action_steps"] = core.repeated_action_steps
                collected["repeated_action_rate"] = (
                    core.repeated_action_steps / max(core.policy_steps, 1))
                transition_queue.put(collected, timeout=10.0)
                last_time = now
                if core.policy_steps >= int(train_cfg.max_steps):
                    # Collection, rather than the lagging learner, owns the
                    # interaction budget. Stop issuing actions at the exact
                    # requested policy-step boundary.
                    channel.clear_action()
                    break
            decision = core.next_decision(message.observation)
            action_id = channel.send_action(decision.action)
            core.remember_action(action_id, message.observation, decision)
    finally:
        channel.close()
