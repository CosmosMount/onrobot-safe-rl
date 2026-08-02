"""Fixed-rate policy inference runtime for Go2."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from rl.agents.saferaw.agent import SafeRawSupervisor
from runtime.inference.actions import ActionApplier, ActionFilterButter
from runtime.inference.dds import DdsConfig, StateReader
from runtime.inference.ipc import PolicyClient
from runtime.inference.observations import (
    build_observation,
    get_run_reward_from_state,
    get_terminal_penalty,
)
from runtime.inference.transport import (
    SharedMemoryReceiver,
    SharedMemoryRingQueue,
    SharedMemorySender,
)


PolicyFn = Callable[[np.ndarray], np.ndarray]
CONTROLLER_PHASE_RECOVER = 1
CONTROLLER_PHASE_STAND_UP = 2
CONTROLLER_PHASE_POLICY = 3


class PolicyInferenceRuntime:
    """Receive observations and send policy actions at a fixed command rate."""

    def __init__(
        self,
        *,
        robot_cfg: Any,
        dds_config: DdsConfig,
        frequency_hz: float = 50.0,
        ipc_socket: str | None = None,
        max_joint_delta: float | None = None,
        use_action_filter: bool = True,
        action_socket: str | None = None,
        train_state_socket: str | None = None,
        max_episode_steps: int = 400,
        reset_hold_steps: int = 220,
        reset_joint_tolerance: float = 0.30,
        recovery_stable_steps: int = 10,
        standup_timeout_steps: int = 200,
        abort_on_unstable_reset: bool = True,
        ordered_state_queue: bool = False,
        ordered_state_queue_capacity: int = 2048,
        ordered_state_queue_slot_size: int = 16 * 1024,
    ):
        if frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        self.robot_cfg = robot_cfg
        self.dds_config = dds_config
        self.frequency_hz = float(frequency_hz)
        self.control_dt = 1.0 / self.frequency_hz
        self._state_reader = StateReader(
            socket_path=robot_cfg.state_socket,
            sport_velocity_world_frame=robot_cfg.sport_velocity_world_frame)
        self._policy_client = PolicyClient(ipc_socket or robot_cfg.ipc_socket)
        action_filter = (
            ActionFilterButter(
                num_joints=robot_cfg.num_joints,
                sampling_rate=self.frequency_hz,
                highcut=robot_cfg.action_filter_highcut,
            )
            if use_action_filter
            else None
        )
        self._action_applier = ActionApplier(
            init_qpos=robot_cfg.init_qpos,
            action_offset=robot_cfg.action_offset,
            joint_min=robot_cfg.joint_min,
            joint_max=robot_cfg.joint_max,
            max_joint_delta=max_joint_delta,
            action_filter=action_filter,
        )
        self._prev_requested_action = robot_cfg.init_qpos.astype(np.float32).copy()
        self._last_send_time: float | None = None
        self._latest_action = np.zeros(robot_cfg.num_joints, dtype=np.float32)
        self._policy_action_cleared = True
        self._motion_request_sent = False
        self._motion_request_recovery = False
        self._last_motion_request_time = 0.0
        self._recovery_upgrade_sent = False
        self._last_debug_time = 0.0
        self._last_debug_signature: tuple[Any, ...] | None = None
        self._step_count = 0
        self._runtime_step_id = 0
        self._episode_id = 0
        self._latest_action_id = -1
        self._action_sequence_latest = -1
        self._action_interaction_step_latest = -1
        self._action_policy_update_step_latest = -1
        self._action_sent_time_ns_latest = 0
        self._action_repeated_steps = 0
        self._awaiting_reset_pose = False
        self._reset_pose_stable_count = 0
        self._reset_pose_wait_steps = 0
        self._reset_pose_timed_out = False
        self._reset_pose_stable_steps = int(recovery_stable_steps)
        self._reset_hold_steps = int(reset_hold_steps)
        self._reset_joint_tolerance = float(reset_joint_tolerance)
        self._abort_on_unstable_reset = bool(abort_on_unstable_reset)
        self._action_rx = SharedMemoryReceiver(action_socket or robot_cfg.runtime_action_shm)
        self._train_tx = SharedMemorySender(train_state_socket or robot_cfg.runtime_state_shm)
        self._ordered_train_tx = (
            SharedMemoryRingQueue(
                f"{train_state_socket or robot_cfg.runtime_state_shm}.ordered",
                capacity=ordered_state_queue_capacity,
                slot_size=ordered_state_queue_slot_size,
            )
            if ordered_state_queue else None
        )
        self._safety = SafeRawSupervisor(
            inverted_acc_z_threshold=robot_cfg.imu_upside_down_acc_z,
            inverted_body_up_cos_threshold=robot_cfg.imu_upside_down_up_cos,
            fallen_roll_pitch_limit_rad=robot_cfg.success_orientation_rad,
            stable_steps=recovery_stable_steps,
            timeout_steps=standup_timeout_steps,
        )
        self._max_episode_steps = int(max_episode_steps)

    def _begin_reset_pose_wait(self) -> None:
        self._awaiting_reset_pose = True
        self._reset_pose_stable_count = 0
        self._reset_pose_wait_steps = 0
        self._reset_pose_timed_out = False

    def _end_reset_pose_wait(self) -> None:
        self._awaiting_reset_pose = False
        self._reset_pose_stable_count = 0
        self._reset_pose_wait_steps = 0
        self._reset_pose_timed_out = False

    @property
    def state_reader(self) -> StateReader:
        return self._state_reader

    @property
    def policy_client(self) -> PolicyClient:
        return self._policy_client

    @property
    def action_applier(self) -> ActionApplier:
        return self._action_applier

    def connect(self) -> None:
        self._state_reader.connect()
        state = self._state_reader.wait_for_state(timeout=10.0)
        self._state_reader.require_fresh_sport_state(
            self.robot_cfg.sport_state_max_age_ms / 1000.0)
        self._policy_client.connect()
        self._action_applier.reset_filter()
        self._action_applier.init_filter_history(state.joint_q)
        self._action_rx.bind()

    def observe(self) -> np.ndarray:
        state = self._state_reader.get_state()
        return build_observation(
            state,
            self._prev_requested_action,
            self.robot_cfg,
        )

    def send_action(self, action: np.ndarray) -> dict[str, Any]:
        self._motion_request_sent = False
        self._motion_request_recovery = False
        self._last_motion_request_time = 0.0
        self._recovery_upgrade_sent = False
        state = self._state_reader.get_state()
        projection = self._action_applier.project(action, state.joint_q)
        self._policy_client.send_target(projection.action_q_target)
        now = time.perf_counter()
        interval_ms = (
            (now - self._last_send_time) * 1000.0
            if self._last_send_time is not None
            else float("nan")
        )
        self._last_send_time = now
        self._prev_requested_action = projection.action_q_target.astype(np.float32).copy()
        self._policy_action_cleared = False
        runtime_delta = projection.action_executed - projection.action_requested
        runtime_norm = float(np.linalg.norm(runtime_delta))
        return {
            "action_requested": projection.action_requested.copy(),
            "action_executed": projection.action_executed.copy(),
            "action_q_target": projection.action_q_target.copy(),
            "projected_action": projection.action_requested.copy(),
            "executed_q_target": projection.action_q_target.copy(),
            "action_runtime_intervened": runtime_norm > 1e-6,
            "action_runtime_intervention_norm": runtime_norm,
            "action_interval_ms": interval_ms,
            "action_frequency_hz": (
                1000.0 / interval_ms
                if np.isfinite(interval_ms) and interval_ms > 0.0
                else float("nan")
            ),
        }

    def send_standup(self, *, with_recovery: bool) -> None:
        self._clear_policy_action()
        now = time.perf_counter()
        if (
            self._motion_request_sent
            and self._motion_request_recovery == with_recovery
            and now - self._last_motion_request_time < 0.2
        ):
            return
        self._policy_client.send_standup(
            with_recovery=with_recovery,
            q_target=self.robot_cfg.init_qpos,
        )
        self._motion_request_sent = True
        self._motion_request_recovery = with_recovery
        self._last_motion_request_time = now

    def step_standup(self, *, with_recovery: bool) -> tuple[np.ndarray, dict[str, Any]]:
        started = time.perf_counter()
        self.send_standup(with_recovery=with_recovery)
        elapsed = time.perf_counter() - started
        time.sleep(max(0.0, self.control_dt - elapsed))
        self._state_reader.require_fresh_sport_state(
            self.robot_cfg.sport_state_max_age_ms / 1000.0)
        return self.observe(), {
            "projected_action": np.zeros(self.robot_cfg.num_joints, dtype=np.float32),
            "executed_q_target": self.robot_cfg.init_qpos.copy(),
            "action_interval_ms": float("nan"),
            "action_frequency_hz": float("nan"),
            "control_hold_overrun_ms": max(0.0, elapsed - self.control_dt) * 1000.0,
        }

    def step_wait_controller(self) -> tuple[np.ndarray, dict[str, Any]]:
        started = time.perf_counter()
        self._clear_policy_action()
        elapsed = time.perf_counter() - started
        time.sleep(max(0.0, self.control_dt - elapsed))
        self._state_reader.require_fresh_sport_state(
            self.robot_cfg.sport_state_max_age_ms / 1000.0)
        return self.observe(), {
            "projected_action": np.zeros(self.robot_cfg.num_joints, dtype=np.float32),
            "executed_q_target": self.robot_cfg.init_qpos.copy(),
            "action_interval_ms": float("nan"),
            "action_frequency_hz": float("nan"),
            "control_hold_overrun_ms": max(0.0, elapsed - self.control_dt) * 1000.0,
        }

    def _runtime_debug(self, *, state: Any, safety: Any, action: str) -> None:
        now = time.perf_counter()
        signature = (
            int(state.phase),
            safety.mode.value,
            safety.reason,
            bool(safety.recovery_requested),
            bool(safety.inverted),
            bool(safety.fallen),
            action,
        )
        if signature == self._last_debug_signature and now - self._last_debug_time < 1.0:
            return
        self._last_debug_signature = signature
        self._last_debug_time = now
        print(
            "[runtime] "
            f"ctrl={int(state.phase)} mode={safety.mode.value} reason={safety.reason} "
            f"action={action} recover={int(bool(safety.recovery_requested))} "
            f"fallen={int(bool(safety.fallen))} inverted={int(bool(safety.inverted))} "
            f"roll={safety.roll:+.2f} pitch={safety.pitch:+.2f} "
            f"up_cos={safety.body_up_cos:+.2f} acc_z={safety.acc_z:+.2f}",
            flush=True,
        )

    def _reset_pose_error(self, state: Any) -> float:
        return float(np.linalg.norm(state.joint_q.astype(np.float32) - self.robot_cfg.init_qpos))

    def _reset_pose_ready(self, state: Any) -> bool:
        return self._reset_pose_error(state) < self._reset_joint_tolerance

    def step_recovery(self) -> tuple[np.ndarray, dict[str, Any]]:
        return self.step_standup(with_recovery=True)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        started = time.perf_counter()
        info = self.send_action(action)
        elapsed = time.perf_counter() - started
        time.sleep(max(0.0, self.control_dt - elapsed))
        self._state_reader.require_fresh_sport_state(
            self.robot_cfg.sport_state_max_age_ms / 1000.0)
        info["control_hold_overrun_ms"] = max(0.0, elapsed - self.control_dt) * 1000.0
        return self.observe(), info

    def run(self, policy_fn: PolicyFn, *, max_steps: int | None = None) -> None:
        self.connect()
        step = 0
        while max_steps is None or step < max_steps:
            observation = self.observe()
            action = policy_fn(observation)
            self.step(action)
            step += 1

    def close(self) -> None:
        self._clear_policy_action()
        self._policy_client.close()
        self._state_reader.close()
        self._action_rx.close()
        self._train_tx.close()
        if self._ordered_train_tx is not None:
            self._ordered_train_tx.close(unlink=self._ordered_train_tx.owner)

    def _clear_policy_action(self, *, reset_episode: bool = False) -> None:
        if reset_episode:
            self._step_count = 0
            self._episode_id = getattr(self, "_episode_id", 0) + 1
        if self._policy_action_cleared:
            return
        self._latest_action.fill(0.0)
        self._prev_requested_action = self.robot_cfg.init_qpos.astype(np.float32).copy()
        self._action_applier.reset_filter()
        self._action_rx.clear()
        self._action_sequence_latest = -1
        self._action_interaction_step_latest = -1
        self._action_policy_update_step_latest = -1
        self._action_sent_time_ns_latest = 0
        self._action_repeated_steps = 0
        self._policy_action_cleared = True

    def _receive_action(self) -> None:
        message = self._action_rx.recv_latest()
        if not message:
            return
        if (bool(message.get("clear", False))
                or message.get("command") == "clear"):
            # An explicit client handshake starts a fresh policy episode and
            # discards an action left in shared memory by an earlier client.
            self._clear_policy_action(reset_episode=True)
            return
        action = np.asarray(message.get("action", self._latest_action), dtype=np.float32)
        if action.shape == self._latest_action.shape and np.all(np.isfinite(action)):
            action_id = int(message.get("action_id", self._latest_action_id + 1))
            action_sequence = int(message.get("action_sequence", action_id))
            if action_id < self._latest_action_id:
                return
            if action_sequence < self._action_sequence_latest:
                return
            self._latest_action = np.clip(action, -1.0, 1.0)
            self._latest_action_id = action_id
            if action_sequence > self._action_sequence_latest:
                self._action_sequence_latest = action_sequence
                self._action_interaction_step_latest = int(message.get("action_interaction_step", -1))
                self._action_policy_update_step_latest = int(message.get("action_policy_update_step", -1))
                self._action_sent_time_ns_latest = int(message.get("action_sent_time_ns", 0))
                self._action_repeated_steps = 0
            self._policy_action_cleared = False
            self._recovery_upgrade_sent = False

    def _runtime_step(self) -> dict[str, Any]:
        self._runtime_step_id += 1
        previous_sequence = self._action_sequence_latest
        self._receive_action()
        if previous_sequence == self._action_sequence_latest and self._action_sequence_latest >= 0:
            self._action_repeated_steps += 1
        state = self._state_reader.get_state()
        controller_policy = int(state.phase) == CONTROLLER_PHASE_POLICY
        safety_before = self._safety.update(state)
        if not controller_policy:
            if (
                safety_before.recovery_requested
                and int(state.phase) != CONTROLLER_PHASE_RECOVER
                and not self._recovery_upgrade_sent
            ):
                observation, runtime_info = self.step_standup(with_recovery=True)
                action_debug = "request_recovery"
                self._recovery_upgrade_sent = True
            else:
                observation, runtime_info = self.step_wait_controller()
                action_debug = "wait_controller"
            state = self._state_reader.get_state()
            safety = self._safety.update(state)
        elif safety_before.restart_required:
            self._begin_reset_pose_wait()
            self._clear_policy_action()
            observation, runtime_info = self.step_standup(
                with_recovery=safety_before.recovery_requested)
            action_debug = "request_recovery" if safety_before.recovery_requested else "request_standup"
            self._recovery_upgrade_sent = bool(safety_before.recovery_requested)
            safety = safety_before
        elif safety_before.policy_enabled and self._awaiting_reset_pose:
            reset_ready_before = self._reset_pose_ready(state)
            if reset_ready_before:
                self._reset_pose_stable_count += 1
            else:
                self._reset_pose_stable_count = 0
            self._reset_pose_wait_steps += 1

            if self._reset_pose_stable_count >= self._reset_pose_stable_steps:
                self._end_reset_pose_wait()
                self._motion_request_sent = False
                self._last_motion_request_time = 0.0
                self._recovery_upgrade_sent = False
                observation, runtime_info = self.step_wait_controller()
                action_debug = "reset_pose_ready"
            else:
                timed_out = self._reset_pose_wait_steps >= self._reset_hold_steps
                self._reset_pose_timed_out = timed_out
                request_recovery = (
                    timed_out
                    and self._abort_on_unstable_reset
                    and not self._recovery_upgrade_sent
                )
                if timed_out:
                    self._reset_pose_wait_steps = 0
                    self._reset_pose_stable_count = 0
                    self._recovery_upgrade_sent = bool(request_recovery)
                if request_recovery:
                    observation, runtime_info = self.step_standup(with_recovery=True)
                    action_debug = "reset_pose_timeout_recovery"
                else:
                    observation, runtime_info = self.step_wait_controller()
                    action_debug = "reset_pose_timeout" if timed_out else "wait_reset_pose"
            state = self._state_reader.get_state()
            safety = self._safety.update(state)
        elif safety_before.policy_enabled:
            self._end_reset_pose_wait()
            if self._policy_action_cleared:
                # Do not manufacture policy transitions from a default zero
                # action before the collector reset handshake has completed.
                observation, runtime_info = self.step_wait_controller()
                action_debug = "wait_policy_action"
            else:
                observation, runtime_info = self.step(self._latest_action)
                action_debug = "policy"
            state = self._state_reader.get_state()
            safety = self._safety.update(state)
            if not safety.policy_enabled:
                self._clear_policy_action()
        elif not safety_before.inverted and not safety_before.fallen:
            self._safety.reset()
            self._motion_request_sent = False
            self._last_motion_request_time = 0.0
            self._recovery_upgrade_sent = False
            observation, runtime_info = self.step_wait_controller()
            state = self._state_reader.get_state()
            safety = self._safety.update(state)
            action_debug = "wait_recovered"
        else:
            self._clear_policy_action()
            observation, runtime_info = self.step_standup(
                with_recovery=safety_before.recovery_requested)
            action_debug = "request_recovery" if safety_before.recovery_requested else "request_standup"
            self._recovery_upgrade_sent = bool(safety_before.recovery_requested)
            state = self._state_reader.get_state()
            safety = self._safety.update(state)

        state = self._state_reader.get_state()
        reset_pose_error = self._reset_pose_error(state)
        reset_pose_ready = reset_pose_error < self._reset_joint_tolerance
        reward, reward_info = get_run_reward_from_state(state, self.robot_cfg)
        controller_policy = int(state.phase) == CONTROLLER_PHASE_POLICY
        policy_step = action_debug == "policy"
        replay_enabled = policy_step
        count_policy_step = policy_step and not safety.fallen and not safety.inverted
        if not controller_policy and safety.policy_enabled:
            safety.reason = "controller_nonpolicy"
        self._runtime_debug(state=state, safety=safety, action=action_debug)
        self._step_count += int(count_policy_step)
        episode_step = self._step_count
        truncated = self._step_count >= self._max_episode_steps
        terminated = bool(safety.terminated)
        terminal_penalty = get_terminal_penalty(terminated=terminated, cfg=self.robot_cfg)
        done = terminated or truncated
        if done:
            if terminated:
                self._begin_reset_pose_wait()
            self._clear_policy_action()
            self._step_count = 0
            if truncated and not safety.restart_required:
                self._safety.reset()

        info = {
            "policy_step": policy_step,
            "count_policy_step": count_policy_step,
            "replay_enabled": replay_enabled,
            "restart_required": bool(safety.restart_required),
            "terminated": terminated,
            "truncated": truncated,
            "inverted": bool(safety.inverted),
            "fallen": bool(safety.fallen),
            "near_failure": bool(
                not safety.terminated
                and (
                    abs(float(safety.roll))
                    >= float(self.robot_cfg.fallen_risk_rad)
                    or abs(float(safety.pitch))
                    >= float(self.robot_cfg.fallen_risk_rad)
                )
            ),
            "upright_gate": float(not safety.inverted and not safety.fallen),
            "safety_mode": safety.mode.value,
            "safety_reason": safety.reason,
            "controller_phase": int(state.phase),
            "safety_roll": float(safety.roll),
            "safety_pitch": float(safety.pitch),
            "safety_acc_z": float(safety.acc_z),
            "safety_body_up_cos": float(safety.body_up_cos),
            "reset_pose_error": reset_pose_error,
            "reset_pose_ready": bool(reset_pose_ready),
            "awaiting_reset_pose": bool(self._awaiting_reset_pose),
            "reset_pose_stable_count": float(self._reset_pose_stable_count),
            "reset_pose_wait_steps": float(self._reset_pose_wait_steps),
            "reset_pose_timed_out": bool(self._reset_pose_timed_out),
            # Preserve the terminal step number in the terminal message. The
            # internal counter may already have been reset for the next
            # episode, but consumers need the completed trajectory boundary.
            "step_count": episode_step,
            "runtime_step_id": self._runtime_step_id,
            "episode_id": self._episode_id,
            "episode_step": episode_step,
            # -1 explicitly means no policy action was executed on this
            # runtime tick (stand-up/recovery/reset supervision).
            "applied_action_id": self._latest_action_id if policy_step else -1,
            "action_sequence": self._action_sequence_latest if policy_step else -1,
            "action_interaction_step": self._action_interaction_step_latest if policy_step else -1,
            "action_policy_update_step": self._action_policy_update_step_latest if policy_step else -1,
            "action_age_ms": (
                (time.monotonic_ns() - self._action_sent_time_ns_latest) / 1_000_000.0
                if policy_step and self._action_sent_time_ns_latest > 0 else float("nan")
            ),
            "action_repeated_steps": self._action_repeated_steps if policy_step else 0,
            "terminal_penalty": float(terminal_penalty),
            **runtime_info,
            **reward_info,
        }
        info["joint_q"] = state.joint_q.copy()
        info["joint_dq"] = state.joint_dq.copy()
        q_target = info.get("executed_q_target")
        if q_target is not None:
            info["joint_tracking_error"] = (
                state.joint_q.astype(np.float32) - np.asarray(q_target, dtype=np.float32)
            )
        return {
            "observation": observation,
            "reward": float(reward + terminal_penalty),
            "done": done,
            "info": info,
        }

    def run_process(self) -> None:
        self.connect()
        if self._ordered_train_tx is not None:
            self._ordered_train_tx.create()
            if not self._ordered_train_tx.owner:
                raise RuntimeError(
                    "ordered state queue already exists; refusing to reset "
                    "an active/stale collector stream. Stop the old runtime "
                    "and unlink the stale queue before restarting.")
        print(
            f"[runtime] ready action_shm={self._action_rx.socket_path} "
            f"state_shm={self._train_tx.socket_path} hz={self.frequency_hz}",
            flush=True,
        )
        while True:
            message = self._runtime_step()
            # Compatibility mailbox remains available for the existing
            # synchronous client. The async collector consumes the ordered
            # queue, which never silently overwrites a terminal transition.
            self._train_tx.send(message)
            if self._ordered_train_tx is not None:
                self._ordered_train_tx.write(message)


def main(argv=None) -> int:
    import argparse
    from train.config import load_app_config

    parser = argparse.ArgumentParser(description="Go2 fixed-rate runtime")
    parser.add_argument(
        "--config-profile",
        choices=("go2", "simulation", "real_robot"),
        default="go2",
    )
    parser.add_argument(
        "--ordered-state-queue",
        action="store_true",
        help="Publish every runtime step to the ordered async collector queue.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML overlay path; overrides --config-profile.",
    )
    args = parser.parse_args(argv)
    robot_cfg, train_cfg, _ = load_app_config(
        path=args.config,
        profile=args.config_profile,
    )
    runtime = PolicyInferenceRuntime(
        robot_cfg=robot_cfg,
        dds_config=DdsConfig(robot_cfg.domain_id, robot_cfg.interface),
        frequency_hz=train_cfg.control_frequency,
        ipc_socket=robot_cfg.ipc_socket,
        max_joint_delta=train_cfg.max_joint_delta,
        use_action_filter=train_cfg.use_action_filter,
        max_episode_steps=train_cfg.max_episode_steps,
        reset_hold_steps=train_cfg.reset_hold_steps,
        reset_joint_tolerance=train_cfg.reset_joint_tolerance,
        recovery_stable_steps=train_cfg.recovery_stable_steps,
        standup_timeout_steps=train_cfg.standup_timeout_steps,
        abort_on_unstable_reset=train_cfg.abort_on_unstable_reset,
        ordered_state_queue=args.ordered_state_queue,
    )
    runtime.run_process()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
