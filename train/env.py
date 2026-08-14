"""A synchronous 20 Hz Go2 environment for online locomotion training.

The controller emits exactly one state at each policy tick.  ``step`` sends
one action and waits for the next state carrying the action id that was
actually applied during that interval; no state is dropped or reused.
"""

from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


POLICY_SOF = 0xA5
STATE_SOF = 0x5A
FLAG_STAND_UP = 0x01
FLAG_RECOVERY = 0x02
FLAG_STOP = 0x04
POLICY_HZ = 20.0
PHASE_AWAIT_STATE = 0
PHASE_RECOVER = 1
PHASE_STAND_UP = 2
PHASE_POLICY = 3
PHASE_RETURN_HOME = 4
PHASE_SHUTDOWN = 5
PHASE_SAFETY_HOLD = 6

EVENT_NONE = 0
EVENT_FALLEN_STANDUP = 1
EVENT_UPSIDE_DOWN_RECOVERY = 2
EVENT_STANDUP_FAILED = 3

POLICY_STRUCT = struct.Struct("=BBQd12f")
STATE_STRUCT = struct.Struct("=BBBQQQIdII12f12f4f3f3f3f3f12f")

REWARD_MOVE_SPEED = 0.50
REWARD_SCALE = 10.0
REWARD_YAW_RATE_WEIGHT = 0.10
INIT_QPOS = np.asarray([0.05, 0.7, -1.4] * 4, dtype=np.float32)


def _reward_tolerance(value, lower, upper, margin):
    if lower <= value <= upper:
        return 1.0
    if margin <= 0.0:
        return 0.0
    distance = lower - value if value < lower else value - upper
    if distance >= margin:
        return 0.0
    return 1.0 - distance / margin


def _reward_pitch_cosine(quat):
    q = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-6:
        q = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    else:
        q = q / norm
    _, pitch = _reward_euler(q)
    return float(np.cos(pitch))


def _reward_euler(quat):
    q = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-6:
        q = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        q = q / norm
    w, x, y, z = q
    roll = np.arctan2(2.0 * (w * x + y * z),
                      1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = np.copysign(np.pi / 2.0, sin_pitch) if abs(sin_pitch) >= 1.0 else np.arcsin(sin_pitch)
    return float(roll), float(pitch)


def _locomotion_straight_reward(state, action, previous_action, body_velocity,
                                terminated, return_info=False):
    """Original walk_in_the_park forward-walking reward."""
    del action, previous_action, terminated
    vx = float(body_velocity[0])
    dyaw = float(state.imu_gyro[2])
    cos_pitch = _reward_pitch_cosine(state.imu_quat)
    forward = _reward_tolerance(
        cos_pitch * vx, REWARD_MOVE_SPEED, 2.0 * REWARD_MOVE_SPEED,
        2.0 * REWARD_MOVE_SPEED)
    reward = float(REWARD_SCALE * (forward - REWARD_YAW_RATE_WEIGHT * abs(dyaw)))
    reward_terms = {"vx": vx, "cos_pitch": cos_pitch, "dyaw": dyaw,
                    "forward_reward": float(REWARD_SCALE * forward),
                    "yaw_penalty": float(REWARD_SCALE * REWARD_YAW_RATE_WEIGHT * abs(dyaw)),
                    "total": reward}
    return (reward, reward_terms) if return_info else reward


@dataclass(frozen=True)
class Go2State:
    policy_sequence: int
    applied_action_id: int
    event: int
    event_action_id: int
    event_confirm_ms: int
    timestamp: float
    joint_q: np.ndarray
    joint_dq: np.ndarray
    imu_quat: np.ndarray
    imu_gyro: np.ndarray
    imu_accel: np.ndarray
    velocity: np.ndarray
    position: np.ndarray
    q_target: np.ndarray
    phase: int


def decode_state(payload: bytes) -> Go2State:
    if len(payload) != STATE_STRUCT.size:
        raise ValueError(
            f"controller protocol mismatch: received state datagram of {len(payload)} "
            f"bytes, expected {STATE_STRUCT.size}; stop any old go2_control and "
            "start /tmp/go2-build/go2_control from this worktree")
    values = STATE_STRUCT.unpack(payload)
    if values[0] != STATE_SOF:
        raise ValueError("invalid state SOF")
    i = 8  # SOF, phase, event, policy seq, applied/event ids, confirm ms, timestamp
    low_count, sport_count = values[i:i + 2]
    del low_count, sport_count
    i += 2
    joint_q = np.asarray(values[i:i + 12], np.float32); i += 12
    joint_dq = np.asarray(values[i:i + 12], np.float32); i += 12
    quat = np.asarray(values[i:i + 4], np.float32); i += 4
    gyro = np.asarray(values[i:i + 3], np.float32); i += 3
    accel = np.asarray(values[i:i + 3], np.float32); i += 3
    velocity = np.asarray(values[i:i + 3], np.float32); i += 3
    position = np.asarray(values[i:i + 3], np.float32); i += 3
    target = np.asarray(values[i:i + 12], np.float32)
    return Go2State(
        policy_sequence=int(values[3]), applied_action_id=int(values[4]),
        event=int(values[2]), event_action_id=int(values[5]),
        event_confirm_ms=int(values[6]), timestamp=float(values[7]),
        joint_q=joint_q, joint_dq=joint_dq,
        imu_quat=quat, imu_gyro=gyro, imu_accel=accel, velocity=velocity,
        position=position, q_target=target, phase=int(values[1]))


class Go2SyncEnv:
    """Gym-like real Go2 adapter with synchronous 20 Hz step semantics."""

    def __init__(self, *, policy_socket: str = "/tmp/go2_policy.v3.sock",
                 state_socket: str = "/tmp/go2_policy.v3.sock.state",
                 init_qpos=None, action_offset=None, max_episode_steps=400,
                 timeout=2.0, sport_velocity_world_frame=True):
        self.policy_socket = policy_socket
        self.state_socket = state_socket
        self.init_qpos = np.asarray(
            [0.05, 0.7, -1.4] * 4 if init_qpos is None else init_qpos,
            np.float32)
        self.action_offset = np.asarray(
            [0.2, 0.4, 0.4] * 4 if action_offset is None else action_offset,
            np.float32)
        if self.init_qpos.shape != (12,) or self.action_offset.shape != (12,):
            raise ValueError("Go2 init_qpos and action_offset must have length 12")
        if (not np.all(np.isfinite(self.init_qpos)) or
                not np.all(np.isfinite(self.action_offset)) or
                np.any(self.action_offset <= 0.0)):
            raise ValueError("Go2 action mapping must contain finite positive offsets")
        self.max_episode_steps = int(max_episode_steps)
        self.timeout = float(timeout)
        self.sport_velocity_world_frame = bool(sport_velocity_world_frame)
        self._rx = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._rx.bind(state_socket)
        self._rx.settimeout(self.timeout)
        self._tx = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._next_action_id = 1
        self._last_state = None
        self.state_intervals_ms = []
        self._last_receive_time = None
        self.action_intervals_ms = []
        self._last_send_time = None
        self._last_payload = None
        self._steps = 0
        self._closing = False
        self._closed = False
        self.last_safety_hold_info = None
        self._previous_action = np.zeros(12, dtype=np.float32)
        # The observable includes the previous absolute qpos action.
        self._previous_qtarget = self.init_qpos.copy()
        self.observation_space_shape = (46,)
        self.action_space_shape = (12,)

    def close(self):
        if self._closed:
            return
        self._closing = True
        try:
            if self._tx.fileno() >= 0:
                stop_id, _ = self._send_shutdown()
                # A stop issued while inverted must finish recovery,
                # stand-up verification, and return-home before local IPC is
                # closed. This can legitimately take more than ten seconds.
                deadline = time.perf_counter() + max(20.0, self.timeout * 10.0)
                while time.perf_counter() < deadline:
                    try:
                        state = self._recv()
                    except socket.timeout:
                        self._resend_last()
                        continue
                    if state.phase in (PHASE_SHUTDOWN, PHASE_SAFETY_HOLD):
                        break
                    if state.applied_action_id < stop_id:
                        self._resend_last()
        except (OSError, RuntimeError, ValueError) as exc:
            # The controller may already be gone; local cleanup must still run.
            print(f"[train] controller shutdown warning: {exc}", flush=True)
        self._last_payload = None
        self._previous_action.fill(0.0)
        self._previous_qtarget = self.init_qpos.copy()
        self._rx.close(); self._tx.close()
        try: Path(self.state_socket).unlink()
        except FileNotFoundError: pass
        self._closed = True

    def _send_shutdown(self):
        """Send a stop packet without allowing any further learner action."""
        action_id = self._next_action_id; self._next_action_id += 1
        q_target = self.init_qpos.copy()
        payload = POLICY_STRUCT.pack(POLICY_SOF, FLAG_STOP, action_id,
                                     time.time(), *q_target.tolist())
        self._tx.sendto(payload, self.policy_socket)
        self._last_payload = payload
        return action_id, q_target

    def _recv(self):
        while True:
            state = decode_state(self._rx.recv(STATE_STRUCT.size))
            now = time.perf_counter()
            if self._last_receive_time is not None:
                self.state_intervals_ms.append(
                    (now - self._last_receive_time) * 1000.0)
            self._last_receive_time = now
            if (self._last_state is not None and
                    state.policy_sequence != self._last_state.policy_sequence + 1):
                raise RuntimeError(
                    f"controller state sequence gap: got {state.policy_sequence}, "
                    f"expected {self._last_state.policy_sequence + 1}")
            self._last_state = state
            return state

    def _recv_for_action(self, action_id: int) -> Go2State:
        """Wait for the first policy state that acknowledges ``action_id``.

        A state already queued at the instant an action is sent can
        legitimately acknowledge the preceding target.  It is consumed and
        accounted for, but must not be used as the transition for the new
        action.  An acknowledgement for a newer id means a packet was lost,
        so fail closed.  The transport wait is bounded by ``self.timeout``;
        the controller itself continues to publish policy states at 20 Hz.
        """
        # The action may be sent immediately after a controller tick, so its
        # acknowledgement can legitimately arrive almost 50 ms later.  Do
        # not turn that phase offset into a false deadline miss.
        deadline = time.perf_counter() + self.timeout
        resend_count = 0
        try:
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    last = self._last_state
                    raise RuntimeError(
                        f"controller timeout waiting for action {action_id}; "
                        f"last_state=(sequence={getattr(last, 'policy_sequence', None)}, "
                        f"phase={getattr(last, 'phase', None)}, "
                        f"applied_action_id={getattr(last, 'applied_action_id', None)}), "
                        f"retries={resend_count}")
                self._rx.settimeout(min(self.timeout, remaining))
                try:
                    state = self._recv()
                except socket.timeout as exc:
                    last = self._last_state
                    raise RuntimeError(
                        f"controller timeout waiting for action {action_id}; "
                        f"last_state=(sequence={getattr(last, 'policy_sequence', None)}, "
                        f"phase={getattr(last, 'phase', None)}, "
                        f"applied_action_id={getattr(last, 'applied_action_id', None)}), "
                        f"retries={resend_count}") from exc
                # Recovery/stand-up owns the target and can begin before the
                # controller acknowledges the policy action that exposed the
                # fall.  Treat that terminal phase as the transition result;
                # retrying the learner action would otherwise wait forever
                # because recovery intentionally does not advance the policy
                # action id.
                if state.phase == PHASE_SAFETY_HOLD:
                    return state
                if (state.event != EVENT_NONE and
                        state.event_action_id <= action_id):
                    return state
                if (state.phase in (PHASE_RECOVER, PHASE_STAND_UP) and
                        state.applied_action_id <= action_id):
                    return state
                if state.applied_action_id < action_id:
                    resend_count += 1
                    # AF_UNIX datagrams are not reliable. Retransmit the same
                    # id; the controller applies each id at most once, so the
                    # operation is idempotent and cannot create a duplicate
                    # learner transition.
                    self._resend_last()
                    continue
                if state.applied_action_id > action_id:
                    raise RuntimeError(
                        f"controller skipped action {action_id}; state "
                        f"{state.policy_sequence} acknowledges {state.applied_action_id}")
                return state
        finally:
            self._rx.settimeout(self.timeout)

    @staticmethod
    def body_velocity(state: Go2State, world_frame=True) -> np.ndarray:
        if not world_frame:
            return state.velocity
        q = np.asarray(state.imu_quat, dtype=np.float64)
        norm = float(np.linalg.norm(q))
        if norm < 1e-8:
            return state.velocity
        qw, qx, qy, qz = q / norm
        rotation_body_to_world = np.asarray([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz),
             2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz),
             2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx),
             1 - 2 * (qx * qx + qy * qy)],
        ])
        return (rotation_body_to_world.T @ state.velocity).astype(np.float32)

    def observation(self, state: Go2State) -> np.ndarray:
        # This is the A1 walker observable order: joints_pos, joints_vel,
        # gyro, velocimeter, framequat, and one previous absolute qpos action.
        return np.concatenate((state.joint_q, state.joint_dq, state.imu_gyro,
                               self.body_velocity(
                                   state, self.sport_velocity_world_frame),
                               state.imu_quat,
                               self._previous_qtarget)).astype(np.float32)

    def _send(self, action: np.ndarray, flags: int = 0):
        if self._closing or self._closed:
            raise RuntimeError("environment is shutting down; action rejected")
        action = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        q_target = self.init_qpos + self.action_offset * action
        action_id = self._next_action_id; self._next_action_id += 1
        now = time.perf_counter()
        if self._last_send_time is not None:
            self.action_intervals_ms.append(
                (now - self._last_send_time) * 1000.0)
        self._last_send_time = now
        payload = POLICY_STRUCT.pack(POLICY_SOF, int(flags), action_id,
                                     time.time(), *q_target.tolist())
        self._tx.sendto(payload, self.policy_socket)
        self._last_payload = payload
        return action_id, q_target

    def _resend_last(self):
        if self._last_payload is None:
            raise RuntimeError("cannot retransmit before sending an action")
        self._tx.sendto(self._last_payload, self.policy_socket)

    def reset(self):
        if self._closing or self._closed:
            raise RuntimeError("environment is closed")
        self.last_safety_hold_info = None
        # A normal episode timeout is not a physical reset.  Re-requesting
        # stand-up while the robot is already in POLICY needlessly drives it
        # through pose_1 -> pose_2 and can make the feet slide.  Only request
        # stand-up on the first reset or after the controller has left POLICY.
        request_standup = (
            self._last_state is None or self._last_state.phase != PHASE_POLICY)
        if request_standup:
            self._send(np.zeros(12, np.float32), FLAG_STAND_UP)
        # Worst case is recovery (~3.4 s), stand-up (~5 s), and the 5 s
        # stability verification timeout. Leave enough room to receive the
        # explicit SAFETY_HOLD result instead of reporting a generic timeout.
        deadline = time.perf_counter() + max(20.0, self.timeout * 10.0)
        state = None
        while time.perf_counter() < deadline:
            try:
                state = self._recv()
            except socket.timeout:
                if request_standup:
                    # AF_UNIX datagrams are best effort.  Re-send the same
                    # lifecycle request while waiting for the controller to
                    # leave recovery/stand-up/shutdown.
                    self._resend_last()
                continue
            if state.phase == PHASE_POLICY:
                break
            if state.phase == PHASE_SAFETY_HOLD:
                roll, pitch = _reward_euler(state.imu_quat)
                self.last_safety_hold_info = {
                    "event": state.event,
                    "event_action_id": state.event_action_id,
                    "event_confirm_ms": state.event_confirm_ms,
                    "roll": roll,
                    "pitch": pitch,
                    "up_cos": float(np.cos(roll) * np.cos(pitch)),
                    "acc_z": float(state.imu_accel[2]),
                }
                raise RuntimeError(
                    "controller entered SAFETY_HOLD: stand-up verification "
                    "failed; policy actions are latched off until the "
                    "controller is restarted after manual inspection")
        if state is None or state.phase != PHASE_POLICY:
            raise RuntimeError(
                "controller reset timeout: expected POLICY phase, got "
                f"phase={getattr(state, 'phase', None)} "
                f"sequence={getattr(state, 'policy_sequence', None)}; "
                "restart/rebuild the matching go2_control if it remains in "
                "SHUTDOWN or uses an old IPC protocol")
        # The controller can outlive this Python process and deliberately
        # keeps applied_action_id monotonic across learner sessions.  Do not
        # restart a new session at action id 1: an old ACK (for example 5)
        # would otherwise look like the controller skipped the new action 2.
        self._next_action_id = max(
            self._next_action_id, int(state.applied_action_id) + 1)
        # Stand-up is an environment lifecycle phase, not a policy interval.
        self._last_send_time = None
        self.action_intervals_ms.clear()
        self._previous_action.fill(0.0)
        self._previous_qtarget = self.init_qpos.copy()
        self._steps = 0
        return self.observation(state), {"policy_sequence": state.policy_sequence}

    def step(self, action):
        if self._closing or self._closed:
            raise RuntimeError("environment is closed")
        action = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        action_id, q_target = self._send(action)
        state = self._recv_for_action(action_id)
        if state.phase == PHASE_AWAIT_STATE:
            raise RuntimeError("controller is still awaiting a valid state")
        if state.phase == PHASE_SAFETY_HOLD:
            roll, pitch = _reward_euler(state.imu_quat)
            self.last_safety_hold_info = {
                "event": state.event,
                "event_action_id": state.event_action_id,
                "event_confirm_ms": state.event_confirm_ms,
                "roll": roll,
                "pitch": pitch,
                "up_cos": float(np.cos(roll) * np.cos(pitch)),
                "acc_z": float(state.imu_accel[2]),
            }
            raise RuntimeError(
                "controller entered SAFETY_HOLD: stand-up verification failed")
        if state.phase not in (PHASE_RECOVER, PHASE_STAND_UP, PHASE_POLICY):
            raise RuntimeError(f"unknown controller phase {state.phase}")
        body_velocity = self.body_velocity(
            state, self.sport_velocity_world_frame)
        failure_event = state.event in (
            EVENT_FALLEN_STANDUP, EVENT_UPSIDE_DOWN_RECOVERY)
        event_matches_action = bool(
            failure_event and state.event_action_id == action_id and
            state.applied_action_id == action_id)
        causal_mismatch = bool(
            failure_event and not event_matches_action)
        event_action_lag = 0
        if causal_mismatch and state.event_action_id < action_id:
            event_action_lag = int(action_id - state.event_action_id)
        recovery_motion = bool(
            state.phase in (PHASE_RECOVER, PHASE_STAND_UP) and
            not event_matches_action)
        # Recovery/stand-up is a controller lifecycle interruption, not an
        # MDP failure transition.  Truncate the learner episode and let the
        # training loop pause until reset() observes POLICY again.
        terminated = event_matches_action
        reward, reward_terms = _locomotion_straight_reward(
            state, action, self._previous_action, body_velocity,
            terminated, return_info=True)
        action_applied = state.applied_action_id == action_id
        if action_applied:
            self._steps += 1
            self._previous_action = action.copy()
            self._previous_qtarget = q_target.copy()
        time_limit = self._steps >= self.max_episode_steps
        truncated = bool(not terminated and (time_limit or recovery_motion))
        info = {"policy_sequence": state.policy_sequence,
                "requested_action_id": action_id,
                "applied_action_id": state.applied_action_id,
                "phase": state.phase,
                "event": state.event,
                "event_action_id": state.event_action_id,
                "event_confirm_ms": state.event_confirm_ms,
                "event_action_lag": event_action_lag,
                "action_applied": action_applied,
                "policy_transition": bool(action_applied and
                                           not recovery_motion),
                "causal_mismatch": causal_mismatch,
                "time_limit": time_limit,
                "terminal_patch_action_id": (
                    state.event_action_id if causal_mismatch else None),
                # Motions in these phases are generated by the controller,
                # not by the learner.  Their observations must never become
                # policy replay transitions.
                "recovery_motion": recovery_motion,
                "terminated": terminated, "truncated": truncated}
        roll, pitch = _reward_euler(state.imu_quat)
        joint_tracking_error = state.q_target - state.joint_q
        info.update({"safety/roll": roll,
                     "safety/pitch": pitch,
                     "safety/up_cos": float(np.cos(roll) * np.cos(pitch)),
                     "safety/acc_z": float(state.imu_accel[2]),
                     "safety/joint_tracking_error_rms": float(
                         np.linalg.norm(joint_tracking_error) /
                         np.sqrt(joint_tracking_error.size)),
                     "safety/joint_tracking_error_max": float(
                         np.max(np.abs(joint_tracking_error)))})
        info.update({f"reward/{key}": value
                     for key, value in reward_terms.items()})
        return self.observation(state), float(reward), terminated, truncated, info
