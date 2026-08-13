"""A synchronous 20 Hz Go2 environment for the walk_in_the_park protocol.

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
POLICY_HZ = 20.0
PHASE_AWAIT_STATE = 0
PHASE_RECOVER = 1
PHASE_STAND_UP = 2
PHASE_POLICY = 3

POLICY_STRUCT = struct.Struct("=BBQd12f")
STATE_STRUCT = struct.Struct("=BBQQdII12f12f4f3f3f3f3f12f")

# Copied from ../_worktree_execution_aware_droq's
# config/rewards/locomotion_straight.yaml plus its Go2 overlay.  These values
# are intentionally local to the reward; the transport/observation protocol
# remains unchanged.
REWARD_COMMAND_VX = 0.30
REWARD_TRACKING_SIGMA = 0.25
REWARD_UPRIGHT_MIN_COS = 0.94
REWARD_UPRIGHT_EXPONENT = 2.0
REWARD_TRACKING_LIN_VEL_WEIGHT = 8.0
REWARD_TRACKING_ANG_VEL_WEIGHT = 4.0
REWARD_ROLL_PITCH_RATE_WEIGHT = 0.4
REWARD_LATERAL_VELOCITY_WEIGHT = 0.5
REWARD_VERTICAL_VELOCITY_WEIGHT = 0.2
REWARD_ACTION_RATE_WEIGHT = 0.05
REWARD_ACTION_RATE_SCALE = 0.25
REWARD_ACTION_RATE_PENALTY_MAX = 4.0
REWARD_ACTION_MAGNITUDE_WEIGHT = 0.02
REWARD_ACTION_MAGNITUDE_SCALE = 0.60
REWARD_ACTION_MAGNITUDE_PENALTY_MAX = 2.0
REWARD_ANGULAR_RATE_SCALE = 2.0
REWARD_LATERAL_VELOCITY_SCALE = 0.35
REWARD_VERTICAL_VELOCITY_SCALE = 0.40
REWARD_VERTICAL_VELOCITY_PENALTY_MAX = 4.0
REWARD_LEG_ACTIVITY_EPSILON = 0.01
REWARD_LEG_BALANCE_SPEED_GATE = 0.05
REWARD_LEG_ACTIVITY_BALANCE_WEIGHT = 0.05
REWARD_LEG_ACTION_ACTIVITY_SCALE = 0.05
REWARD_LEG_JOINT_VELOCITY_SCALE = 1.0
REWARD_SIMILAR_TO_DEFAULT_WEIGHT = 0.05
REWARD_BASE_HEIGHT_WEIGHT = 15.0
REWARD_BASE_HEIGHT_TARGET = 0.445
REWARD_ORIENTATION_WEIGHT = 1.0
REWARD_ORIENTATION_PENALTY_MAX = 4.0
REWARD_DOF_VELOCITY_WEIGHT = 0.05
REWARD_DOF_VELOCITY_SCALE = 4.0
REWARD_JOINT_LIMIT_WEIGHT = 0.20
REWARD_JOINT_LIMIT_MARGIN = 0.10
REWARD_FORWARD_TILT_WEIGHT = 2.0
REWARD_PITCH_FREE_RAD = 0.10
REWARD_PITCH_DANGER_RAD = 0.40
REWARD_FORWARD_PITCH_RATE_WEIGHT = 0.50
REWARD_PITCH_RATE_SCALE = 1.0
REWARD_PITCH_RATE_PENALTY_MAX = 2.0
FALL_TERMINAL_PENALTY = -100.0
INIT_QPOS = np.asarray([0.05, 0.7, -1.4] * 4, dtype=np.float32)
JOINT_MIN = np.asarray(
    [-1.05, -1.57, -2.72, -1.05, -1.57, -2.72,
     -1.05, -0.52, -2.72, -1.05, -0.52, -2.72], dtype=np.float32)
JOINT_MAX = np.asarray(
    [1.05, 3.49, -0.84, 1.05, 3.49, -0.84,
     1.05, 4.54, -0.84, 1.05, 4.54, -0.84], dtype=np.float32)


def _reward_tolerance(value, lower, upper, margin):
    if lower <= value <= upper:
        return 1.0
    if margin <= 0.0:
        return 0.0
    distance = lower - value if value < lower else value - upper
    if distance >= margin:
        return 0.0
    return 1.0 - distance / margin


def _reward_body_up(quat):
    q = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-6:
        q = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    else:
        q = q / norm
    return float(1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))


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


def _reward_leg_balance(action_activity, joint_velocity):
    action_activity = np.asarray(action_activity, dtype=np.float32).reshape(4)
    joint_velocity = np.asarray(joint_velocity, dtype=np.float32).reshape(4)
    values = np.sqrt((action_activity / REWARD_LEG_ACTION_ACTIVITY_SCALE) ** 2
                     + (joint_velocity / REWARD_LEG_JOINT_VELOCITY_SCALE) ** 2)
    front = abs(values[0] - values[1]) / (values[0] + values[1] + REWARD_LEG_ACTIVITY_EPSILON)
    rear = abs(values[2] - values[3]) / (values[2] + values[3] + REWARD_LEG_ACTIVITY_EPSILON)
    return float(0.5 * (front + rear))


def _upstream_reward(state, body_velocity, return_info=False):
    """Reward copied from walk_in_the_park/sim/tasks/run.py."""
    vx = float(body_velocity[0])
    vy = float(body_velocity[1])
    vz = float(body_velocity[2])
    _, pitch = _reward_euler(state.imu_quat)
    dyaw = float(state.imu_gyro[2])
    forward = _reward_tolerance(
        np.cos(pitch) * vx, 0.5, 1.0, 1.0)
    reward = 10.0 * (forward - 0.1 * abs(dyaw))
    info = {"vx": vx, "vy": vy, "vz": vz, "pitch": float(pitch),
            "forward": float(forward), "dyaw": dyaw, "total": reward}
    return (reward, info) if return_info else reward


def _locomotion_straight_reward(state, action, previous_action, body_velocity,
                                terminated, return_info=False):
    """Exact scalar dense reward from locomotion_straight."""
    vx, vy, vz = (float(body_velocity[i]) for i in range(3))
    roll, pitch = _reward_euler(state.imu_quat)
    droll, dpitch, dyaw = (float(state.imu_gyro[i]) for i in range(3))
    body_up = _reward_body_up(state.imu_quat)

    forward_raw = _reward_tolerance(vx, REWARD_COMMAND_VX,
                                    2.0 * REWARD_COMMAND_VX,
                                    2.0 * REWARD_COMMAND_VX)
    idle_forward = _reward_tolerance(0.0, REWARD_COMMAND_VX,
                                     2.0 * REWARD_COMMAND_VX,
                                     2.0 * REWARD_COMMAND_VX)
    forward_zero_based = float(np.clip(
        (forward_raw - idle_forward) / (1.0 - idle_forward + 1e-6), 0.0, 1.0))
    if vx < 0.0:
        forward_zero_based = 0.0

    upright_gate = float(np.clip(
        (body_up - REWARD_UPRIGHT_MIN_COS)
        / (1.0 - REWARD_UPRIGHT_MIN_COS + 1e-6), 0.0, 1.0
    ) ** REWARD_UPRIGHT_EXPONENT)
    x_error = (REWARD_COMMAND_VX - vx) ** 2
    x_tracking_raw = float(np.exp(-x_error / REWARD_TRACKING_SIGMA))
    x_tracking_idle = float(np.exp(-(REWARD_COMMAND_VX ** 2) / REWARD_TRACKING_SIGMA))
    x_tracking_zero_based = float(np.clip(
        (x_tracking_raw - x_tracking_idle) / (1.0 - x_tracking_idle + 1e-6), 0.0, 1.0))
    if vx < 0.0:
        x_tracking_zero_based = 0.0
    angular_tracking = float(np.exp(-(dyaw ** 2) / REWARD_TRACKING_SIGMA))
    yaw_tracking_penalty = float(np.clip(1.0 - angular_tracking, 0.0, 1.0))

    action_delta = np.asarray(action, dtype=np.float32) - np.asarray(previous_action, dtype=np.float32)
    action_rate_penalty = float(np.minimum(
        np.mean(np.square(action_delta)) / (REWARD_ACTION_RATE_SCALE ** 2),
        REWARD_ACTION_RATE_PENALTY_MAX))
    action_magnitude_penalty = float(np.minimum(
        np.mean(np.square(action)) / (REWARD_ACTION_MAGNITUDE_SCALE ** 2),
        REWARD_ACTION_MAGNITUDE_PENALTY_MAX))
    roll_pitch_penalty = float(np.clip(
        (droll * droll + dpitch * dpitch) / (REWARD_ANGULAR_RATE_SCALE ** 2), 0.0, 1.0))
    lateral_penalty = float(np.minimum(
        vy * vy / (REWARD_LATERAL_VELOCITY_SCALE ** 2), 1.0))
    vertical_penalty = float(np.minimum(
        vz * vz / (REWARD_VERTICAL_VELOCITY_SCALE ** 2),
        REWARD_VERTICAL_VELOCITY_PENALTY_MAX))
    leg_delta_rms = np.asarray([
        np.sqrt(np.mean(np.square(action_delta[3 * i:3 * i + 3]))) for i in range(4)
    ], dtype=np.float32)
    leg_velocity_rms = np.asarray([
        np.sqrt(np.mean(np.square(state.joint_dq[3 * i:3 * i + 3]))) for i in range(4)
    ], dtype=np.float32)
    leg_balance = _reward_leg_balance(leg_delta_rms, leg_velocity_rms)
    if not (vx > REWARD_LEG_BALANCE_SPEED_GATE or x_tracking_zero_based > 0.01):
        leg_balance = 0.0

    pose_penalty = float(np.sum(np.abs(state.joint_q - INIT_QPOS)))
    base_height = float(state.position[2])
    base_height_penalty = (base_height - REWARD_BASE_HEIGHT_TARGET) ** 2
    orientation_penalty = float(np.clip(
        (1.0 - body_up) / (1.0 - REWARD_UPRIGHT_MIN_COS + 1e-6),
        0.0, REWARD_ORIENTATION_PENALTY_MAX))
    forward_tilt_penalty = float(np.clip(
        (abs(pitch) - REWARD_PITCH_FREE_RAD)
        / max(REWARD_PITCH_DANGER_RAD - REWARD_PITCH_FREE_RAD, 1e-6),
        0.0, 1.0) ** 2)
    forward_pitch_rate_penalty = float(np.clip(
        abs(dpitch) / REWARD_PITCH_RATE_SCALE,
        0.0, REWARD_PITCH_RATE_PENALTY_MAX) ** 2)
    dof_velocity_penalty = float(np.clip(
        np.mean(np.square(state.joint_dq)) / (REWARD_DOF_VELOCITY_SCALE ** 2),
        0.0, 1.0))
    joint_width = np.maximum(JOINT_MAX - JOINT_MIN, 1e-6)
    limit_margin = REWARD_JOINT_LIMIT_MARGIN * joint_width
    near_lower = np.clip((JOINT_MIN + limit_margin - state.joint_q) / limit_margin, 0.0, 1.0)
    near_upper = np.clip((state.joint_q - (JOINT_MAX - limit_margin)) / limit_margin, 0.0, 1.0)
    joint_limit_penalty = float(np.mean(np.maximum(near_lower, near_upper)))

    reward_terms = {
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "position_z": float(state.position[2]),
        "body_up": body_up,
        "x_tracking_zero_based": x_tracking_zero_based,
        "upright_gate": upright_gate,
        "yaw_tracking_penalty": yaw_tracking_penalty,
        "base_height_penalty": float(base_height_penalty),
        "pose_penalty": pose_penalty,
        "orientation_penalty": orientation_penalty,
        "dof_velocity_penalty": dof_velocity_penalty,
        "joint_limit_penalty": joint_limit_penalty,
        "action_rate_penalty": action_rate_penalty,
        "action_magnitude_penalty": action_magnitude_penalty,
        "dense_total": 0.0,
    }
    dense_total = (
        REWARD_TRACKING_LIN_VEL_WEIGHT * x_tracking_zero_based * upright_gate
        - REWARD_TRACKING_ANG_VEL_WEIGHT * yaw_tracking_penalty
        - REWARD_ROLL_PITCH_RATE_WEIGHT * roll_pitch_penalty
        - REWARD_LATERAL_VELOCITY_WEIGHT * lateral_penalty
        - REWARD_VERTICAL_VELOCITY_WEIGHT * vertical_penalty
        - REWARD_ACTION_RATE_WEIGHT * action_rate_penalty
        - REWARD_ACTION_MAGNITUDE_WEIGHT * action_magnitude_penalty
        - REWARD_SIMILAR_TO_DEFAULT_WEIGHT * pose_penalty
        - REWARD_BASE_HEIGHT_WEIGHT * base_height_penalty
        - REWARD_LEG_ACTIVITY_BALANCE_WEIGHT * leg_balance
        - REWARD_ORIENTATION_WEIGHT * orientation_penalty
        - REWARD_DOF_VELOCITY_WEIGHT * dof_velocity_penalty
        - REWARD_JOINT_LIMIT_WEIGHT * joint_limit_penalty
        - REWARD_FORWARD_TILT_WEIGHT * forward_tilt_penalty
        - REWARD_FORWARD_PITCH_RATE_WEIGHT * forward_pitch_rate_penalty)
    reward = float(dense_total + (FALL_TERMINAL_PENALTY if terminated else 0.0))
    reward_terms["dense_total"] = float(dense_total)
    reward_terms["terminal_penalty"] = float(FALL_TERMINAL_PENALTY if terminated else 0.0)
    reward_terms["total"] = reward
    return (reward, reward_terms) if return_info else reward


@dataclass(frozen=True)
class Go2State:
    policy_sequence: int
    applied_action_id: int
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
    i = 5  # SOF, phase, policy sequence, applied action id, timestamp
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
        policy_sequence=int(values[2]), applied_action_id=int(values[3]),
        timestamp=float(values[4]), joint_q=joint_q, joint_dq=joint_dq,
        imu_quat=quat, imu_gyro=gyro, imu_accel=accel, velocity=velocity,
        position=position, q_target=target, phase=int(values[1]))


class Go2SyncEnv:
    """Gym-like real Go2 adapter with the upstream 20 Hz step semantics."""

    def __init__(self, *, policy_socket: str = "/tmp/go2_policy.sock",
                 state_socket: str = "/tmp/go2_policy.sock.state",
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
        self._previous_action = np.zeros(12, dtype=np.float32)
        # Upstream's A1 observable exposes the previous absolute qpos action.
        self._previous_qtarget = self.init_qpos.copy()
        self.observation_space_shape = (46,)
        self.action_space_shape = (12,)

    def close(self):
        self._rx.close(); self._tx.close()
        try: Path(self.state_socket).unlink()
        except FileNotFoundError: pass

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
        # A normal episode timeout is not a physical reset.  Re-requesting
        # stand-up while the robot is already in POLICY needlessly drives it
        # through pose_1 -> pose_2 and can make the feet slide.  Only request
        # stand-up on the first reset or after the controller has left POLICY.
        request_standup = (
            self._last_state is None or self._last_state.phase != PHASE_POLICY)
        if request_standup:
            self._send(np.zeros(12, np.float32), FLAG_STAND_UP)
        state = self._recv()
        while state.phase != PHASE_POLICY:
            state = self._recv()
        # Stand-up is an environment lifecycle phase, not a policy interval.
        self._last_send_time = None
        self.action_intervals_ms.clear()
        self._previous_action.fill(0.0)
        self._previous_qtarget = self.init_qpos.copy()
        self._steps = 0
        return self.observation(state), {"policy_sequence": state.policy_sequence}

    def step(self, action):
        action = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        action_id, q_target = self._send(action)
        state = self._recv_for_action(action_id)
        if state.phase == PHASE_AWAIT_STATE:
            raise RuntimeError("controller is still awaiting a valid state")
        if state.phase not in (PHASE_RECOVER, PHASE_STAND_UP, PHASE_POLICY):
            raise RuntimeError(f"unknown controller phase {state.phase}")
        self._steps += 1
        body_velocity = self.body_velocity(
            state, self.sport_velocity_world_frame)
        terminated = bool(state.phase in (PHASE_RECOVER, PHASE_STAND_UP))
        reward, reward_terms = _upstream_reward(
            state, body_velocity, return_info=True)
        self._previous_action = action.copy()
        self._previous_qtarget = q_target.copy()
        truncated = self._steps >= self.max_episode_steps
        info = {"policy_sequence": state.policy_sequence,
                "applied_action_id": state.applied_action_id,
                "phase": state.phase,
                # Motions in these phases are generated by the controller,
                # not by the learner.  Their observations must never become
                # policy replay transitions.
                "recovery_motion": state.phase in (PHASE_RECOVER, PHASE_STAND_UP),
                "terminated": terminated, "truncated": truncated}
        info.update({f"reward/{key}": value
                     for key, value in reward_terms.items()})
        return self.observation(state), float(reward), terminated, truncated, info
