"""Legacy Go2 walk environment adapter.

The P2 architecture moves rollout ownership to the C++ controller and keeps this
module only as the in-process compatibility adapter for `python -m train`.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from train.action_filter import ActionFilterButter
from train.config import Go2Config, load_app_config
from train.dds import DdsConfig, StateReader
from train.ipc import PolicyClient
from train.obs import (build_observation, body_up_cos, get_run_reward_from_state,
                       get_terminal_penalty, gravity_acc_z, gravity_up_cos,
                       is_belly_up, is_fallen, is_pose_stable,
                       quat_to_euler_xyz)
from train.types import RobotState
from jaxrl.env.specs import BoxSpec

try:
    from train._debug_agent_log import debug_log as _debug_log
except ImportError:
    def _debug_log(*_a, **_k):
        pass

# Consecutive belly-up frames before mid-episode recovery (20Hz → 0.25s).
_BELLY_UP_TRIGGER_STEPS = 5


class UnstableResetError(RuntimeError):
    """Raised when standup/recovery did not produce a safe policy start state."""


def action_to_qpos(action: np.ndarray, cfg: Go2Config) -> np.ndarray:
    action = np.clip(action, -1.0, 1.0)
    return np.clip(cfg.init_qpos + action * cfg.action_offset,
                   cfg.action_joint_min, cfg.action_joint_max)


def qpos_to_action(q_target: np.ndarray, cfg: Go2Config) -> np.ndarray:
    action = (np.asarray(q_target, dtype=np.float32) - cfg.init_qpos) / np.maximum(
        cfg.action_offset, 1e-6)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


class Go2Env:

    def __init__(self,
                 dds_config: DdsConfig,
                 go2_config: Go2Config | None = None,
                 control_frequency: float = 20.0,
                 max_episode_steps: int = 400,
                 ipc_socket: str | None = None,
                 max_joint_delta: float | None = None,
                 use_action_filter: bool = True,
                 reset_grace_steps: int = 20,
                 reset_hold_steps: int = 220,
                 reset_joint_tolerance: float = 0.30,
                 recovery_stable_steps: int = 10,
                 standup_timeout_steps: int = 200,
                 abort_on_unstable_reset: bool = True,
                 seed: int = 0):
        self.cfg = go2_config or load_app_config()[0]
        self.dds_config = dds_config
        self.control_dt = 1.0 / control_frequency
        self.control_frequency = control_frequency
        self.max_episode_steps = max_episode_steps
        self.ipc_socket = ipc_socket or self.cfg.ipc_socket
        self.max_joint_delta = max_joint_delta
        self.use_action_filter = use_action_filter
        self.reset_grace_steps = reset_grace_steps
        self.reset_hold_steps = reset_hold_steps
        self.reset_joint_tolerance = reset_joint_tolerance
        self.recovery_stable_steps = recovery_stable_steps
        self.standup_timeout_steps = standup_timeout_steps
        self.abort_on_unstable_reset = abort_on_unstable_reset

        n = self.cfg.num_joints
        self.observation_space = BoxSpec(shape=(self.cfg.obs_dim,),
                                         low=-np.inf,
                                         high=np.inf)
        self.action_space = BoxSpec(shape=(n,),
                                    low=-np.ones(n, dtype=np.float32),
                                    high=np.ones(n, dtype=np.float32))

        self._state_reader: Optional[StateReader] = None
        self._policy_client: Optional[PolicyClient] = None
        self._action_filter: Optional[ActionFilterButter] = None
        if use_action_filter:
            self._action_filter = ActionFilterButter(
                num_joints=n,
                sampling_rate=control_frequency,
                highcut=self.cfg.action_filter_highcut,
            )
        self._prev_requested_action = np.zeros(n, dtype=np.float32)
        self._prev_executed_action = np.zeros(n, dtype=np.float32)
        self._step_count = 0
        self._steps_since_reset = 0
        self._standup_active = False
        self._standup_with_recovery = False
        self._standup_stable_count = 0
        self._standup_step_count = 0
        self._belly_up_count = 0
        self._not_belly_up_count = 0
        self._last_policy_send_time: float | None = None
        self._np_random = np.random.RandomState(seed)

    def seed(self, seed: Optional[int] = None) -> int:
        if seed is None:
            seed = int(self._np_random.randint(0, 2**31 - 1))
        self._np_random.seed(seed)
        return seed

    def _ensure_connected(self) -> None:
        if self._state_reader is None:
            self._state_reader = StateReader(
                sport_velocity_world_frame=self.cfg.sport_velocity_world_frame)
            self._state_reader.init_dds(self.dds_config.domain_id,
                                        self.dds_config.interface)
            self._state_reader.wait_for_state(timeout=10.0)
            time.sleep(0.5)
            self._state_reader.require_fresh_sport_state(
                self.cfg.sport_state_max_age_ms / 1000.0)
        if self._policy_client is None:
            self._policy_client = PolicyClient(self.ipc_socket)
            self._policy_client.connect()

    def _init_action_filter(self) -> None:
        if self._action_filter is not None:
            self._action_filter.reset()
            self._action_filter.init_history(self.cfg.init_qpos)

    def _send_q_target(self, q_target: np.ndarray) -> np.ndarray:
        assert self._policy_client is not None
        q = np.asarray(q_target, dtype=np.float32)
        if self._action_filter is not None:
            q = self._action_filter.filter(q)
        self._policy_client.send_target(q)
        return q.copy()

    def _send_standup_request(self, *, with_recovery: bool) -> None:
        assert self._policy_client is not None
        self._policy_client.send_standup(with_recovery=with_recovery,
                                         q_target=self.cfg.init_qpos)

    def _wait_standup(self, *, with_recovery: bool) -> RobotState:
        """Block until controller standup_fsm finishes (recovery→standup or standup only)."""
        assert self._state_reader is not None
        state = self._state_reader.get_state()

        def wait_phase(request_recovery: bool,
                       initial_state: RobotState) -> tuple[RobotState, bool]:
            stable_count = 0
            phase_state = initial_state
            for _ in range(self.reset_hold_steps):
                self._send_standup_request(
                    with_recovery=request_recovery)
                time.sleep(self.control_dt)
                phase_state = self._state_reader.get_state()
                if is_pose_stable(
                        phase_state,
                        self.cfg,
                        joint_tolerance=self.reset_joint_tolerance):
                    stable_count += 1
                    if stable_count >= self.recovery_stable_steps:
                        return phase_state, True
                else:
                    stable_count = 0
            return phase_state, False

        state, reset_stable = wait_phase(with_recovery, state)

        # A tilted fall can cross the belly-up threshold only after the terminal
        # frame was captured. A standup-only request cannot recover that pose,
        # so escalate once instead of starting policy or aborting immediately.
        escalated_to_recovery = False
        if not reset_stable and not with_recovery and is_belly_up(
                state, self.cfg):
            escalated_to_recovery = True
            print('[env] standup ended belly-up; escalating to '
                  'recovery→standup', flush=True)
            state, reset_stable = wait_phase(True, state)

        if not reset_stable and self.abort_on_unstable_reset:
            roll, pitch, _ = quat_to_euler_xyz(state.imu_quat)
            joint_err = float(np.linalg.norm(state.joint_q -
                                             self.cfg.init_qpos))
            raise UnstableResetError(
                'Standup/recovery did not reach a stable policy start state '
                f'within {self.reset_hold_steps} steps: '
                f'roll={roll:.3f} pitch={pitch:.3f} '
                f'joint_error={joint_err:.3f} '
                f'belly_up={is_belly_up(state, self.cfg)} '
                f'recovery_requested={with_recovery} '
                f'recovery_escalated={escalated_to_recovery}. '
                'Policy rollout was aborted so unstable transitions cannot '
                'enter replay.')
        # Signal controller to leave standup FSM and accept policy targets.
        self._policy_client.send_target(state.joint_q)
        return state

    def reset(self, *, standup: bool = False,
              with_recovery: bool = False,
              grace_period: bool = True,
              preserve_policy_state: bool = False) -> np.ndarray:
        self._ensure_connected()
        assert self._policy_client is not None

        self._step_count = 0
        # A time-limit truncation is only a logical episode boundary: physics
        # is not reset, so granting a new fall-detection grace period would
        # admit up to reset_grace_steps unsafe transitions into replay.
        self._steps_since_reset = (
            0 if grace_period else self.reset_grace_steps + 1)
        self._standup_active = False
        self._standup_with_recovery = False
        self._standup_stable_count = 0
        self._standup_step_count = 0
        self._belly_up_count = 0
        self._not_belly_up_count = 0
        self._last_policy_send_time = None
        if not preserve_policy_state:
            self._prev_requested_action = np.zeros(self.cfg.num_joints,
                                                   dtype=np.float32)
            self._prev_executed_action = np.zeros(self.cfg.num_joints,
                                                  dtype=np.float32)
            self._init_action_filter()

        if standup:
            state = self._wait_standup(with_recovery=with_recovery)
        else:
            # No forced init_qpos — resume policy from current pose (e.g. after
            # truncate or training start).
            state = self._state_reader.get_state()
            if self._action_filter is not None and not preserve_policy_state:
                self._action_filter.init_history(state.joint_q)

        return build_observation(state, self._prev_requested_action, self.cfg,
                                 self._prev_executed_action)

    def _resume_policy(self, state: RobotState) -> None:
        """Leave standup mode and tell controller to track policy targets again."""
        assert self._policy_client is not None
        self._standup_active = False
        self._standup_with_recovery = False
        self._standup_stable_count = 0
        self._standup_step_count = 0
        self._belly_up_count = 0
        self._not_belly_up_count = 0
        self._policy_client.send_target(state.joint_q)
        if self._action_filter is not None:
            self._action_filter.init_history(state.joint_q)

    def _needs_standup(self, state: RobotState) -> bool:
        if self._standup_active:
            return True
        # Mid-episode: only confirmed belly-up → recovery (never standup-only).
        if is_belly_up(state, self.cfg):
            self._belly_up_count += 1
        else:
            self._belly_up_count = 0
        return self._belly_up_count >= _BELLY_UP_TRIGGER_STEPS

    def _tick_standup(self, state: RobotState) -> None:
        belly_up = is_belly_up(state, self.cfg)
        if not belly_up:
            self._not_belly_up_count += 1
            if self._not_belly_up_count >= self.recovery_stable_steps:
                # Flipped back or false trigger — resume policy, no standup.
                # region agent log
                _debug_log(
                    'G',
                    'env.py:_tick_standup',
                    'abort standup (not belly-up)',
                    {
                        'not_belly_up_count': self._not_belly_up_count,
                        'body_up_cos': float(body_up_cos(state.imu_quat)),
                    },
                    run_id='post-fix',
                )
                # endregion
                self._resume_policy(state)
            return
        self._not_belly_up_count = 0
        if is_pose_stable(state, self.cfg):
            self._standup_stable_count += 1
            if self._standup_stable_count >= self.recovery_stable_steps:
                self._resume_policy(state)
        else:
            self._standup_stable_count = 0

    def step(
        self, action: np.ndarray, during_hold=None
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self._ensure_connected()
        assert self._policy_client is not None
        assert self._state_reader is not None

        state = self._state_reader.get_state()
        executed_q_target = state.joint_q.copy()
        action_interval_ms = float('nan')

        if self._standup_active:
            policy_step = False
            policy_action = np.zeros(self.cfg.num_joints, dtype=np.float32)
            self._standup_step_count += 1
            self._send_standup_request(with_recovery=self._standup_with_recovery)
        else:
            policy_step = True
            send_time = time.perf_counter()
            if self._last_policy_send_time is not None:
                action_interval_ms = (
                    send_time - self._last_policy_send_time) * 1000.0
            self._last_policy_send_time = send_time
            policy_action = np.clip(np.asarray(action, dtype=np.float32),
                                    -1.0, 1.0)
            q_desired = action_to_qpos(policy_action, self.cfg)
            if self.max_joint_delta is not None:
                delta = np.clip(q_desired - state.joint_q,
                                -self.max_joint_delta,
                                self.max_joint_delta)
                q_send = state.joint_q + delta
            else:
                q_send = q_desired
            executed_q_target = self._send_q_target(q_send)

        hold_start = time.perf_counter()
        if policy_step and during_hold is not None:
            during_hold()
        hold_elapsed = time.perf_counter() - hold_start
        hold_overrun_s = max(0.0, hold_elapsed - self.control_dt)
        time.sleep(max(0.0, self.control_dt - hold_elapsed))
        self._state_reader.require_fresh_sport_state(
            self.cfg.sport_state_max_age_ms / 1000.0)
        state = self._state_reader.get_state()
        belly_up = is_belly_up(state, self.cfg)
        if self._standup_active:
            self._tick_standup(state)
        executed_action = (qpos_to_action(executed_q_target, self.cfg)
                           if policy_step else policy_action)
        self._prev_requested_action = policy_action
        self._prev_executed_action = executed_action
        obs = build_observation(state, policy_action, self.cfg,
                                executed_action)
        reward, reward_info = get_run_reward_from_state(state, self.cfg)
        self._steps_since_reset += 1
        if policy_step:
            self._step_count += 1

        past_grace = self._steps_since_reset > self.reset_grace_steps
        fallen = past_grace and is_fallen(state, self.cfg)
        standup_timed_out = False
        if (self._standup_active
                and self._standup_step_count >= self.standup_timeout_steps):
            # Do not truncate the episode — abort and resume policy.
            # region agent log
            _debug_log(
                'G',
                'env.py:step',
                'standup timeout abort',
                {
                    'standup_steps': self._standup_step_count,
                    'still_belly_up': is_belly_up(state, self.cfg),
                    'body_up_cos': float(body_up_cos(state.imu_quat)),
                },
                run_id='post-fix',
            )
            # endregion
            still_belly_up = is_belly_up(state, self.cfg)
            self._resume_policy(state)
            standup_timed_out = still_belly_up
        if policy_step and past_grace and belly_up:
            self._standup_active = True
            self._standup_with_recovery = True
            self._standup_stable_count = 0
            self._standup_step_count = 0
            self._not_belly_up_count = 0
            # region agent log
            _debug_log(
                'G',
                'env.py:step',
                'terminal before recovery',
                {
                    'body_up_cos': float(body_up_cos(state.imu_quat)),
                    'gravity_up_cos': float(gravity_up_cos(state)),
                    'acc_z': float(gravity_acc_z(state)),
                },
                run_id='post-fix',
            )
            # endregion

        terminated = fallen or standup_timed_out or (policy_step and past_grace
                                                    and belly_up)
        truncated = self._step_count >= self.max_episode_steps
        done = terminated or truncated
        terminal_penalty = get_terminal_penalty(
            terminated=terminated, cfg=self.cfg)
        reward += terminal_penalty
        reward_info['terminal_penalty'] = float(terminal_penalty)

        return obs, reward, done, {
            'is_fallen': fallen,
            'is_belly_up': belly_up,
            'is_recovering': self._standup_active,
            'standup_with_recovery': self._standup_with_recovery,
            'standup_timed_out': standup_timed_out,
            'policy_step': policy_step,
            'terminated': terminated,
            'truncated': truncated,
            'projected_action': policy_action.copy(),
            'executed_q_target': executed_q_target.copy(),
            'executed_q_target_norm': float(np.linalg.norm(executed_q_target)),
            'intervention_mask': bool(
                policy_step and np.linalg.norm(executed_action - policy_action)
                > 1e-5),
            'sport_state_age_ms': float(
                self._state_reader.sport_state_age() * 1000.0),
            'action_interval_ms': action_interval_ms,
            'action_frequency_hz': (
                1000.0 / action_interval_ms
                if np.isfinite(action_interval_ms) and action_interval_ms > 0.0
                else float('nan')),
            'control_hold_overrun_ms': hold_overrun_s * 1000.0,
            'world_x': float(state.world_position[0]),
            'world_y': float(state.world_position[1]),
            'world_z': float(state.world_position[2]),
            'step_count': self._step_count,
            'standup_step_count': self._standup_step_count,
            **reward_info,
        }

    def close(self) -> None:
        if self._policy_client is not None:
            self._policy_client.close()
            self._policy_client = None
