"""Go2 MDP termination with debounce."""

from __future__ import annotations

from dataclasses import dataclass

from common.transition import TerminationReason
from train.config import Go2Config, TrainConfig
from train.obs import is_belly_up, quat_to_euler_xyz
from train.types import RobotState


@dataclass
class TerminationState:
    tilt_violation_frames: int = 0

    def reset(self) -> None:
        self.tilt_violation_frames = 0


def update_termination_state(
    state: RobotState,
    cfg: Go2Config,
    train_cfg: TrainConfig,
    term_state: TerminationState,
    *,
    past_grace: bool,
    policy_step: bool,
    step_count: int,
) -> tuple[bool, bool, TerminationReason, dict[str, float]]:
    roll, pitch, _ = quat_to_euler_xyz(state.imu_quat)
    tilt_failed_now = (
        abs(roll) > train_cfg.termination_tilt_rad
        or abs(pitch) > train_cfg.termination_tilt_rad)

    if past_grace and policy_step and tilt_failed_now:
        term_state.tilt_violation_frames += 1
    elif (abs(roll) < train_cfg.termination_warning_tilt_rad
          and abs(pitch) < train_cfg.termination_warning_tilt_rad):
        term_state.tilt_violation_frames = 0

    belly_up = past_grace and policy_step and is_belly_up(state, cfg)
    tilt_confirmed = (
        term_state.tilt_violation_frames
        >= train_cfg.termination_confirm_frames)
    terminated = bool(belly_up or tilt_confirmed)
    truncated = bool(step_count >= train_cfg.max_episode_steps)

    if belly_up:
        reason = TerminationReason.EXCESSIVE_TILT
    elif tilt_confirmed:
        reason = TerminationReason.EXCESSIVE_TILT
    elif truncated:
        reason = TerminationReason.TIME_LIMIT
    else:
        reason = TerminationReason.NONE

    info = {
        'roll': float(roll),
        'pitch': float(pitch),
        'tilt_violation_frames': float(term_state.tilt_violation_frames),
        'tilt_failed_now': float(tilt_failed_now),
    }
    return terminated, truncated, reason, info
