"""Legacy environment adapter used by the in-process collector path.

The long-term P2 runtime receives ObservationFrame/TransitionFrame objects from
the C++ rollout executor.  This adapter keeps the current `python -m train`
entrypoint usable while the controller protocol is migrated.
"""

from __future__ import annotations

from train.config import TrainConfig
from train.dds import DdsConfig
from train.env import Go2Env


def build_legacy_env(robot_cfg, train_cfg: TrainConfig, seed: int) -> Go2Env:
    return Go2Env(
        dds_config=DdsConfig(domain_id=robot_cfg.domain_id,
                             interface=robot_cfg.interface),
        go2_config=robot_cfg,
        control_frequency=train_cfg.control_frequency,
        max_episode_steps=train_cfg.max_episode_steps,
        ipc_socket=robot_cfg.ipc_socket,
        max_joint_delta=train_cfg.max_joint_delta,
        use_action_filter=train_cfg.use_action_filter,
        reset_grace_steps=train_cfg.reset_grace_steps,
        reset_hold_steps=train_cfg.reset_hold_steps,
        reset_joint_tolerance=train_cfg.reset_joint_tolerance,
        recovery_stable_steps=train_cfg.recovery_stable_steps,
        standup_timeout_steps=train_cfg.standup_timeout_steps,
        abort_on_unstable_reset=train_cfg.abort_on_unstable_reset,
        train_cfg=train_cfg,
        seed=seed,
    )
