"""Narrow adapter around the validated target-aligned MjLab Go2 task."""

from __future__ import annotations

from typing import Any

import torch

from safety_data.mjlab_natural_falls import MJLAB_TO_TARGET_JOINT
from safety_data.mjlab_target_alignment import (
    configure_target_aligned_go2,
    validate_target_aligned_go2,
)


TARGET_PERMUTATION = tuple(MJLAB_TO_TARGET_JOINT)


def configure_command(cfg: Any, command_vx: float) -> Any:
    """Reuse the validated 0.30 contract, then change only command vx."""
    cfg = configure_target_aligned_go2(cfg)
    validate_target_aligned_go2(cfg)
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (float(command_vx), float(command_vx))
    return cfg


def validate_command(cfg: Any, command_vx: float) -> None:
    twist = cfg.commands["twist"]
    if tuple(twist.ranges.lin_vel_x) != (float(command_vx), float(command_vx)):
        raise ValueError("PPO-SQRL command drifted")
    if tuple(twist.ranges.lin_vel_y) != (0.0, 0.0) or tuple(
            twist.ranges.ang_vel_z) != (0.0, 0.0):
        raise ValueError("PPO-SQRL non-forward command drifted")
    if "push_robot" in cfg.events or set(cfg.terminations) != {
            "time_out", "target_fall"}:
        raise ValueError("PPO-SQRL force or termination contract drifted")


def corrected_observation(environment: Any) -> torch.Tensor:
    robot = environment.scene.entities["robot"]
    permutation = torch.as_tensor(
        TARGET_PERMUTATION, dtype=torch.long, device=robot.data.joint_pos.device)
    result = torch.cat((
        robot.data.joint_pos[:, permutation],
        robot.data.joint_vel[:, permutation],
        robot.data.root_link_ang_vel_b,
        robot.data.root_link_lin_vel_b,
        robot.data.root_link_quat_w,
        robot.data.joint_pos_target[:, permutation],
    ), dim=1).to(torch.float32)
    if result.ndim != 2 or result.shape[1] != 46:
        raise RuntimeError("corrected Go2 observation is not 46D")
    return result


def initialize_history(observation: torch.Tensor, frames: int = 5) -> torch.Tensor:
    return observation[:, None, :].expand(-1, frames, -1).clone()


def advance_history(history: torch.Tensor, observation: torch.Tensor,
                    done: torch.Tensor) -> torch.Tensor:
    result = torch.roll(history, shifts=-1, dims=1)
    result[:, -1] = observation
    mask = done.to(torch.bool)
    if bool(mask.any().item()):
        result[mask] = observation[mask, None, :].expand(-1, history.shape[1], -1)
    return result


def target_order_action(action: torch.Tensor) -> torch.Tensor:
    if action.shape[-1] != 12:
        raise ValueError("policy action must end in 12 joints")
    permutation = torch.as_tensor(
        TARGET_PERMUTATION, dtype=torch.long, device=action.device)
    return action.clamp(-1.0, 1.0).index_select(-1, permutation)


def project_environment_action(action: torch.Tensor) -> torch.Tensor:
    """Apply the normalized-action projection used by the native controller."""
    if action.shape[-1] != 12:
        raise ValueError("policy action must end in 12 joints")
    return action.clamp(-1.0, 1.0)


def target_to_mjlab_action(action: torch.Tensor) -> torch.Tensor:
    if action.shape[-1] != 12:
        raise ValueError("critic action must end in 12 joints")
    permutation = torch.as_tensor(
        TARGET_PERMUTATION, dtype=torch.long, device=action.device)
    inverse = torch.argsort(permutation)
    return action.index_select(-1, inverse)


def forward_velocity(environment: Any) -> torch.Tensor:
    return environment.scene.entities["robot"].data.root_link_lin_vel_b[:, 0]


def make_environment(*, command_vx: float, environments: int, seed: int,
                     device: str = "cuda:0"):
    import mjlab.tasks  # noqa: F401
    import src.tasks  # type: ignore  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    cfg = configure_command(load_env_cfg("Unitree-Go2-Flat"), command_vx)
    cfg.seed = int(seed)
    cfg.scene.num_envs = int(environments)
    validate_command(cfg, command_vx)
    agent_cfg = load_rl_cfg("Unitree-Go2-Flat")
    agent_cfg.seed = int(seed)
    environment = ManagerBasedRlEnv(cfg=cfg, device=device)
    if bool(torch.any(environment.sim.data.xfrc_applied != 0.0).item()):
        raise RuntimeError("PPO-SQRL environment contains an external force")
    wrapped = RslRlVecEnvWrapper(
        environment, clip_actions=agent_cfg.clip_actions)
    return environment, wrapped, cfg, agent_cfg
