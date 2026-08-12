#!/usr/bin/env python3
"""Collect stochastic frozen-PPO transitions for the SQRL master dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from safety_data.mjlab_natural_falls import (
    MJLAB_TO_TARGET_JOINT,
    target_order_action_and_qtarget,
)
from safety_data.mjlab_target_alignment import (
    configure_target_aligned_go2,
    target_alignment_manifest,
    validate_target_aligned_go2,
)
from safety_data.ppo_sqrl_master import TransitionShardWriter, sha256_file
from safety_data.ppo_sqrl_protocol import (
    load_ppo_sqrl_protocol,
    ppo_sqrl_protocol_sha256,
)


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _git_clean(path: Path) -> bool:
    return not bool(subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=path, check=True, capture_output=True,
    ).stdout)


def _corrected_observation(env: object) -> torch.Tensor:
    robot = env.scene.entities["robot"]  # type: ignore[attr-defined]
    permutation = torch.as_tensor(
        MJLAB_TO_TARGET_JOINT, dtype=torch.long, device=robot.data.joint_pos.device)
    return torch.cat((
        robot.data.joint_pos[:, permutation],
        robot.data.joint_vel[:, permutation],
        robot.data.root_link_ang_vel_b,
        robot.data.root_link_lin_vel_b,
        robot.data.root_link_quat_w,
        robot.data.joint_pos_target[:, permutation],
    ), dim=1).to(torch.float32)


def _mix_u64(seed: int, env: torch.Tensor, episode: torch.Tensor,
             step: int) -> np.ndarray:
    # Use Python integer arithmetic to keep unsigned wraparound explicit.
    values = []
    for environment, episode_id in zip(
            env.detach().cpu().tolist(), episode.detach().cpu().tolist(), strict=True):
        payload = f"{seed}:{environment}:{episode_id}:{step}".encode("ascii")
        values.append(int.from_bytes(hashlib.sha256(payload).digest()[:8], "little"))
    return np.asarray(values, dtype=np.uint64)


def _flatten(chunks: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(chunks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ppo-seed", type=int, choices=(137, 138), required=True)
    parser.add_argument("--stage", choices=("early", "boundary", "mature"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=2000)
    parser.add_argument("--aggregate-transitions", type=int, default=1_000_000)
    parser.add_argument("--shard-vector-steps", type=int, default=25)
    args = parser.parse_args()

    protocol = load_ppo_sqrl_protocol()
    locked = protocol["ppo_master_dataset"]
    expected = locked["stages"][args.stage]
    if args.envs != protocol["task"]["parallel_environments"] or (
            args.aggregate_transitions != locked["transitions_per_seed_stage"]):
        raise ValueError("collector dimensions differ from the locked protocol")
    if args.aggregate_transitions % args.envs or args.shard_vector_steps <= 0:
        raise ValueError("aggregate transitions must be divisible by environments")
    if args.checkpoint.name != expected["checkpoint"] or not args.checkpoint.is_file():
        raise ValueError("checkpoint differs from the preregistered collector stage")
    if not _git_clean(REPOSITORY_ROOT):
        raise RuntimeError("formal PPO transition collection requires a clean worktree")

    import mjlab.tasks  # noqa: F401
    import src.tasks  # type: ignore  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    cfg = configure_target_aligned_go2(load_env_cfg("Unitree-Go2-Flat"))
    agent_cfg = load_rl_cfg("Unitree-Go2-Flat")
    cfg.seed = args.ppo_seed
    cfg.scene.num_envs = args.envs
    validate_target_aligned_go2(cfg)
    agent_cfg.seed = args.ppo_seed
    agent_cfg.upload_model = False
    agent_cfg.logger = "tensorboard"
    environment = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    if "push_robot" in environment.cfg.events or bool(torch.any(
            environment.sim.data.xfrc_applied != 0.0).item()):
        raise RuntimeError("PPO SQRL collector contains an external force")
    wrapped = RslRlVecEnvWrapper(environment, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device="cuda:0")
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True,
                map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    if policy.distribution is None:
        raise RuntimeError("collector policy has no stochastic distribution")

    writer = TransitionShardWriter(args.output)
    vector_steps = args.aggregate_transitions // args.envs
    environment_ids = torch.arange(args.envs, device="cuda:0")
    episode_ids = torch.zeros(args.envs, dtype=torch.long, device="cuda:0")
    policy_observation = wrapped.get_observations()
    corrected = _corrected_observation(environment)
    history = corrected[:, None, :].expand(-1, 5, -1).clone()
    previous_critic_action = corrected[:, 34:46].clone()
    pending: dict[str, list[np.ndarray]] = {}
    started = time.perf_counter()

    def append(name: str, value: torch.Tensor | np.ndarray) -> None:
        array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
        pending.setdefault(name, []).append(np.asarray(array))

    with torch.inference_mode():
        for vector_step in range(vector_steps):
            raw_action = policy(policy_observation, stochastic_output=True)
            action_mean = policy.output_mean
            action_std = policy.output_std
            log_probability = policy.get_output_log_prob(raw_action)
            entropy = policy.output_entropy
            pre_projection = raw_action
            requested_for_env = raw_action
            if agent_cfg.clip_actions is not None:
                requested_for_env = torch.clamp(
                    requested_for_env, -agent_cfg.clip_actions, agent_cfg.clip_actions)
            robot = environment.scene.entities["robot"]
            action_term = environment.action_manager.get_term("joint_pos")
            encoder_bias = robot.data.encoder_bias[:, action_term.target_ids]
            action_requested, absolute_target = target_order_action_and_qtarget(
                requested_for_env,
                scale=action_term.scale,
                offset=action_term.offset,
                encoder_bias=encoder_bias,
            )
            critic_action = absolute_target
            saturation = pre_projection.abs() >= 1.0
            action_change = torch.linalg.vector_norm(
                critic_action - previous_critic_action, dim=1)
            current_episode = episode_ids.clone()
            current_policy_actor = policy_observation["actor"].clone()
            next_policy_observation, _, dones, extras = wrapped.step(raw_action)
            timeouts = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))
            done = dones.to(torch.bool)
            truncated = done & timeouts.to(torch.bool)
            terminated = done & ~truncated
            if bool(torch.any(environment.sim.data.xfrc_applied != 0.0).item()):
                raise RuntimeError("runtime external force became non-zero")
            next_corrected = _corrected_observation(environment)
            next_history = torch.roll(history, shifts=-1, dims=1)
            next_history[:, -1] = next_corrected
            if bool(done.any().item()):
                next_history[done] = next_corrected[done, None, :].expand(-1, 5, -1)
            next_encoder_bias = robot.data.encoder_bias[:, action_term.target_ids]
            next_encoder_bias = next_encoder_bias[:, torch.as_tensor(
                MJLAB_TO_TARGET_JOINT, device="cuda:0")].to(torch.float32)

            append("observation_history_t", history)
            append("critic_action", critic_action)
            append("next_observation_history", next_history)
            append("c_t_plus_1", terminated)
            append("terminated", terminated)
            append("truncated", truncated)
            append("action_requested", action_requested)
            append("action_pre_projection", pre_projection[:, torch.as_tensor(
                MJLAB_TO_TARGET_JOINT, device="cuda:0")])
            append("absolute_q_target", absolute_target)
            append("action_log_probability", log_probability)
            append("policy_entropy", entropy)
            append("action_std", action_std[:, torch.as_tensor(
                MJLAB_TO_TARGET_JOINT, device="cuda:0")])
            append("action_saturation", saturation[:, torch.as_tensor(
                MJLAB_TO_TARGET_JOINT, device="cuda:0")])
            append("action_change_rate", action_change)
            append("ppo_seed", np.full(args.envs, args.ppo_seed, np.int32))
            append("collector_stage", np.full(args.envs, args.stage, "U8"))
            append("collector_checkpoint", np.full(
                args.envs, args.checkpoint.name, "U32"))
            append("env_id", environment_ids)
            append("episode_id", current_episode)
            append("vector_step", np.full(args.envs, vector_step, np.int64))
            append("randomization_identity", _mix_u64(
                args.ppo_seed, environment_ids, current_episode, 0))
            append("rng_identity", _mix_u64(
                args.ppo_seed, environment_ids, current_episode, vector_step + 1))
            append("policy_observation_t", current_policy_actor)
            append("next_policy_observation", next_policy_observation["actor"])
            append("next_action_encoder_bias", next_encoder_bias)

            history = next_history
            policy_observation = next_policy_observation
            previous_critic_action = critic_action
            episode_ids += done.to(torch.long)
            if len(pending["c_t_plus_1"]) == args.shard_vector_steps or (
                    vector_step + 1 == vector_steps):
                writer.write({name: _flatten(values) for name, values in pending.items()})
                pending.clear()

    elapsed = time.perf_counter() - started
    manifest = writer.close({
        "protocol_sha256": ppo_sqrl_protocol_sha256(),
        "generator_commit": _git_head(REPOSITORY_ROOT),
        "generator_worktree_clean": True,
        "ppo_seed": args.ppo_seed,
        "collector_stage": args.stage,
        "collector_training_transitions": expected["aggregate_training_transitions"],
        "collector_checkpoint": str(args.checkpoint.resolve()),
        "collector_checkpoint_sha256": sha256_file(args.checkpoint),
        "environments": args.envs,
        "vector_steps": vector_steps,
        "aggregate_environment_transitions": args.aggregate_transitions,
        "elapsed_seconds": elapsed,
        "target_alignment": target_alignment_manifest(),
        "external_force_verified_zero": True,
        "recovery_or_get_up_used": False,
    })
    print(json.dumps({
        "manifest": str(manifest.resolve()),
        "transitions": writer.transition_count,
        "elapsed_seconds": elapsed,
    }, indent=2))
    environment.close()


if __name__ == "__main__":
    main()
