#!/usr/bin/env python3
"""Paired exact-snapshot evaluation of nominal versus Q_safe-selected actions."""

from __future__ import annotations

import argparse
import json

import gymnasium as gym
import numpy as np
import torch

from rl.agents import create_agent
from safety_data.paths import (
    assert_development_path,
    assert_safe_evidence_output,
    require_v3_audit_consumed_or_safe_input,
)
from train.config import load_app_config
from train.mujoco_snapshot_env import MujocoSnapshotEnv


def _spaces(robot_cfg):
    observation_space = gym.spaces.Box(
        -100.0, 100.0, (robot_cfg.obs_dim,), dtype=np.float32)
    action_space = gym.spaces.Box(
        -1.0, 1.0, (robot_cfg.num_joints,), dtype=np.float32)
    return observation_space, action_space


@torch.no_grad()
def _actor_actions(agent, observation, count: int, *, sample: bool,
                   seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    obs = torch.as_tensor(
        np.repeat(observation[None, :], count, axis=0),
        dtype=torch.float32, device=agent._device)
    actions, _ = agent._actor(
        observations=agent._actor_observations(obs),
        training=False, sample=sample)
    return actions.cpu().numpy().astype(np.float32)


@torch.no_grad()
def _risks(agent, observation, actions) -> np.ndarray:
    obs = torch.as_tensor(
        np.repeat(observation[None, :], len(actions), axis=0),
        dtype=torch.float32, device=agent._device)
    act = torch.as_tensor(
        actions, dtype=torch.float32, device=agent._device)
    return agent._risks(obs, act).cpu().numpy().reshape(-1)


def _select(agent, observation, nominal, *, seed: int):
    """Run the exact online selector, including all configured gates."""
    torch.manual_seed(seed)
    interaction_step = (
        int(agent.cfg.safety_activation_step)
        + int(agent.cfg.safety_masking_ramp_steps)
        + 1
    )
    selected = np.asarray(agent.filter_nominal_action(
        interaction_step,
        {"next_observation": observation[None, :]},
        nominal,
        training=True,
    )[0], dtype=np.float32)
    metrics = agent.get_metrics()
    return {
        "action": selected,
        "replaced": bool(metrics.get("safety/replaced", 0.0)),
        "no_safe": bool(metrics.get("safety/no_safe_candidate", 0.0)),
        "nominal_risk": float(metrics.get("safety/nominal_risk", np.nan)),
        "selected_risk": float(metrics.get("safety/selected_risk", np.nan)),
        "nominal_reward_q": float(
            metrics.get("safety/nominal_reward_q", np.nan)),
        "selected_reward_q": float(
            metrics.get("safety/selected_reward_q", np.nan)),
    }


def _branch(env, agent, snapshot, first_action, horizon):
    env.restore(snapshot)
    action = np.asarray(first_action, dtype=np.float32)
    first_application = None
    failure_step = None
    near_failure = False
    max_tilt = 0.0
    min_height = np.inf
    for step in range(1, horizon + 1):
        step_result = env.step(action)
        measurement = step_result.measurement
        if first_application is None:
            first_application = step_result.application
        near_failure |= measurement.near_failure
        max_tilt = max(max_tilt, measurement.tilt_rad)
        min_height = min(min_height, measurement.height_m)
        if measurement.failure:
            failure_step = step
            break
        observation = env.record_observation()[-1]
        action = _actor_actions(
            agent, observation, 1, sample=False,
            seed=0)[0]
    assert first_application is not None
    return {
        "failure": failure_step is not None,
        "failure_step": failure_step,
        "near_failure": bool(near_failure),
        "max_tilt_rad": float(max_tilt),
        "min_height_m": float(min_height),
        "first_action_requested": first_application.action_requested.tolist(),
        "first_action_executed": first_application.action_executed.tolist(),
        "first_action_q_target": first_application.action_q_target.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/go2_50hz_safe.yaml")
    parser.add_argument(
        "--checkpoint",
        default=("saved/experiments/p16_030_qsafe_masking/"
                 "step_000000015000/agent"))
    parser.add_argument(
        "--model",
        default=("/home/xyz/code/unitree_mujoco/"
                 "unitree_robots/go2/scene_empty.xml"))
    parser.add_argument("--natural-steps", type=int, default=12000)
    parser.add_argument("--max-pairs", type=int, default=500)
    parser.add_argument("--episode-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--disturbance-interval", type=int, default=25,
        help="Apply the same pre-snapshot base-velocity disturbance every N steps.")
    parser.add_argument("--linear-impulse-std", type=float, default=0.35)
    parser.add_argument("--angular-impulse-std", type=float, default=1.25)
    parser.add_argument("--output",
                        default="saved/experiments/p16_snapshot_pairs.json")
    args = parser.parse_args()
    config_path = assert_development_path(
        require_v3_audit_consumed_or_safe_input(args.config))
    checkpoint_path = assert_development_path(
        require_v3_audit_consumed_or_safe_input(args.checkpoint))
    model_path = assert_development_path(
        require_v3_audit_consumed_or_safe_input(args.model))
    output_path = assert_development_path(
        assert_safe_evidence_output(args.output))

    robot_cfg, train_cfg, agent_cfg = load_app_config(
        config_path, agent="safe_droq")
    agent_cfg.device_type = (
        "cuda" if torch.cuda.is_available() else "cpu")
    agent_cfg.buffer_device_type = agent_cfg.device_type
    observation_space, action_space = _spaces(robot_cfg)
    agent = create_agent(
        observation_space, action_space, {}, agent_cfg)
    agent.load(str(checkpoint_path))
    # A full agent checkpoint contains a trained safety critic even when the
    # YAML does not specify safety_pretrained_path. Mark it ready so the exact
    # online filtering path is exercised without loading replay data.
    agent._cfg.safety_pretrained_path = str(
        checkpoint_path / "safety_critic.pt")
    env = MujocoSnapshotEnv(
        model_path, robot_cfg,
        policy_frequency=train_cfg.control_frequency,
        max_joint_delta=train_cfg.max_joint_delta,
        use_action_filter=train_cfg.use_action_filter)
    rng = np.random.default_rng(args.seed)
    env.reset_standing(rng=rng)
    episode_step = 0
    pairs = []
    selector_counts = {
        "evaluated": 0,
        "nominal_unsafe": 0,
        "replaced": 0,
        "no_safe": 0,
    }
    nominal_risks = []
    selected_risks = []

    # Exact restore must reproduce derived state before collecting evidence.
    verification = env.capture()
    before = env.observation()
    env.step(np.zeros(robot_cfg.num_joints, dtype=np.float32))
    env.restore(verification)
    restore_error = float(np.max(np.abs(
        before - env.observation())))
    if restore_error != 0.0:
        raise RuntimeError(f"snapshot restore error {restore_error}")

    for natural_step in range(args.natural_steps):
        if (
            args.disturbance_interval > 0
            and episode_step > 0
            and episode_step % args.disturbance_interval == 0
        ):
            env.apply_base_velocity_impulse(
                linear_velocity_delta=rng.normal(
                    0.0, args.linear_impulse_std, size=3),
                angular_velocity_delta=rng.normal(
                    0.0, args.angular_impulse_std, size=3))
        observation = env.record_observation()[-1]
        nominal = _actor_actions(
            agent, observation, 1, sample=True,
            seed=args.seed * 1_000_000 + natural_step)[0]
        selection = _select(
            agent, observation, nominal,
            seed=args.seed * 2_000_000 + natural_step)
        selector_counts["evaluated"] += 1
        selector_counts["nominal_unsafe"] += int(
            selection["nominal_risk"] > float(agent.cfg.safety_epsilon))
        selector_counts["replaced"] += int(selection["replaced"])
        selector_counts["no_safe"] += int(selection["no_safe"])
        nominal_risks.append(selection["nominal_risk"])
        selected_risks.append(selection["selected_risk"])
        selected = selection["action"]
        snapshot = env.capture()
        if selection["replaced"]:
            record = {
                "natural_step": natural_step,
                "episode_step": episode_step,
                "nominal_risk": selection["nominal_risk"],
                "selected_risk": selection["selected_risk"],
                "predicted_improvement": float(
                    selection["nominal_risk"]
                    - selection["selected_risk"]),
                "nominal_reward_q": selection["nominal_reward_q"],
                "selected_reward_q": selection["selected_reward_q"],
                "action_l2": float(np.linalg.norm(selected - nominal)),
                "horizons": {},
            }
            for horizon in (8, 16, 32):
                nominal_result = _branch(
                    env, agent, snapshot, nominal, horizon)
                selected_result = _branch(
                    env, agent, snapshot, selected, horizon)
                record["horizons"][str(horizon)] = {
                    "nominal": nominal_result,
                    "selected": selected_result,
                }
            pairs.append(record)
            env.restore(snapshot)
            if len(pairs) >= args.max_pairs:
                break

        measurement = env.step(nominal)
        episode_step += 1
        if measurement.failure or episode_step >= args.episode_steps:
            env.reset_standing(rng=rng)
            episode_step = 0

    summary = {
        "seed": args.seed,
        "checkpoint": str(checkpoint_path),
        "model": str(model_path),
        "natural_steps_requested": args.natural_steps,
        "pairs": len(pairs),
        "selector_counts": selector_counts,
        "nominal_risk_quantiles": {
            str(q): float(np.nanquantile(nominal_risks, q))
            for q in (0.5, 0.9, 0.95, 0.99, 1.0)
        },
        "selected_risk_quantiles": {
            str(q): float(np.nanquantile(selected_risks, q))
            for q in (0.5, 0.9, 0.95, 0.99, 1.0)
        },
        "snapshot_restore_max_observation_error": restore_error,
        "horizons": {},
    }
    for horizon in (8, 16, 32):
        results = [p["horizons"][str(horizon)] for p in pairs]
        improved = sum(
            r["nominal"]["failure"] and not r["selected"]["failure"]
            for r in results)
        worsened = sum(
            not r["nominal"]["failure"] and r["selected"]["failure"]
            for r in results)
        both_fail = sum(
            r["nominal"]["failure"] and r["selected"]["failure"]
            for r in results)
        neither = len(results) - improved - worsened - both_fail
        nominal_fail = improved + both_fail
        selected_fail = worsened + both_fail
        informative = improved + worsened
        summary["horizons"][str(horizon)] = {
            "improved": improved,
            "worsened": worsened,
            "both_fail": both_fail,
            "neither_fail": neither,
            "nominal_failures": nominal_fail,
            "selected_failures": selected_fail,
            "absolute_failure_delta": selected_fail - nominal_fail,
            "pairwise_ranking_accuracy": (
                improved / informative if informative else None),
            "net_improvement_rate": (
                (improved - worsened) / len(results)
                if results else None),
        }
    payload = {"summary": summary, "records": pairs}
    output = output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
