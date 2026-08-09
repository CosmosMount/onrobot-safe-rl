#!/usr/bin/env python3
"""Collect same-snapshot action outcomes for Q_safe ranking supervision."""

from __future__ import annotations

import argparse
import os

import gymnasium as gym
import numpy as np
import torch

from rl.agents import create_agent
from safety_data.paths import (
    assert_development_path,
    assert_safe_evidence_output,
    require_v3_audit_consumed_or_safe_input,
)
from scripts.evaluate_p16_snapshot_replacements import (
    _actor_actions,
    _branch,
    _risks,
)
from scripts.collect_native_grouped_qsafe import (
    _prepare_staged_outputs,
    _publish_staged_outputs,
)
from train.config import load_app_config
from train.mujoco_snapshot_env import MujocoSnapshotEnv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/go2_50hz_safe_adaptive_gated_v3.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--states", type=int, default=2000)
    parser.add_argument("--policy-candidates", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--disturbance-interval", type=int, default=10)
    parser.add_argument("--linear-impulse-std", type=float, default=1.0)
    parser.add_argument("--angular-impulse-std", type=float, default=4.0)
    parser.add_argument("--local-rms", type=float, default=0.15)
    parser.add_argument(
        "--mixed-only", action="store_true",
        help="Keep only states where candidate actions have different outcomes.")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model",
        default=("/home/xyz/code/unitree_mujoco/"
                 "unitree_robots/go2/scene_empty.xml"))
    args = parser.parse_args()
    config_path = assert_development_path(
        require_v3_audit_consumed_or_safe_input(args.config))
    checkpoint_path = assert_development_path(
        require_v3_audit_consumed_or_safe_input(args.checkpoint))
    model_path = assert_development_path(
        require_v3_audit_consumed_or_safe_input(args.model))
    output_path = assert_development_path(
        assert_safe_evidence_output(args.output))
    if output_path.suffix != ".npz":
        parser.error("counterfactual dataset output must use .npz")
    if os.path.lexists(os.fspath(output_path)):
        raise FileExistsError(f"refusing to overwrite output: {output_path}")

    robot_cfg, train_cfg, agent_cfg = load_app_config(
        config_path, agent="safe_droq")
    agent_cfg.device_type = "cuda" if torch.cuda.is_available() else "cpu"
    agent_cfg.buffer_device_type = agent_cfg.device_type
    observation_space = gym.spaces.Box(
        -100.0, 100.0, (robot_cfg.obs_dim,), dtype=np.float32)
    action_space = gym.spaces.Box(
        -1.0, 1.0, (robot_cfg.num_joints,), dtype=np.float32)
    agent = create_agent(
        observation_space, action_space, {}, agent_cfg)
    agent.load(str(checkpoint_path))

    env = MujocoSnapshotEnv(
        model_path, robot_cfg,
        policy_frequency=train_cfg.control_frequency,
        max_joint_delta=train_cfg.max_joint_delta,
        use_action_filter=train_cfg.use_action_filter)
    rng = np.random.default_rng(args.seed)
    env.reset_standing(rng=rng)
    episode_step = 0
    observations = []
    actions = []
    actions_executed = []
    action_q_targets = []
    failures = []
    failure_steps = []
    max_tilts = []
    min_heights = []
    state_ids = []
    candidate_kinds = []
    nominal_risks = []
    safety_contexts = []
    observation_histories = []

    natural_step = 0
    while len(set(state_ids)) < args.states:
        if episode_step > 0 and episode_step % args.disturbance_interval == 0:
            env.apply_base_velocity_impulse(
                linear_velocity_delta=rng.normal(
                    0.0, args.linear_impulse_std, size=3),
                angular_velocity_delta=rng.normal(
                    0.0, args.angular_impulse_std, size=3))
        observation_history = env.record_observation()
        observation = observation_history[-1]
        previous = env.previous_action_requested
        nominal = _actor_actions(
            agent, observation, 1, sample=True,
            seed=args.seed * 1_000_000 + natural_step)[0]
        risk = float(_risks(agent, observation, nominal[None, :])[0])

        # Retain boundary-heavy states plus a random normal sample so the
        # critic learns both discrimination and local action ranking.
        measurement_before = env.measurement()
        keep = bool(
            risk >= 0.05
            or measurement_before.near_failure
            or rng.random() < 0.10)
        if keep:
            state_id = len(set(state_ids))
            raw_policy = _actor_actions(
                agent, observation, args.policy_candidates,
                sample=True, seed=args.seed * 2_000_000 + natural_step)
            delta = raw_policy - nominal[None, :]
            rms = np.sqrt(np.mean(np.square(delta), axis=1, keepdims=True))
            scale = np.minimum(1.0, args.local_rms / np.maximum(rms, 1e-8))
            local_policy = np.clip(
                nominal[None, :] + scale * delta, -1.0, 1.0)
            candidates = [
                ("nominal", nominal),
                ("previous", previous.copy()),
                ("contracted_075", np.clip(
                    0.75 * nominal + 0.25 * previous, -1.0, 1.0)),
            ]
            candidates.extend(
                (f"local_policy_{index}", action)
                for index, action in enumerate(local_policy))
            snapshot = env.capture()
            state_rows = []
            for kind, candidate in candidates:
                outcome = _branch(
                    env, agent, snapshot, candidate, args.horizon)
                state_rows.append((
                    kind,
                    outcome["first_action_requested"],
                    outcome["first_action_executed"],
                    outcome["first_action_q_target"],
                    float(outcome["failure"]),
                    (
                        outcome["failure_step"]
                        if outcome["failure_step"] is not None
                        else args.horizon + 1),
                    float(outcome["max_tilt_rad"]),
                    float(outcome["min_height_m"]),
                ))
            state_labels = [row[4] for row in state_rows]
            if not args.mixed_only or min(state_labels) != max(state_labels):
                for (kind, candidate, candidate_executed, candidate_q_target,
                     failure, failure_step,
                     max_tilt, min_height) in state_rows:
                    observations.append(observation.copy())
                    actions.append(candidate)
                    actions_executed.append(candidate_executed)
                    action_q_targets.append(candidate_q_target)
                    failures.append(failure)
                    failure_steps.append(failure_step)
                    max_tilts.append(max_tilt)
                    min_heights.append(min_height)
                    state_ids.append(state_id)
                    candidate_kinds.append(kind)
                    nominal_risks.append(risk)
                    safety_contexts.append(np.asarray([
                        measurement_before.height_m,
                        measurement_before.tilt_rad,
                        float(measurement_before.contact_count),
                    ], dtype=np.float32))
                    observation_histories.append(observation_history)
            env.restore(snapshot)

        measurement = env.step(nominal)
        episode_step += 1
        natural_step += 1
        if measurement.failure or episode_step >= 400:
            env.reset_standing(rng=rng)
            episode_step = 0

    output = output_path
    staged = _prepare_staged_outputs((output,))
    staging = staged[0][0]
    try:
        np.savez_compressed(
            staging,
            observations=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            actions_executed=np.asarray(actions_executed, dtype=np.float32),
            action_q_targets=np.asarray(action_q_targets, dtype=np.float32),
            failures=np.asarray(failures, dtype=np.float32),
            failure_steps=np.asarray(failure_steps, dtype=np.int16),
            max_tilts=np.asarray(max_tilts, dtype=np.float32),
            min_heights=np.asarray(min_heights, dtype=np.float32),
            state_ids=np.asarray(state_ids, dtype=np.int32),
            candidate_kinds=np.asarray(candidate_kinds),
            nominal_risks=np.asarray(nominal_risks, dtype=np.float32),
            safety_contexts=np.asarray(safety_contexts, dtype=np.float32),
            observation_histories=np.asarray(
                observation_histories, dtype=np.float32),
            seed=np.asarray(args.seed, dtype=np.int32),
            horizon=np.asarray(args.horizon, dtype=np.int32),
        )
        _publish_staged_outputs(staged)
    finally:
        staging.unlink(missing_ok=True)
    mixed = 0
    for state_id in range(args.states):
        labels = np.asarray(failures)[np.asarray(state_ids) == state_id]
        mixed += int(labels.min() != labels.max())
    print({
        "states": args.states,
        "transitions": len(failures),
        "failures": int(sum(failures)),
        "mixed_outcome_states": mixed,
        "output": str(output),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
