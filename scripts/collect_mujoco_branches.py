#!/usr/bin/env python3
"""Collect exact-state MuJoCo counterfactual branches from a frozen SAC policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jaxrl.agents import DroQLearner
from jaxrl.env.specs import BoxSpec
from learner.checkpoint import restore_training_snapshot
from learner.counterfactual_dataset import (
    evaluate_snapshot_candidates,
    make_candidate_actions,
    save_counterfactual_artifact,
)
from train.config import load_app_config
from train.mujoco_branch import MujocoBranchBackend


DEFAULT_MODEL = (
    '/home/xyz/code/unitree_mujoco/unitree_robots/go2/scene_empty.xml')


def _load_agent(checkpoint: Path, robot_cfg, train_cfg, droq_cfg):
    observation_spec = BoxSpec(
        shape=(robot_cfg.obs_dim,), low=-np.inf, high=np.inf)
    action_spec = BoxSpec(
        shape=(robot_cfg.num_joints,),
        low=-np.ones(robot_cfg.num_joints, np.float32),
        high=np.ones(robot_cfg.num_joints, np.float32))
    template = DroQLearner.create(
        train_cfg.seed, observation_spec, action_spec, **droq_cfg)
    return restore_training_snapshot(
        checkpoint, agent=template)['agent']


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/go2.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument(
        '--output',
        default='saved/safety_datasets/counterfactual_branches_v1.pkl')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--natural-steps', type=int, default=400)
    parser.add_argument(
        '--max-episodes', type=int, default=100,
        help='Reset after natural-policy failure until this episode limit.')
    parser.add_argument('--snapshot-interval', type=int, default=10)
    parser.add_argument('--settle-seconds', type=float, default=1.0)
    parser.add_argument('--perturbation-count', type=int, default=8)
    parser.add_argument('--perturbation-std', type=float, default=0.15)
    parser.add_argument('--contraction', type=float, default=0.90)
    parser.add_argument('--horizons', default='8,16,32')
    args = parser.parse_args()

    robot_cfg, train_cfg, droq_cfg = load_app_config(args.config)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    agent = _load_agent(
        checkpoint, robot_cfg, train_cfg, droq_cfg)
    backend = MujocoBranchBackend(
        args.model, robot_cfg,
        policy_frequency=train_cfg.control_frequency)
    backend.reset_standing(settle_seconds=args.settle_seconds)
    rng = np.random.default_rng(args.seed)
    horizons = tuple(int(value) for value in args.horizons.split(','))

    def policy(observation):
        return np.clip(
            np.asarray(agent.eval_actions(observation), dtype=np.float32),
            -1.0, 1.0)

    snapshots = []
    branches = []
    previous = np.zeros(robot_cfg.num_joints, dtype=np.float32)
    executed = previous.copy()
    total_steps = 0
    episode_id = 0
    episode_step = 0
    natural_failures = 0
    while (total_steps < args.natural_steps
           and episode_id < args.max_episodes):
        observation = backend.observation(
            previous, executed, robot_cfg.move_speed)
        nominal = policy(observation)
        if total_steps % args.snapshot_interval == 0:
            snapshot = backend.snapshot(
                previous_action=previous,
                previous_executed_action=executed,
                command_speed=robot_cfg.move_speed,
                episode_id=episode_id,
                policy_step=episode_step)
            candidates = make_candidate_actions(
                nominal, previous, rng=rng,
                perturbation_count=args.perturbation_count,
                perturbation_std=args.perturbation_std,
                contraction=args.contraction)
            snapshots.append(snapshot)
            branches.extend(evaluate_snapshot_candidates(
                backend, snapshot, candidates, policy,
                snapshot_index=len(snapshots) - 1,
                horizons=horizons))
            # Candidate branches leave the backend at a counterfactual state.
            backend.restore_state(snapshot.simulator_state)
        measurement = backend.step_action(nominal)
        previous = nominal.copy()
        executed = nominal.copy()
        total_steps += 1
        episode_step += 1
        if measurement.failure:
            natural_failures += 1
            episode_id += 1
            episode_step = 0
            if total_steps < args.natural_steps:
                backend.reset_standing(settle_seconds=args.settle_seconds)
                previous = np.zeros(
                    robot_cfg.num_joints, dtype=np.float32)
                executed = previous.copy()

    output = save_counterfactual_artifact(
        args.output, snapshots=snapshots, branches=branches,
        metadata={
            'checkpoint': str(checkpoint),
            'model': str(Path(args.model).expanduser().resolve()),
            'seed': args.seed,
            'horizons': horizons,
            'natural_steps_requested': args.natural_steps,
            'natural_steps_completed': total_steps,
            'natural_failures': natural_failures,
            'episodes_started': episode_id + int(
                total_steps > 0 and episode_step > 0),
            'snapshot_interval': args.snapshot_interval,
            'perturbation_count': args.perturbation_count,
            'perturbation_std': args.perturbation_std,
            'command_speed': robot_cfg.move_speed,
        })
    summary = {
        'output': str(output),
        'snapshots': len(snapshots),
        'branches': len(branches),
        'natural_steps_completed': total_steps,
        'natural_failures': natural_failures,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
