"""Learner orchestration for in-process and split collector modes."""

from __future__ import annotations

import signal

from pathlib import Path

import numpy as np

from collector.legacy_env import build_legacy_env
from jaxrl.agents import DroQLearner
from jaxrl.data import ReplayBuffer
from jaxrl.env.env import prepare_env
from learner.checkpoint import (latest_snapshot, load_training_snapshot_metadata,
                                restore_training_snapshot)
from train.loop import run_training
from train.warmup import warmup_agent


class UpdateCredit:
    def __init__(self, utd_ratio: int, max_credit: int | None = None):
        self.utd_ratio = int(utd_ratio)
        self.max_credit = max_credit
        self.credit = 0

    def add_transition(self) -> None:
        self.credit += self.utd_ratio
        if self.max_credit is not None:
            self.credit = min(self.credit, self.max_credit)

    def consume_one(self) -> bool:
        if self.credit <= 0:
            return False
        self.credit -= 1
        return True


def build_agent_and_replay(env, train_cfg, droq_cfg):
    agent = DroQLearner.create(
        train_cfg.seed,
        env.observation_spec,
        env.action_spec,
        **droq_cfg,
    )
    replay_buffer = ReplayBuffer(env.observation_spec, env.action_spec,
                                 train_cfg.buffer_size)
    replay_buffer.seed(train_cfg.seed)
    return agent, replay_buffer


def run_in_process(robot_cfg, train_cfg, droq_cfg) -> int:
    """Run collector and learner in one process through the legacy adapter."""
    env = prepare_env(build_legacy_env(robot_cfg, train_cfg, train_cfg.seed),
                      rescale_actions=False,
                      seed=train_cfg.seed)

    def _shutdown_handler(signum, frame):
        print(f'\n[train] shutting down (signal {signum}), closing IPC...',
              flush=True)
        env.close()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    agent, replay_buffer = build_agent_and_replay(env, train_cfg, droq_cfg)

    if train_cfg.warmup:
        agent, warmup_ms, warmup_metrics = warmup_agent(
            agent, env, train_cfg.batch_size, train_cfg.utd_ratio)
        print(f'[train] JIT warmup compile_ms={warmup_ms:.1f} '
              f'steady_ms={warmup_metrics.get("warmup_steady_ms", warmup_ms):.1f} '
              f'{ {k: v for k, v in warmup_metrics.items() if not k.startswith("warmup_")} }',
              flush=True)

    run_training(agent, env, replay_buffer, train_cfg)
    return 0


def run_play(robot_cfg,
             train_cfg,
             droq_cfg,
             *,
             checkpoint: str | None = None,
             episodes: int = 1) -> int:
    """Run a deterministic policy rollout from a saved training snapshot."""
    env = prepare_env(build_legacy_env(robot_cfg, train_cfg, train_cfg.seed),
                      rescale_actions=False,
                      seed=train_cfg.seed)
    agent = DroQLearner.create(
        train_cfg.seed,
        env.observation_spec,
        env.action_spec,
        **droq_cfg,
    )

    path = Path(checkpoint) if checkpoint else latest_snapshot(train_cfg.save_dir)
    if path is None:
        raise RuntimeError(
            f'No training snapshot found in {train_cfg.save_dir}. '
            'Train first or pass --checkpoint.')

    metadata = load_training_snapshot_metadata(path)
    snapshot_obs_dim = metadata.get('obs_dim')
    if snapshot_obs_dim is not None:
        current_obs_dim = int(env.observation_space.shape[0])
        if int(snapshot_obs_dim) != current_obs_dim:
            raise RuntimeError(
                'Refusing to play an incompatible snapshot: '
                f'{path} has obs_dim={snapshot_obs_dim}, '
                f'current obs_dim={current_obs_dim}.')

    snapshot = restore_training_snapshot(path, agent=agent)
    agent = snapshot['agent']
    print(f'[play] loaded {path} step={snapshot["step"]} '
          f'obs={env.observation_space.shape} episodes={episodes}',
          flush=True)

    try:
        for episode in range(episodes):
            observation = env.reset()
            done = False
            episode_return = 0.0
            episode_length = 0
            last_info = {}
            while not done:
                action = np.clip(agent.eval_actions(observation), -1.0, 1.0)
                observation, reward, done, info = env.step(action)
                episode_return += reward
                last_info = info
                if info.get('policy_step', True):
                    episode_length += 1
            reason = (
                'fallen' if last_info.get('terminated') else
                'standup-timeout' if last_info.get('standup_timed_out') else
                'truncated')
            print(f'[play] episode={episode + 1} reason={reason} '
                  f'return={episode_return:.2f} length={episode_length} '
                  f'x={last_info.get("world_x", 0.0):.3f} '
                  f'x_vel={last_info.get("x_velocity", 0.0):.3f} '
                  f'belly_up={last_info.get("is_belly_up", False)}',
                  flush=True)
    finally:
        env.close()
    return 0


def run_split(robot_cfg, train_cfg, droq_cfg) -> int:
    """P2 split-mode entrypoint.

    The module boundaries are fixed now; until the framed controller protocol is
    switched on end-to-end, split mode intentionally reuses the in-process path
    instead of exposing a partially realtime execution path.
    """
    print('[train] split mode requested; using in-process compatibility path '
          'until framed controller protocol is enabled.',
          flush=True)
    return run_in_process(robot_cfg, train_cfg, droq_cfg)
