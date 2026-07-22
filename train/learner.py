"""Learner orchestration for in-process online training."""

from __future__ import annotations

import signal

from pathlib import Path

import numpy as np

from rl.droq.data import ReplayBuffer
from train.checkpoint import (latest_snapshot, load_training_snapshot_metadata,
                              restore_training_snapshot)
from rl.droq import DroQLearner
from train.loop import (_is_finite_array, run_training)
from train.warmup import warmup_agent


def build_agent_and_replay(env, train_cfg, agent_cfgs):
    if train_cfg.agent == 'flashsac':
        from rl.flashsac import FlashSACBackend
        agent = FlashSACBackend(
            env.observation_space,
            env.action_space,
            seed=train_cfg.seed,
            **agent_cfgs.get('flashsac', {}),
        )
        return agent, None
    else:
        agent = DroQLearner.create(
            train_cfg.seed,
            env.observation_space,
            env.action_space,
            **agent_cfgs.get('droq', {}),
        )
        replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                     train_cfg.buffer_size)
        replay_buffer.seed(train_cfg.seed)
        return agent, replay_buffer


def run_in_process(robot_cfg, train_cfg, agent_cfgs) -> int:
    """Run collector and learner in one process through the legacy adapter."""
    from train.collector.legacy_env import build_legacy_env
    from train.gym_env import prepare_env

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

    agent, replay_buffer = build_agent_and_replay(env, train_cfg, agent_cfgs)

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
             agent_cfgs,
             *,
             checkpoint: str | None = None,
             episodes: int = 1) -> int:
    """Run a deterministic policy rollout from a saved training snapshot."""
    from train.collector.legacy_env import build_legacy_env
    from train.gym_env import prepare_env

    env = prepare_env(build_legacy_env(robot_cfg, train_cfg, train_cfg.seed),
                      rescale_actions=False,
                      seed=train_cfg.seed)
    agent, _ = build_agent_and_replay(env, train_cfg, agent_cfgs)

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
            observation, _ = env.reset()
            done = False
            episode_return = 0.0
            episode_length = 0
            last_info = {}
            while not done:
                action = np.clip(agent.eval_actions(observation), -1.0, 1.0)
                observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
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
