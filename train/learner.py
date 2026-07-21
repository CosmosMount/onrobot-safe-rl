"""Learner orchestration for in-process and split collector modes."""

from __future__ import annotations

import os
import signal
import threading
import time

from pathlib import Path

import numpy as np

from collector.legacy_env import build_legacy_env
from collector.transition_builder import build_transition
from rl.droq.data import ReplayBuffer
from train.gym_env import prepare_env
from train.checkpoint import (latest_snapshot, load_training_snapshot_metadata,
                              restore_training_snapshot,
                              save_training_snapshot)
from rl.droq import DroQLearner
from train.logging import TrainLogger
from train.loop import (_apply_agent_update, _is_finite_array,
                        _snapshot_metadata, _validate_snapshot_metadata,
                        run_training)
from train.profiling import StepProfiler
from train.rolling_metrics import RollingTrainingSummary
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


def build_agent_and_replay(env, train_cfg, agent_cfgs):
    if train_cfg.agent == 'flashsac':
        from rl.flashsac import FlashSACBackend
        agent = FlashSACBackend(
            env.observation_space,
            env.action_space,
            seed=train_cfg.seed,
            **agent_cfgs.get('flashsac', {}),
        )
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


def run_split(robot_cfg, train_cfg, agent_cfgs) -> int:
    """Run a 20 Hz collector loop with learner updates on a background thread."""
    os.makedirs(train_cfg.save_dir, exist_ok=True)
    env = prepare_env(build_legacy_env(robot_cfg, train_cfg, train_cfg.seed),
                      rescale_actions=False,
                      seed=train_cfg.seed)
    agent, replay_buffer = build_agent_and_replay(env, train_cfg, agent_cfgs)

    start_i = 0
    if (train_cfg.save_checkpoints and train_cfg.resume_checkpoint
            and not train_cfg.benchmark_only):
        latest = latest_snapshot(train_cfg.save_dir)
        if latest is not None:
            metadata = load_training_snapshot_metadata(latest)
            _validate_snapshot_metadata(latest, metadata, env)
            snapshot_experiment = metadata.get('experiment_name')
            if snapshot_experiment not in (None, train_cfg.experiment_name):
                raise RuntimeError(
                    'Refusing to restore a snapshot from another experiment: '
                    f'snapshot={snapshot_experiment!r} '
                    f'current={train_cfg.experiment_name!r}')
            snapshot = restore_training_snapshot(
                latest, agent=agent, replay_buffer=replay_buffer)
            agent = snapshot['agent']
            replay_buffer = snapshot['replay_buffer']
            start_i = int(snapshot['step'])
            print(f'[split] resumed complete snapshot {latest} step {start_i}',
                  flush=True)
    elif train_cfg.save_checkpoints and not train_cfg.benchmark_only:
        latest = latest_snapshot(train_cfg.save_dir)
        if latest is not None:
            print(f'[split] starting from scratch; ignoring checkpoint {latest}',
                  flush=True)

    if train_cfg.warmup:
        agent, warmup_ms, warmup_metrics = warmup_agent(
            agent, env, train_cfg.batch_size, train_cfg.utd_ratio)
        print(f'[split] JIT warmup compile_ms={warmup_ms:.1f} '
              f'steady_ms={warmup_metrics.get("warmup_steady_ms", warmup_ms):.1f} '
              f'{ {k: v for k, v in warmup_metrics.items() if not k.startswith("warmup_")} }',
              flush=True)

    stop_event = threading.Event()
    replay_lock = threading.Lock()
    policy_lock = threading.Lock()
    credit_cv = threading.Condition()
    shared = {
        'policy_agent': agent,
        'policy_version': 0,
        'train_agent': agent,
        'pending_updates': 0,
        'completed_step': start_i,
        'latest_update_info': None,
        'latest_update_ms': float('nan'),
        'latest_update_source_step': None,
        'latest_update_version': 0,
        'learner_error': None,
    }

    def _shutdown_handler(signum, frame):
        print(f'\n[split] shutting down (signal {signum})...', flush=True)
        stop_event.set()
        with credit_cv:
            credit_cv.notify_all()

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    update_batch_size = train_cfg.batch_size * train_cfg.utd_ratio

    def learner_loop() -> None:
        nonlocal replay_buffer
        train_agent = agent
        try:
            while not stop_event.is_set():
                with credit_cv:
                    while (not stop_event.is_set()
                           and shared['pending_updates'] <= 0):
                        credit_cv.wait(timeout=0.1)
                    if stop_event.is_set():
                        break
                    shared['pending_updates'] -= 1
                    source_step = int(shared['completed_step'])

                with replay_lock:
                    if len(replay_buffer) <= 0:
                        continue
                    batch = replay_buffer.sample_jax(update_batch_size)

                train_agent, info, corrupted, elapsed = _apply_agent_update(
                    train_agent, batch, train_cfg, source_step)
                if corrupted:
                    stop_event.set()
                with policy_lock:
                    shared['train_agent'] = train_agent
                    shared['policy_agent'] = train_agent
                    shared['policy_version'] += 1
                    shared['latest_update_info'] = info
                    shared['latest_update_ms'] = elapsed * 1000.0
                    shared['latest_update_source_step'] = source_step
                    shared['latest_update_version'] += 1
        except BaseException as exc:  # Surface background failures in collector.
            with policy_lock:
                shared['learner_error'] = exc
            stop_event.set()
            with credit_cv:
                credit_cv.notify_all()

    learner_thread = threading.Thread(
        target=learner_loop, name='split-learner', daemon=True)
    learner_thread.start()

    logger = TrainLogger(
        enabled=train_cfg.wandb and not train_cfg.benchmark_only,
        project=train_cfg.wandb_project,
        run_name=train_cfg.wandb_run_name,
        config={
            'mode': 'split',
            'experiment_name': train_cfg.experiment_name,
            'agent': train_cfg.agent,
            'seed': train_cfg.seed,
            'max_steps': train_cfg.max_steps,
            'start_training': train_cfg.start_training,
            'batch_size': train_cfg.batch_size,
            'utd_ratio': train_cfg.utd_ratio,
            'metrics_interval': train_cfg.metrics_interval,
            'explore_action_scale': train_cfg.explore_action_scale,
            'control_frequency': train_cfg.control_frequency,
            'resume_checkpoint': train_cfg.resume_checkpoint,
        },
    )
    profiler = StepProfiler(
        control_dt=1.0 / train_cfg.control_frequency,
        utd_ratio=train_cfg.utd_ratio,
        enabled=train_cfg.profile or train_cfg.benchmark_only,
    )
    rolling = RollingTrainingSummary(
        window=train_cfg.rolling_summary_window,
        action_dim=env.action_space.shape[0],
    )

    observation, _ = env.reset()
    local_policy = agent
    local_policy_version = 0
    episode_return = 0.0
    episode_length = 0
    done = False
    completed_step = start_i
    last_saved_step = start_i if latest_snapshot(train_cfg.save_dir) else -1
    max_steps = (
        train_cfg.benchmark_steps if train_cfg.benchmark_only
        else train_cfg.max_steps)
    seen_update_version = 0
    print(f'[split] collector ready obs={observation.shape} '
          f'start_training={train_cfg.start_training} '
          f'utd_ratio={train_cfg.utd_ratio} replay={len(replay_buffer)}',
          flush=True)

    try:
        for i in range(start_i, max_steps):
            if stop_event.is_set():
                break
            with policy_lock:
                if shared['learner_error'] is not None:
                    raise shared['learner_error']
                if shared['policy_version'] != local_policy_version:
                    local_policy = shared['policy_agent']
                    local_policy_version = int(shared['policy_version'])

            loop_t0 = time.perf_counter()
            profiler.begin_step()
            sample_t0 = time.perf_counter()
            if i < train_cfg.start_training:
                action = env.sample_action() * train_cfg.explore_action_scale
            else:
                if i == train_cfg.start_training:
                    print(f'[split] === Entering policy training at step {i} ===',
                          flush=True)
                action, local_policy = local_policy.sample_actions(observation)
            action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            if not _is_finite_array(action):
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            profiler.record_sample(time.perf_counter() - sample_t0)

            step_t0 = time.perf_counter()
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            profiler.record_step(time.perf_counter() - step_t0)
            episode_return += reward
            if info.get('policy_step', True):
                episode_length += 1

            policy_step = info.get('policy_step', True)
            insert_ok = (policy_step
                         and _is_finite_array(observation)
                         and _is_finite_array(next_observation)
                         and _is_finite_array(action)
                         and np.isfinite(reward))
            if insert_ok:
                transition = build_transition(
                    observation, action, reward, next_observation, done, info,
                    projected_action=info.get('projected_action'),
                    executed_q_target=info.get('executed_q_target'),
                    policy_version=local_policy_version)
                with replay_lock:
                    replay_buffer.insert(transition.replay_dict())
                    replay_size = len(replay_buffer)
                update_due = (
                    i >= train_cfg.start_training
                    and train_cfg.split_update_interval_steps > 0
                    and completed_step % train_cfg.split_update_interval_steps == 0)
                if update_due:
                    with credit_cv:
                        if shared['pending_updates'] < train_cfg.split_max_pending_updates:
                            shared['pending_updates'] += 1
                            credit_cv.notify()
            else:
                replay_size = len(replay_buffer)
            observation = next_observation

            profiler.end_loop(time.perf_counter() - loop_t0)
            completed_step = i + 1
            with policy_lock:
                shared['completed_step'] = completed_step
                latest_update_version = int(shared['latest_update_version'])
                if latest_update_version != seen_update_version:
                    update_info = shared['latest_update_info']
                    update_ms = float(shared['latest_update_ms'])
                    update_source_step = shared['latest_update_source_step']
                    seen_update_version = latest_update_version
                else:
                    update_info = None
                    update_ms = float('nan')
                    update_source_step = None
                policy_version = int(shared['policy_version'])
            with credit_cv:
                pending_updates = int(shared['pending_updates'])
            if np.isfinite(update_ms):
                profiler.record_update(update_ms / 1000.0)
            timing_metrics = profiler.metrics()
            rolling.record_step(
                action=action,
                info=info,
                timing=timing_metrics,
                update_info=update_info,
            )

            if (i % train_cfg.log_interval == 0
                    or i == train_cfg.start_training):
                phase = 'explore' if i < train_cfg.start_training else 'train'
                print(f'[step {i}] split phase={phase} reward={reward:.3f} '
                      f'x_vel={info.get("x_velocity", 0):.3f} '
                      f'|action|={float(np.linalg.norm(action)):.2f} '
                      f'policy_v={policy_version} pending={pending_updates} '
                      f'policy_len={info.get("step_count", 0)} '
                      f'ep_return={episode_return:.2f} buffer={replay_size}',
                      flush=True)
                if update_info is not None and update_source_step is not None:
                    metrics = {
                        k: float(v) if hasattr(v, 'item') else v
                        for k, v in update_info.items()
                    }
                    print(f'[step {i}] learner source_step={update_source_step} '
                          f'update_ms={update_ms:.1f} {metrics}',
                          flush=True)

            metrics_due = (
                train_cfg.metrics_interval <= 1
                or i % train_cfg.metrics_interval == 0
                or done
                or i == max_steps - 1)
            rolling_metrics = (
                rolling.metrics(replay_size) if metrics_due else {})
            if metrics_due:
                log_metrics = {
                    'env/reward': float(reward),
                    'env/task_reward': float(info.get('task_reward', reward)),
                    'env/terminal_penalty': float(
                        info.get('terminal_penalty', 0.0)),
                    'env/upright_gate': float(info.get('upright_gate', 1.0)),
                    'env/body_up_cos': float(info.get('body_up_cos', 1.0)),
                    'env/x_velocity': float(info.get('x_velocity', 0.0)),
                    'env/world_x': float(info.get('world_x', 0.0)),
                    'env/episode_return': float(episode_return),
                    'env/episode_length': float(episode_length),
                    'env/action_frequency_hz': float(
                        info.get('action_frequency_hz', np.nan)),
                    'env/control_hold_overrun_ms': float(
                        info.get('control_hold_overrun_ms', 0.0)),
                    'split/pending_updates': float(pending_updates),
                    'split/policy_version': float(policy_version),
                }
                if update_info is not None:
                    for k, v in update_info.items():
                        fv = float(v) if hasattr(v, 'item') else float(v)
                        if np.isfinite(fv):
                            log_metrics[f'training/{k}'] = fv
                log_metrics.update(timing_metrics)
                log_metrics.update(rolling_metrics)
                logger.log(log_metrics, step=i)

            if i % train_cfg.log_interval == 0 and rolling_metrics:
                print(
                    f'[step {i}] rolling n={int(rolling_metrics["rolling/window_steps"])} '
                    f'forward_vel={rolling_metrics["rolling/forward_velocity_mean"]:.3f} '
                    f'dx={rolling_metrics["rolling/world_x_delta"]:.3f} '
                    f'upright={rolling_metrics["rolling/upright_ratio"]:.3f} '
                    f'action_sat={rolling_metrics["rolling/action_saturation_rate"]:.3f} '
                    f'falls={int(rolling_metrics["rolling/falls_total"])} '
                    f'loop_hz={rolling_metrics["rolling/effective_hz_mean"]:.1f} '
                    f'action_hz={rolling_metrics["rolling/action_frequency_hz_mean"]:.1f}',
                    flush=True)

            if done:
                reason = (
                    'standup-timeout' if info.get('standup_timed_out')
                    else 'fallen' if info.get('terminated')
                    else 'truncated')
                print(f'[step {i}] episode done ({reason}) '
                      f'return={episode_return:.2f} '
                      f'policy_len={info.get("step_count", episode_length)}',
                      flush=True)
                rolling.record_episode(episode_return, episode_length)
                logger.log({
                    'training/return': episode_return,
                    'training/length': float(episode_length),
                }, step=i)
                observation, _ = env.reset(
                    standup=info.get('terminated', False)
                    or info.get('standup_timed_out', False),
                    with_recovery=info.get('is_belly_up', False),
                    grace_period=not info.get('truncated', False),
                    preserve_policy_state=info.get('truncated', False),
                )
                if not _is_finite_array(observation):
                    observation = np.zeros(env.observation_space.shape,
                                           dtype=np.float32)
                done = False
                episode_return = 0.0
                episode_length = 0

            if (train_cfg.save_checkpoints and not train_cfg.benchmark_only
                    and train_cfg.checkpoint_interval > 0
                    and completed_step % train_cfg.checkpoint_interval == 0):
                with policy_lock:
                    checkpoint_agent = shared['train_agent']
                with replay_lock:
                    path = save_training_snapshot(
                        train_cfg.save_dir,
                        agent=checkpoint_agent,
                        replay_buffer=replay_buffer,
                        step=completed_step,
                        metadata=_snapshot_metadata(train_cfg, env),
                    )
                last_saved_step = completed_step
                print(f'[step {i}] checkpoint saved: {path}', flush=True)
    finally:
        stop_event.set()
        with credit_cv:
            credit_cv.notify_all()
        learner_thread.join(timeout=5.0)
        if (train_cfg.save_checkpoints and not train_cfg.benchmark_only
                and completed_step > 0 and completed_step != last_saved_step):
            with policy_lock:
                checkpoint_agent = shared['train_agent']
            with replay_lock:
                path = save_training_snapshot(
                    train_cfg.save_dir,
                    agent=checkpoint_agent,
                    replay_buffer=replay_buffer,
                    step=completed_step,
                    metadata=_snapshot_metadata(train_cfg, env),
                )
            print(f'[split] final checkpoint saved: {path}', flush=True)
        logger.finish()
        env.close()

    return 0
