"""Online DROQ training loop."""

from __future__ import annotations

import os
import time

import numpy as np

try:
    import tqdm as tqdm_module
except ImportError:
    tqdm_module = None

from train.config import TrainConfig
from train.logging import TrainLogger
from train.profiling import StepProfiler
from collector.transition_builder import build_transition
from learner.checkpoint import (has_legacy_agent_checkpoint, latest_snapshot,
                                restore_training_snapshot,
                                save_training_snapshot)
from jaxrl.env.evaluation import evaluate


def _log(msg: str) -> None:
    print(msg, flush=True)


def _to_float_dict(info: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in info.items():
        if isinstance(v, (int, float, np.floating)):
            out[k] = float(v)
    return out


def _is_finite_array(x) -> bool:
    a = np.asarray(x)
    return bool(np.all(np.isfinite(a)))


def _batch_is_finite(batch: dict) -> bool:
    for v in batch.values():
        arr = np.asarray(v)
        if not np.all(np.isfinite(arr)):
            return False
    return True


def run_training(agent, env, replay_buffer, cfg: TrainConfig):
    os.makedirs(cfg.save_dir, exist_ok=True)

    start_i = 0
    if cfg.save_checkpoints and not cfg.benchmark_only:
        latest = latest_snapshot(cfg.save_dir)
        if latest is not None:
            snapshot = restore_training_snapshot(latest)
            agent = snapshot['agent']
            replay_buffer = snapshot['replay_buffer']
            start_i = int(snapshot['step'])
            _log(f'[train] resumed complete snapshot {latest} step {start_i}')
        elif has_legacy_agent_checkpoint(cfg.save_dir):
            raise RuntimeError(
                'Found legacy agent-only checkpoint in '
                f'{cfg.save_dir}. Online training requires an agent+replay '
                'snapshot. Delete the old checkpoint directory or start a new '
                'run from step 0.')

    update_batch_size = cfg.batch_size * cfg.utd_ratio
    inner = getattr(env, '_env', env)
    control_dt = inner.control_dt
    control_frequency = inner.control_frequency
    profiler = StepProfiler(control_dt=control_dt,
                            utd_ratio=cfg.utd_ratio,
                            enabled=cfg.profile or cfg.benchmark_only)
    logger = TrainLogger(
        enabled=cfg.wandb and not cfg.benchmark_only,
        project=cfg.wandb_project,
        run_name=cfg.wandb_run_name,
        config={
            'seed': cfg.seed,
            'max_steps': cfg.max_steps,
            'start_training': cfg.start_training,
            'batch_size': cfg.batch_size,
            'utd_ratio': cfg.utd_ratio,
            'explore_action_scale': cfg.explore_action_scale,
            'control_frequency': control_frequency,
        },
    )

    observation = env.reset()
    nan_policy_warned = False
    policy_corrupted = False
    _log(f'[train] env ready obs={observation.shape} '
         f'start_training={cfg.start_training} '
         f'explore_action_scale={cfg.explore_action_scale} '
         f'log_interval={cfg.log_interval} utd_ratio={cfg.utd_ratio} '
         f'no_eval={cfg.no_eval} profile={cfg.profile}')

    episode_return = 0.0
    episode_length = 0
    done = False
    max_steps = cfg.benchmark_steps if cfg.benchmark_only else cfg.max_steps
    iterator = range(start_i, max_steps)
    if cfg.use_tqdm and tqdm_module is not None:
        iterator = tqdm_module.tqdm(iterator, smoothing=0.1)

    try:
        for i in iterator:
            loop_t0 = time.perf_counter()
            profiler.begin_step()

            sample_t0 = time.perf_counter()
            skip_update = policy_corrupted
            if i < cfg.start_training:
                action = env.sample_action() * cfg.explore_action_scale
            else:
                if i == cfg.start_training:
                    _log(f'[train] === Entering policy training at step {i} ===')
                action, agent = agent.sample_actions(observation)
                if not _is_finite_array(action):
                    if not nan_policy_warned:
                        _log('[train] WARNING: policy returned non-finite action; '
                             'using zeros. Delete saved/checkpoints and restart '
                             'if this persists.')
                        nan_policy_warned = True
                    action = np.zeros(env.action_space.shape, dtype=np.float32)
                    skip_update = True
                    policy_corrupted = True
            action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            if not _is_finite_array(action):
                action = np.zeros(env.action_space.shape, dtype=np.float32)
                skip_update = True
            profiler.record_sample(time.perf_counter() - sample_t0)

            step_t0 = time.perf_counter()
            next_observation, reward, done, info = env.step(action)
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
                transition = build_transition(observation, action, reward,
                                              next_observation, done, info,
                                              projected_action=info.get(
                                                  'projected_action'),
                                              executed_q_target=info.get(
                                                  'executed_q_target'))
                replay_buffer.insert(transition.replay_dict())
            elif i >= cfg.start_training:
                skip_update = True
            observation = next_observation

            update_info = None
            if (not skip_update and i >= cfg.start_training
                    and len(replay_buffer) > 0):
                update_t0 = time.perf_counter()
                batch = replay_buffer.sample_jax(update_batch_size)
                if _batch_is_finite(batch):
                    agent, update_info = agent.update(batch, cfg.utd_ratio)
                    if update_info is not None and not all(
                            np.isfinite(float(v) if hasattr(v, 'item') else v)
                            for v in update_info.values()):
                        _log(f'[train] WARNING: non-finite update at step {i}, '
                             f'skipping future updates until restart')
                        skip_update = True
                        policy_corrupted = True
                        update_info = None
                else:
                    _log(f'[train] WARNING: non-finite batch at step {i}, skip update')
                    skip_update = True
                profiler.record_update(time.perf_counter() - update_t0)

            profiler.end_loop(time.perf_counter() - loop_t0)

            if (i % cfg.log_interval == 0 or i == cfg.start_training
                    or (i >= cfg.start_training and i < cfg.start_training + 5)):
                phase = 'explore' if i < cfg.start_training else 'train'
                _log(f'[step {i}] phase={phase} reward={reward:.3f} '
                     f'x_vel={info.get("x_velocity", 0):.3f} '
                     f'|action|={float(np.linalg.norm(action)):.2f} '
                     f'recovering={info.get("is_recovering", False)} '
                     f'policy_len={info.get("step_count", 0)} '
                     f'ep_return={episode_return:.2f} buffer={len(replay_buffer)}')

            log_metrics: dict[str, float] = {
                'env/reward': float(reward),
                'env/task_reward': float(info.get('task_reward', reward)),
                'env/terminal_penalty': float(
                    info.get('terminal_penalty', 0.0)),
                'env/upright_gate': float(info.get('upright_gate', 1.0)),
                'env/body_up_cos': float(info.get('body_up_cos', 1.0)),
                'env/x_velocity': float(info.get('x_velocity', 0.0)),
                'env/world_x': float(info.get('world_x', 0.0)),
                'env/world_y': float(info.get('world_y', 0.0)),
                'env/world_z': float(info.get('world_z', 0.0)),
                'env/forward_term': float(info.get('forward_term', 0.0)),
                'env/episode_return': float(episode_return),
                'env/episode_length': float(episode_length),
            }
            if update_info is not None:
                for k, v in update_info.items():
                    fv = float(v) if hasattr(v, 'item') else float(v)
                    if np.isfinite(fv):
                        log_metrics[f'training/{k}'] = fv
            log_metrics.update(profiler.metrics())
            logger.log(log_metrics, step=i)

            if update_info is not None and (
                    i % cfg.log_interval == 0 or i == cfg.start_training):
                metrics = {
                    k: float(v) if hasattr(v, 'item') else v
                    for k, v in update_info.items()
                }
                timing = profiler.metrics()
                _log(f'[step {i}] update {metrics}')
                if timing:
                    _log(f'[step {i}] timing step_ms={timing["timing/step_ms"]:.1f} '
                         f'update_ms={timing["timing/update_ms"]:.1f} '
                         f'effective_hz={timing["timing/effective_hz"]:.1f} '
                         f'critic/s={timing["timing/critic_updates_per_sec"]:.0f}')

            if done:
                if info.get('standup_timed_out'):
                    reason = 'standup-timeout'
                elif info.get('terminated'):
                    reason = 'fallen'
                else:
                    reason = 'truncated'
                _log(f'[step {i}] episode done ({reason}) '
                     f'return={episode_return:.2f} '
                     f'policy_len={info.get("step_count", episode_length)}')
                logger.log({
                    'training/return': episode_return,
                    'training/length': float(episode_length),
                }, step=i)
                if info.get('terminated') or info.get('standup_timed_out'):
                    kind = ('belly-up recovery→standup'
                            if info.get('standup_with_recovery')
                            else 'stand-up')
                    if info.get('standup_timed_out'):
                        kind = f'standup-timeout ({kind})'
                    _log(f'[step {i}] reset: {kind}')
                observation = env.reset(
                    standup=info.get('terminated', False)
                    or info.get('standup_timed_out', False),
                    with_recovery=info.get('is_belly_up', False),
                    grace_period=not info.get('truncated', False),
                )
                if not _is_finite_array(observation):
                    observation = np.zeros(env.observation_space.shape,
                                           dtype=np.float32)
                done = False
                episode_return = 0.0
                episode_length = 0

            train_step = i - cfg.start_training
            if (not cfg.no_eval and not cfg.benchmark_only
                    and cfg.eval_interval > 0 and train_step > 0
                    and i >= cfg.start_training
                    and train_step % cfg.eval_interval == 0):
                _log(f'[step {i}] eval ({cfg.eval_episodes} ep)...')
                eval_t0 = time.time()
                eval_info = evaluate(agent, env, num_episodes=cfg.eval_episodes)
                observation = env.reset()
                done = False
                episode_return = 0.0
                episode_length = 0
                _log(f'[step {i}] eval {time.time() - eval_t0:.1f}s '
                     f'return={eval_info["return"]:.2f} '
                     f'length={eval_info["length"]:.1f}')
                logger.log({
                    'eval/return': float(eval_info['return']),
                    'eval/length': float(eval_info['length']),
                }, step=i)

                if cfg.save_checkpoints:
                    save_training_snapshot(
                        cfg.save_dir,
                        agent=agent,
                        replay_buffer=replay_buffer,
                        step=i + 1,
                        metadata={
                            'start_training': cfg.start_training,
                            'batch_size': cfg.batch_size,
                            'utd_ratio': cfg.utd_ratio,
                        },
                    )
    finally:
        logger.finish()

    if cfg.benchmark_only:
        timing = profiler.metrics()
        _log('[benchmark] done')
        if timing:
            _log(f'[benchmark] effective_hz={timing["timing/effective_hz"]:.2f} '
                 f'update_ms={timing["timing/update_ms"]:.1f} '
                 f'critic/s={timing["timing/avg_critic_updates_per_sec"]:.0f}')

    return agent
