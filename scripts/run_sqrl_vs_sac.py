#!/usr/bin/env python3
"""Stabilize, eval SAC baseline, SQRL-pretrain, eval SQRL, optional finetune.

Compares falls / episode length / return under the same held-out noise.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from collector.legacy_env import build_legacy_env
from jaxrl.agents import DroQLearner
from jaxrl.agents.safety_critic import SafetyCritic
from jaxrl.agents.sqrl import select_sqrl_action
from jaxrl.env.env import prepare_env
from learner.checkpoint import latest_snapshot, restore_training_snapshot
from learner.learner import run_in_process
from train.config import load_app_config
from train.env import UnstableResetError
from train.main import _configure_sqrl_mode


def _stabilize(robot_cfg, train_cfg, attempts: int = 8) -> None:
    env = prepare_env(
        build_legacy_env(robot_cfg, train_cfg, train_cfg.seed),
        rescale_actions=False, seed=train_cfg.seed)
    try:
        for attempt in range(attempts):
            try:
                env.reset(standup=True, with_recovery=True, grace_period=True)
                print(f'[exp] stabilized attempt={attempt}', flush=True)
                time.sleep(1.0)
                return
            except UnstableResetError:
                time.sleep(2.0)
        raise RuntimeError('Could not stabilize robot')
    finally:
        env.close()


def _eval_policy(*, robot_cfg, train_cfg, droq_cfg, checkpoint: str,
                 episodes: int, action_noise_std: float, rollout_seed: int,
                 use_sqrl: bool, epsilon: float, num_candidates: int) -> dict:
    env = prepare_env(
        build_legacy_env(robot_cfg, train_cfg, train_cfg.seed),
        rescale_actions=False, seed=train_cfg.seed)
    agent = DroQLearner.create(
        train_cfg.seed, env.observation_spec, env.action_spec, **droq_cfg)
    safety = SafetyCritic.create(
        seed=train_cfg.seed + 10_000,
        observation_dim=int(env.observation_space.shape[0]),
        action_dim=int(env.action_space.shape[0]),
        hidden_dims=train_cfg.safety_critic_hidden_dims,
        learning_rate=train_cfg.safety_critic_learning_rate,
        discount=train_cfg.safety_discount,
        tau=train_cfg.safety_critic_tau,
        future_loss_weight=train_cfg.safety_future_loss_weight)
    print(f'[eval] loading {checkpoint}', flush=True)
    snapshot = restore_training_snapshot(
        Path(checkpoint), agent=agent,
        safety_critic=safety if use_sqrl else None)
    agent = snapshot['agent']
    if use_sqrl:
        if 'safety_critic' not in snapshot:
            env.close()
            raise RuntimeError(f'{checkpoint} missing safety_critic_state')
        safety = snapshot['safety_critic']
    print('[eval] checkpoint loaded', flush=True)

    rng = np.random.default_rng(rollout_seed)
    sqrl_rng = jax.random.PRNGKey(rollout_seed + 1)
    ep_returns = []
    ep_lengths = []
    falls = 0
    no_safe_steps = 0
    total_steps = 0
    reset_kwargs = {'standup': True, 'grace_period': True}
    try:
        for episode in range(episodes):
            observation = env.reset(**reset_kwargs)
            reset_kwargs = {}
            done = False
            ep_ret = 0.0
            ep_len = 0
            last_info = {}
            while not done:
                if use_sqrl:
                    action, info, sqrl_rng = select_sqrl_action(
                        agent, safety, observation, sqrl_rng,
                        num_candidates=num_candidates,
                        epsilon_safe=epsilon)
                    no_safe_steps += int(info['sqrl_no_safe_candidate'] > 0.5)
                else:
                    action = np.clip(agent.eval_actions(observation), -1.0, 1.0)
                if action_noise_std > 0.0:
                    action = np.clip(
                        action + rng.normal(
                            0.0, action_noise_std, size=action.shape),
                        -1.0, 1.0).astype(np.float32)
                observation, reward, done, last_info = env.step(action)
                if last_info.get('policy_step', True):
                    ep_ret += float(reward)
                    ep_len += 1
                    total_steps += 1
            fell = bool(
                last_info.get('terminated', False)
                or last_info.get('unsafe_label', False))
            falls += int(fell)
            ep_returns.append(ep_ret)
            ep_lengths.append(ep_len)
            reset_kwargs = {
                'standup': bool(last_info.get('terminated', False)
                                or last_info.get('standup_timed_out', False)),
                'with_recovery': bool(last_info.get('is_belly_up', False)),
                'grace_period': not bool(last_info.get('truncated', False)),
                'preserve_policy_state': bool(
                    last_info.get('truncated', False)),
            }
            print(
                f'[eval] ep={episode + 1}/{episodes} len={ep_len} '
                f'return={ep_ret:.1f} fell={fell}',
                flush=True)
    finally:
        env.close()
    return {
        'checkpoint': checkpoint,
        'use_sqrl': use_sqrl,
        'rollout_seed': rollout_seed,
        'action_noise_std': action_noise_std,
        'episodes': episodes,
        'falls': falls,
        'average_episode_length': float(np.mean(ep_lengths)),
        'average_return': float(np.mean(ep_returns)),
        'no_safe_rate': (
            float(no_safe_steps / max(total_steps, 1)) if use_sqrl else 0.0),
        'episode_lengths': ep_lengths,
        'episode_returns': ep_returns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/go2.yaml')
    parser.add_argument(
        '--sac-checkpoint',
        default='saved/checkpoints_58d/training_snapshot_000000012584.pkl')
    parser.add_argument('--save-dir', default='saved/checkpoints_sqrl')
    parser.add_argument('--pretrain-steps', type=int, default=3000)
    parser.add_argument('--finetune-steps', type=int, default=2000)
    parser.add_argument('--skip-sac', action='store_true',
                        help='Reuse SAC rows from --reuse-results JSON')
    parser.add_argument('--skip-pretrain', action='store_true',
                        help='Eval existing SQRL checkpoint in --save-dir')
    parser.add_argument('--skip-finetune', action='store_true')
    parser.add_argument(
        '--skip-pretrain-eval', action='store_true',
        help='Skip SQRL-pretrain held-out eval (reuse prior rows)')
    parser.add_argument(
        '--skip-finetune-train', action='store_true',
        help='Skip finetune training; eval latest checkpoint in --save-dir')
    parser.add_argument(
        '--reuse-results', default='',
        help='JSON with prior results[] to merge when skipping stages')
    parser.add_argument('--play-episodes', type=int, default=2)
    parser.add_argument('--action-noise-std', type=float, default=0.50)
    parser.add_argument('--rollout-seeds', default='9010,9011')
    parser.add_argument(
        '--output', default='saved/safety_evaluation/sqrl_vs_sac_summary.json')
    args = parser.parse_args()

    seeds = [int(x) for x in args.rollout_seeds.split(',') if x.strip()]
    robot_cfg, train_cfg, droq_cfg = load_app_config(path=args.config)
    droq_cfg = dict(droq_cfg)
    results = []
    if args.reuse_results:
        prior = json.loads(Path(args.reuse_results).read_text(encoding='utf-8'))
        for row in prior.get('results', []):
            cell = str(row.get('cell', ''))
            keep = (
                (args.skip_sac and cell.startswith('sac_'))
                or (args.skip_pretrain_eval
                    and cell.startswith('sqrl_pretrain_')))
            if keep:
                results.append(row)

    if not args.skip_sac:
        print('[exp] === SAC baseline eval ===', flush=True)
        for seed in seeds:
            _stabilize(robot_cfg, train_cfg)
            row = _eval_policy(
                robot_cfg=robot_cfg, train_cfg=train_cfg, droq_cfg=droq_cfg,
                checkpoint=args.sac_checkpoint, episodes=args.play_episodes,
                action_noise_std=args.action_noise_std, rollout_seed=seed,
                use_sqrl=False, epsilon=train_cfg.sqrl_epsilon,
                num_candidates=train_cfg.sqrl_num_candidates)
            row['cell'] = f'sac_seed{seed}'
            results.append(row)
            print(f'[exp] {row["cell"]} falls={row["falls"]} '
                  f'len={row["average_episode_length"]:.1f} '
                  f'return={row["average_return"]:.1f}', flush=True)
    else:
        print('[exp] === SAC baseline eval (skipped) ===', flush=True)

    # --- SQRL pretrain ---
    warm_step = 12584
    pre_cfg_max_steps = warm_step + int(args.pretrain_steps)
    if not args.skip_pretrain:
        print('[exp] === SQRL pretrain ===', flush=True)
        _stabilize(robot_cfg, train_cfg)
        ns = argparse.Namespace(
            mode='sqrl_pretrain', checkpoint=args.sac_checkpoint,
            save_dir=args.save_dir, from_scratch=False)
        pre_cfg, pre_droq = _configure_sqrl_mode(ns, train_cfg, dict(droq_cfg))
        # Continue N steps past the warm-start step index.
        pre_cfg.max_steps = pre_cfg_max_steps
        pre_cfg.checkpoint_interval = min(1000, int(args.pretrain_steps))
        pre_cfg.warmup = True
        pre_cfg.wandb = False
        pre_cfg.resume_checkpoint = True
        print(f'[exp] pretrain max_steps={pre_cfg.max_steps} '
              f'save_dir={pre_cfg.save_dir}', flush=True)
        run_in_process(robot_cfg, pre_cfg, pre_droq)
        pre_ckpt = latest_snapshot(pre_cfg.save_dir)
    else:
        print('[exp] === SQRL pretrain (skipped) ===', flush=True)
        pre_ckpt = latest_snapshot(args.save_dir)
    if pre_ckpt is None:
        raise RuntimeError('SQRL pretrain produced no snapshot')
    print(f'[exp] pretrain checkpoint={pre_ckpt}', flush=True)

    if not args.skip_pretrain_eval:
        print('[exp] === SQRL pretrain eval ===', flush=True)
        for seed in seeds:
            _stabilize(robot_cfg, train_cfg)
            row = _eval_policy(
                robot_cfg=robot_cfg, train_cfg=train_cfg, droq_cfg=droq_cfg,
                checkpoint=str(pre_ckpt), episodes=args.play_episodes,
                action_noise_std=args.action_noise_std, rollout_seed=seed,
                use_sqrl=True, epsilon=train_cfg.sqrl_epsilon,
                num_candidates=train_cfg.sqrl_num_candidates)
            row['cell'] = f'sqrl_pretrain_seed{seed}'
            results.append(row)
            print(f'[exp] {row["cell"]} falls={row["falls"]} '
                  f'len={row["average_episode_length"]:.1f} '
                  f'return={row["average_return"]:.1f} '
                  f'no_safe={row["no_safe_rate"]:.3f}', flush=True)
    else:
        print('[exp] === SQRL pretrain eval (skipped) ===', flush=True)

    ft_ckpt = None
    if not args.skip_finetune and (
            args.finetune_steps > 0 or args.skip_finetune_train):
        if not args.skip_finetune_train:
            print('[exp] === SQRL finetune ===', flush=True)
            _stabilize(robot_cfg, train_cfg)
            # Reload config so we don't mutate pre_cfg leftovers oddly.
            robot_cfg, train_cfg, droq_cfg = load_app_config(path=args.config)
            ns = argparse.Namespace(
                mode='sqrl_finetune', checkpoint=str(pre_ckpt),
                save_dir=args.save_dir, from_scratch=False)
            ft_cfg, ft_droq = _configure_sqrl_mode(
                ns, train_cfg, dict(droq_cfg))
            ft_cfg.max_steps = (
                int(pre_cfg_max_steps) + int(args.finetune_steps))
            ft_cfg.checkpoint_interval = min(1000, int(args.finetune_steps))
            ft_cfg.warmup = True
            ft_cfg.wandb = False
            ft_cfg.resume_checkpoint = True
            # Prefer resume from save_dir latest, not warm_start overwrite.
            ft_cfg.warm_start_checkpoint = None
            print(f'[exp] finetune max_steps={ft_cfg.max_steps}', flush=True)
            run_in_process(robot_cfg, ft_cfg, ft_droq)
            ft_ckpt = latest_snapshot(ft_cfg.save_dir)
        else:
            print('[exp] === SQRL finetune train (skipped) ===', flush=True)
            robot_cfg, train_cfg, droq_cfg = load_app_config(path=args.config)
            droq_cfg = dict(droq_cfg)
            ft_ckpt = latest_snapshot(args.save_dir)
        print(f'[exp] finetune checkpoint={ft_ckpt}', flush=True)

        print('[exp] === SQRL finetune eval ===', flush=True)
        for seed in seeds:
            _stabilize(robot_cfg, train_cfg)
            print(f'[exp] finetune eval start seed={seed}', flush=True)
            row = _eval_policy(
                robot_cfg=robot_cfg, train_cfg=train_cfg, droq_cfg=droq_cfg,
                checkpoint=str(ft_ckpt), episodes=args.play_episodes,
                action_noise_std=args.action_noise_std, rollout_seed=seed,
                use_sqrl=True, epsilon=train_cfg.sqrl_epsilon,
                num_candidates=train_cfg.sqrl_num_candidates)
            row['cell'] = f'sqrl_finetune_seed{seed}'
            results.append(row)
            print(f'[exp] {row["cell"]} falls={row["falls"]} '
                  f'len={row["average_episode_length"]:.1f} '
                  f'return={row["average_return"]:.1f} '
                  f'no_safe={row["no_safe_rate"]:.3f}', flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        'sac_checkpoint': args.sac_checkpoint,
        'pretrain_checkpoint': str(pre_ckpt),
        'finetune_checkpoint': str(ft_ckpt) if ft_ckpt else None,
        'pretrain_steps': args.pretrain_steps,
        'finetune_steps': 0 if args.skip_finetune else args.finetune_steps,
        'action_noise_std': args.action_noise_std,
        'results': results,
    }
    out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'[exp] summary={out}', flush=True)

    # Compact verdict
    def _agg(prefix: str):
        rows = [r for r in results if r['cell'].startswith(prefix)]
        if not rows:
            return None
        return {
            'falls': sum(r['falls'] for r in rows),
            'episodes': sum(r['episodes'] for r in rows),
            'mean_len': float(np.mean([r['average_episode_length'] for r in rows])),
            'mean_return': float(np.mean([r['average_return'] for r in rows])),
        }

    sac = _agg('sac_')
    pre = _agg('sqrl_pretrain_')
    ft = _agg('sqrl_finetune_')
    print('[exp] VERDICT', json.dumps({'sac': sac, 'sqrl_pretrain': pre,
                                       'sqrl_finetune': ft}, indent=2),
          flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
