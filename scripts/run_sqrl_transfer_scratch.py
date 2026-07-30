#!/usr/bin/env python3
"""Paper-style SQRL: from-scratch slow pretrain → faster finetune.

Protocol (Srinivasan et al. 2020 Minitaur analogue on Go2):
  T_pre:    scene_empty, move_speed=0.30, sqrl_pretrain --from-scratch
  T_target: scene_empty, move_speed=0.40, sqrl_finetune (+ Lagrange nu)

Control: SAC transfer with the same speed schedule and step budgets, no Q_safe.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
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
from train.main import _configure_sqrl_mode, apply_move_speed

_REPO = Path(__file__).resolve().parents[1]
_CTRL_BIN = _REPO / 'controller' / 'build' / 'go2_control'
_CTRL_CFG = _REPO / 'config' / 'go2.yaml'
_SOCK = Path('/tmp/go2_policy.sock')


def _bounce_controller(reason: str = '') -> None:
    """Kill and restart go2_control to clear stuck Unix sockets."""
    print(f'[xfer] bouncing controller{" (" + reason + ")" if reason else ""}',
          flush=True)
    try:
        out = subprocess.check_output(
            ['pgrep', '-f', 'go2_control.*go2.yaml'], text=True)
        for pid in out.strip().split():
            try:
                os.kill(int(pid), signal.SIGTERM)
            except (ProcessLookupError, ValueError):
                pass
    except subprocess.CalledProcessError:
        pass
    time.sleep(1.5)
    if _SOCK.exists():
        try:
            _SOCK.unlink()
        except OSError:
            pass
    if not _CTRL_BIN.is_file():
        raise RuntimeError(f'missing controller binary: {_CTRL_BIN}')
    subprocess.Popen(
        [str(_CTRL_BIN), str(_CTRL_CFG)],
        cwd=str(_REPO),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True)
    for _ in range(40):
        if _SOCK.exists():
            print('[xfer] controller ready', flush=True)
            return
        time.sleep(0.25)
    raise RuntimeError('controller sock did not appear after bounce')


def _stabilize(robot_cfg, train_cfg, attempts: int = 8) -> None:
    env = prepare_env(
        build_legacy_env(robot_cfg, train_cfg, train_cfg.seed),
        rescale_actions=False, seed=train_cfg.seed)
    try:
        for attempt in range(attempts):
            try:
                env.reset(standup=True, with_recovery=True, grace_period=True)
                print(f'[xfer] stabilized attempt={attempt}', flush=True)
                time.sleep(1.0)
                return
            except UnstableResetError:
                time.sleep(2.0)
        raise RuntimeError('Could not stabilize robot')
    finally:
        env.close()


def _eval_policy(*, robot_cfg, train_cfg, droq_cfg, checkpoint: str,
                 episodes: int, action_noise_std: float, rollout_seed: int,
                 use_sqrl: bool, epsilon: float, num_candidates: int,
                 noise_mode: str = 'candidate') -> dict:
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
                    # candidate: noise before Q_safe (SQRL can reject unsafe).
                    # post: legacy post-filter noise (defeats the constraint).
                    cand_noise = (
                        float(action_noise_std)
                        if noise_mode == 'candidate' else 0.0)
                    action, info, sqrl_rng = select_sqrl_action(
                        agent, safety, observation, sqrl_rng,
                        num_candidates=num_candidates,
                        epsilon_safe=epsilon,
                        candidate_noise_std=cand_noise)
                    no_safe_steps += int(info['sqrl_no_safe_candidate'] > 0.5)
                    if noise_mode == 'post' and action_noise_std > 0.0:
                        action = np.clip(
                            action + rng.normal(
                                0.0, action_noise_std, size=action.shape),
                            -1.0, 1.0).astype(np.float32)
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
        'noise_mode': noise_mode if use_sqrl else 'post',
        'move_speed': float(robot_cfg.move_speed),
        'episodes': episodes,
        'falls': falls,
        'average_episode_length': float(np.mean(ep_lengths)),
        'average_return': float(np.mean(ep_returns)),
        'no_safe_rate': (
            float(no_safe_steps / max(total_steps, 1)) if use_sqrl else 0.0),
        'episode_lengths': ep_lengths,
        'episode_returns': ep_returns,
    }


def _heldout_fixed_speed(*, robot_cfg, train_cfg, droq_cfg, checkpoint: str,
                         max_steps: int, action_noise_std: float,
                         rollout_seed: int, use_sqrl: bool, epsilon: float,
                         num_candidates: int,
                         noise_mode: str = 'candidate',
                         log_qsafe: bool = False,
                         sample_policy: bool = False) -> dict:
    """Short held-out rollout at fixed move_speed; no gradient updates.

    Counts fallen terminations over ``max_steps`` policy steps.
    """
    # Disable curriculum so move_speed stays at robot_cfg value.
    train_cfg.cmd_speed_curriculum = False

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
        future_loss_weight=train_cfg.safety_future_loss_weight,
        ensemble_size=train_cfg.safety_critic_ensemble_size)
    print(f'[heldout] loading {checkpoint} speed={robot_cfg.move_speed}',
          flush=True)
    snapshot = restore_training_snapshot(
        Path(checkpoint), agent=agent,
        safety_critic=safety if (use_sqrl or log_qsafe) else None)
    agent = snapshot['agent']
    if sample_policy:
        # A restored checkpoint also restores the actor RNG. Re-seed held-out
        # policy sampling so different rollout_seed values are true stochastic
        # replicates instead of replaying the same action-noise sequence.
        agent = agent.replace(rng=jax.random.PRNGKey(rollout_seed))
    if use_sqrl or log_qsafe:
        if 'safety_critic' not in snapshot:
            env.close()
            raise RuntimeError(f'{checkpoint} missing safety_critic_state')
        safety = snapshot['safety_critic']

    rng = np.random.default_rng(rollout_seed)
    sqrl_rng = jax.random.PRNGKey(rollout_seed + 1)
    falls = 0
    no_safe_steps = 0
    total_steps = 0
    vel_sum = 0.0
    vel_n = 0
    qsafe_sum = 0.0
    qsafe_disagreement_sum = 0.0
    qsafe_n = 0
    emergency_steps = 0
    previous_action = np.zeros(
        env.action_space.shape, dtype=np.float32)
    reset_kwargs = {'standup': True, 'grace_period': True}
    try:
        observation = env.reset(**reset_kwargs)
        while total_steps < int(max_steps):
            if use_sqrl:
                cand_noise = (
                    float(action_noise_std)
                    if noise_mode == 'candidate' else 0.0)
                action, info, sqrl_rng = select_sqrl_action(
                    agent, safety, observation, sqrl_rng,
                    num_candidates=num_candidates,
                    epsilon_safe=epsilon,
                    candidate_noise_std=cand_noise,
                    previous_action=previous_action,
                    local_candidate_count=train_cfg.sqrl_local_candidate_count,
                    local_action_std=train_cfg.sqrl_local_action_std,
                    fallback_contraction=train_cfg.sqrl_fallback_contraction,
                    fallback_emergency_risk=(
                        train_cfg.sqrl_fallback_emergency_risk),
                    uncertainty_penalty=train_cfg.sqrl_uncertainty_penalty)
                no_safe_steps += int(info['sqrl_no_safe_candidate'] > 0.5)
                emergency_steps += int(
                    info['sqrl_emergency_supervisor'] > 0.5)
                if noise_mode == 'post' and action_noise_std > 0.0:
                    action = np.clip(
                        action + rng.normal(
                            0.0, action_noise_std, size=action.shape),
                        -1.0, 1.0).astype(np.float32)
                qsafe_sum += float(info['selected_Q_safe'])
                qsafe_disagreement_sum += float(
                    info['candidate_Q_safe_disagreement_mean'])
                qsafe_n += 1
            else:
                if sample_policy:
                    action, agent = agent.sample_actions(observation)
                    action = np.clip(action, -1.0, 1.0)
                else:
                    action = np.clip(
                        agent.eval_actions(observation), -1.0, 1.0)
                if not sample_policy and action_noise_std > 0.0:
                    action = np.clip(
                        action + rng.normal(
                            0.0, action_noise_std, size=action.shape),
                        -1.0, 1.0).astype(np.float32)
                if log_qsafe:
                    q_value, disagreement = safety.predict_with_uncertainty(
                        observation[None, :], action[None, :])
                    qsafe_sum += float(q_value[0])
                    qsafe_disagreement_sum += float(disagreement[0])
                    qsafe_n += 1
            previous_action = np.asarray(action, dtype=np.float32).copy()
            observation, _reward, done, last_info = env.step(action)
            if last_info.get('policy_step', True):
                total_steps += 1
                fv = last_info.get('forward_velocity')
                if fv is not None and np.isfinite(fv):
                    vel_sum += float(fv)
                    vel_n += 1
            if done:
                fell = bool(
                    last_info.get('terminated', False)
                    or last_info.get('unsafe_label', False))
                falls += int(fell)
                observation = env.reset(
                    standup=bool(last_info.get('terminated', False)
                                 or last_info.get('standup_timed_out', False)),
                    with_recovery=bool(last_info.get('is_belly_up', False)),
                    grace_period=not bool(last_info.get('truncated', False)),
                    preserve_policy_state=bool(
                        last_info.get('truncated', False)),
                )
                previous_action = np.zeros(
                    env.action_space.shape, dtype=np.float32)
    finally:
        env.close()
    return {
        'checkpoint': checkpoint,
        'phase': 'heldout',
        'use_sqrl': use_sqrl,
        'move_speed': float(robot_cfg.move_speed),
        'max_steps': int(max_steps),
        'steps': total_steps,
        'falls': falls,
        'mean_forward_vel': (vel_sum / vel_n) if vel_n else None,
        'no_safe_rate': (
            float(no_safe_steps / max(total_steps, 1)) if use_sqrl else 0.0),
        'emergency_supervisor_rate': float(
            emergency_steps / max(total_steps, 1)),
        'mean_Q_safe': (
            float(qsafe_sum / qsafe_n) if qsafe_n else None),
        'mean_Q_safe_disagreement': (
            float(qsafe_disagreement_sum / qsafe_n) if qsafe_n else None),
        'rollout_seed': rollout_seed,
        'action_noise_std': action_noise_std,
        'sample_policy': bool(sample_policy),
    }


def _fresh_dir(path: str | Path) -> Path:
    root = Path(path)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _scale_schedule_for_short_run(cfg, n_steps: int) -> None:
    """Shrink explore/SQRL warmup so tiny smoke budgets still exercise updates."""
    if n_steps < int(cfg.start_training):
        cfg.start_training = max(n_steps // 4, 1)
    if n_steps < int(cfg.sqrl_activation_steps):
        cfg.sqrl_activation_steps = max(n_steps // 4, 1)
        cfg.sqrl_epsilon_anneal_steps = max(n_steps // 4, 1)


class _tee_stdout:
    """Duplicate process stdout/stderr into a stage log file."""

    def __init__(self, path: Path):
        self.path = path
        self._fp = None
        self._stdout = None
        self._stderr = None

    def write(self, data):
        self._stdout.write(data)
        self._fp.write(data)
        self._fp.flush()

    def flush(self):
        self._stdout.flush()
        self._fp.flush()

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.path, 'w', encoding='utf-8')
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._fp.close()
        return False


def _qsafe_log_health(log_path: Path | None, *, min_auroc: float = 0.70,
                      min_gap: float = 0.05) -> dict:
    """Parse train stdout for last Q_safe metrics + constraint activation."""
    info = {
        'constraint_on': False,
        'last_auroc': None,
        'last_pos': None,
        'last_neg': None,
        'last_gap': None,
        'ok': False,
        'n_qsafe_lines': 0,
    }
    if log_path is None or not log_path.is_file():
        return info
    text = log_path.read_text(encoding='utf-8', errors='replace')
    info['constraint_on'] = 'SQRL constraint ON' in text
    pat = re.compile(
        r'pos=(?P<pos>[-+eE0-9.]+)\s+neg=(?P<neg>[-+eE0-9.]+).*?'
        r'auroc=(?P<auroc>[-+eE0-9.]+)')
    matches = list(pat.finditer(text))
    info['n_qsafe_lines'] = len(matches)
    if not matches:
        return info
    m = matches[-1]
    pos = float(m.group('pos'))
    neg = float(m.group('neg'))
    auroc = float(m.group('auroc'))
    gap = pos - neg
    info.update({
        'last_auroc': auroc,
        'last_pos': pos,
        'last_neg': neg,
        'last_gap': gap,
        'ok': (
            np.isfinite(auroc) and np.isfinite(gap)
            and auroc >= min_auroc and gap >= min_gap),
    })
    return info


def _agg(results: list[dict], prefix: str):
    rows = [r for r in results if str(r.get('cell', '')).startswith(prefix)]
    if not rows:
        return None
    return {
        'falls': sum(r['falls'] for r in rows),
        'episodes': sum(r['episodes'] for r in rows),
        'mean_len': float(np.mean([r['average_episode_length'] for r in rows])),
        'mean_return': float(np.mean([r['average_return'] for r in rows])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/go2.yaml')
    parser.add_argument('--pre-speed', type=float, default=0.30)
    parser.add_argument('--ft-speed', type=float, default=0.40)
    parser.add_argument('--pretrain-steps', type=int, default=12000)
    parser.add_argument('--finetune-steps', type=int, default=4000)
    parser.add_argument(
        '--sqrl-pre-dir', default='saved/checkpoints_sqrl_transfer_pre')
    parser.add_argument(
        '--sqrl-ft-dir', default='saved/checkpoints_sqrl_transfer_ft')
    parser.add_argument(
        '--sac-pre-dir', default='saved/checkpoints_sac_transfer_pre')
    parser.add_argument(
        '--sac-ft-dir', default='saved/checkpoints_sac_transfer_ft')
    parser.add_argument('--skip-sqrl', action='store_true')
    parser.add_argument(
        '--skip-sqrl-pre', action='store_true',
        help='Reuse existing SQRL slow pretrain checkpoint; only run SQRL ft')
    parser.add_argument('--skip-sac', action='store_true')
    parser.add_argument(
        '--skip-sac-pre', action='store_true',
        help='Reuse existing SAC slow pretrain checkpoint; only run SAC ft')
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument(
        '--eval-existing', action='store_true',
        help='Skip all training; evaluate whatever checkpoints exist')
    parser.add_argument('--play-episodes', type=int, default=2)
    parser.add_argument('--action-noise-std', type=float, default=0.50)
    parser.add_argument(
        '--noise-mode', choices=('candidate', 'post'), default='candidate',
        help='SQRL: apply action noise before (candidate) or after (post) '
             'Q_safe filtering')
    parser.add_argument('--eval-epsilon', type=float, default=None,
                        help='Override sqrl_epsilon at eval')
    parser.add_argument('--rollout-seeds', default='9010,9011')
    parser.add_argument(
        '--output',
        default='saved/safety_evaluation/sqrl_transfer_scratch_summary.json')
    parser.add_argument(
        '--keep-dirs', action='store_true',
        help='Do not wipe stage save_dirs before training')
    parser.add_argument(
        '--no-bounce', action='store_true',
        help='Do not restart go2_control between stages')
    parser.add_argument(
        '--stage-log-dir',
        default='saved/safety_evaluation/sqrl_transfer_stage_logs')
    parser.add_argument(
        '--wandb', action='store_true',
        help='Enable Weights & Biases for training stages')
    parser.add_argument(
        '--wandb-prefix', default='xfer',
        help='Prefix for wandb run names')
    args = parser.parse_args()

    if args.ft_speed <= args.pre_speed:
        raise SystemExit('--ft-speed must be greater than --pre-speed')

    seeds = [int(x) for x in args.rollout_seeds.split(',') if x.strip()]
    n_pre = int(args.pretrain_steps)
    n_ft = int(args.finetune_steps)
    results: list[dict] = []
    qsafe_health: dict = {}
    sqrl_pre_ckpt = None
    sqrl_ft_ckpt = None
    sac_pre_ckpt = None
    sac_ft_ckpt = None
    stage_log_dir = Path(args.stage_log_dir)
    stage_log_dir.mkdir(parents=True, exist_ok=True)
    if args.eval_existing:
        args.skip_sqrl = True
        args.skip_sac = True

    def _prep_stage(label: str):
        if not args.no_bounce:
            _bounce_controller(label)

    # --- SQRL from-scratch pretrain (slow) ---
    if not args.skip_sqrl:
        if args.skip_sqrl_pre:
            print('[xfer] === SQRL pretrain (skipped) ===', flush=True)
            sqrl_pre_ckpt = latest_snapshot(args.sqrl_pre_dir)
            if sqrl_pre_ckpt is None:
                raise RuntimeError(
                    f'No SQRL pre checkpoint in {args.sqrl_pre_dir}')
            print(f'[xfer] SQRL pre checkpoint={sqrl_pre_ckpt}', flush=True)
        else:
            print('[xfer] === SQRL pretrain (from-scratch, slow) ===',
                  flush=True)
            robot_cfg, train_cfg, droq_cfg = load_app_config(path=args.config)
            robot_cfg = apply_move_speed(robot_cfg, args.pre_speed)
            droq_cfg = dict(droq_cfg)
            if not args.keep_dirs:
                _fresh_dir(args.sqrl_pre_dir)
            _prep_stage('before SQRL pre')
            _stabilize(robot_cfg, train_cfg)
            ns = argparse.Namespace(
                mode='sqrl_pretrain', checkpoint=None,
                save_dir=args.sqrl_pre_dir, from_scratch=True)
            pre_cfg, pre_droq = _configure_sqrl_mode(ns, train_cfg, droq_cfg)
            pre_cfg.max_steps = n_pre
            pre_cfg.checkpoint_interval = min(1000, max(n_pre // 4, 1))
            pre_cfg.warmup = True
            pre_cfg.wandb = bool(args.wandb)
            pre_cfg.wandb_run_name = f'{args.wandb_prefix}_sqrl_pre'
            _scale_schedule_for_short_run(pre_cfg, n_pre)
            print(f'[xfer] SQRL pre move_speed={robot_cfg.move_speed} '
                  f'max_steps={pre_cfg.max_steps} save_dir={pre_cfg.save_dir} '
                  f'warm_start={pre_cfg.warm_start_checkpoint}',
                  flush=True)
            pre_log = stage_log_dir / 'sqrl_pre.log'
            with _tee_stdout(pre_log):
                run_in_process(robot_cfg, pre_cfg, pre_droq)
            sqrl_pre_ckpt = latest_snapshot(args.sqrl_pre_dir)
            if sqrl_pre_ckpt is None:
                raise RuntimeError('SQRL pretrain produced no snapshot')
            print(f'[xfer] SQRL pre checkpoint={sqrl_pre_ckpt}', flush=True)
            qsafe_health['pre'] = _qsafe_log_health(pre_log)
            print(f'[xfer] Q_safe health (pre)={qsafe_health["pre"]}',
                  flush=True)
            if not qsafe_health['pre'].get('ok'):
                raise RuntimeError(
                    'SQRL pretrain finished but Q_safe is not discriminative '
                    f'({qsafe_health["pre"]}). Aborting before finetune.')

        print('[xfer] === SQRL finetune (fast + nu) ===', flush=True)
        robot_cfg, train_cfg, droq_cfg = load_app_config(path=args.config)
        robot_cfg = apply_move_speed(robot_cfg, args.ft_speed)
        droq_cfg = dict(droq_cfg)
        if not args.keep_dirs:
            _fresh_dir(args.sqrl_ft_dir)
        _prep_stage('before SQRL ft')
        _stabilize(robot_cfg, train_cfg)
        ns = argparse.Namespace(
            mode='sqrl_finetune', checkpoint=str(sqrl_pre_ckpt),
            save_dir=args.sqrl_ft_dir, from_scratch=False)
        ft_cfg, ft_droq = _configure_sqrl_mode(ns, train_cfg, droq_cfg)
        # Warm-start into a fresh dir; continue for N_ft steps past pre step.
        ft_cfg.resume_checkpoint = False
        ft_cfg.warm_start_checkpoint = str(sqrl_pre_ckpt)
        ft_cfg.max_steps = n_pre + n_ft
        ft_cfg.checkpoint_interval = min(1000, max(n_ft // 4, 1))
        # Target-speed phase should constrain sooner; critic is already trained.
        ft_cfg.sqrl_activation_steps = min(int(ft_cfg.sqrl_activation_steps), 200)
        ft_cfg.sqrl_epsilon_anneal_steps = min(
            int(ft_cfg.sqrl_epsilon_anneal_steps), 500)
        ft_cfg.warmup = True
        ft_cfg.wandb = bool(args.wandb)
        ft_cfg.wandb_run_name = f'{args.wandb_prefix}_sqrl_ft'
        _scale_schedule_for_short_run(ft_cfg, n_ft)
        print(f'[xfer] SQRL ft move_speed={robot_cfg.move_speed} '
              f'max_steps={ft_cfg.max_steps} save_dir={ft_cfg.save_dir} '
              f'activation={ft_cfg.sqrl_activation_steps}',
              flush=True)
        ft_log = stage_log_dir / 'sqrl_ft.log'
        with _tee_stdout(ft_log):
            run_in_process(robot_cfg, ft_cfg, ft_droq)
        sqrl_ft_ckpt = latest_snapshot(args.sqrl_ft_dir)
        print(f'[xfer] SQRL ft checkpoint={sqrl_ft_ckpt}', flush=True)
        qsafe_health['ft'] = _qsafe_log_health(ft_log)
        print(f'[xfer] Q_safe health (ft)={qsafe_health["ft"]}', flush=True)
    else:
        print('[xfer] === SQRL stages (skipped) ===', flush=True)
        sqrl_pre_ckpt = latest_snapshot(args.sqrl_pre_dir)
        sqrl_ft_ckpt = latest_snapshot(args.sqrl_ft_dir)

    # --- SAC transfer control ---
    if not args.skip_sac:
        if not args.skip_sac_pre:
            print('[xfer] === SAC pretrain (from-scratch, slow) ===',
                  flush=True)
            robot_cfg, train_cfg, droq_cfg = load_app_config(path=args.config)
            robot_cfg = apply_move_speed(robot_cfg, args.pre_speed)
            droq_cfg = dict(droq_cfg)
            if not args.keep_dirs:
                _fresh_dir(args.sac_pre_dir)
            _prep_stage('before SAC pre')
            _stabilize(robot_cfg, train_cfg)
            sac_pre = train_cfg
            sac_pre.sqrl_enabled = False
            sac_pre.safety_critic_enabled = False
            sac_pre.experiment_name = 'sac_transfer_pre'
            sac_pre.save_dir = args.sac_pre_dir
            sac_pre.max_steps = n_pre
            sac_pre.checkpoint_interval = min(1000, max(n_pre // 4, 1))
            sac_pre.warmup = True
            sac_pre.wandb = bool(args.wandb)
            sac_pre.wandb_run_name = f'{args.wandb_prefix}_sac_pre'
            sac_pre.resume_checkpoint = False
            sac_pre.warm_start_checkpoint = None
            _scale_schedule_for_short_run(sac_pre, n_pre)
            print(f'[xfer] SAC pre move_speed={robot_cfg.move_speed} '
                  f'max_steps={sac_pre.max_steps}', flush=True)
            sac_pre_log = stage_log_dir / 'sac_pre.log'
            with _tee_stdout(sac_pre_log):
                run_in_process(robot_cfg, sac_pre, droq_cfg)
            sac_pre_ckpt = latest_snapshot(args.sac_pre_dir)
            if sac_pre_ckpt is None:
                raise RuntimeError('SAC pretrain produced no snapshot')
            print(f'[xfer] SAC pre checkpoint={sac_pre_ckpt}', flush=True)
        else:
            print('[xfer] === SAC pretrain (skipped) ===', flush=True)
            sac_pre_ckpt = latest_snapshot(args.sac_pre_dir)
            if sac_pre_ckpt is None:
                raise RuntimeError(
                    f'No SAC pre checkpoint in {args.sac_pre_dir}')
            print(f'[xfer] SAC pre checkpoint={sac_pre_ckpt}', flush=True)

        print('[xfer] === SAC finetune (fast, no SQRL) ===', flush=True)
        robot_cfg, train_cfg, droq_cfg = load_app_config(path=args.config)
        robot_cfg = apply_move_speed(robot_cfg, args.ft_speed)
        droq_cfg = dict(droq_cfg)
        if not args.keep_dirs:
            _fresh_dir(args.sac_ft_dir)
        _prep_stage('before SAC ft')
        _stabilize(robot_cfg, train_cfg)
        sac_ft = train_cfg
        sac_ft.sqrl_enabled = False
        sac_ft.safety_critic_enabled = False
        sac_ft.experiment_name = 'sac_transfer_ft'
        sac_ft.save_dir = args.sac_ft_dir
        sac_ft.max_steps = n_pre + n_ft
        sac_ft.checkpoint_interval = min(1000, max(n_ft // 4, 1))
        sac_ft.warmup = True
        sac_ft.wandb = bool(args.wandb)
        sac_ft.wandb_run_name = f'{args.wandb_prefix}_sac_ft'
        sac_ft.resume_checkpoint = False
        sac_ft.warm_start_checkpoint = str(sac_pre_ckpt)
        _scale_schedule_for_short_run(sac_ft, n_ft)
        print(f'[xfer] SAC ft move_speed={robot_cfg.move_speed} '
              f'max_steps={sac_ft.max_steps}', flush=True)
        sac_ft_log = stage_log_dir / 'sac_ft.log'
        with _tee_stdout(sac_ft_log):
            run_in_process(robot_cfg, sac_ft, droq_cfg)
        sac_ft_ckpt = latest_snapshot(args.sac_ft_dir)
        print(f'[xfer] SAC ft checkpoint={sac_ft_ckpt}', flush=True)
    else:
        print('[xfer] === SAC stages (skipped) ===', flush=True)
        sac_pre_ckpt = latest_snapshot(args.sac_pre_dir)
        sac_ft_ckpt = latest_snapshot(args.sac_ft_dir)

    # --- Held-out eval on target speed ---
    if not args.skip_eval:
        robot_cfg, train_cfg, droq_cfg = load_app_config(path=args.config)
        robot_cfg = apply_move_speed(robot_cfg, args.ft_speed)
        droq_cfg = dict(droq_cfg)
        eval_eps = (
            float(args.eval_epsilon)
            if args.eval_epsilon is not None
            else float(train_cfg.sqrl_epsilon))

        if sqrl_ft_ckpt is not None:
            print(
                f'[xfer] === SQRL finetune eval (target speed, '
                f'noise_mode={args.noise_mode}, eps={eval_eps}) ===',
                flush=True)
            for seed in seeds:
                _prep_stage(f'before SQRL eval seed={seed}')
                _stabilize(robot_cfg, train_cfg)
                row = _eval_policy(
                    robot_cfg=robot_cfg, train_cfg=train_cfg, droq_cfg=droq_cfg,
                    checkpoint=str(sqrl_ft_ckpt), episodes=args.play_episodes,
                    action_noise_std=args.action_noise_std, rollout_seed=seed,
                    use_sqrl=True, epsilon=eval_eps,
                    num_candidates=train_cfg.sqrl_num_candidates,
                    noise_mode=args.noise_mode)
                row['cell'] = f'sqrl_ft_seed{seed}'
                results.append(row)
                print(f'[xfer] {row["cell"]} falls={row["falls"]} '
                      f'len={row["average_episode_length"]:.1f} '
                      f'return={row["average_return"]:.1f} '
                      f'no_safe={row["no_safe_rate"]:.3f}', flush=True)

        # Eval SAC whenever a finetune checkpoint exists (including
        # --eval-existing). Training-only --skip-sac without eval-existing
        # still skips stale SAC comparison.
        eval_sac = sac_ft_ckpt is not None and (
            args.eval_existing or not args.skip_sac)
        if eval_sac:
            print('[xfer] === SAC finetune eval (target speed) ===', flush=True)
            for seed in seeds:
                _prep_stage(f'before SAC eval seed={seed}')
                _stabilize(robot_cfg, train_cfg)
                row = _eval_policy(
                    robot_cfg=robot_cfg, train_cfg=train_cfg, droq_cfg=droq_cfg,
                    checkpoint=str(sac_ft_ckpt), episodes=args.play_episodes,
                    action_noise_std=args.action_noise_std, rollout_seed=seed,
                    use_sqrl=False, epsilon=eval_eps,
                    num_candidates=train_cfg.sqrl_num_candidates)
                row['cell'] = f'sac_ft_seed{seed}'
                results.append(row)
                print(f'[xfer] {row["cell"]} falls={row["falls"]} '
                      f'len={row["average_episode_length"]:.1f} '
                      f'return={row["average_return"]:.1f}', flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        'protocol': 'sqrl_from_scratch_slow_to_fast',
        'pre_speed': args.pre_speed,
        'ft_speed': args.ft_speed,
        'pretrain_steps': n_pre,
        'finetune_steps': n_ft,
        'sqrl_pre_checkpoint': str(sqrl_pre_ckpt) if sqrl_pre_ckpt else None,
        'sqrl_ft_checkpoint': str(sqrl_ft_ckpt) if sqrl_ft_ckpt else None,
        'sac_pre_checkpoint': str(sac_pre_ckpt) if sac_pre_ckpt else None,
        'sac_ft_checkpoint': str(sac_ft_ckpt) if sac_ft_ckpt else None,
        'action_noise_std': args.action_noise_std,
        'noise_mode': args.noise_mode,
        'qsafe_health': qsafe_health,
        'results': results,
        'verdict': {
            'sqrl_ft': _agg(results, 'sqrl_ft_'),
            'sac_ft': _agg(results, 'sac_ft_'),
        },
    }
    out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'[xfer] summary={out}', flush=True)
    print('[xfer] VERDICT', json.dumps(summary['verdict'], indent=2),
          flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
