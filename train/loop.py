"""Online DROQ training loop."""

from __future__ import annotations

import os
import time
from collections import deque

import numpy as np

try:
    import tqdm as tqdm_module
except ImportError:
    tqdm_module = None

from pathlib import Path

from train.config import TrainConfig
from train.logging import TrainLogger
from train.profiling import StepProfiler
from train.rolling_metrics import RollingTrainingSummary
from train.speed_curriculum import PerformanceSpeedCurriculum
from collector.transition_builder import build_transition
from learner.checkpoint import (experiments_compatible,
                                has_legacy_agent_checkpoint, latest_snapshot,
                                load_training_snapshot_metadata,
                                restore_training_snapshot,
                                save_training_snapshot)
from jaxrl.env.evaluation import evaluate
from jaxrl.data.safety_replay import SafetyReplayManager
from jaxrl.agents.safety_critic import (SafetyCritic,
                                        binary_prediction_metrics)
from jaxrl.agents.sqrl import select_sqrl_action
import jax


def _log(msg: str) -> None:
    print(msg, flush=True)


def _to_float_dict(info: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in info.items():
        if isinstance(v, (int, float, np.floating)):
            out[k] = float(v)
        elif np.asarray(v).shape == ():
            out[k] = float(np.asarray(v))
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


def _sqrl_gate_decision(latest_safety_info: dict | None,
                        candidate_no_safe: deque[float],
                        candidate_ranges: deque[float],
                        cfg: TrainConfig,
                        control_metrics: dict | None = None
                        ) -> tuple[bool, str]:
    if latest_safety_info is None:
        return False, 'no-natural-calibration'
    required = (
        'Q_safe_AUROC', 'Q_safe_label_pos', 'Q_safe_label_neg',
        'Q_safe_calibration_ece', 'Q_safe_brier',
        'Q_safe_num_samples', 'Q_safe_positive_rate')
    if any(key not in latest_safety_info for key in required):
        return False, 'incomplete-natural-calibration'
    samples = float(latest_safety_info['Q_safe_num_samples'])
    positive_rate = float(latest_safety_info['Q_safe_positive_rate'])
    if samples < float(cfg.sqrl_min_gate_samples):
        return False, 'too-few-calibration-samples'
    if not 0.0 < positive_rate < 1.0:
        return False, 'calibration-missing-class'
    auroc = float(latest_safety_info['Q_safe_AUROC'])
    gap = (
        float(latest_safety_info['Q_safe_label_pos'])
        - float(latest_safety_info['Q_safe_label_neg']))
    ece = float(latest_safety_info['Q_safe_calibration_ece'])
    brier = float(latest_safety_info['Q_safe_brier'])
    values = (auroc, gap, ece, brier)
    if not all(np.isfinite(value) for value in values):
        return False, 'non-finite-calibration'
    if auroc < float(cfg.sqrl_min_auroc):
        return False, 'auroc'
    if gap < float(cfg.sqrl_min_pos_neg_gap):
        return False, 'pos-neg-gap'
    if ece > float(cfg.sqrl_max_ece):
        return False, 'ece'
    if brier > float(cfg.sqrl_max_brier):
        return False, 'brier'
    required_window = max(int(cfg.sqrl_gate_candidate_window), 1)
    if (len(candidate_no_safe) < required_window
            or len(candidate_ranges) < required_window):
        return False, 'candidate-window'
    if float(np.mean(candidate_no_safe)) > float(cfg.sqrl_max_no_safe_rate):
        return False, 'no-safe-rate'
    if float(np.mean(candidate_ranges)) < float(
            cfg.sqrl_min_candidate_range):
        return False, 'candidate-range'
    if cfg.sqrl_control_gate_required:
        if control_metrics is None:
            return False, 'no-control-evaluation'
        control_required = (
            'control_pairwise_risk_ranking_accuracy',
            'control_selected_false_safe_rate',
            'control_coverage',
            'control_nominal_relative_failure_reduction')
        if any(key not in control_metrics for key in control_required):
            return False, 'incomplete-control-evaluation'
        pairwise, false_safe, coverage, reduction = (
            float(control_metrics[control_required[0]]),
            float(control_metrics[control_required[1]]),
            float(control_metrics[control_required[2]]),
            float(control_metrics[control_required[3]]))
        if not all(np.isfinite(value) for value in (
                pairwise, false_safe, coverage, reduction)):
            return False, 'non-finite-control-evaluation'
        if pairwise < cfg.sqrl_control_min_pairwise_accuracy:
            return False, 'control-pairwise-ranking'
        if false_safe > cfg.sqrl_control_max_false_safe_rate:
            return False, 'control-false-safe'
        if coverage < cfg.sqrl_control_min_coverage:
            return False, 'control-coverage'
        if reduction < cfg.sqrl_control_min_failure_reduction:
            return False, 'control-failure-reduction'
    return True, 'ready'


def _snapshot_metadata(cfg: TrainConfig, env=None) -> dict:
    metadata = {
        'experiment_name': cfg.experiment_name,
        'start_training': cfg.start_training,
        'batch_size': cfg.batch_size,
        'utd_ratio': cfg.utd_ratio,
        'seed': cfg.seed,
    }
    if env is not None:
        metadata['obs_dim'] = int(env.observation_space.shape[0])
    return metadata


def _validate_snapshot_metadata(path, metadata: dict, env) -> None:
    snapshot_obs_dim = metadata.get('obs_dim')
    if snapshot_obs_dim is None:
        return
    current_obs_dim = int(env.observation_space.shape[0])
    if int(snapshot_obs_dim) != current_obs_dim:
        raise RuntimeError(
            'Refusing to restore an incompatible training snapshot: '
            f'{path} has obs_dim={snapshot_obs_dim}, '
            f'current obs_dim={current_obs_dim}. Start a new save_dir or '
            'switch back to the code/config that produced this snapshot.')


def _apply_agent_update(agent, batch, cfg: TrainConfig, source_step: int,
                        *, safety_critic=None):
    """Apply one complete UTD update and report whether the agent corrupted."""
    update_t0 = time.perf_counter()
    if not _batch_is_finite(batch):
        _log(f'[train] WARNING: non-finite batch at step {source_step}, '
             'skip update')
        return agent, None, False, time.perf_counter() - update_t0

    use_lagrange = (
        bool(cfg.sqrl_enabled)
        and str(cfg.sqrl_phase) == 'finetune'
        and safety_critic is not None)
    agent, update_info = agent.update(
        batch, cfg.utd_ratio,
        safety_critic=safety_critic if use_lagrange else None,
        epsilon_safe=float(cfg.sqrl_epsilon),
        sqrl_use_lagrange=use_lagrange)
    corrupted = (
        update_info is not None
        and not all(
            np.isfinite(float(v) if hasattr(v, 'item') else v)
            for v in update_info.values()
        )
    )
    if corrupted:
        _log(f'[train] WARNING: non-finite update at step {source_step}, '
             'skipping future updates until restart')
        update_info = None
    return agent, update_info, corrupted, time.perf_counter() - update_t0


def run_training(agent, env, replay_buffer, cfg: TrainConfig):
    os.makedirs(cfg.save_dir, exist_ok=True)

    safety_replay = None
    if cfg.safety_replay_enabled:
        safety_replay = SafetyReplayManager(
            recent_capacity=cfg.safety_recent_capacity,
            failure_capacity=cfg.safety_failure_capacity,
            boundary_capacity=cfg.safety_boundary_capacity,
            recovery_capacity=cfg.safety_recovery_capacity,
            all_capacity=cfg.buffer_size,
            failure_history=cfg.safety_failure_history,
            n_step=cfg.safety_critic_n_step,
            failure_horizons=cfg.safety_failure_horizons,
            seed=cfg.seed,
        )
    safety_critic = None
    safety_validator = None
    if cfg.safety_critic_enabled:
        if safety_replay is None:
            raise ValueError('safety_critic_enabled requires safety replay')
        safety_critic = SafetyCritic.create(
            seed=cfg.seed + 10_000,
            observation_dim=int(env.observation_space.shape[0]),
            action_dim=int(env.action_space.shape[0]),
            hidden_dims=cfg.safety_critic_hidden_dims,
            learning_rate=cfg.safety_critic_learning_rate,
            discount=cfg.safety_discount,
            tau=cfg.safety_critic_tau,
            future_loss_weight=cfg.safety_future_loss_weight,
            ensemble_size=cfg.safety_critic_ensemble_size,
            conservative_weight=cfg.safety_conservative_weight,
            conservative_num_actions=cfg.safety_conservative_num_actions)
        if cfg.sqrl_double_critic_enabled:
            # A completely separate initialization, parameter tree, optimizer
            # state, target network and RNG. This is intentionally not another
            # head on SafetyCritic A's shared representation.
            safety_validator = SafetyCritic.create(
                seed=cfg.seed + 20_000,
                observation_dim=int(env.observation_space.shape[0]),
                action_dim=int(env.action_space.shape[0]),
                hidden_dims=cfg.safety_critic_hidden_dims,
                learning_rate=cfg.safety_critic_learning_rate,
                discount=cfg.safety_discount,
                tau=cfg.safety_critic_tau,
                future_loss_weight=cfg.safety_future_loss_weight,
                ensemble_size=cfg.safety_critic_ensemble_size,
                conservative_weight=cfg.safety_conservative_weight,
                conservative_num_actions=(
                    cfg.safety_conservative_num_actions))

    start_i = 0
    sqrl_finetune = bool(cfg.sqrl_enabled) and str(cfg.sqrl_phase) == 'finetune'
    if cfg.save_checkpoints and cfg.resume_checkpoint and not cfg.benchmark_only:
        latest = latest_snapshot(cfg.save_dir)
        if latest is not None:
            metadata = load_training_snapshot_metadata(latest)
            _validate_snapshot_metadata(latest, metadata, env)
            snapshot_experiment = metadata.get('experiment_name')
            if not experiments_compatible(
                    snapshot_experiment, cfg.experiment_name):
                raise RuntimeError(
                    'Refusing to restore a snapshot from another experiment: '
                    f'snapshot={snapshot_experiment!r} '
                    f'current={cfg.experiment_name!r}')
            snapshot = restore_training_snapshot(
                latest, agent=agent, replay_buffer=replay_buffer,
                safety_replay=safety_replay, safety_critic=safety_critic,
                safety_validator=safety_validator)
            agent = snapshot['agent']
            replay_buffer = snapshot['replay_buffer']
            safety_critic = snapshot.get('safety_critic', safety_critic)
            safety_validator = snapshot.get(
                'safety_validator', safety_validator)
            start_i = int(snapshot['step'])
            _log(f'[train] resumed complete snapshot {latest} step {start_i}')
        elif has_legacy_agent_checkpoint(cfg.save_dir):
            raise RuntimeError(
                'Found legacy agent-only checkpoint in '
                f'{cfg.save_dir}. Online training requires an agent+replay '
                'snapshot. Delete the old checkpoint directory or start a new '
                'run from step 0.')
        elif cfg.warm_start_checkpoint:
            warm_path = Path(cfg.warm_start_checkpoint)
            if not warm_path.exists():
                raise RuntimeError(
                    f'warm_start_checkpoint not found: {warm_path}')
            metadata = load_training_snapshot_metadata(warm_path)
            _validate_snapshot_metadata(warm_path, metadata, env)
            # Load actor/replay from warm-start. Pretrain keeps a fresh Q_safe
            # and empty D_safe; finetune restores both from the SQRL snapshot.
            snapshot = restore_training_snapshot(
                warm_path, agent=agent, replay_buffer=replay_buffer,
                safety_replay=safety_replay if sqrl_finetune else None,
                safety_critic=safety_critic if sqrl_finetune else None,
                safety_validator=(
                    safety_validator if sqrl_finetune else None))
            agent = snapshot['agent']
            replay_buffer = snapshot.get('replay_buffer', replay_buffer)
            if sqrl_finetune:
                if 'safety_critic' not in snapshot:
                    raise RuntimeError(
                        'sqrl_finetune warm_start requires safety_critic_state '
                        f'in {warm_path}')
                safety_critic = snapshot['safety_critic']
                safety_validator = snapshot.get(
                    'safety_validator', safety_validator)
            start_i = int(snapshot['step'])
            _log(f'[train] warm-started from {warm_path} step {start_i} '
                 f'sqrl_phase={cfg.sqrl_phase} '
                 f'q_safe={"restored" if sqrl_finetune else "fresh"}')
    elif cfg.save_checkpoints and not cfg.benchmark_only:
        latest = latest_snapshot(cfg.save_dir)
        if latest is not None:
            _log(f'[train] starting from scratch; ignoring checkpoint {latest}')
        elif cfg.warm_start_checkpoint:
            warm_path = Path(cfg.warm_start_checkpoint)
            if not warm_path.exists():
                raise RuntimeError(
                    f'warm_start_checkpoint not found: {warm_path}')
            metadata = load_training_snapshot_metadata(warm_path)
            _validate_snapshot_metadata(warm_path, metadata, env)
            snapshot = restore_training_snapshot(
                warm_path, agent=agent, replay_buffer=replay_buffer,
                safety_replay=safety_replay if sqrl_finetune else None,
                safety_critic=safety_critic if sqrl_finetune else None,
                safety_validator=(
                    safety_validator if sqrl_finetune else None))
            agent = snapshot['agent']
            replay_buffer = snapshot.get('replay_buffer', replay_buffer)
            if sqrl_finetune:
                if 'safety_critic' not in snapshot:
                    raise RuntimeError(
                        'sqrl_finetune warm_start requires safety_critic_state '
                        f'in {warm_path}')
                safety_critic = snapshot['safety_critic']
                safety_validator = snapshot.get(
                    'safety_validator', safety_validator)
            start_i = int(snapshot['step'])
            _log(f'[train] warm-started from {warm_path} step {start_i} '
                 f'sqrl_phase={cfg.sqrl_phase} '
                 f'q_safe={"restored" if sqrl_finetune else "fresh"}')

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
            'experiment_name': cfg.experiment_name,
            'seed': cfg.seed,
            'max_steps': cfg.max_steps,
            'start_training': cfg.start_training,
            'batch_size': cfg.batch_size,
            'utd_ratio': cfg.utd_ratio,
            'metrics_interval': cfg.metrics_interval,
            'explore_action_scale': cfg.explore_action_scale,
            'control_frequency': control_frequency,
            'pipeline_updates': cfg.pipeline_updates,
            'resume_checkpoint': cfg.resume_checkpoint,
            'sqrl_enabled': cfg.sqrl_enabled,
            'sqrl_phase': cfg.sqrl_phase,
            'sqrl_epsilon': cfg.sqrl_epsilon,
            'save_dir': cfg.save_dir,
        },
    )
    if logger.enabled:
        try:
            meta_path = Path(cfg.save_dir) / 'wandb_run.json'
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(
                __import__('json').dumps({
                    'name': logger.run_name,
                    'url': logger.run_url,
                    'dir': logger.run_path,
                    'mode': logger.mode,
                    'project': cfg.wandb_project,
                }, indent=2),
                encoding='utf-8')
        except Exception as exc:
            _log(f'[wandb] could not write wandb_run.json: {exc}')

    speed_curriculum = None
    if cfg.cmd_speed_curriculum:
        restored_upper_speed = cfg.cmd_speed_min
        if safety_replay is not None and len(safety_replay.all) > 0:
            restored_upper_speed = max(
                float(item.get('command_speeds', cfg.cmd_speed_min))
                for item in safety_replay.all._items)
        speed_curriculum = PerformanceSpeedCurriculum(
            min_speed=cfg.cmd_speed_min,
            max_speed=cfg.cmd_speed_max,
            increment=cfg.cmd_speed_increment,
            window=cfg.cmd_speed_promotion_window,
            min_episode_length=cfg.cmd_speed_min_episode_length,
            min_velocity_ratio=cfg.cmd_speed_min_velocity_ratio,
            max_fall_rate=cfg.cmd_speed_max_fall_rate,
            new_stage_exploration_scale=(
                cfg.cmd_speed_new_stage_exploration_scale),
            exploration_recovery_episodes=(
                cfg.cmd_speed_exploration_recovery_episodes),
            initial_upper_speed=restored_upper_speed,
        )
        if hasattr(inner, 'set_curriculum_upper_speed'):
            inner.set_curriculum_upper_speed(speed_curriculum.upper_speed)
        if restored_upper_speed > cfg.cmd_speed_min + 1e-9:
            _log(
                f'[train] restored cmd_speed curriculum frontier='
                f'{restored_upper_speed:.2f} from safety replay')
    observation = env.reset()
    nan_policy_warned = False
    policy_corrupted = False
    sqrl_rng = jax.random.PRNGKey(cfg.seed + 50_000)
    previous_policy_action = np.zeros(
        env.action_space.shape, dtype=np.float32)
    sqrl_run_start = start_i
    latest_sqrl_info = {
        'sqrl_rejected_fraction': 0.0,
        'sqrl_no_safe_candidate': 0.0,
        'selected_Q_safe': 0.0,
        'sqrl_fallback_min_risk': 0.0,
        'sqrl_epsilon_eff': float(cfg.sqrl_epsilon),
        'sqrl_active': 0.0,
    }
    candidate_window = max(int(cfg.sqrl_gate_candidate_window), 1)
    sqrl_candidate_no_safe: deque[float] = deque(maxlen=candidate_window)
    sqrl_candidate_ranges: deque[float] = deque(maxlen=candidate_window)
    last_gate_reason = 'not-evaluated'
    control_gate_metrics = None
    if cfg.sqrl_control_metrics_path:
        metrics_path = Path(cfg.sqrl_control_metrics_path)
        if not metrics_path.exists():
            raise RuntimeError(
                f'sqrl_control_metrics_path not found: {metrics_path}')
        control_gate_metrics = __import__('json').loads(
            metrics_path.read_text(encoding='utf-8'))
        _log(f'[train] loaded SQRL control gate metrics: {metrics_path}')
    if cfg.sqrl_enabled:
        if safety_critic is None:
            raise ValueError('sqrl_enabled requires safety_critic_enabled')
        _log(f'[train] SQRL enabled phase={cfg.sqrl_phase} '
             f'epsilon={cfg.sqrl_epsilon} K={cfg.sqrl_num_candidates} '
             f'recent_only={cfg.sqrl_qsafe_recent_only} '
             f'activation_steps={cfg.sqrl_activation_steps} '
             f'min_auroc={cfg.sqrl_min_auroc} '
             f'min_pos_neg_gap={cfg.sqrl_min_pos_neg_gap} '
             f'max_ece={cfg.sqrl_max_ece} '
             f'max_brier={cfg.sqrl_max_brier} '
             f'max_no_safe={cfg.sqrl_max_no_safe_rate} '
             f'min_candidate_range={cfg.sqrl_min_candidate_range} '
             f'eps_start={cfg.sqrl_epsilon_start} '
             f'eps_anneal={cfg.sqrl_epsilon_anneal_steps} '
             f'double_critic={cfg.sqrl_double_critic_enabled} '
             f'validation_margin={cfg.sqrl_validation_improvement_margin}')
    if cfg.cmd_speed_curriculum:
        _log(f'[train] cmd_speed curriculum '
             f'min={cfg.cmd_speed_min} max={cfg.cmd_speed_max} '
             f'increment={cfg.cmd_speed_increment} '
             f'window={cfg.cmd_speed_promotion_window}')
    _log(f'[train] env ready obs={observation.shape} '
         f'start_training={cfg.start_training} '
         f'explore_action_scale={cfg.explore_action_scale} '
         f'log_interval={cfg.log_interval} utd_ratio={cfg.utd_ratio} '
         f'pipeline_updates={cfg.pipeline_updates} '
         f'no_eval={cfg.no_eval} profile={cfg.profile} '
         f'cmd_speed={float(getattr(inner.cfg, "move_speed", float("nan")))}')

    episode_return = 0.0
    episode_length = 0
    episode_safety_cost = 0.0
    episode_forward_velocity_sum = 0.0
    episode_id = int(start_i)
    completed_step = start_i
    last_saved_step = start_i if latest_snapshot(cfg.save_dir) else -1
    rolling = RollingTrainingSummary(
        window=cfg.rolling_summary_window,
        action_dim=env.action_space.shape[0],
    )
    done = False
    pending_update = None
    latest_safety_info = None
    sqrl_constraint_start = None
    sqrl_ready_logged = False

    def apply_pending_update():
        nonlocal agent, pending_update, policy_corrupted
        if pending_update is None:
            return None, 0.0, None
        source_step, batch = pending_update
        pending_update = None
        agent, info, corrupted, elapsed = _apply_agent_update(
            agent, batch, cfg, source_step, safety_critic=safety_critic)
        if corrupted:
            policy_corrupted = True
        return info, elapsed, source_step

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
                sqrl_steps = i - sqrl_run_start
                qsafe_ready, gate_reason = _sqrl_gate_decision(
                    latest_safety_info,
                    sqrl_candidate_no_safe,
                    sqrl_candidate_ranges,
                    cfg,
                    control_gate_metrics)
                sqrl_active = (
                    cfg.sqrl_enabled and safety_critic is not None
                    and sqrl_steps >= int(cfg.sqrl_activation_steps)
                    and qsafe_ready)
                if cfg.sqrl_enabled and gate_reason != last_gate_reason:
                    _log(
                        f'[train] SQRL gate={gate_reason} '
                        f'candidate_window={len(sqrl_candidate_no_safe)}/'
                        f'{candidate_window}')
                    last_gate_reason = gate_reason
                if sqrl_active:
                    if sqrl_constraint_start is None:
                        sqrl_constraint_start = i
                        if not sqrl_ready_logged:
                            _log(
                                f'[train] SQRL constraint ON at step {i} '
                                '(natural calibration and candidate coverage '
                                'gate passed)')
                            sqrl_ready_logged = True
                    anneal = max(int(cfg.sqrl_epsilon_anneal_steps), 1)
                    progress = min(
                        1.0,
                        (i - int(sqrl_constraint_start)) / anneal)
                    eps_eff = (
                        float(cfg.sqrl_epsilon_start)
                        + progress * (float(cfg.sqrl_epsilon)
                                      - float(cfg.sqrl_epsilon_start)))
                else:
                    sqrl_constraint_start = None
                    eps_eff = float(cfg.sqrl_epsilon_start)

                if cfg.sqrl_enabled and safety_critic is not None:
                    candidate_action, latest_sqrl_info, sqrl_rng = (
                        select_sqrl_action(
                        agent, safety_critic, observation, sqrl_rng,
                        validation_critic=safety_validator,
                        num_candidates=cfg.sqrl_num_candidates,
                        epsilon_safe=eps_eff,
                        candidate_noise_std=float(
                            cfg.sqrl_train_candidate_noise_std),
                        previous_action=previous_policy_action,
                        local_candidate_count=cfg.sqrl_local_candidate_count,
                        local_action_std=cfg.sqrl_local_action_std,
                        fallback_contraction=cfg.sqrl_fallback_contraction,
                        fallback_emergency_risk=(
                            cfg.sqrl_fallback_emergency_risk),
                        uncertainty_penalty=(
                            cfg.sqrl_uncertainty_penalty),
                        support_gate_enabled=(
                            cfg.sqrl_support_gate_enabled),
                        min_behavior_log_prob_per_dim=(
                            cfg.sqrl_min_behavior_log_prob_per_dim),
                        max_nominal_action_distance=(
                            cfg.sqrl_max_nominal_action_distance),
                        validation_improvement_margin=(
                            cfg.sqrl_validation_improvement_margin)))
                    sqrl_candidate_no_safe.append(float(
                        latest_sqrl_info['sqrl_no_safe_candidate']))
                    sqrl_candidate_ranges.append(float(
                        latest_sqrl_info['candidate_Q_safe_range']))
                    action = candidate_action if sqrl_active else None
                    latest_sqrl_info['sqrl_epsilon_eff'] = float(eps_eff)
                    latest_sqrl_info['sqrl_active'] = float(sqrl_active)
                    latest_sqrl_info['sqrl_qsafe_ready'] = float(qsafe_ready)
                    latest_sqrl_info['sqrl_gate_no_safe_rate'] = float(
                        np.mean(sqrl_candidate_no_safe))
                    latest_sqrl_info['sqrl_gate_candidate_range'] = float(
                        np.mean(sqrl_candidate_ranges))
                else:
                    action = None
                if action is None:
                    action, agent = agent.sample_actions(observation)
                    if speed_curriculum is not None:
                        explore_multiplier = (
                            speed_curriculum.exploration_multiplier)
                        if explore_multiplier < 1.0:
                            mean_action = agent.eval_actions(observation)
                            action = (
                                mean_action
                                + explore_multiplier * (action - mean_action))
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
            previous_policy_action = action.copy()
            profiler.record_sample(time.perf_counter() - sample_t0)

            update_info = None
            update_elapsed = 0.0
            update_source_step = None

            def update_during_hold():
                nonlocal update_info, update_elapsed, update_source_step
                update_info, update_elapsed, update_source_step = (
                    apply_pending_update())

            step_t0 = time.perf_counter()
            hold_callback = (
                update_during_hold
                if cfg.pipeline_updates and pending_update is not None
                else None
            )
            next_observation, reward, done, info = env.step(
                action, during_hold=hold_callback)
            profiler.record_step(time.perf_counter() - step_t0)
            if update_elapsed > 0.0:
                profiler.record_update(update_elapsed)

            episode_return += reward
            if info.get('policy_step', True):
                episode_length += 1
                episode_forward_velocity_sum += float(
                    info.get('x_velocity', 0.0))

            policy_step = info.get('policy_step', True)
            step_costs = info.get('costs') or {}
            if policy_step:
                episode_safety_cost += sum(float(v)
                                           for v in step_costs.values())
            transition_ok = (_is_finite_array(observation)
                             and _is_finite_array(next_observation)
                             and _is_finite_array(action)
                             and np.isfinite(reward))
            insert_ok = policy_step and transition_ok
            if transition_ok:
                transition = build_transition(observation, action, reward,
                                              next_observation, done, info,
                                              projected_action=info.get(
                                                  'projected_action'),
                                              executed_q_target=info.get(
                                                  'executed_q_target'),
                                              policy_version=i,
                                              episode_id=episode_id)
                if safety_replay is not None:
                    safety_replay.insert(transition,
                                         policy_step=policy_step)
                if policy_step:
                    replay_buffer.insert(transition.replay_dict())

            if (safety_critic is not None and safety_replay is not None
                    and len(safety_replay) > 0
                    and cfg.safety_critic_update_interval > 0
                    and i % cfg.safety_critic_update_interval == 0):
                # Skip until at least one failure is labeled: all-negative BCE
                # collapses sigmoid Q_safe to ~0 and blocks later recovery.
                if len(safety_replay.failure) == 0:
                    pass
                else:
                    use_recent = (
                        bool(cfg.sqrl_enabled)
                        and bool(cfg.sqrl_qsafe_recent_only)
                        and len(safety_replay.recent) > 0)
                    if use_recent:
                        safety_batch = safety_replay.sample_recent_balanced(
                            cfg.safety_critic_batch_size)
                    else:
                        safety_batch = safety_replay.sample_mixed(
                            cfg.safety_critic_batch_size)
                    safety_critic, safety_info = SafetyCritic.update(
                        safety_critic, agent.actor, safety_batch)
                    latest_safety_info = _to_float_dict(safety_info)
                    if safety_validator is not None:
                        safety_validator, validator_info = SafetyCritic.update(
                            safety_validator, agent.actor, safety_batch)
                        latest_safety_info.update({
                            f'validator_{key}': value
                            for key, value in _to_float_dict(
                                validator_info).items()
                        })
                    training_predictions = safety_critic.predict(
                        safety_batch['observations'], safety_batch['actions'])
                    training_labels = safety_batch['future_failure_labels']
                    training_metrics = binary_prediction_metrics(
                        training_labels, training_predictions)
                    latest_safety_info.update({
                        f'Q_safe_training_{key.removeprefix("Q_safe_")}': value
                        for key, value in training_metrics.items()
                    })

                    # Gate metrics use the natural recent distribution, never
                    # the deliberately 50/50 balanced optimization batch.
                    natural_batch = safety_replay.sample_recent(
                        min(cfg.safety_critic_batch_size,
                            len(safety_replay.recent)))
                    raw_logits = safety_critic.predict_logits(
                        natural_batch['observations'],
                        natural_batch['actions'])
                    if (i % max(cfg.safety_calibration_interval, 1) == 0
                            and len(raw_logits)
                            >= cfg.safety_calibration_min_samples):
                        safety_critic, calibration_info = (
                            safety_critic.calibrate(
                                natural_batch['future_failure_labels'],
                                raw_logits))
                        latest_safety_info.update(calibration_info)
                        if safety_validator is not None:
                            validator_logits = (
                                safety_validator.predict_logits(
                                    natural_batch['observations'],
                                    natural_batch['actions']))
                            safety_validator, validator_calibration = (
                                safety_validator.calibrate(
                                    natural_batch['future_failure_labels'],
                                    validator_logits))
                            latest_safety_info.update({
                                f'validator_{key}': value
                                for key, value in
                                validator_calibration.items()
                            })
                    predictions = safety_critic.predict(
                        natural_batch['observations'],
                        natural_batch['actions'])
                    labels = natural_batch['future_failure_labels']
                    latest_safety_info.update(
                        binary_prediction_metrics(labels, predictions))
                    pos = labels >= 0.5
                    neg = ~pos
                    latest_safety_info['Q_safe_label_pos'] = float(
                        np.mean(predictions[pos]) if np.any(pos) else 0.0)
                    latest_safety_info['Q_safe_label_neg'] = float(
                        np.mean(predictions[neg]) if np.any(neg) else 0.0)
                    latest_safety_info['future_fail_batch_rate'] = float(
                        np.mean(labels))
            if not insert_ok and i >= cfg.start_training:
                skip_update = True
            observation = next_observation

            if (not skip_update and i >= cfg.start_training
                    and len(replay_buffer) > 0):
                batch = replay_buffer.sample_jax(update_batch_size)
                if cfg.pipeline_updates:
                    if pending_update is not None:
                        raise RuntimeError(
                            'A policy transition tried to queue an update '
                            'before the previous update was consumed')
                    pending_update = (i, batch)
                else:
                    agent, update_info, corrupted, update_elapsed = (
                        _apply_agent_update(
                            agent, batch, cfg, i,
                            safety_critic=safety_critic))
                    if corrupted:
                        policy_corrupted = True
                    profiler.record_update(update_elapsed)

            profiler.end_loop(time.perf_counter() - loop_t0)
            completed_step = i + 1
            timing_metrics = profiler.metrics()
            rolling.record_step(
                action=action,
                info=info,
                timing=timing_metrics,
                update_info=update_info,
            )

            if (i % cfg.log_interval == 0 or i == cfg.start_training
                    or (i >= cfg.start_training and i < cfg.start_training + 5)):
                phase = 'explore' if i < cfg.start_training else 'train'
                _log(f'[step {i}] phase={phase} reward={reward:.3f} '
                     f'x_vel={info.get("x_velocity", 0):.3f} '
                     f'|action|={float(np.linalg.norm(action)):.2f} '
                     f'recovering={info.get("is_recovering", False)} '
                     f'policy_len={info.get("step_count", 0)} '
                     f'ep_return={episode_return:.2f} buffer={len(replay_buffer)}')

            metrics_due = (
                cfg.metrics_interval <= 1
                or i % cfg.metrics_interval == 0
                or done
                or i == max_steps - 1
            )
            rolling_metrics = (
                rolling.metrics(len(replay_buffer)) if metrics_due else {})
            if metrics_due:
                log_metrics: dict[str, float] = {
                    'env/reward': float(reward),
                    'env/task_reward': float(info.get('task_reward', reward)),
                    'env/terminal_penalty': float(
                        info.get('terminal_penalty', 0.0)),
                    'env/upright_gate': float(
                        info.get('upright_gate', 1.0)),
                    'env/body_up_cos': float(
                        info.get('body_up_cos', 1.0)),
                    'env/x_velocity': float(
                        info.get('x_velocity', 0.0)),
                    'env/world_x': float(info.get('world_x', 0.0)),
                    'env/world_y': float(info.get('world_y', 0.0)),
                    'env/world_z': float(info.get('world_z', 0.0)),
                    'env/forward_term': float(
                        info.get('forward_term', 0.0)),
                    'env/episode_return': float(episode_return),
                    'env/episode_length': float(episode_length),
                    'env/action_frequency_hz': float(
                        info.get('action_frequency_hz', np.nan)),
                    'env/control_hold_overrun_ms': float(
                        info.get('control_hold_overrun_ms', 0.0)),
                    'safety/unsafe_label': float(
                        bool(info.get('unsafe_label', False))),
                    'safety/near_failure_label': float(
                        bool(info.get('near_failure_label', False))),
                    'safety/intervention_mask': float(
                        bool(info.get('intervention_mask', False))),
                    'safety/step_cost': float(
                        sum(float(v) for v in step_costs.values())),
                    'safety/episode_cost_return': float(
                        episode_safety_cost),
                }
                for cost_key, cost_value in step_costs.items():
                    log_metrics[f'safety/{cost_key}'] = float(cost_value)
                if update_info is not None:
                    for k, v in update_info.items():
                        fv = float(v) if hasattr(v, 'item') else float(v)
                        if np.isfinite(fv):
                            log_metrics[f'training/{k}'] = fv
                if latest_safety_info is not None:
                    for key, value in latest_safety_info.items():
                        if np.isfinite(value):
                            log_metrics[f'safety_critic/{key}'] = value
                if cfg.sqrl_enabled:
                    for key, value in latest_sqrl_info.items():
                        if np.isfinite(value):
                            log_metrics[f'sqrl/{key}'] = float(value)
                log_metrics.update(timing_metrics)
                log_metrics.update(rolling_metrics)
                if safety_replay is not None:
                    sizes = safety_replay.sizes
                    log_metrics.update({
                        'safety_replay/recent_size': float(sizes.recent),
                        'safety_replay/failure_size': float(sizes.failure),
                        'safety_replay/boundary_size': float(sizes.boundary),
                        'safety_replay/recovery_size': float(sizes.recovery),
                        'safety_replay/all_size': float(sizes.all),
                    })
                logger.log(log_metrics, step=i)

            if update_info is not None and (
                    i % cfg.log_interval == 0 or i == cfg.start_training):
                metrics = {
                    k: float(v) if hasattr(v, 'item') else v
                    for k, v in update_info.items()
                }
                timing = timing_metrics
                _log(f'[step {i}] update {metrics}')
                if (cfg.pipeline_updates and update_source_step is not None):
                    _log(f'[step {i}] pipelined update source_step='
                         f'{update_source_step}')
                if timing:
                    _log(f'[step {i}] timing step_ms={timing["timing/step_ms"]:.1f} '
                         f'update_ms={timing["timing/update_ms"]:.1f} '
                         f'effective_hz={timing["timing/effective_hz"]:.1f} '
                         f'critic/s={timing["timing/critic_updates_per_sec"]:.0f}')
            if (latest_safety_info is not None
                    and (i % cfg.log_interval == 0
                         or i == cfg.start_training)):
                _log(
                    f'[step {i}] Q_safe '
                    f'loss={latest_safety_info["safety_critic_loss"]:.4f} '
                    f'mean={latest_safety_info["mean_Q_safe"]:.4f} '
                    f'pos={latest_safety_info.get("Q_safe_label_pos", float("nan")):.4f} '
                    f'neg={latest_safety_info.get("Q_safe_label_neg", float("nan")):.4f} '
                    f'fail_rate={latest_safety_info.get("future_fail_batch_rate", float("nan")):.3f} '
                    f'auroc={latest_safety_info["Q_safe_AUROC"]:.3f}')
            if (cfg.sqrl_enabled
                    and (i % cfg.log_interval == 0
                         or i == cfg.start_training)
                    and i >= cfg.start_training):
                _log(
                    f'[step {i}] SQRL '
                    f'active={latest_sqrl_info.get("sqrl_active", 0):.0f} '
                    f'eps={latest_sqrl_info.get("sqrl_epsilon_eff", cfg.sqrl_epsilon):.3f} '
                    f'reject={latest_sqrl_info["sqrl_rejected_fraction"]:.3f} '
                    f'no_safe={latest_sqrl_info["sqrl_no_safe_candidate"]:.3f} '
                    f'Q={latest_sqrl_info["selected_Q_safe"]:.3f} '
                    f'cand_range={latest_sqrl_info.get("candidate_Q_safe_range", 0):.3f} '
                    f'gate_no_safe={latest_sqrl_info.get("sqrl_gate_no_safe_rate", 0):.3f} '
                    f'min_risk_fb='
                    f'{latest_sqrl_info["sqrl_fallback_min_risk"]:.3f}')
            if i % cfg.log_interval == 0 and rolling_metrics:
                _log(
                    f'[step {i}] rolling n={int(rolling_metrics["rolling/window_steps"])} '
                    f'forward_vel={rolling_metrics["rolling/forward_velocity_mean"]:.3f} '
                    f'dx={rolling_metrics["rolling/world_x_delta"]:.3f} '
                    f'upright={rolling_metrics["rolling/upright_ratio"]:.3f} '
                    f'action_sat={rolling_metrics["rolling/action_saturation_rate"]:.3f} '
                    f'falls={int(rolling_metrics["rolling/falls_total"])} '
                    f'loop_hz={rolling_metrics["rolling/effective_hz_mean"]:.1f} '
                    f'action_hz={rolling_metrics["rolling/action_frequency_hz_mean"]:.1f}')

            if done:
                if info.get('standup_timed_out'):
                    reason = 'standup-timeout'
                elif info.get('terminated'):
                    reason = 'fallen'
                else:
                    reason = 'truncated'
                _log(f'[step {i}] episode done ({reason}) '
                     f'return={episode_return:.2f} '
                     f'policy_len={info.get("step_count", episode_length)} '
                     f'cmd_speed={float(info.get("cmd_speed", float("nan"))):.3f}')
                rolling.record_episode(episode_return, episode_length)
                logger.log({
                    'training/return': episode_return,
                    'training/length': float(episode_length),
                    'training/safety_cost_return': episode_safety_cost,
                    'env/cmd_speed': float(info.get('cmd_speed', float('nan'))),
                }, step=i)
                if speed_curriculum is not None:
                    episode_cmd_speed = float(
                        info.get('cmd_speed', speed_curriculum.upper_speed))
                    curriculum_update = speed_curriculum.record_episode(
                        command_speed=episode_cmd_speed,
                        mean_forward_velocity=(
                            episode_forward_velocity_sum
                            / max(episode_length, 1)),
                        episode_length=episode_length,
                        fell=bool(info.get('terminated')
                                  or info.get('standup_timed_out')),
                    )
                    if curriculum_update.promoted:
                        _log(
                            f'[train] cmd_speed promoted to '
                            f'{curriculum_update.upper_speed:.2f} m/s; '
                            f'exploration_multiplier='
                            f'{curriculum_update.exploration_multiplier:.2f}')
                    logger.log({
                        'curriculum/upper_speed':
                            curriculum_update.upper_speed,
                        'curriculum/frontier_episodes':
                            float(curriculum_update.frontier_episodes),
                        'curriculum/mean_velocity_ratio':
                            curriculum_update.mean_velocity_ratio,
                        'curriculum/mean_episode_length':
                            curriculum_update.mean_episode_length,
                        'curriculum/fall_rate':
                            curriculum_update.fall_rate,
                        'curriculum/exploration_multiplier':
                            curriculum_update.exploration_multiplier,
                        'curriculum/promoted':
                            float(curriculum_update.promoted),
                    }, step=i)
                    if hasattr(inner, 'set_curriculum_upper_speed'):
                        inner.set_curriculum_upper_speed(
                            curriculum_update.upper_speed)
                if info.get('terminated') or info.get('standup_timed_out'):
                    kind = ('belly-up recovery→standup'
                            if info.get('standup_with_recovery')
                            else 'stand-up')
                    if info.get('standup_timed_out'):
                        kind = f'standup-timeout ({kind})'
                    _log(f'[step {i}] reset: {kind}')
                if hasattr(inner, 'set_curriculum_step'):
                    inner.set_curriculum_step(i + 1)
                observation = env.reset(
                    standup=info.get('terminated', False)
                    or info.get('standup_timed_out', False),
                    with_recovery=info.get('is_belly_up', False),
                    grace_period=not info.get('truncated', False),
                    preserve_policy_state=info.get('truncated', False),
                )
                if safety_replay is not None:
                    for recovery_transition in (
                            inner.drain_recovery_transitions()):
                        safety_replay.insert(recovery_transition,
                                             policy_step=False)
                if not _is_finite_array(observation):
                    observation = np.zeros(env.observation_space.shape,
                                           dtype=np.float32)
                done = False
                previous_policy_action = np.zeros(
                    env.action_space.shape, dtype=np.float32)
                episode_id += 1
                episode_return = 0.0
                episode_length = 0
                episode_safety_cost = 0.0
                episode_forward_velocity_sum = 0.0

            train_step = i - cfg.start_training
            if (not cfg.no_eval and not cfg.benchmark_only
                    and cfg.eval_interval > 0 and train_step > 0
                    and i >= cfg.start_training
                    and train_step % cfg.eval_interval == 0):
                if pending_update is not None:
                    apply_pending_update()
                _log(f'[step {i}] eval ({cfg.eval_episodes} ep)...')
                eval_t0 = time.time()
                eval_info = evaluate(agent, env, num_episodes=cfg.eval_episodes)
                observation = env.reset()
                done = False
                episode_return = 0.0
                episode_length = 0
                episode_safety_cost = 0.0
                _log(f'[step {i}] eval {time.time() - eval_t0:.1f}s '
                     f'return={eval_info["return"]:.2f} '
                     f'length={eval_info["length"]:.1f}')
                logger.log({
                    'eval/return': float(eval_info['return']),
                    'eval/length': float(eval_info['length']),
                }, step=i)

            if (cfg.save_checkpoints and not cfg.benchmark_only
                    and cfg.checkpoint_interval > 0
                    and completed_step % cfg.checkpoint_interval == 0):
                if pending_update is not None:
                    apply_pending_update()
                path = save_training_snapshot(
                    cfg.save_dir,
                    agent=agent,
                    replay_buffer=replay_buffer,
                    safety_replay=safety_replay,
                    safety_critic=safety_critic,
                    safety_validator=safety_validator,
                    step=completed_step,
                    metadata=_snapshot_metadata(cfg, env),
                )
                last_saved_step = completed_step
                _log(f'[step {i}] checkpoint saved: {path}')
    finally:
        if pending_update is not None:
            apply_pending_update()
        if (cfg.save_checkpoints and not cfg.benchmark_only
                and completed_step > 0 and completed_step != last_saved_step):
            path = save_training_snapshot(
                cfg.save_dir,
                agent=agent,
                replay_buffer=replay_buffer,
                safety_replay=safety_replay,
                safety_critic=safety_critic,
                safety_validator=safety_validator,
                step=completed_step,
                metadata=_snapshot_metadata(cfg, env),
            )
            _log(f'[train] final checkpoint saved: {path}')
        logger.finish()
        if logger.enabled:
            try:
                meta_path = Path(cfg.save_dir) / 'wandb_run.json'
                meta_path.write_text(
                    __import__('json').dumps({
                        'name': logger.run_name,
                        'url': logger.run_url,
                        'dir': logger.run_path,
                        'mode': logger.mode,
                        'project': cfg.wandb_project,
                    }, indent=2),
                    encoding='utf-8')
            except Exception:
                pass

    if cfg.benchmark_only:
        timing = profiler.metrics()
        _log('[benchmark] done')
        if timing:
            _log(f'[benchmark] effective_hz={timing["timing/effective_hz"]:.2f} '
                 f'update_ms={timing["timing/update_ms"]:.1f} '
                 f'critic/s={timing["timing/avg_critic_updates_per_sec"]:.0f}')

    return agent
