"""FlashSAC online training loop for Go2."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

try:
    import tqdm as tqdm_module
except ImportError:
    tqdm_module = None

from rl.utils.logger import AverageMeterDict, WandbTrainerLogger
from train.config import TrainConfig


def _log(msg: str) -> None:
    print(msg, flush=True)


def _is_finite_array(x: Any) -> bool:
    a = np.asarray(x)
    return bool(np.all(np.isfinite(a)))


def _batched_transition(
    observation: np.ndarray,
    action: np.ndarray,
    reward: float,
    next_observation: np.ndarray,
    info: dict[str, Any],
) -> dict[str, np.ndarray]:
    return {
        "observation": np.asarray(observation, dtype=np.float32)[None, ...],
        "action": np.asarray(action, dtype=np.float32)[None, ...],
        "reward": np.asarray([reward], dtype=np.float32),
        "terminated": np.asarray([bool(info.get("terminated", False))], dtype=np.float32),
        "truncated": np.asarray([bool(info.get("truncated", False))], dtype=np.float32),
        "next_observation": np.asarray(next_observation, dtype=np.float32)[None, ...],
    }


def _update_meter(meter: AverageMeterDict, metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        if np.isfinite(float(value)):
            meter.update(key, float(value))


class _NullTrainerLogger:
    def update_metric(self, **kwargs: Any) -> None:
        pass

    def log_metric(self, step: int) -> None:
        pass

    def reset(self) -> None:
        pass


def _prev_transition(observation: np.ndarray) -> dict[str, np.ndarray]:
    return {"next_observation": np.asarray(observation, dtype=np.float32)[None, ...]}


def _safe_action(action: np.ndarray, shape: tuple[int, ...]) -> tuple[np.ndarray, bool]:
    action = np.asarray(action, dtype=np.float32).reshape(shape)
    if not _is_finite_array(action):
        return np.zeros(shape, dtype=np.float32), False
    return np.clip(action, -1.0, 1.0).astype(np.float32), True


def _snapshot_dir(save_dir: str, step: int) -> Path:
    return Path(save_dir) / f"step_{step:012d}"


def latest_snapshot(save_dir: str) -> Path | None:
    root = Path(save_dir)
    if not root.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in root.iterdir():
        match = re.fullmatch(r"step_(\d+)", path.name)
        if path.is_dir() and match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _snapshot_step(path: Path) -> int:
    match = re.fullmatch(r"step_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Invalid snapshot path: {path}")
    return int(match.group(1))


def save_snapshot(agent, cfg: TrainConfig, step: int) -> Path:
    path = _snapshot_dir(cfg.save_dir, step)
    agent.save(str(path / "agent"))
    agent.save_replay_buffer(str(path / "replay"))
    return path


def restore_snapshot(agent, cfg: TrainConfig, checkpoint: str | None = None) -> int:
    path = Path(checkpoint) if checkpoint is not None else latest_snapshot(cfg.save_dir)
    if path is None:
        return 0
    agent.load(str(path / "agent"))
    agent.load_replay_buffer(str(path / "replay"))
    return _snapshot_step(path)


def _sample_policy_action(agent, observation: np.ndarray, step: int, *, training: bool) -> np.ndarray:
    sampled = agent.sample_actions(step, _prev_transition(observation), training=training)
    return np.asarray(sampled[0], dtype=np.float32)


def _update_agent(agent, cfg: TrainConfig, source_step: int) -> tuple[dict[str, float] | None, float]:
    update_t0 = time.perf_counter()
    if not agent.can_start_training():
        return None, 0.0

    last_info: dict[str, float] | None = None
    for _ in range(max(1, int(cfg.utd_ratio))):
        info = agent.update()
        finite_info: dict[str, float] = {}
        for key, value in info.items():
            fv = float(value)
            if not np.isfinite(fv):
                _log(f"[train] WARNING: non-finite update metric {key} at step {source_step}")
                return last_info, time.perf_counter() - update_t0
            finite_info[key] = fv
        last_info = finite_info
    return last_info, time.perf_counter() - update_t0


def run_training(agent, env, cfg: TrainConfig):
    os.makedirs(cfg.save_dir, exist_ok=True)

    start_i = 0
    if cfg.save_checkpoints and cfg.resume_checkpoint and not cfg.benchmark_only:
        start_i = restore_snapshot(agent, cfg)
        if start_i > 0:
            _log(f"[train] resumed FlashSAC snapshot step {start_i}")

    inner = getattr(env, "_env", env)
    logger_cfg = OmegaConf.create({
        "project_name": cfg.wandb_project,
        "entity_name": None,
        "group_name": cfg.experiment_name,
        "run_name": cfg.wandb_run_name,
        "config": {
            "experiment_name": cfg.experiment_name,
            "agent": cfg.agent,
            "seed": cfg.seed,
            "max_steps": cfg.max_steps,
            "start_training": cfg.start_training,
            "batch_size": cfg.batch_size,
            "utd_ratio": cfg.utd_ratio,
            "buffer_size": cfg.buffer_size,
            "control_frequency": inner.control_frequency,
        },
    })
    logger = (
        WandbTrainerLogger(logger_cfg)
        if cfg.wandb and not cfg.benchmark_only
        else _NullTrainerLogger()
    )

    observation = env.reset()
    if not _is_finite_array(observation):
        observation = np.zeros(env.observation_space.shape, dtype=np.float32)

    episode_return = 0.0
    episode_length = 0
    completed_step = start_i
    last_saved_step = start_i
    rolling = AverageMeterDict()
    total_policy_steps = 0
    total_falls = 0
    total_recoveries = 0
    total_timeouts = 0

    max_steps = cfg.benchmark_steps if cfg.benchmark_only else cfg.max_steps
    iterator = range(start_i, max_steps)
    if cfg.use_tqdm and tqdm_module is not None:
        iterator = tqdm_module.tqdm(iterator, smoothing=0.1)

    _log(
        f"[train] env ready obs={observation.shape} action={env.action_space.shape} "
        f"start_training={cfg.start_training} utd_ratio={cfg.utd_ratio}"
    )

    try:
        for i in iterator:
            loop_t0 = time.perf_counter()

            sample_t0 = time.perf_counter()
            if i < cfg.start_training:
                action = env.sample_action()
            else:
                if i == cfg.start_training:
                    _log(f"[train] === Entering FlashSAC updates at step {i} ===")
                action = _sample_policy_action(agent, observation, i, training=True)
            action, action_ok = _safe_action(action, env.action_space.shape)
            sample_ms = (time.perf_counter() - sample_t0) * 1000.0

            step_t0 = time.perf_counter()
            next_observation, reward, done, info = env.step(action)
            step_ms = (time.perf_counter() - step_t0) * 1000.0

            if not _is_finite_array(next_observation):
                next_observation = np.zeros(env.observation_space.shape, dtype=np.float32)
                action_ok = False

            policy_step = bool(info.get("policy_step", True))
            replay_enabled = bool(info.get("replay_enabled", policy_step))
            insert_ok = (
                policy_step
                and replay_enabled
                and action_ok
                and _is_finite_array(observation)
                and _is_finite_array(next_observation)
                and np.isfinite(reward)
            )
            if insert_ok:
                transition = _batched_transition(observation, action, reward, next_observation, info)
                repeats = max(1, int(cfg.terminal_replay_repeats) if info.get("terminated") else 1)
                for _ in range(repeats):
                    agent.process_transition(transition)

            update_info = None
            update_elapsed = 0.0
            if i >= cfg.start_training and insert_ok:
                update_info, update_elapsed = _update_agent(agent, cfg, i)
            update_ms = update_elapsed * 1000.0

            observation = next_observation
            episode_return += float(reward)
            if policy_step:
                episode_length += 1
                total_policy_steps += 1
                total_falls += int(bool(info.get("terminated", False)))
                total_recoveries += int(info.get("safety_mode") == "recovery")
                total_timeouts += int(info.get("safety_reason") == "recovery_timeout")

            loop_elapsed = time.perf_counter() - loop_t0
            completed_step = i + 1
            timing_metrics = {
                "timing/step_ms": step_ms,
                "timing/sample_ms": sample_ms,
                "timing/update_ms": update_ms,
                "timing/loop_ms": loop_elapsed * 1000.0,
                "timing/effective_hz": 1.0 / loop_elapsed if loop_elapsed > 0 else 0.0,
                "timing/critic_updates_per_sec": (
                    cfg.utd_ratio / loop_elapsed if update_info is not None and loop_elapsed > 0 else 0.0
                ),
            }
            if policy_step:
                _update_meter(
                    rolling,
                    {
                        "rolling/forward_velocity_mean": float(info.get("forward_velocity", info.get("x_velocity", 0.0))),
                        "rolling/upright_ratio": float(info.get("upright_gate", 1.0)),
                        "rolling/action_frequency_hz_mean": float(info.get("action_frequency_hz", np.nan)),
                        "rolling/control_hold_overrun_ms_mean": float(info.get("control_hold_overrun_ms", 0.0)),
                        "rolling/action_mean": float(np.mean(action)),
                        "rolling/action_std": float(np.std(action)),
                        "rolling/action_saturation_rate": float(np.mean(np.abs(action) >= 0.99)),
                        "rolling/effective_hz_mean": timing_metrics["timing/effective_hz"],
                        "rolling/update_ms_mean": update_ms,
                    },
                )

            if i % cfg.log_interval == 0 or i == cfg.start_training:
                phase = "explore" if i < cfg.start_training else "train"
                _log(
                    f"[step {i}] phase={phase} mode={info.get('safety_mode', 'policy')} "
                    f"reward={reward:.3f} "
                    f"x_vel={info.get('x_velocity', 0):.3f} "
                    f"replay={insert_ok} update={update_info is not None} "
                    f"reason={info.get('safety_reason', 'policy')} "
                    f"ctrl={info.get('controller_phase', -1)} "
                    f"roll={info.get('safety_roll', 0.0):+.2f} "
                    f"pitch={info.get('safety_pitch', 0.0):+.2f} "
                    f"up_cos={info.get('safety_body_up_cos', 1.0):+.2f} "
                    f"acc_z={info.get('safety_acc_z', 0.0):+.2f} "
                    f"fallen={int(bool(info.get('fallen', False)))} "
                    f"inverted={int(bool(info.get('inverted', False)))} "
                    f"ep_return={episode_return:.2f}"
                )

            metrics_due = (
                cfg.metrics_interval <= 1
                or i % cfg.metrics_interval == 0
                or done
                or i == max_steps - 1
            )
            rolling_metrics = rolling.averages() if metrics_due else {}
            if metrics_due:
                log_metrics: dict[str, float] = {
                    "env/reward": float(reward),
                    "env/task_reward": float(info.get("task_reward", reward)),
                    "env/terminal_penalty": float(info.get("terminal_penalty", 0.0)),
                    "env/upright_gate": float(info.get("upright_gate", 1.0)),
                    "env/x_velocity": float(info.get("x_velocity", 0.0)),
                    "env/world_x": float(info.get("world_x", 0.0)),
                    "env/world_y": float(info.get("world_y", 0.0)),
                    "env/world_z": float(info.get("world_z", 0.0)),
                    "env/episode_return": float(episode_return),
                    "env/episode_length": float(episode_length),
                    "env/action_frequency_hz": float(info.get("action_frequency_hz", np.nan)),
                    "env/control_hold_overrun_ms": float(info.get("control_hold_overrun_ms", 0.0)),
                    "env/controller_phase": float(info.get("controller_phase", -1)),
                    "env/safety_roll": float(info.get("safety_roll", 0.0)),
                    "env/safety_pitch": float(info.get("safety_pitch", 0.0)),
                    "env/safety_acc_z": float(info.get("safety_acc_z", 0.0)),
                    "env/safety_body_up_cos": float(info.get("safety_body_up_cos", 1.0)),
                    "env/fallen": float(bool(info.get("fallen", False))),
                    "env/inverted": float(bool(info.get("inverted", False))),
                }
                if update_info is not None:
                    log_metrics.update({f"training/{k}": float(v) for k, v in update_info.items()})
                log_metrics.update(timing_metrics)
                log_metrics.update(rolling_metrics)
                log_metrics.update({
                    "rolling/total_policy_steps": float(total_policy_steps),
                    "rolling/falls_total": float(total_falls),
                    "rolling/recoveries_total": float(total_recoveries),
                    "rolling/timeouts_total": float(total_timeouts),
                })
                logger.update_metric(**log_metrics)
                logger.log_metric(step=i)
                logger.reset()

            if update_info is not None and (i % cfg.log_interval == 0 or i == cfg.start_training):
                _log(f"[step {i}] update {update_info}")

            if done:
                if hasattr(env, "clear_action"):
                    env.clear_action()
                reason = info.get("safety_reason", "truncated" if info.get("truncated") else "terminated")
                _log(
                    f"[step {i}] episode done ({reason}) "
                    f"return={episode_return:.2f} policy_len={episode_length}"
                )
                logger.update_metric(
                    **{
                        "training/return": episode_return,
                        "training/length": float(episode_length),
                    }
                )
                logger.log_metric(step=i)
                logger.reset()
                observation = env.reset()
                if not _is_finite_array(observation):
                    observation = np.zeros(env.observation_space.shape, dtype=np.float32)
                episode_return = 0.0
                episode_length = 0

            if (
                cfg.save_checkpoints
                and not cfg.benchmark_only
                and cfg.checkpoint_interval > 0
                and completed_step % cfg.checkpoint_interval == 0
            ):
                path = save_snapshot(agent, cfg, completed_step)
                last_saved_step = completed_step
                _log(f"[step {i}] checkpoint saved: {path}")
    finally:
        if hasattr(env, "close"):
            env.close()
        if cfg.save_checkpoints and not cfg.benchmark_only and completed_step > 0 and completed_step != last_saved_step:
            path = save_snapshot(agent, cfg, completed_step)
            _log(f"[train] final checkpoint saved: {path}")

    if cfg.benchmark_only:
        _log("[benchmark] done")

    return agent


def run_play(agent, env, cfg: TrainConfig, *, checkpoint: str | None, episodes: int) -> int:
    restore_snapshot(agent, cfg, checkpoint)
    for episode in range(episodes):
        observation = env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0
        while not done:
            action, _ = _safe_action(
                _sample_policy_action(agent, observation, episode_length, training=False),
                env.action_space.shape,
            )
            observation, reward, done, info = env.step(action)
            episode_return += float(reward)
            if info.get("policy_step", True):
                episode_length += 1
        _log(f"[play] episode={episode} return={episode_return:.2f} length={episode_length}")
    return 0
