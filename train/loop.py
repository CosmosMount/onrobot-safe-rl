"""FlashSAC online training loop for Go2."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
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


LEG_NAMES = ("FR", "FL", "RR", "RL")


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
        "unsafe_label": np.asarray([
            bool(info.get("terminated", False))
            or bool(info.get("fallen", False))
            or bool(info.get("inverted", False))
        ], dtype=np.float32),
        "near_failure_label": np.asarray([
            bool(info.get("near_failure", False))
        ], dtype=np.float32),
    }


def _update_meter(meter: AverageMeterDict, metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        if np.isfinite(float(value)):
            meter.update(key, float(value))


class _NullTrainerLogger:
    run_id = None
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


def _leg_action_metrics(action: np.ndarray) -> dict[str, float]:
    values: dict[str, float] = {}
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size < 12:
        return values
    for leg_idx, leg_action in enumerate(action[:12].reshape(4, 3)):
        leg_name = LEG_NAMES[leg_idx]
        values[f"env/action_leg_{leg_idx}_mean"] = float(np.mean(leg_action))
        values[f"env/action_leg_{leg_idx}_std"] = float(np.std(leg_action))
        values[f"env/action_leg_{leg_idx}_saturation"] = float(np.mean(np.abs(leg_action) >= 0.99))
        values[f"env/action_{leg_name}_mean"] = values[f"env/action_leg_{leg_idx}_mean"]
        values[f"env/action_{leg_name}_std"] = values[f"env/action_leg_{leg_idx}_std"]
        values[f"env/action_{leg_name}_saturation"] = values[f"env/action_leg_{leg_idx}_saturation"]
    return values


def _q_target_leg_metrics(info: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    q_target = info.get("executed_q_target")
    if q_target is None:
        return values
    q = np.asarray(q_target, dtype=np.float32).reshape(-1)
    if q.size < 12:
        return values
    for leg_idx, leg_q in enumerate(q[:12].reshape(4, 3)):
        leg_name = LEG_NAMES[leg_idx]
        values[f"env/q_target_leg_{leg_idx}_mean"] = float(np.mean(leg_q))
        values[f"env/q_target_leg_{leg_idx}_std"] = float(np.std(leg_q))
        values[f"env/q_target_{leg_name}_mean"] = values[f"env/q_target_leg_{leg_idx}_mean"]
        values[f"env/q_target_{leg_name}_std"] = values[f"env/q_target_leg_{leg_idx}_std"]
    return values


def _joint_feedback_leg_metrics(info: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    joint_q = info.get("joint_q")
    joint_dq = info.get("joint_dq")
    if joint_q is None or joint_dq is None:
        return values
    q = np.asarray(joint_q, dtype=np.float32).reshape(-1)
    dq = np.asarray(joint_dq, dtype=np.float32).reshape(-1)
    if q.size < 12 or dq.size < 12:
        return values
    tracking_error = info.get("joint_tracking_error")
    err = None if tracking_error is None else np.asarray(tracking_error, dtype=np.float32).reshape(-1)
    for leg_idx, (leg_q, leg_dq) in enumerate(zip(q[:12].reshape(4, 3), dq[:12].reshape(4, 3))):
        leg_name = LEG_NAMES[leg_idx]
        values[f"env/joint_{leg_name}_q_std"] = float(np.std(leg_q))
        values[f"env/joint_{leg_name}_dq_abs_mean"] = float(np.mean(np.abs(leg_dq)))
        values[f"env/joint_leg_{leg_idx}_q_std"] = values[f"env/joint_{leg_name}_q_std"]
        values[f"env/joint_leg_{leg_idx}_dq_abs_mean"] = values[f"env/joint_{leg_name}_dq_abs_mean"]
        if err is not None and err.size >= 12:
            leg_err = err[:12].reshape(4, 3)[leg_idx]
            values[f"env/joint_{leg_name}_tracking_abs_mean"] = float(np.mean(np.abs(leg_err)))
            values[f"env/joint_{leg_name}_tracking_norm"] = float(np.linalg.norm(leg_err))
            values[f"env/joint_leg_{leg_idx}_tracking_abs_mean"] = values[
                f"env/joint_{leg_name}_tracking_abs_mean"
            ]
            values[f"env/joint_leg_{leg_idx}_tracking_norm"] = values[
                f"env/joint_{leg_name}_tracking_norm"
            ]
    return values


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


def _network_hash(network: Any) -> str | None:
    module = getattr(network, "network", network)
    state_dict = getattr(module, "state_dict", None)
    if not callable(state_dict):
        return None
    digest = hashlib.sha256()
    for key, value in sorted(state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(np.ascontiguousarray(
            value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def _agent_hashes(agent: Any) -> dict[str, str | None]:
    return {
        "actor_hash": _network_hash(getattr(agent, "_actor", None)),
        "reward_critic_hash": _network_hash(
            getattr(agent, "_critic", None)),
        "safety_critic_hash": _network_hash(
            getattr(agent, "_safety_critic", None)),
    }


def _git_metadata() -> dict[str, Any]:
    def output(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], text=True).strip()
    try:
        return {
            "commit": output("rev-parse", "HEAD"),
            "branch": output("branch", "--show-current"),
            "dirty": bool(output("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


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
    run_started_at = datetime.now(timezone.utc)
    manifest_path = Path(cfg.save_dir) / "manifest.json"
    initial_hashes = _agent_hashes(agent)
    manifest: dict[str, Any] = {
        **_git_metadata(),
        **{f"initial_{key}": value
           for key, value in initial_hashes.items()},
        "seed": int(cfg.seed),
        "target_speed_mps": float(getattr(env.cfg, "move_speed", np.nan)),
        "control_frequency_hz": float(inner.control_frequency),
        "max_steps": int(cfg.max_steps),
        "start_training": int(cfg.start_training),
        "agent": str(cfg.agent),
        "wandb_run_id": logger.run_id,
        "started_at": run_started_at.isoformat(),
        "status": "running",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    observation = env.reset()
    if not _is_finite_array(observation):
        observation = np.zeros(env.observation_space.shape, dtype=np.float32)

    episode_return = 0.0
    episode_length = 0
    episode_forward_velocity_sum = 0.0
    episode_tracking_error_sum = 0.0
    completed_step = start_i
    last_saved_step = start_i
    rolling = AverageMeterDict()
    total_policy_steps = start_i
    total_falls = 0
    total_recoveries = 0
    total_timeouts = 0
    replacement_history: deque[bool] = deque(maxlen=32)
    pending_replacements: list[dict[str, Any]] = []
    replacement_evaluated = {8: 0, 16: 0, 32: 0}
    replacement_failures = {8: 0, 16: 0, 32: 0}
    falls_with_recent_replacement = {8: 0, 16: 0, 32: 0}
    false_negative_falls_32 = 0
    policy_replacements = 0
    policy_no_safe = 0
    policy_safety_active = 0
    episode_records: list[dict[str, Any]] = []
    warned_runtime_time_limit = False

    max_steps = cfg.benchmark_steps if cfg.benchmark_only else cfg.max_steps
    progress = (
        tqdm_module.tqdm(
            total=max_steps, initial=start_i, smoothing=0.1)
        if cfg.use_tqdm and tqdm_module is not None
        else None
    )

    _log(
        f"[train] env ready obs={observation.shape} action={env.action_space.shape} "
        f"start_training={cfg.start_training} utd_ratio={cfg.utd_ratio}"
    )

    try:
        i = start_i
        while total_policy_steps < max_steps:
            loop_t0 = time.perf_counter()
            policy_index = total_policy_steps

            sample_t0 = time.perf_counter()
            if policy_index < cfg.start_training:
                action = env.sample_action() * cfg.explore_action_scale
                filter_nominal = getattr(agent, "filter_nominal_action", None)
                if callable(filter_nominal):
                    action = np.asarray(filter_nominal(
                        policy_index,
                        _prev_transition(observation),
                        action,
                        training=True,
                    )[0], dtype=np.float32)
            else:
                if policy_index == cfg.start_training:
                    _log(
                        "[train] === Entering FlashSAC updates at "
                        f"policy step {policy_index} ===")
                action = _sample_policy_action(
                    agent, observation, policy_index, training=True)
            action, action_ok = _safe_action(action, env.action_space.shape)
            sample_ms = (time.perf_counter() - sample_t0) * 1000.0

            step_t0 = time.perf_counter()
            next_observation, reward, done, info = env.step(action)
            step_ms = (time.perf_counter() - step_t0) * 1000.0

            if not _is_finite_array(next_observation):
                next_observation = np.zeros(env.observation_space.shape, dtype=np.float32)
                action_ok = False

            policy_step = bool(info.get("policy_step", True))
            count_policy_step = bool(info.get("count_policy_step", policy_step))
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
                for repeat_index in range(repeats):
                    transition["replay_repeat_index"] = np.asarray(
                        [repeat_index], dtype=np.int32)
                    agent.process_transition(transition)

            update_info = None
            update_elapsed = 0.0
            if policy_index >= cfg.start_training and insert_ok:
                update_info, update_elapsed = _update_agent(
                    agent, cfg, policy_index)
            update_ms = update_elapsed * 1000.0

            observation = next_observation
            if insert_ok:
                episode_return += float(reward)
            if policy_step:
                episode_length += 1
                forward_velocity = float(
                    info.get("forward_velocity",
                             info.get("x_velocity", 0.0)))
                episode_forward_velocity_sum += forward_velocity
                episode_tracking_error_sum += abs(
                    forward_velocity - float(env.cfg.move_speed))
                total_policy_steps += 1
                if progress is not None:
                    progress.update(1)
                if (
                    not done
                    and not warned_runtime_time_limit
                    and episode_length > cfg.max_episode_steps
                ):
                    _log(
                        "[train] warning: policy episode length exceeded "
                        f"max_episode_steps={cfg.max_episode_steps} without runtime truncation; "
                        "restart the standalone runtime so it uses the current config."
                    )
                    warned_runtime_time_limit = True
            if policy_step:
                total_falls += int(bool(info.get("terminated", False)))
                total_recoveries += int(info.get("safety_mode") == "recovery")
                total_timeouts += int(info.get("safety_reason") == "recovery_timeout")
            if policy_step:
                terminated_now = bool(info.get("terminated", False))
                for event in pending_replacements:
                    event["age"] += 1
                    event["failed"] = bool(event["failed"] or terminated_now)
                remaining_events: list[dict[str, Any]] = []
                for event in pending_replacements:
                    age = int(event["age"])
                    for horizon in (8, 16, 32):
                        if age == horizon:
                            replacement_evaluated[horizon] += 1
                            replacement_failures[horizon] += int(
                                bool(event["failed"]))
                    if age < 32:
                        remaining_events.append(event)
                pending_replacements = remaining_events

                replaced_now = bool(
                    agent.get_metrics().get("safety/replaced", 0.0))
                active_now = bool(
                    agent.get_metrics().get("safety/active", 0.0))
                no_safe_now = bool(
                    agent.get_metrics().get(
                        "safety/no_safe_candidate", 0.0))
                policy_safety_active += int(active_now)
                policy_replacements += int(replaced_now)
                policy_no_safe += int(no_safe_now)
                replacement_history.append(replaced_now)
                if replaced_now:
                    pending_replacements.append({
                        "age": 0,
                        "failed": terminated_now,
                    })
                if terminated_now:
                    history = list(replacement_history)
                    for horizon in (8, 16, 32):
                        falls_with_recent_replacement[horizon] += int(
                            any(history[-horizon:]))
                    false_negative_falls_32 += int(
                        not any(history[-32:]))

            loop_elapsed = time.perf_counter() - loop_t0
            completed_step = total_policy_steps
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

            if (
                policy_index % cfg.log_interval == 0
                or policy_index == cfg.start_training
            ):
                phase = (
                    "explore"
                    if policy_index < cfg.start_training else "train")
                _log(
                    f"[step {policy_index}] phase={phase} "
                    f"mode={info.get('safety_mode', 'policy')} "
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
                or policy_index % cfg.metrics_interval == 0
                or done
                or total_policy_steps >= max_steps
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
                    "env/count_policy_step": float(count_policy_step),
                    "env/reset_pose_error": float(info.get("reset_pose_error", np.nan)),
                    "env/reset_pose_ready": float(bool(info.get("reset_pose_ready", True))),
                    "env/awaiting_reset_pose": float(bool(info.get("awaiting_reset_pose", False))),
                    "env/reset_pose_stable_count": float(info.get("reset_pose_stable_count", 0.0)),
                    "env/reset_pose_wait_steps": float(info.get("reset_pose_wait_steps", 0.0)),
                    "env/reset_pose_timed_out": float(bool(info.get("reset_pose_timed_out", False))),
                    "env/safety_roll": float(info.get("safety_roll", 0.0)),
                    "env/safety_pitch": float(info.get("safety_pitch", 0.0)),
                    "env/safety_acc_z": float(info.get("safety_acc_z", 0.0)),
                    "env/safety_body_up_cos": float(info.get("safety_body_up_cos", 1.0)),
                    "env/fallen": float(bool(info.get("fallen", False))),
                    "env/inverted": float(bool(info.get("inverted", False))),
                }
                if update_info is not None:
                    log_metrics.update({f"training/{k}": float(v) for k, v in update_info.items()})
                log_metrics.update({
                    str(key): float(value)
                    for key, value in agent.get_metrics().items()
                    if np.isfinite(float(value))
                })
                log_metrics.update(_leg_action_metrics(action))
                log_metrics.update(_q_target_leg_metrics(info))
                log_metrics.update(_joint_feedback_leg_metrics(info))
                log_metrics.update(timing_metrics)
                log_metrics.update(rolling_metrics)
                log_metrics.update({
                    "rolling/total_policy_steps": float(total_policy_steps),
                    "rolling/falls_total": float(total_falls),
                    "rolling/recoveries_total": float(total_recoveries),
                    "rolling/timeouts_total": float(total_timeouts),
                    "safety_control/false_negative_falls_h32": float(
                        false_negative_falls_32),
                    "safety_control/policy_active_steps": float(
                        policy_safety_active),
                    "safety_control/policy_replacements": float(
                        policy_replacements),
                    "safety_control/policy_replacement_rate": float(
                        policy_replacements
                        / max(policy_safety_active, 1)),
                    "safety_control/policy_no_safe": float(policy_no_safe),
                    "safety_control/policy_no_safe_rate": float(
                        policy_no_safe / max(policy_safety_active, 1)),
                })
                for horizon in (8, 16, 32):
                    evaluated = replacement_evaluated[horizon]
                    log_metrics.update({
                        f"safety_control/replacement_evaluated_h{horizon}":
                            float(evaluated),
                        f"safety_control/replacement_failures_h{horizon}":
                            float(replacement_failures[horizon]),
                        f"safety_control/replacement_failure_rate_h{horizon}":
                            float(replacement_failures[horizon]
                                  / max(evaluated, 1)),
                        f"safety_control/falls_with_replacement_h{horizon}":
                            float(falls_with_recent_replacement[horizon]),
                    })
                logger.update_metric(**log_metrics)
                logger.log_metric(step=i)
                logger.reset()

            if update_info is not None and (
                policy_index % cfg.log_interval == 0
                or policy_index == cfg.start_training
            ):
                _log(f"[step {policy_index}] update {update_info}")

            if done:
                if hasattr(env, "clear_action"):
                    env.clear_action()
                reason = info.get("safety_reason", "truncated" if info.get("truncated") else "terminated")
                _log(
                    f"[step {i}] episode done ({reason}) "
                    f"return={episode_return:.2f} policy_len={episode_length}"
                )
                if episode_length > 0:
                    episode_records.append({
                    "end_step": int(policy_index),
                        "return": float(episode_return),
                        "policy_length": int(episode_length),
                        "reason": str(reason),
                        "terminated_policy_transition": bool(
                            policy_step and info.get("terminated", False)),
                        "forward_velocity_mean": float(
                            episode_forward_velocity_sum
                            / episode_length),
                        "tracking_error_mean": float(
                            episode_tracking_error_sum
                            / episode_length),
                    })
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
                episode_forward_velocity_sum = 0.0
                episode_tracking_error_sum = 0.0
                warned_runtime_time_limit = False

            if (
                cfg.save_checkpoints
                and not cfg.benchmark_only
                and cfg.checkpoint_interval > 0
                and completed_step % cfg.checkpoint_interval == 0
                and completed_step != last_saved_step
            ):
                path = save_snapshot(agent, cfg, completed_step)
                last_saved_step = completed_step
                _log(
                    f"[step {completed_step}] checkpoint saved: {path}")
            i += 1
    finally:
        if progress is not None:
            progress.close()
        if hasattr(env, "close"):
            env.close()
        if cfg.save_checkpoints and not cfg.benchmark_only and completed_step > 0 and completed_step != last_saved_step:
            path = save_snapshot(agent, cfg, completed_step)
            _log(f"[train] final checkpoint saved: {path}")
        final_hashes = _agent_hashes(agent)
        manifest.update({
            **{f"final_{key}": value
               for key, value in final_hashes.items()},
            "status": "finished" if completed_step >= max_steps else "stopped",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "completed_steps": int(completed_step),
            "policy_steps": int(total_policy_steps),
            "falls": int(total_falls),
            "falls_per_1000_policy_steps": float(
                1000.0 * total_falls / max(total_policy_steps, 1)),
            "episodes": len(episode_records),
            "episode_fall_rate": float(
                sum(r["terminated_policy_transition"]
                    for r in episode_records)
                / max(len(episode_records), 1)),
            "false_negative_falls_h32": int(false_negative_falls_32),
            "policy_safety_active_steps": int(policy_safety_active),
            "policy_replacements": int(policy_replacements),
            "policy_replacement_rate": float(
                policy_replacements / max(policy_safety_active, 1)),
            "policy_no_safe": int(policy_no_safe),
            "policy_no_safe_rate": float(
                policy_no_safe / max(policy_safety_active, 1)),
            "replacement_evaluated": replacement_evaluated,
            "replacement_failures": replacement_failures,
            "falls_with_recent_replacement":
                falls_with_recent_replacement,
            "agent_metrics": {
                str(key): float(value)
                for key, value in agent.get_metrics().items()
                if np.isfinite(float(value))
            },
        })
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (Path(cfg.save_dir) / "episodes.json").write_text(
            json.dumps(episode_records, indent=2) + "\n")

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
            count_policy_step = bool(
                info.get("count_policy_step", info.get("policy_step", True))
            )
            if count_policy_step:
                episode_length += 1
        _log(f"[play] episode={episode} return={episode_return:.2f} length={episode_length}")
    return 0
