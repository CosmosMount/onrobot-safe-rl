"""Learner side of the asynchronous paper-SQRL training pipeline."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from rl.utils.logger import WandbTrainerLogger
from train.async_collector import run_async_collector
from train.loop import (
    _agent_hashes,
    _git_metadata,
    latest_snapshot,
    prepare_save_dir,
    restore_snapshot,
    save_snapshot,
)
from train.update_schedule import UTDUpdateScheduler


class _NullTrainerLogger:
    run_id = None

    def update_metric(self, **kwargs: Any) -> None:
        del kwargs

    def log_metric(self, step: int) -> None:
        del step

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


def _replace_latest(q: Any, payload: Any) -> None:
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(payload)
    except queue.Full:
        pass


def run_async_training(agent: Any, env: Any, cfg: Any,
                       robot_cfg: Any) -> Any:
    """Collect at runtime rate while this process performs learner updates."""
    prepare_save_dir(
        cfg.save_dir,
        resume=bool(cfg.resume_checkpoint and cfg.save_checkpoints),
        benchmark=bool(cfg.benchmark_only),
    )
    start_i = 0
    resume_state: dict[str, Any] = {}
    if cfg.resume_checkpoint and cfg.save_checkpoints:
        resume_path = latest_snapshot(cfg.save_dir)
        if resume_path is not None:
            start_i = restore_snapshot(agent, cfg, str(resume_path))
            state_path = resume_path / "async_state.json"
            if state_path.exists():
                resume_state = json.loads(state_path.read_text())
    ctx = mp.get_context("spawn")
    transition_queue = ctx.Queue(
        maxsize=int(cfg.async_transition_queue_capacity))
    weight_queue = ctx.Queue(maxsize=2)
    control_queue = ctx.Queue(maxsize=4)
    initial_update_version = int(resume_state.get("snapshot_version", 0))
    snapshot_version = initial_update_version
    agent.cfg.actor_observation_dim = int(agent.get_inference_observation_dim())
    weight_queue.put(agent.export_inference_snapshot(
        snapshot_version=initial_update_version))
    collector = ctx.Process(
        target=run_async_collector,
        kwargs={
            "robot_cfg": robot_cfg,
            "agent_cfg": agent.cfg,
            "train_cfg": cfg,
            "observation_dim": int(env.observation_space.shape[-1]),
            "action_dim": int(env.action_space.shape[-1]),
            "action_space": env.action_space,
            "transition_queue": transition_queue,
            "weight_queue": weight_queue,
            "control_queue": control_queue,
            "initial_policy_steps": start_i,
        },
        name="ordered-async-collector",
    )

    logger_cfg = OmegaConf.create({
        "project_name": cfg.wandb_project,
        "entity_name": None,
        "group_name": cfg.experiment_name,
        "run_name": cfg.wandb_run_name,
        "config": {
            "experiment_name": cfg.experiment_name,
            "agent": str(agent.cfg.agent_type),
            "seed": cfg.seed,
            "max_steps": cfg.max_steps,
            "start_training": cfg.start_training,
            "batch_size": cfg.batch_size,
            "utd_ratio": cfg.utd_ratio,
            "buffer_size": cfg.buffer_size,
            "control_frequency": cfg.control_frequency,
            "async_collection": True,
            "target_speed_mps": robot_cfg.move_speed,
            "reward_profile": robot_cfg.reward_profile,
        },
    })
    logger = (
        WandbTrainerLogger(logger_cfg)
        if cfg.wandb and not cfg.benchmark_only
        else _NullTrainerLogger()
    )
    manifest_path = Path(cfg.save_dir) / "manifest.json"
    initial_hashes = _agent_hashes(agent)
    manifest = {
        **_git_metadata(),
        **{f"initial_{key}": value for key, value in initial_hashes.items()},
        "seed": int(cfg.seed),
        "target_speed_mps": float(robot_cfg.move_speed),
        "control_frequency_hz": float(cfg.control_frequency),
        "max_steps": int(cfg.max_steps),
        "agent": str(agent.cfg.agent_type),
        "collector_architecture": "ordered_async_process_v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "resumed_from_step": start_i,
        "wandb_run_id": logger.run_id,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    steps = start_i
    falls = int(resume_state.get("falls", 0))
    updates = int(agent.get_update_counters().get("critic_steps", 0))
    learner_calls = int(resume_state.get("learner_calls", 0))
    last_published_critic_steps = int(agent.get_update_counters().get("critic_steps", 0))
    counters = agent.get_update_counters()
    last_published_actor_steps = int(counters.get("actor_steps", 0))
    last_published_auxiliary_steps = int(counters.get("auxiliary_steps", 0))
    last_saved = start_i
    checkpoint_interval = int(cfg.checkpoint_interval)
    next_checkpoint = (
        (start_i // checkpoint_interval + 1) * checkpoint_interval
        if checkpoint_interval > 0 else 0)
    episode_return = float(resume_state.get("episode_return", 0.0))
    episode_length = int(resume_state.get("episode_length", 0))
    episodes: list[dict[str, Any]] = list(
        resume_state.get("episodes", []))
    intervals: list[float] = []
    max_runtime_queue_depth = 0
    max_transition_queue_depth = 0
    final_inference_weight_version = initial_update_version
    utd_scheduler = UTDUpdateScheduler(cfg.utd_ratio)
    last_update_info: dict[str, float] = {}
    repeated_action_steps_base = int(resume_state.get(
        "repeated_action_steps", 0))
    repeated_action_steps = repeated_action_steps_base
    repeated_action_rate = 0.0
    status = "stopped"
    collector.start()
    try:
        while steps < int(cfg.max_steps):
            try:
                item = transition_queue.get(timeout=10.0)
            except queue.Empty:
                if not collector.is_alive():
                    if collector.exitcode == 0:
                        # A clean collector exit is a normal shutdown path.
                        # Drain anything already handed off, then let the
                        # finally block stop the runtime producer.
                        status = "finished"
                        break
                    raise RuntimeError(
                        "collector exited unexpectedly: "
                        f"exitcode={collector.exitcode}")
                continue
            transition = item["transition"]
            info = item["info"]
            agent.process_transition(transition)
            steps += 1
            episode_return += float(transition["reward"][0])
            episode_length += 1
            falls += int(bool(transition["terminated"][0]))
            intervals.append(float(item["collector_interval_ms"]))
            if len(intervals) > 5000:
                del intervals[:1000]
            max_runtime_queue_depth = max(
                max_runtime_queue_depth,
                int(item.get("runtime_queue_depth", 0)))
            final_inference_weight_version = max(
                final_inference_weight_version,
                int(item.get("inference_weight_version", 0)))
            repeated_action_steps = repeated_action_steps_base + int(
                item.get("repeated_action_steps", 0))
            repeated_action_rate = (
                repeated_action_steps / max(steps, 1))
            try:
                max_transition_queue_depth = max(
                    max_transition_queue_depth, transition_queue.qsize())
            except (NotImplementedError, AttributeError):
                pass

            if steps >= int(cfg.start_training) and agent.can_start_training():
                request = utd_scheduler.next_request()
                if request is None:
                    continue
                update_info = agent.update_policy_steps(request)
                if not all(np.isfinite(float(v))
                           for v in update_info.values()):
                    raise FloatingPointError(
                        f"non-finite learner update at step {steps}")
                last_update_info = {key: float(value)
                                    for key, value in update_info.items()}
                learner_calls += 1
                updates = int(agent.get_update_counters().get(
                    "critic_steps", updates))
                sync_period = int(cfg.inference_sync_updates)
                counters = agent.get_update_counters()
                actor_steps = int(counters.get("actor_steps", 0))
                auxiliary_steps = int(counters.get("auxiliary_steps", 0))
                crossed_sync = (sync_period > 0 and updates > last_published_critic_steps
                                 and updates // sync_period > last_published_critic_steps // sync_period)
                changed = (actor_steps > last_published_actor_steps
                           or auxiliary_steps > last_published_auxiliary_steps)
                if crossed_sync and changed:
                    snapshot_version += 1
                    _replace_latest(
                        weight_queue,
                        agent.export_inference_snapshot(snapshot_version=snapshot_version))
                    last_published_critic_steps = updates
                    last_published_actor_steps = actor_steps
                    last_published_auxiliary_steps = auxiliary_steps
                    final_inference_weight_version = snapshot_version

            done = bool(transition["terminated"][0]
                        or transition["truncated"][0])
            completed_episode_return = episode_return
            completed_episode_length = episode_length
            if done:
                episodes.append({
                    "end_step": steps,
                    "return": episode_return,
                    "policy_length": episode_length,
                    "terminated": bool(transition["terminated"][0]),
                    "reason": str(info.get("safety_reason", "done")),
                })
                episode_return = 0.0
                episode_length = 0
            if steps % max(1, int(cfg.log_interval)) == 0:
                recent = intervals[-min(len(intervals), 500):]
                # Temporary diagnostics only.  Keep these out of the metrics
                # dict so the W&B schema remains unchanged.
                diag_keys = (
                    "critic/q_mean", "critic/target_q_mean",
                    "critic/pred_boundary_mass_low",
                    "critic/pred_boundary_mass_high",
                    "critic/target_boundary_mass_low",
                    "critic/target_boundary_mass_high",
                    "temperature/value", "actor/entropy",
                    "actor/action_saturation",
                )
                diagnostics = " ".join(
                    f"{key.rsplit('/', 1)[-1]}={last_update_info[key]:.3g}"
                    for key in diag_keys if key in last_update_info
                )
                print(
                    f"[async step {steps}] falls={falls} "
                    f"critic_updates={updates} "
                    f"actor_updates={int(agent.get_update_counters().get('actor_steps', 0))} "
                    f"collector_ms_p50={np.median(recent):.2f} "
                    f"runtime_q={item.get('runtime_queue_depth', 0)} "
                    f"reward={float(transition['reward'][0]):.3g} "
                    f"xvel={float(info.get('x_velocity', 0.0)):.3g} "
                    f"{diagnostics}",
                    flush=True)
            if (cfg.metrics_interval <= 1
                    or steps % int(cfg.metrics_interval) == 0
                    or done):
                metrics = {
                    "env/reward": float(transition["reward"][0]),
                    "env/episode_return": float(completed_episode_return),
                    "env/episode_length": float(completed_episode_length),
                    "env/x_velocity": float(info.get("x_velocity", 0.0)),
                    "env/dyaw": float(info.get("dyaw", 0.0)),
                    "env/vy": float(info.get("vy", 0.0)),
                    "rolling/falls_total": float(falls),
                    "timing/collector_interval_ms": float(
                        item.get("collector_interval_ms", 0.0)),
                    "timing/runtime_queue_depth": float(
                        item.get("runtime_queue_depth", 0)),
                    "timing/repeated_action_rate": float(repeated_action_rate),
                }
                if done:
                    metrics.update({
                        "episode/return": float(completed_episode_return),
                        "episode/length": float(completed_episode_length),
                        "episode/terminated": float(
                            bool(transition["terminated"][0])),
                    })
                if last_update_info:
                    metrics.update({
                        f"training/{key}": float(value)
                        for key, value in last_update_info.items()
                    })
                metrics.update({
                    str(key): float(value)
                    for key, value in info.items()
                    if (str(key).startswith(("reward/", "env/"))
                        and np.isscalar(value)
                        and np.isfinite(float(value)))
                })
                metrics.update({
                    str(key): float(value)
                    for key, value in agent.get_metrics().items()
                    if np.isfinite(float(value))
                })
                logger.update_metric(**metrics)
                logger.log_metric(step=steps)
                logger.reset()
            if (cfg.save_checkpoints and checkpoint_interval > 0
                    and done and steps >= next_checkpoint):
                save_snapshot(agent, cfg, steps)
                state_path = Path(cfg.save_dir) / f"step_{steps:012d}" / "async_state.json"
                state_path.write_text(json.dumps({
                    "falls": falls,
                    "episodes": episodes,
                    "episode_return": episode_return,
                    "episode_length": episode_length,
                    "repeated_action_steps": repeated_action_steps,
                    "learner_calls": learner_calls,
                    "snapshot_version": snapshot_version,
                }, indent=2) + "\n")
                last_saved = steps
                while next_checkpoint <= steps:
                    next_checkpoint += checkpoint_interval
        status = "finished"
    finally:
        logger.close()
        try:
            control_queue.put_nowait("stop")
        except queue.Full:
            pass
        collector.join(timeout=5.0)
        if collector.is_alive():
            collector.terminate()
            collector.join(timeout=5.0)
        safe_final_checkpoint = (
            status == "finished" or episode_length == 0)
        if (cfg.save_checkpoints and steps and steps != last_saved
                and safe_final_checkpoint):
            save_snapshot(agent, cfg, steps)
            state_path = Path(cfg.save_dir) / f"step_{steps:012d}" / "async_state.json"
            state_path.write_text(json.dumps({
                "falls": falls,
                "episodes": episodes,
                "episode_return": episode_return,
                "episode_length": episode_length,
                "repeated_action_steps": repeated_action_steps,
                "learner_calls": learner_calls,
                "snapshot_version": snapshot_version,
            }, indent=2) + "\n")
        final_hashes = _agent_hashes(agent)
        recent = np.asarray(intervals, dtype=np.float64)
        manifest.update({
            **{f"final_{key}": value for key, value in final_hashes.items()},
            "status": status,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "completed_steps": steps,
            "policy_steps": steps,
            "falls": falls,
            "falls_per_1000_policy_steps": 1000.0 * falls / max(steps, 1),
            "episodes": len(episodes),
            "episode_fall_rate": (
                sum(bool(ep["terminated"]) for ep in episodes)
                / max(len(episodes), 1)),
            "learner_updates": updates,
            "learner_calls": learner_calls,
            "update_counters": agent.get_update_counters(),
            "last_update_metrics": last_update_info,
            "snapshot_version": snapshot_version,
            "collector_interval_ms_p50": (
                float(np.percentile(recent, 50)) if recent.size else None),
            "collector_interval_ms_p95": (
                float(np.percentile(recent, 95)) if recent.size else None),
            "max_runtime_queue_depth": max_runtime_queue_depth,
            "max_transition_queue_depth": max_transition_queue_depth,
            "collector_exitcode": collector.exitcode,
            "final_inference_weight_version": (
                final_inference_weight_version),
            "repeated_action_steps": repeated_action_steps,
            "repeated_action_rate": repeated_action_rate,
            "agent_metrics": {
                str(k): float(v) for k, v in agent.get_metrics().items()
                if np.isfinite(float(v))},
        })
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        (Path(cfg.save_dir) / "episodes.json").write_text(
            json.dumps(episodes, indent=2) + "\n")
    return agent
