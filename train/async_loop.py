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

from rl.agents.paper_sqrl.inference import export_inference_weights
from train.async_collector import run_async_collector
from train.loop import (
    _agent_hashes,
    _git_metadata,
    latest_snapshot,
    restore_snapshot,
    save_snapshot,
)


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


def _manifest_lineage(existing: dict[str, Any],
                      current_hashes: dict[str, str | None],
                      resume_step: int) -> dict[str, Any]:
    """Keep true run-origin hashes when a resumed process rewrites manifest."""
    initial = {}
    for key, value in current_hashes.items():
        manifest_key = f"initial_{key}"
        initial[manifest_key] = (
            existing.get(manifest_key, value) if resume_step > 0 else value)
    if resume_step <= 0:
        return initial
    return {
        **initial,
        **{f"resume_{key}": value for key, value in current_hashes.items()},
    }


def run_async_training(agent: Any, env: Any, cfg: Any,
                       robot_cfg: Any) -> Any:
    """Collect at runtime rate while this process performs learner updates."""
    os.makedirs(cfg.save_dir, exist_ok=True)
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
    initial_update_version = int(getattr(agent, "_update_step", 0))
    weight_queue.put(export_inference_weights(
        agent, version=initial_update_version))
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
        name="go2-sqrl-collector",
    )

    manifest_path = Path(cfg.save_dir) / "manifest.json"
    existing_manifest = {}
    if manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            existing_manifest = {}
    current_hashes = _agent_hashes(agent)
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        **_git_metadata(),
        **_manifest_lineage(existing_manifest, current_hashes, start_i),
        "seed": int(cfg.seed),
        "target_speed_mps": float(robot_cfg.move_speed),
        "control_frequency_hz": float(cfg.control_frequency),
        "max_steps": int(cfg.max_steps),
        "agent": str(agent.cfg.agent_type),
        "collector_architecture": "ordered_async_process_v1",
        "started_at": existing_manifest.get("started_at", now),
        "last_started_at": now,
        "status": "running",
        "resumed_from_step": start_i,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    steps = start_i
    falls = int(resume_state.get("falls", 0))
    updates = initial_update_version
    learner_calls = int(resume_state.get("learner_calls", 0))
    last_published_version = initial_update_version
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
                    raise RuntimeError(
                        f"collector exited unexpectedly: {collector.exitcode}")
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
                for _ in range(max(1, int(cfg.utd_ratio))):
                    update_info = agent.update()
                    if not all(np.isfinite(float(v))
                               for v in update_info.values()):
                        raise FloatingPointError(
                            f"non-finite learner update at step {steps}")
                    learner_calls += 1
                updates = int(getattr(agent, "_update_step", updates))
                sync_period = int(cfg.inference_sync_updates)
                if (sync_period > 0
                        and updates > last_published_version
                        and updates // sync_period
                        > last_published_version // sync_period):
                    _replace_latest(
                        weight_queue,
                        export_inference_weights(agent, version=updates))
                    last_published_version = updates

            done = bool(transition["terminated"][0]
                        or transition["truncated"][0])
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
                print(
                    f"[async step {steps}] falls={falls} updates={updates} "
                    f"collector_ms_p50={np.median(recent):.2f} "
                    f"runtime_q={item.get('runtime_queue_depth', 0)}",
                    flush=True)
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
                }, indent=2) + "\n")
                last_saved = steps
                while next_checkpoint <= steps:
                    next_checkpoint += checkpoint_interval
        status = "finished"
    finally:
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
