"""Run SQRL Algorithm 1 against the standalone Go2 runtime."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
from pathlib import Path

from ..algo.checkpoint import save_pretrain_checkpoint
from ..algo.pretrain import SQRLPretrainer
from ..config import load_config
from ..env.async_collector import inference_snapshot, run_async_collector
from .common import (
    append_jsonl, build_core, launch_owned_runtime, module_sha256,
    stop_owned_runtime, write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="reproductions/sqrl_go2/config/pretrain_030.yaml")
    parser.add_argument("--steps", type=int, required=True,
                        help="Development budget selected during Phase 0; no hidden formal default.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--protocol-id", default="development")
    parser.add_argument(
        "--launch-runtime", action="store_true",
        help="Own the ordered Python runtime lifecycle; simulator/controller must already run.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if cfg.phase != "pretrain" or args.steps <= 0:
        raise SystemExit("pretrain config and positive --steps are required")
    seed = cfg.training.seed if args.seed is None else args.seed
    device = cfg.training.device if args.device is None else args.device
    sac, safety, task_replay, safety_replay, policy = build_core(
        cfg, seed=seed, device=device)
    trainer = SQRLPretrainer(
        sac, safety, task_replay, safety_replay, policy,
        batch_size=cfg.replay.batch_size,
        minimum_task_transitions=cfg.replay.minimum_task_transitions,
        minimum_safety_transitions=cfg.replay.minimum_safety_transitions,
        task_steps_per_cycle=cfg.sqrl.task_steps_per_cycle,
        safety_trajectories_per_cycle=cfg.sqrl.safety_trajectories_per_cycle,
        safety_updates_per_cycle=cfg.sqrl.safety_updates_per_cycle)
    output_root = Path(args.output_root or cfg.training.output_dir)
    output = output_root / f"seed_{seed}" / "pretrain_030"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite run directory: {output}")
    metrics_path = output / "metrics.jsonl"
    manifest_path = output / "manifest.json"
    write_json(manifest_path, {
        "status": "running", "phase": "pretrain", "seed": seed,
        "move_speed": cfg.move_speed, "max_steps": args.steps,
        "protocol_id": args.protocol_id,
    })
    context = mp.get_context("spawn")
    weight_queue = context.Queue(maxsize=2)
    transition_queue = context.Queue(maxsize=8192)
    control_queue = context.Queue(maxsize=1)
    weight_queue.put(inference_snapshot(sac.actor, safety.critic))
    collector = context.Process(target=run_async_collector, kwargs={
        "config_path": args.config, "branch": "pretrain", "seed": seed,
        "weight_queue": weight_queue, "transition_queue": transition_queue,
        "control_queue": control_queue,
    })
    collector.start()
    runtime_process = launch_owned_runtime(args.config) if args.launch_runtime else None
    falls = 0
    episode_return = 0.0
    episode_tracking_error = 0.0
    episode_steps = 0
    completed_episodes = 0
    safety_collection_falls = 0
    try:
        for step in range(1, args.steps + 1):
            for _ in range(120):
                try:
                    collected = transition_queue.get(timeout=1.0)
                    break
                except queue.Empty:
                    if not collector.is_alive():
                        raise RuntimeError(
                            "50 Hz SQRL collector exited before producing a transition: "
                            f"exitcode={collector.exitcode}")
            else:
                raise RuntimeError("50 Hz SQRL collector produced no transition for 120 seconds")
            metrics = trainer.observe(
                collected.transition, collection_phase=collected.phase)
            falls += int(collected.transition.cost)
            episode_return += collected.transition.reward
            velocity = float(collected.info.get(
                "forward_velocity", collected.info.get("x_velocity", 0.0)))
            if collected.phase == "safety":
                safety_collection_falls += int(collected.transition.cost)
            episode_tracking_error += abs(velocity - cfg.move_speed)
            episode_steps += 1
            if collected.mask is not None:
                metrics.update({
                    "mask/risk": collected.mask.risk,
                    "mask/accepted": float(collected.mask.accepted),
                    "mask/no_safe_candidate": float(collected.mask.no_safe_candidate),
                    "mask/candidate_count": float(collected.mask.candidate_count),
                    "mask/intervened": float(collected.mask.candidate_count > 1),
                    "safety/q_mean": collected.mask.risk_mean,
                    "safety/q_p50": collected.mask.risk_p50,
                    "safety/q_p90": collected.mask.risk_p90,
                })
            metrics.update({
                "step": step, "falls": falls,
                "falls_per_1000_steps": 1000.0 * falls / step,
                "reward": collected.transition.reward,
                "forward_velocity": velocity,
                "velocity_tracking_error": abs(velocity - cfg.move_speed),
                "phase": collected.phase,
                "safety/collection_falls": safety_collection_falls,
                "collector/ordered_queue_depth": collected.queue_depth,
            })
            if collected.transition.terminated or collected.transition.truncated:
                completed_episodes += 1
                metrics.update({
                    "episode/return": episode_return,
                    "episode/length": episode_steps,
                    "episode/count": completed_episodes,
                    "episode/velocity_tracking_error": (
                        episode_tracking_error / max(episode_steps, 1)),
                    "episode/fall": collected.transition.cost,
                })
            append_jsonl(metrics_path, metrics)
            if collected.transition.terminated or collected.transition.truncated:
                episode_return = 0.0
                episode_tracking_error = 0.0
                episode_steps = 0
            if step % 10 == 0:
                while True:
                    try:
                        weight_queue.get_nowait()
                    except queue.Empty:
                        break
                weight_queue.put_nowait(inference_snapshot(sac.actor, safety.critic))
            if step % cfg.training.checkpoint_interval == 0:
                save_pretrain_checkpoint(
                    output / f"step_{step:012d}.pt", sac, safety,
                    {"seed": seed, "step": step, "falls": falls})
        save_pretrain_checkpoint(
            output / "final.pt", sac, safety,
            {"seed": seed, "step": args.steps, "falls": falls,
             "actor_sha256": module_sha256(sac.actor),
             "safety_sha256": module_sha256(safety.critic)})
        write_json(manifest_path, {
            "status": "finished", "phase": "pretrain", "seed": seed,
            "move_speed": cfg.move_speed, "completed_steps": args.steps,
            "falls": falls, "actor_sha256": module_sha256(sac.actor),
            "safety_sha256": module_sha256(safety.critic),
            "checkpoint": str(output / "final.pt"),
            "completed_episodes": completed_episodes,
            "safety_collection_falls": safety_collection_falls,
            "protocol_id": args.protocol_id,
        })
    except BaseException as exc:
        write_json(manifest_path, {
            "status": "failed", "phase": "pretrain", "seed": seed,
            "move_speed": cfg.move_speed, "max_steps": args.steps,
            "protocol_id": args.protocol_id,
            "error": repr(exc),
        })
        raise
    finally:
        if collector.is_alive():
            try:
                control_queue.put_nowait("stop")
            except queue.Full:
                pass
            collector.join(timeout=10.0)
        if collector.is_alive():
            collector.terminate()
            collector.join(timeout=5.0)
        stop_owned_runtime(runtime_process)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
