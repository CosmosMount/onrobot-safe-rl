#!/usr/bin/env python3
"""Benchmark one MuJoCo-Warp Go2 capacity rung without artificial pushes.

Run this script from an isolated MjLab environment.  One process benchmarks
one capacity so Warp and CUDA allocations are released between ladder rungs.
The actor has the official Go2 PPO MLP dimensions; this command measures
environment stepping, automatic terminal resets, observations and inference,
but does not update PPO weights.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from safety_data.mjlab_target_alignment import (
    configure_target_aligned_go2,
    target_alignment_manifest,
    validate_target_aligned_go2,
)


class GpuSampler:
    def __init__(self, gpu_index: int = 0) -> None:
        self.gpu_index = int(gpu_index)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples_mib: list[int] = []
        self.samples_utilization: list[int] = []
        self.measurement_start_index: int | None = None
        self.error: str | None = None

    def _sample_once(self) -> None:
        value = subprocess.run(
            ["nvidia-smi", f"--id={self.gpu_index}",
             "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        memory, utilization = value.splitlines()[0].split(",")
        self.samples_mib.append(int(memory.strip()))
        self.samples_utilization.append(int(utilization.strip()))

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample_once()
            except Exception as exc:  # fail closed in the main thread
                self.error = f"{type(exc).__name__}: {exc}"
                self._stop.set()
                return
            self._stop.wait(0.5)

    def __enter__(self) -> "GpuSampler":
        # A synchronous baseline guarantees that initialization/JIT is inside
        # the monitored interval rather than racing the sampler thread.
        self._sample_once()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def mark_measurement_start(self) -> None:
        self.measurement_start_index = len(self.samples_mib)

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        assert self._thread is not None
        self._thread.join(timeout=5.0)


class OfficialSizeActor(nn.Module):
    def __init__(self, observation_size: int, action_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, action_size),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(observation))


def _versions() -> dict[str, str]:
    import mujoco
    import mujoco_warp
    import warp
    return {
        "mujoco": str(mujoco.__version__),
        "mujoco_warp": str(mujoco_warp.__version__),
        "warp": str(warp.__version__),
    }


def _git_head() -> str:
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True).stdout.strip()


def _edge_median(values: list[int], *, first: bool) -> float:
    if not values:
        return 0.0
    count = max(1, len(values) // 5)
    selected = values[:count] if first else values[-count:]
    return float(np.median(selected))


def capacity_run_passes(
    *, elapsed_seconds: float, minimum_seconds: float, peak_vram_mib: int,
    memory_growth_mib: float, nonfinite: bool, external_force_nonzero: bool,
    push_event_present: bool, gpu_sampling_error: bool = False,
    initialization_peak_monitored: bool = True,
    measured_gpu_memory_samples: int = 0,
) -> bool:
    return bool(
        elapsed_seconds >= minimum_seconds
        and peak_vram_mib <= 20480
        and memory_growth_mib <= 128.0
        and not nonfinite
        and not external_force_nonzero
        and not push_event_present
        and not gpu_sampling_error
        and initialization_peak_monitored
        and (minimum_seconds == 0.0 or measured_gpu_memory_samples >= 10)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--minimum-measured-seconds", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.envs <= 0 or args.warmup_steps < 0 or args.steps <= 0 or (
            args.minimum_measured_seconds < 0.0):
        raise ValueError("envs and steps must be positive")

    import mjlab.tasks  # noqa: F401
    import src.tasks  # type: ignore  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    torch.manual_seed(args.seed)
    cfg = configure_target_aligned_go2(load_env_cfg("Unitree-Go2-Flat"))
    cfg.seed = args.seed
    cfg.scene.num_envs = args.envs
    validate_target_aligned_go2(cfg)

    terminated_count = 0
    truncated_count = 0
    nonfinite = False
    external_force_nonzero = False
    env = None
    with GpuSampler() as gpu:
        # Establish PyTorch's CUDA context inside the monitored interval before
        # asking its allocator for peak statistics.
        torch.empty(0, device="cuda:0")
        torch.cuda.reset_peak_memory_stats()
        started_initialization = time.perf_counter()
        env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
        observation, _ = env.reset()
        actor_observation = observation["actor"]
        actor = OfficialSizeActor(actor_observation.shape[1], 12).to("cuda:0")
        torch.cuda.synchronize()
        initialization_seconds = time.perf_counter() - started_initialization
        try:
            with torch.inference_mode():
                started = time.perf_counter()
                if args.warmup_steps == 0:
                    gpu.mark_measurement_start()
                for index in range(args.warmup_steps + args.steps):
                    action = actor(actor_observation)
                    observation, _, terminated, truncated, _ = env.step(action)
                    actor_observation = observation["actor"]
                    if index >= args.warmup_steps:
                        terminated_count += int(terminated.sum().item())
                        truncated_count += int(truncated.sum().item())
                        nonfinite = nonfinite or not bool(torch.isfinite(
                            actor_observation).all().item())
                        force = env.sim.data.xfrc_applied
                        external_force_nonzero = external_force_nonzero or bool(
                            torch.any(force != 0.0).item())
                    if args.warmup_steps > 0 and index + 1 == args.warmup_steps:
                        torch.cuda.synchronize()
                        gpu.mark_measurement_start()
                        started = time.perf_counter()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
        finally:
            env.close()
        process_peak_allocated_mib = int(
            torch.cuda.max_memory_allocated() / (1024 * 1024))
        process_peak_reserved_mib = int(
            torch.cuda.max_memory_reserved() / (1024 * 1024))

    measured_start = gpu.measurement_start_index
    measured_memory = (
        gpu.samples_mib[measured_start:] if measured_start is not None else [])
    memory_growth_mib = (
        _edge_median(measured_memory, first=False)
        - _edge_median(measured_memory, first=True)
    )
    peak_vram_mib = max(gpu.samples_mib, default=0)
    push_event_present = "push_robot" in cfg.events
    result = {
        "schema_version": "qsafe.mjlab_go2_capacity.v1",
        "backend": "unitree_mjlab_mujoco_warp",
        "algorithm_path": "official_size_ppo_actor_inference_no_update",
        "generator_commit": _git_head(),
        "envs": args.envs,
        "policy_steps": args.steps,
        "policy_env_steps": args.envs * args.steps,
        "elapsed_seconds": elapsed,
        "minimum_measured_seconds": args.minimum_measured_seconds,
        "policy_env_steps_per_second": args.envs * args.steps / elapsed,
        "initialization_seconds": initialization_seconds,
        "terminated_count": terminated_count,
        "resets_per_second": (terminated_count + truncated_count) / elapsed,
        "truncated_count": truncated_count,
        "nonfinite": nonfinite,
        "external_force_nonzero": external_force_nonzero,
        "peak_total_gpu_memory_mib": peak_vram_mib,
        "initialization_peak_monitored": True,
        "process_peak_allocated_mib": process_peak_allocated_mib,
        "process_peak_reserved_mib": process_peak_reserved_mib,
        "mean_total_gpu_memory_mib": float(
            np.mean(gpu.samples_mib) if gpu.samples_mib else 0.0),
        "memory_growth_mib": memory_growth_mib,
        "mean_gpu_utilization_percent": float(
            np.mean(gpu.samples_utilization)
            if gpu.samples_utilization else 0.0),
        "peak_gpu_utilization_percent": max(gpu.samples_utilization, default=0),
        "gpu_memory_samples": len(gpu.samples_mib),
        "measured_gpu_memory_samples": len(measured_memory),
        "gpu_sampling_error": gpu.error,
        "command_distribution": {
            "type": "constant", "vx": 0.30,
            "vy": 0.0, "yaw_rate": 0.0,
        },
        "push_event_present": push_event_present,
        "target_alignment": target_alignment_manifest(),
        "versions": _versions(),
    }
    result["pass"] = capacity_run_passes(
        elapsed_seconds=elapsed,
        minimum_seconds=args.minimum_measured_seconds,
        peak_vram_mib=peak_vram_mib,
        memory_growth_mib=memory_growth_mib,
        nonfinite=nonfinite,
        external_force_nonzero=external_force_nonzero,
        push_event_present=push_event_present,
        gpu_sampling_error=gpu.error is not None,
        initialization_peak_monitored=True,
        measured_gpu_memory_samples=len(measured_memory),
    )
    rendered = json.dumps(result, sort_keys=True, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write((rendered + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, args.output)
    except FileExistsError as exc:
        raise FileExistsError("capacity output path was already consumed") from exc
    temporary.unlink()
    directory = os.open(args.output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    if not result["pass"]:
        raise RuntimeError("parallel PPO capacity rung failed")


if __name__ == "__main__":
    main()
