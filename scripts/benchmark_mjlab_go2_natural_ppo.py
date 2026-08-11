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
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

import numpy as np
import torch
from torch import nn


class GpuSampler:
    def __init__(self, gpu_index: int = 0) -> None:
        self.gpu_index = int(gpu_index)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples_mib: list[int] = []

    def _sample(self) -> None:
        while not self._stop.is_set():
            value = subprocess.run(
                ["nvidia-smi", f"--id={self.gpu_index}",
                 "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            self.samples_mib.append(int(value.splitlines()[0]))
            self._stop.wait(0.5)

    def __enter__(self) -> "GpuSampler":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.envs <= 0 or args.warmup_steps < 0 or args.steps <= 0:
        raise ValueError("envs and steps must be positive")

    import mjlab.tasks  # noqa: F401
    import src.tasks  # type: ignore  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    torch.manual_seed(args.seed)
    cfg = load_env_cfg("Unitree-Go2-Flat")
    cfg.seed = args.seed
    cfg.scene.num_envs = args.envs
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (0.30, 0.30)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)

    started_initialization = time.perf_counter()
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    observation, _ = env.reset()
    actor_observation = observation["actor"]
    actor = OfficialSizeActor(actor_observation.shape[1], 12).to("cuda:0")
    initialization_seconds = time.perf_counter() - started_initialization

    terminated_count = 0
    truncated_count = 0
    nonfinite = False
    external_force_nonzero = False
    with torch.inference_mode(), GpuSampler() as gpu:
        started = time.perf_counter()
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
                started = time.perf_counter()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

    env.close()
    result = {
        "schema_version": "qsafe.mjlab_go2_capacity.v1",
        "backend": "unitree_mjlab_mujoco_warp",
        "algorithm_path": "official_size_ppo_actor_inference_no_update",
        "envs": args.envs,
        "policy_steps": args.steps,
        "policy_env_steps": args.envs * args.steps,
        "elapsed_seconds": elapsed,
        "policy_env_steps_per_second": args.envs * args.steps / elapsed,
        "initialization_seconds": initialization_seconds,
        "terminated_count": terminated_count,
        "truncated_count": truncated_count,
        "nonfinite": nonfinite,
        "external_force_nonzero": external_force_nonzero,
        "peak_total_gpu_memory_mib": max(gpu.samples_mib, default=0),
        "gpu_memory_samples": len(gpu.samples_mib),
        "fixed_command": {"vx": 0.30, "vy": 0.0, "yaw_rate": 0.0},
        "push_event_present": "push_robot" in cfg.events,
        "versions": _versions(),
    }
    rendered = json.dumps(result, sort_keys=True, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
