"""Construction and live-loop helpers shared by SQRL runners."""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from runtime.inference.transport import SharedMemoryReceiver, SharedMemoryRingQueue

from ..algo.buffers import ReplayBuffer, SafetyReplayBuffer
from ..algo.sac import SACConfig, VanillaSAC
from ..algo.safety_critic import SafetyCriticConfig, SafetyCriticLearner
from ..algo.safety_policy import SafetyPolicy
from ..config import ExperimentConfig


def build_core(cfg: ExperimentConfig, *, seed: int, device: str):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sac = VanillaSAC(SACConfig(
        observation_dim=cfg.stacked_observation_dim,
        action_dim=cfg.environment.action_dim,
        hidden_dims=cfg.training.hidden_dims,
    ), device=device)
    safety = SafetyCriticLearner(SafetyCriticConfig(
        observation_dim=cfg.stacked_observation_dim,
        action_dim=cfg.environment.action_dim,
        hidden_dims=cfg.training.hidden_dims,
        gamma=cfg.sqrl.gamma_safe,
    ), device=device)
    task_replay = ReplayBuffer(
        cfg.replay.task_capacity, (cfg.stacked_observation_dim,),
        cfg.environment.action_dim, seed)
    safety_replay = SafetyReplayBuffer(cfg.replay.safety_trajectories, seed + 1)
    policy = SafetyPolicy(
        sac.actor, safety.critic, cfg.sqrl.epsilon_safe,
        cfg.sqrl.mask_candidates, sac.device)
    return sac, safety, task_replay, safety_replay, policy
def append_jsonl(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(values, sort_keys=True) + "\n")


def module_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def write_json(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2, sort_keys=True)
        handle.write("\n")


def launch_owned_runtime(config_path: str) -> subprocess.Popen[bytes]:
    """Start the ordered runtime only after the collector is waiting.

    The simulator and C++ controller remain externally owned. This helper
    removes only a stale queue from a previously stopped producer. The runner
    creates and clears the action mailbox before starting the runtime, so a
    fresh OS session does not depend on shared memory surviving an earlier run.
    """
    SharedMemoryRingQueue.unlink_existing("go2_runtime_state.ordered")
    receiver = SharedMemoryReceiver("go2_runtime_action")
    try:
        receiver.bind()
        receiver.clear()
    finally:
        receiver.close()
    return subprocess.Popen([
        sys.executable, "-m", "runtime.inference", "--config", config_path,
        "--ordered-state-queue",
    ])


def stop_owned_runtime(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
