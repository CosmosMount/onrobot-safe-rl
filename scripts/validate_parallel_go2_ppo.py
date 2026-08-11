"""Bounded native-MuJoCo parallel PPO smoke test for the Go2 safety PoC.

This is deliberately a validation harness, not the Q_safe trainer.  It uses
the deployment-like absolute q-target action contract, runs independent Go2
workers, and reports throughput plus a short PPO update trace.  The resulting
states are proposal data only; no safety labels are exported here.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any

import mujoco
import numpy as np
import torch
from torch import nn


INIT_Q = np.tile(np.asarray([0.05, 0.7, -1.4], dtype=np.float64), 4)
ACTION_SCALE = np.tile(np.asarray([0.2, 0.4, 0.4], dtype=np.float64), 4)
KP = np.tile(np.asarray([20.0, 20.0, 40.0], dtype=np.float64), 4)
KD = np.tile(np.asarray([1.0, 1.0, 2.0], dtype=np.float64), 4)


def _projected_gravity(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.asarray([
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        w * w - x * x - y * y + z * z,
    ], dtype=np.float64)


def _observe(data: mujoco.MjData, previous_q: np.ndarray) -> np.ndarray:
    # Generator observation: q/dq, body gyro/velocity, projected gravity and
    # the absolute q target actually sent.  This is intentionally not claimed
    # to be the corrected 46D deployment observation used by Q_safe.
    return np.concatenate((
        data.qpos[7:19], data.qvel[6:18], data.qvel[3:6], data.qvel[:3],
        _projected_gravity(data.qpos[3:7]), previous_q,
    )).astype(np.float32)


def _worker(conn: Any, model_path: str, seed: int, physics_steps: int) -> None:
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    previous_q = INIT_Q.copy()

    def reset() -> np.ndarray:
        nonlocal previous_q
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = np.asarray([0.0, 0.0, 0.45])
        data.qpos[3:7] = np.asarray([1.0, 0.0, 0.0, 0.0])
        data.qpos[7:19] = INIT_Q + rng.normal(0.0, 0.01, size=12)
        data.qvel[:] = 0.0
        previous_q = INIT_Q.copy()
        mujoco.mj_forward(model, data)
        return _observe(data, previous_q)

    try:
        conn.send(("ready", reset()))
        while True:
            command, action = conn.recv()
            if command == "close":
                return
            if command == "reset":
                conn.send(("transition", reset(), 0.0, False, False))
                continue
            if command != "step":
                raise ValueError(f"unknown worker command: {command}")
            action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
            target_q = INIT_Q + ACTION_SCALE * action
            for _ in range(physics_steps):
                torque = KP * (target_q - data.qpos[7:19]) - KD * data.qvel[6:18]
                data.ctrl[:] = np.clip(torque, model.actuator_ctrlrange[:, 0],
                                       model.actuator_ctrlrange[:, 1])
                mujoco.mj_step(model, data)
            previous_q = target_q.copy()
            tilt = 2.0 * np.arccos(np.clip(data.qpos[3], -1.0, 1.0))
            fall = bool(data.qpos[2] < 0.18 or tilt > 0.9)
            reward = float(1.0 - 2.0 * fall - 0.01 * np.mean(action * action))
            conn.send(("transition", _observe(data, previous_q), reward, fall, fall))
    finally:
        conn.close()


@dataclass
class ParallelWorkers:
    parents: list[Any]
    processes: list[mp.Process]

    @classmethod
    def start(cls, model_path: str, count: int, physics_steps: int) -> "ParallelWorkers":
        parents, processes = [], []
        for seed in range(count):
            parent, child = mp.Pipe()
            process = mp.Process(target=_worker,
                                 args=(child, model_path, 1000 + seed, physics_steps))
            process.start()
            child.close()
            parents.append(parent)
            processes.append(process)
        for parent in parents:
            status, _ = parent.recv()
            if status != "ready":
                raise RuntimeError("Go2 worker failed to initialize")
        return cls(parents, processes)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for parent, action in zip(self.parents, actions, strict=True):
            parent.send(("step", action))
        values = [parent.recv()[1:] for parent in self.parents]
        observations, rewards, dones, falls = zip(*values, strict=True)
        return (np.stack(observations), np.asarray(rewards, dtype=np.float32),
                np.asarray(dones, dtype=bool))

    def close(self) -> None:
        for parent in self.parents:
            parent.send(("close", None))
        for process in self.processes:
            process.join(timeout=5.0)
        for parent in self.parents:
            parent.close()


class ActorCritic(nn.Module):
    def __init__(self, observation_size: int, action_size: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(observation_size, 128), nn.Tanh(),
                                  nn.Linear(128, 128), nn.Tanh())
        self.mean = nn.Linear(128, action_size)
        self.value = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.full((action_size,), -1.0))

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(observations)
        return torch.tanh(self.mean(hidden)), self.value(hidden).squeeze(-1)

    def distribution(self, observations: torch.Tensor) -> tuple[torch.distributions.Normal, torch.Tensor]:
        hidden = self.body(observations)
        return torch.distributions.Normal(self.mean(hidden), self.log_std.exp()), self.value(hidden).squeeze(-1)

    def sample(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self.distribution(observations)
        latent = distribution.rsample()
        action = torch.tanh(latent)
        log_probability = distribution.log_prob(latent).sum(-1)
        log_probability -= torch.log(1.0 - action.square() + 1e-6).sum(-1)
        return action, log_probability, value

    def log_probability(self, observations: torch.Tensor,
                        action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distribution, value = self.distribution(observations)
        clipped = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        latent = torch.atanh(clipped)
        log_probability = distribution.log_prob(latent).sum(-1)
        log_probability -= torch.log(1.0 - clipped.square() + 1e-6).sum(-1)
        return log_probability, value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--physics-steps", type=int, default=10)
    parser.add_argument("--model", type=Path,
                        default=Path("/home/xyz/code/unitree_mujoco/unitree_robots/go2/scene_empty.xml"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.envs <= 0 or args.iterations <= 0 or args.rollout_steps <= 0:
        raise ValueError("envs, iterations and rollout-steps must be positive")
    torch.manual_seed(0)
    workers = ParallelWorkers.start(str(args.model), args.envs, args.physics_steps)
    agent: ActorCritic | None = None
    optimizer: torch.optim.Optimizer | None = None
    total_steps = 0
    falls = 0
    started = time.perf_counter()
    try:
        for _ in range(args.iterations):
            observations = []
            rewards = []
            actions = []
            old_log_probabilities = []
            old_values = []
            # Re-obtain initial observations without sharing mutable worker state.
            for parent in workers.parents:
                parent.send(("reset", None))
            current = np.stack([parent.recv()[1] for parent in workers.parents])
            for _ in range(args.rollout_steps):
                if agent is None:
                    agent = ActorCritic(current.shape[1], 12)
                    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
                with torch.no_grad():
                    action, old_log_probability, old_value = agent.sample(
                        torch.from_numpy(current))
                action_np = action.numpy()
                next_obs, reward, done = workers.step(action_np)
                observations.append(current)
                actions.append(action_np)
                rewards.append(reward)
                old_log_probabilities.append(old_log_probability.numpy())
                old_values.append(old_value.numpy())
                falls += int(done.sum())
                current = next_obs
                total_steps += args.envs
            # Minimal PPO-style policy/value update proves the PPO path is live.
            obs_tensor = torch.from_numpy(np.concatenate(observations))
            action_tensor = torch.from_numpy(np.concatenate(actions))
            reward_tensor = torch.from_numpy(np.concatenate(rewards))
            old_log_probability_tensor = torch.from_numpy(
                np.concatenate(old_log_probabilities))
            old_value_tensor = torch.from_numpy(np.concatenate(old_values))
            assert agent is not None and optimizer is not None
            log_probability, value = agent.log_probability(obs_tensor, action_tensor)
            advantage = (reward_tensor - old_value_tensor).detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
            ratio = torch.exp(log_probability - old_log_probability_tensor.detach())
            clipped_ratio = ratio.clamp(1.0 - 0.2, 1.0 + 0.2)
            policy_loss = -torch.minimum(ratio * advantage,
                                         clipped_ratio * advantage).mean()
            value_loss = 0.5 * (value - reward_tensor).square().mean()
            entropy = agent.distribution(obs_tensor)[0].entropy().sum(-1).mean()
            loss = policy_loss + value_loss - 1e-3 * entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    finally:
        workers.close()
    elapsed = time.perf_counter() - started
    result = {
        "schema_version": "qsafe.parallel_go2_ppo_smoke.v1",
        "backend": "native_mujoco_subprocess_workers",
        "algorithm": "ppo_style_actor_critic_update",
        "envs": args.envs,
        "iterations": args.iterations,
        "rollout_steps": args.rollout_steps,
        "physics_steps_per_policy_step": args.physics_steps,
        "policy_steps": total_steps,
        "falls": falls,
        "elapsed_seconds": elapsed,
        "policy_env_steps_per_second": total_steps / max(elapsed, 1e-9),
        "model": str(args.model),
    }
    rendered = json.dumps(result, sort_keys=True, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
