"""Deterministic native inference for the target-aligned natural-PPO actor."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from runtime.inference.velocity import quat_world_to_body


_INIT_Q = np.asarray([0.05, 0.70, -1.40] * 4, dtype=np.float32)
_ACTION_SCALE = np.asarray([0.2, 0.4, 0.4] * 4, dtype=np.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class NaturalPpoRecoveryPolicy:
    """Frozen noise-free RSL-RL PPO mean mapped to native normalized actions."""

    def __init__(self, checkpoint: str | Path) -> None:
        self.path = Path(checkpoint).resolve()
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        state = payload.get("actor_state_dict")
        if not isinstance(state, dict):
            raise ValueError("PPO checkpoint has no actor_state_dict")
        shapes = ((47, 512), (512, 256), (256, 128), (128, 12))
        layers = []
        for index, (input_dim, output_dim) in enumerate(shapes):
            linear = nn.Linear(input_dim, output_dim)
            weight = state[f"mlp.{2 * index}.weight"]
            bias = state[f"mlp.{2 * index}.bias"]
            if tuple(weight.shape) != (output_dim, input_dim) or tuple(
                    bias.shape) != (output_dim,):
                raise ValueError("PPO actor MLP shape differs from target contract")
            linear.weight.data.copy_(weight)
            linear.bias.data.copy_(bias)
            layers.append(linear)
            if index + 1 < len(shapes):
                layers.append(nn.ELU())
        self.model = nn.Sequential(*layers).eval()
        self.mean = np.asarray(state["obs_normalizer._mean"], dtype=np.float32).reshape(47)
        self.std = np.asarray(state["obs_normalizer._std"], dtype=np.float32).reshape(47)
        if not np.all(np.isfinite(self.mean)) or not np.all(self.std >= 0):
            raise ValueError("PPO observation normalizer is invalid")
        self.checkpoint_sha256 = _sha256(self.path)
        self.iteration = int(payload.get("iter", -1))

    def observation(self, env: object, previous_ppo_action: np.ndarray) -> np.ndarray:
        state = env.robot_state()
        gravity_body = quat_world_to_body(
            np.asarray([0.0, 0.0, -1.0], dtype=np.float32), state.imu_quat)
        phase = (float(env.data.time) % 0.6) / 0.6
        phase_pair = np.asarray([
            np.sin(phase * 2.0 * np.pi), np.cos(phase * 2.0 * np.pi)
        ], dtype=np.float32)
        observation = np.concatenate((
            state.imu_gyro,
            gravity_body,
            np.asarray([0.30, 0.0, 0.0], dtype=np.float32),
            phase_pair,
            state.joint_q - _INIT_Q,
            state.joint_dq,
            np.asarray(previous_ppo_action, dtype=np.float32).reshape(12),
        )).astype(np.float32)
        if observation.shape != (47,) or not np.all(np.isfinite(observation)):
            raise RuntimeError("native PPO recovery observation is invalid")
        return observation

    def action(self, env: object, previous_ppo_action: np.ndarray) -> np.ndarray:
        observation = self.observation(env, previous_ppo_action)
        normalized = (observation - self.mean) / (self.std + 1e-2)
        with torch.inference_mode():
            raw = self.model(torch.from_numpy(normalized)).numpy()
        # Native SAC actions are normalized and projected to target joint
        # limits.  Clipping is explicit because MjLab's Gaussian actor itself
        # is unbounded, while the shared deployable actuator API is not.
        return np.clip(raw, -1.0, 1.0).astype(np.float32)

    def initial_previous_action(self, env: object) -> np.ndarray:
        return np.clip(
            (np.asarray(env.previous_action_q_target, dtype=np.float32) - _INIT_Q)
            / _ACTION_SCALE, -1.0, 1.0).astype(np.float32)

    def manifest(self) -> dict[str, object]:
        return {
            "checkpoint": str(self.path), "checkpoint_sha256": self.checkpoint_sha256,
            "iteration": self.iteration, "actor_observation_dim": 47,
            "action_dim": 12, "action_output_projection": "clip_to_native_minus1_plus1",
            "command_vx_mps": 0.30, "phase_period_s": 0.6,
        }
