"""GPU ring-buffer capture for natural falls in MjLab/MuJoCo-Warp.

This adapter is invoked immediately before MjLab resets a terminated world, so
terminal state is preserved while the environment still performs its normal
same-step vector reset.  It never changes physics or actions.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from safety_data.natural_ppo_falls import (
    NORMAL_TERMINAL_DISTANCE,
    PREFALL_OFFSETS,
    RING_POLICY_STEPS,
)


MJLAB_TO_TARGET_JOINT = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)
CAPTURE_RING_STEPS = 97
NORMAL_HASH_MODULUS = 32


def target_fall_predicate(qpos: torch.Tensor) -> torch.Tensor:
    """Apply the target native-MuJoCo height/roll/pitch fall predicate."""
    if qpos.ndim != 2 or qpos.shape[1] < 7:
        raise ValueError("qpos must have shape [N, >=7]")
    quat = qpos[:, 3:7]
    quat = quat / torch.linalg.vector_norm(quat, dim=1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = quat.unbind(dim=1)
    roll = torch.atan2(2.0 * (w * x + y * z),
                       1.0 - 2.0 * (x.square() + y.square()))
    sin_pitch = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0)
    pitch = torch.asin(sin_pitch)
    tilt = torch.maximum(roll.abs(), pitch.abs())
    return (qpos[:, 2] < 0.18) | (tilt >= 1.047198)


def ordered_ring_indices(count: int, capacity: int = RING_POLICY_STEPS) -> np.ndarray:
    if count < 0 or capacity <= 0:
        raise ValueError("count must be non-negative and capacity positive")
    length = min(count, capacity)
    start = (count - length) % capacity
    return (start + np.arange(length, dtype=np.int64)) % capacity


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class MjlabFallShardWriter:
    def __init__(self, root: str | Path, *, events_per_shard: int = 256) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.events_per_shard = int(events_per_shard)
        if self.events_per_shard <= 0:
            raise ValueError("events_per_shard must be positive")
        self.pending: list[dict[str, np.ndarray]] = []
        self.shards: list[dict[str, Any]] = []
        self.identities: set[str] = set()

    def add(self, event: dict[str, np.ndarray]) -> None:
        identity = bytes(event["identity"].item()).decode("ascii")
        if identity in self.identities:
            raise RuntimeError("duplicate MjLab natural-fall identity")
        self.identities.add(identity)
        self.pending.append(event)
        if len(self.pending) >= self.events_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        arrays = {
            name: np.stack([event[name] for event in self.pending])
            for name in self.pending[0]
        }
        path = self.root / f"falls-{len(self.shards):06d}.npz"
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
        np.savez_compressed(temporary, **arrays)
        content = temporary.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        os.replace(temporary, path)
        _fsync_dir(path.parent)
        self.shards.append({
            "path": path.name,
            "sha256": digest,
            "event_count": len(self.pending),
        })
        self.pending.clear()

    def close(self, provenance: Mapping[str, Any]) -> Path:
        self.flush()
        manifest = {
            "schema_version": "qsafe.mjlab_natural_falls.v1",
            "event_count": sum(item["event_count"] for item in self.shards),
            "prefall_offsets": list(PREFALL_OFFSETS),
            "terminal_state_captured_before_vector_reset": True,
            "reset_occurs_in_same_environment_step": True,
            "external_force": "verified_zero",
            "ppo_outcomes_are_qsafe_labels": False,
            "shards": self.shards,
            "provenance": dict(provenance),
        }
        content = (json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                   + "\n").encode("utf-8")
        path = self.root / "manifest.json"
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
        return path


class MjlabNormalShardWriter(MjlabFallShardWriter):
    """Use the same atomic sharding contract for delayed-safe normal frames."""

    def flush(self) -> None:
        if not self.pending:
            return
        arrays = {
            name: np.stack([event[name] for event in self.pending])
            for name in self.pending[0]
        }
        path = self.root / f"normals-{len(self.shards):06d}.npz"
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
        np.savez_compressed(temporary, **arrays)
        content = temporary.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        os.replace(temporary, path)
        _fsync_dir(path.parent)
        self.shards.append({
            "path": path.name,
            "sha256": digest,
            "event_count": len(self.pending),
        })
        self.pending.clear()

    def close(self, provenance: Mapping[str, Any]) -> Path:
        self.flush()
        manifest = {
            "schema_version": "qsafe.mjlab_natural_normals.v1",
            "event_count": sum(item["event_count"] for item in self.shards),
            "minimum_future_nonterminal_steps": NORMAL_TERMINAL_DISTANCE,
            "selection": "deterministic_hash_modulo_32_before_branch_outcomes",
            "shards": self.shards,
            "provenance": dict(provenance),
        }
        path = self.root / "manifest.json"
        content = (json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                   + "\n").encode("utf-8")
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
        return path


class MjlabNaturalFallCapture:
    """Maintain 65-step GPU histories and export every target-predicate fall."""

    def __init__(self, num_envs: int, output: str | Path, *, seed: int) -> None:
        self.num_envs = int(num_envs)
        self.seed = int(seed)
        self.writer = MjlabFallShardWriter(output)
        self.normal_writer = MjlabNormalShardWriter(Path(output) / "normals")
        self.armed = False
        self._allocated = False
        self.global_vector_step = 0
        self.fall_count = 0
        self.normal_count = 0

    def _allocate(self, env: Any) -> None:
        device = torch.device(env.device)
        n = self.num_envs
        c = CAPTURE_RING_STEPS
        self.count = torch.zeros(n, dtype=torch.long, device=device)
        self.episode_id = torch.zeros(n, dtype=torch.long, device=device)
        self.history = torch.zeros((n, 5, 46), dtype=torch.float32, device=device)
        self.qpos = torch.zeros((n, c, 19), dtype=torch.float64, device=device)
        self.qvel = torch.zeros((n, c, 18), dtype=torch.float64, device=device)
        self.ctrl = torch.zeros((n, c, 12), dtype=torch.float64, device=device)
        self.obs_history = torch.zeros((n, c, 5, 46), dtype=torch.float32,
                                       device=device)
        self.action_requested = torch.zeros((n, c, 12), dtype=torch.float32,
                                            device=device)
        self.action_executed = torch.zeros_like(self.action_requested)
        self.q_target = torch.zeros_like(self.action_requested)
        self.command = torch.zeros((n, c, 3), dtype=torch.float32, device=device)
        self.episode_step = torch.zeros((n, c), dtype=torch.long, device=device)
        self.policy_step = torch.zeros((n, c), dtype=torch.long, device=device)
        self._allocated = True

    def arm(self, env: Any) -> None:
        if not self._allocated:
            self._allocate(env)
        self.armed = True

    def before_step(self, env: Any, action: torch.Tensor) -> None:
        if not self.armed:
            return
        if bool(torch.any(env.sim.data.xfrc_applied != 0.0).item()):
            raise RuntimeError("natural PPO encountered non-zero xfrc_applied")
        robot = env.scene.entities["robot"]
        permutation = torch.as_tensor(
            MJLAB_TO_TARGET_JOINT, dtype=torch.long, device=action.device)
        joint_q = robot.data.joint_pos[:, permutation]
        joint_dq = robot.data.joint_vel[:, permutation]
        q_target = robot.data.joint_pos_target[:, permutation]
        observation = torch.cat((
            joint_q, joint_dq, robot.data.root_link_ang_vel_b,
            robot.data.root_link_lin_vel_b, robot.data.root_link_quat_w,
            q_target,
        ), dim=1).to(torch.float32)
        if observation.shape != (self.num_envs, 46):
            raise RuntimeError("corrected MjLab proposal observation is not 46D")
        first = self.count == 0
        self.history = torch.roll(self.history, shifts=-1, dims=1)
        self.history[:, -1] = observation
        if bool(first.any().item()):
            self.history[first] = observation[first, None, :].expand(-1, 5, -1)

        ids = torch.arange(self.num_envs, device=action.device)
        self._capture_mature_normals(env, ids)
        slot = self.count % CAPTURE_RING_STEPS
        self.qpos[ids, slot] = env.sim.data.qpos.to(torch.float64)
        self.qvel[ids, slot] = env.sim.data.qvel.to(torch.float64)
        self.ctrl[ids, slot] = env.sim.data.ctrl.to(torch.float64)
        self.obs_history[ids, slot] = self.history
        previous = env.action_manager.action.to(torch.float32)
        self.action_requested[ids, slot] = previous
        self.action_executed[ids, slot] = previous
        self.q_target[ids, slot] = q_target
        self.command[ids, slot] = env.command_manager.get_command("twist")
        self.episode_step[ids, slot] = env.episode_length_buf
        self.policy_step[ids, slot] = (
            self.global_vector_step * self.num_envs + ids)
        self.count += 1
        self.global_vector_step += 1

    def _capture_mature_normals(self, env: Any, ids: torch.Tensor) -> None:
        mature = self.count >= CAPTURE_RING_STEPS
        if not bool(mature.any().item()):
            return
        old_slot = (self.count - CAPTURE_RING_STEPS) % CAPTURE_RING_STEPS
        key = (
            self.policy_step[ids, old_slot]
            + 131 * self.episode_id
            + 17 * self.episode_step[ids, old_slot]
            + self.seed
        )
        selected = mature & ((key % NORMAL_HASH_MODULUS) == 0)
        for environment_id in selected.nonzero(as_tuple=False).flatten().cpu().tolist():
            self.normal_writer.add(self._normal_event(env, int(environment_id),
                                                      int(old_slot[environment_id].item())))
            self.normal_count += 1

    def before_reset(self, env: Any, env_ids: torch.Tensor) -> None:
        if not self.armed or len(env_ids) == 0:
            return
        if bool(torch.any(env.sim.data.xfrc_applied[env_ids] != 0.0).item()):
            raise RuntimeError("terminal natural PPO state has external force")
        qpos = env.sim.data.qpos[env_ids]
        fell = target_fall_predicate(qpos)
        fall_ids = env_ids[fell]
        for environment_id in fall_ids.detach().cpu().tolist():
            self.writer.add(self._event(env, int(environment_id)))
            self.fall_count += 1

    def after_reset(self, env_ids: torch.Tensor) -> None:
        if not self.armed or len(env_ids) == 0:
            return
        self.count[env_ids] = 0
        self.history[env_ids] = 0.0
        self.episode_id[env_ids] += 1

    def _event(self, env: Any, environment_id: int) -> dict[str, np.ndarray]:
        count = int(self.count[environment_id].item())
        indices = ordered_ring_indices(count, CAPTURE_RING_STEPS)[-RING_POLICY_STEPS:]
        length = len(indices)
        device_indices = torch.as_tensor(indices, dtype=torch.long,
                                         device=self.count.device)

        def padded(value: torch.Tensor) -> np.ndarray:
            selected = value[environment_id, device_indices].detach().cpu().numpy()
            result = np.zeros((RING_POLICY_STEPS, *selected.shape[1:]),
                              dtype=selected.dtype)
            result[:length] = selected
            return result

        availability = np.zeros(len(PREFALL_OFFSETS), dtype=bool)
        prefall_index = np.full(len(PREFALL_OFFSETS), -1, dtype=np.int16)
        for offset_index, offset in enumerate(PREFALL_OFFSETS):
            if length >= offset:
                availability[offset_index] = True
                prefall_index[offset_index] = length - offset

        terminal_qpos = env.sim.data.qpos[environment_id].detach().cpu().numpy()
        terminal_qvel = env.sim.data.qvel[environment_id].detach().cpu().numpy()
        terminal_ctrl = env.sim.data.ctrl[environment_id].detach().cpu().numpy()
        episode = int(self.episode_id[environment_id].item())
        raw_identity = np.asarray([
            self.seed, environment_id, episode, self.global_vector_step,
        ], dtype=np.uint64).tobytes()
        identity = hashlib.sha256(
            b"qsafe.mjlab_natural_fall.v1\0" + raw_identity).hexdigest()
        robot = env.scene.entities["robot"]
        return {
            "identity": np.asarray(identity.encode("ascii"), dtype="S64"),
            "environment_id": np.asarray(environment_id, dtype=np.int32),
            "episode_id": np.asarray(episode, dtype=np.int64),
            "trajectory_length": np.asarray(length, dtype=np.int16),
            "trajectory_mask": np.arange(RING_POLICY_STEPS) < length,
            "prefall_availability": availability,
            "prefall_trajectory_index": prefall_index,
            "trajectory_qpos": padded(self.qpos),
            "trajectory_qvel": padded(self.qvel),
            "trajectory_ctrl": padded(self.ctrl),
            "trajectory_observation_history": padded(self.obs_history),
            "trajectory_action_requested": padded(self.action_requested),
            "trajectory_action_executed": padded(self.action_executed),
            "trajectory_q_target": padded(self.q_target),
            "trajectory_command": padded(self.command),
            "trajectory_episode_step": padded(self.episode_step),
            "trajectory_policy_step": padded(self.policy_step),
            "terminal_qpos": terminal_qpos,
            "terminal_qvel": terminal_qvel,
            "terminal_ctrl": terminal_ctrl,
            "terminal_action_requested": env.action_manager.action[
                environment_id].detach().cpu().numpy().astype(np.float32),
            "terminal_command": env.command_manager.get_command("twist")[
                environment_id].detach().cpu().numpy().astype(np.float32),
            "randomized_geom_friction": env.sim.model.geom_friction[
                environment_id].detach().cpu().numpy(),
            "randomized_body_ipos": env.sim.model.body_ipos[
                environment_id].detach().cpu().numpy(),
            "randomized_encoder_bias": robot.data.encoder_bias[
                environment_id].detach().cpu().numpy(),
        }

    def _normal_event(self, env: Any, environment_id: int,
                      slot: int) -> dict[str, np.ndarray]:
        policy_step = int(self.policy_step[environment_id, slot].item())
        episode = int(self.episode_id[environment_id].item())
        raw_identity = np.asarray([
            self.seed, environment_id, episode, policy_step,
        ], dtype=np.uint64).tobytes()
        identity = hashlib.sha256(
            b"qsafe.mjlab_natural_normal.v1\0" + raw_identity).hexdigest()
        robot = env.scene.entities["robot"]

        def cpu(value: torch.Tensor) -> np.ndarray:
            return value[environment_id, slot].detach().cpu().numpy()

        return {
            "identity": np.asarray(identity.encode("ascii"), dtype="S64"),
            "environment_id": np.asarray(environment_id, dtype=np.int32),
            "episode_id": np.asarray(episode, dtype=np.int64),
            "policy_step": np.asarray(policy_step, dtype=np.int64),
            "episode_step": cpu(self.episode_step),
            "qpos": cpu(self.qpos),
            "qvel": cpu(self.qvel),
            "ctrl": cpu(self.ctrl),
            "observation_history": cpu(self.obs_history),
            "action_requested": cpu(self.action_requested),
            "action_executed": cpu(self.action_executed),
            "q_target": cpu(self.q_target),
            "command": cpu(self.command),
            "qualification_future_nonterminal_steps": np.asarray(
                NORMAL_TERMINAL_DISTANCE, dtype=np.int16),
            "randomized_geom_friction": env.sim.model.geom_friction[
                environment_id].detach().cpu().numpy(),
            "randomized_body_ipos": env.sim.model.body_ipos[
                environment_id].detach().cpu().numpy(),
            "randomized_encoder_bias": robot.data.encoder_bias[
                environment_id].detach().cpu().numpy(),
        }

    def close(self, provenance: Mapping[str, Any]) -> Path:
        normal_manifest = self.normal_writer.close({
            **dict(provenance),
            "normal_candidates": self.normal_count,
        })
        return self.writer.close({
            **dict(provenance),
            "independent_fall_episodes": self.fall_count,
            "recorded_falls": self.fall_count,
            "normal_candidates": self.normal_count,
            "normal_manifest": str(normal_manifest.relative_to(self.writer.root)),
        })
