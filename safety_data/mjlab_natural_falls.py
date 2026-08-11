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
# A 1/32 pilot did not supply a normal for every pre-fall state after the
# preregistered age/randomization/whole-episode-role stratification.  Select
# four times as many outcome-blind candidates before the formal 30M run.
NORMAL_HASH_MODULUS = 8
RISK_HORIZON_POLICY_STEPS = 96


def target_order_action_and_qtarget(
    action: torch.Tensor,
    *,
    scale: torch.Tensor | float,
    offset: torch.Tensor | float,
    encoder_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map the action applied at this state into target joint order."""
    if action.ndim != 2 or action.shape[1] != len(MJLAB_TO_TARGET_JOINT):
        raise ValueError("action must have shape [N, 12]")
    if encoder_bias.shape != action.shape:
        raise ValueError("encoder_bias must match action shape")
    permutation = torch.as_tensor(
        MJLAB_TO_TARGET_JOINT, dtype=torch.long, device=action.device)
    raw = action.to(torch.float32)
    absolute = raw * scale + offset - encoder_bias
    return raw[:, permutation], absolute[:, permutation].to(torch.float32)


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
    def __init__(self, root: str | Path, *, events_per_shard: int = 1024) -> None:
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
            "schema_version": "qsafe.mjlab_natural_falls.v2",
            "event_count": sum(item["event_count"] for item in self.shards),
            "prefall_offsets": list(PREFALL_OFFSETS),
            "terminal_state_captured_before_vector_reset": True,
            "reset_occurs_in_same_environment_step": True,
            "external_force": "verified_zero",
            "direct_qsafe_supervision": {
                "state_risk": True,
                "executed_action_risk_under_ppo_continuation": "diagnostic_only",
                "counterfactual_recovery_action_risk": False,
                "horizon_policy_steps": RISK_HORIZON_POLICY_STEPS,
            },
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
            "schema_version": "qsafe.mjlab_natural_normals.v2",
            "event_count": sum(item["event_count"] for item in self.shards),
            "minimum_future_nonterminal_steps": NORMAL_TERMINAL_DISTANCE,
            "selection": (
                f"deterministic_hash_modulo_{NORMAL_HASH_MODULUS}"
                "_before_branch_outcomes"),
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

    def __init__(
        self,
        num_envs: int,
        output: str | Path,
        *,
        seed: int,
        rollout_steps: int = 125,
        preview_envs: int = 4,
        preview_policy_steps: int = 500,
    ) -> None:
        self.num_envs = int(num_envs)
        self.seed = int(seed)
        self.rollout_steps = int(rollout_steps)
        self.preview_envs = min(int(preview_envs), self.num_envs)
        self.preview_policy_steps = int(preview_policy_steps)
        if self.rollout_steps <= 0 or self.preview_envs <= 0 or (
                self.preview_policy_steps <= 0):
            raise ValueError("capture rollout and preview dimensions must be positive")
        self.writer = MjlabFallShardWriter(output)
        self.normal_writer = MjlabNormalShardWriter(Path(output) / "normals")
        self.armed = False
        self._allocated = False
        self.global_vector_step = 0
        self.fall_count = 0
        self.normal_count = 0
        self.preview_frames: list[dict[str, np.ndarray]] = []
        self.normal_preview: dict[str, np.ndarray] | None = None

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
        self.time = torch.zeros((n, c), dtype=torch.float64, device=device)
        self.act = torch.zeros(
            (n, c, int(env.sim.data.act.shape[1])), dtype=torch.float64,
            device=device)
        self.qacc_warmstart = torch.zeros(
            (n, c, int(env.sim.data.qacc_warmstart.shape[1])),
            dtype=torch.float64, device=device)
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
        previous_q_target = robot.data.joint_pos_target[:, permutation]
        observation = torch.cat((
            joint_q, joint_dq, robot.data.root_link_ang_vel_b,
            robot.data.root_link_lin_vel_b, robot.data.root_link_quat_w,
            previous_q_target,
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
        self.time[ids, slot] = env.sim.data.time.to(torch.float64)
        self.act[ids, slot] = env.sim.data.act.to(torch.float64)
        self.qacc_warmstart[ids, slot] = env.sim.data.qacc_warmstart.to(
            torch.float64)
        self.obs_history[ids, slot] = self.history
        action_term = env.action_manager.get_term("joint_pos")
        encoder_bias = robot.data.encoder_bias[:, action_term.target_ids]
        requested, absolute_target = target_order_action_and_qtarget(
            action,
            scale=action_term.scale,
            offset=action_term.offset,
            encoder_bias=encoder_bias,
        )
        self.action_requested[ids, slot] = requested
        self.action_executed[ids, slot] = requested
        self.q_target[ids, slot] = absolute_target
        self.command[ids, slot] = env.command_manager.get_command("twist")
        self.episode_step[ids, slot] = env.episode_length_buf
        self.policy_step[ids, slot] = (
            self.global_vector_step * self.num_envs + ids)
        self.count += 1
        if len(self.preview_frames) < self.preview_policy_steps:
            preview = self.preview_envs
            self.preview_frames.append({
                "qpos": env.sim.data.qpos[:preview].detach().cpu().numpy().copy(),
                "qvel": env.sim.data.qvel[:preview].detach().cpu().numpy().copy(),
                "action": requested[:preview].detach().cpu().numpy().copy(),
                "q_target": absolute_target[:preview].detach().cpu().numpy().copy(),
                "episode_id": self.episode_id[:preview].detach().cpu().numpy().copy(),
                "episode_step": env.episode_length_buf[:preview].detach().cpu().numpy().copy(),
                "fall_after_action": np.zeros(preview, dtype=bool),
                "terminal_qpos": np.full((preview, 19), np.nan, dtype=np.float64),
            })
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
            if self.normal_preview is None:
                self.normal_preview = self._normal_preview_event(int(environment_id))
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
        if self.preview_frames:
            frame = self.preview_frames[-1]
            for environment_id in fall_ids.detach().cpu().tolist():
                if int(environment_id) < self.preview_envs:
                    frame["fall_after_action"][int(environment_id)] = True
                    frame["terminal_qpos"][int(environment_id)] = env.sim.data.qpos[
                        int(environment_id)].detach().cpu().numpy()
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

        steps_to_fall = np.zeros(RING_POLICY_STEPS, dtype=np.int16)
        steps_to_fall[:length] = np.arange(length, 0, -1, dtype=np.int16)

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
        permutation = torch.as_tensor(
            MJLAB_TO_TARGET_JOINT, dtype=torch.long, device=self.count.device)
        return {
            "identity": np.asarray(identity.encode("ascii"), dtype="S64"),
            "environment_id": np.asarray(environment_id, dtype=np.int32),
            "episode_id": np.asarray(episode, dtype=np.int64),
            "trajectory_length": np.asarray(length, dtype=np.int16),
            "trajectory_mask": np.arange(RING_POLICY_STEPS) < length,
            "prefall_availability": availability,
            "prefall_trajectory_index": prefall_index,
            "trajectory_steps_to_fall": steps_to_fall,
            "trajectory_fall_within_96_steps": np.arange(RING_POLICY_STEPS) < length,
            "trajectory_qpos": padded(self.qpos),
            "trajectory_qvel": padded(self.qvel),
            "trajectory_ctrl": padded(self.ctrl),
            "trajectory_time": padded(self.time),
            "trajectory_act": padded(self.act),
            "trajectory_qacc_warmstart": padded(self.qacc_warmstart),
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
            "terminal_time": env.sim.data.time[
                environment_id].detach().cpu().numpy(),
            "terminal_act": env.sim.data.act[
                environment_id].detach().cpu().numpy(),
            "terminal_qacc_warmstart": env.sim.data.qacc_warmstart[
                environment_id].detach().cpu().numpy(),
            "terminal_action_requested": env.action_manager.action[
                environment_id, permutation].detach().cpu().numpy().astype(np.float32),
            "terminal_action_executed": env.action_manager.action[
                environment_id, permutation].detach().cpu().numpy().astype(np.float32),
            "terminal_q_target": robot.data.joint_pos_target[
                environment_id, permutation].detach().cpu().numpy().astype(np.float32),
            "terminal_command": env.command_manager.get_command("twist")[
                environment_id].detach().cpu().numpy().astype(np.float32),
            "randomized_geom_friction": env.sim.model.geom_friction[
                environment_id].detach().cpu().numpy(),
            "randomized_body_ipos": env.sim.model.body_ipos[
                environment_id].detach().cpu().numpy(),
            "randomized_encoder_bias": robot.data.encoder_bias[
                environment_id].detach().cpu().numpy(),
            "rng_identity": np.asarray(hashlib.sha256(
                b"qsafe.mjlab_rng_identity.v1\0" + raw_identity
            ).hexdigest().encode("ascii"), dtype="S64"),
            "ppo_iteration": np.asarray(
                (self.global_vector_step * self.num_envs)
                // (self.num_envs * self.rollout_steps), dtype=np.int64),
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
            "fall_within_96_steps": np.asarray(False),
            "outcome_horizon_policy_steps": np.asarray(
                RISK_HORIZON_POLICY_STEPS, dtype=np.int16),
            "qualification_future_nonterminal_steps": np.asarray(
                NORMAL_TERMINAL_DISTANCE, dtype=np.int16),
            "randomized_geom_friction": env.sim.model.geom_friction[
                environment_id].detach().cpu().numpy(),
            "randomized_body_ipos": env.sim.model.body_ipos[
                environment_id].detach().cpu().numpy(),
            "randomized_encoder_bias": robot.data.encoder_bias[
                environment_id].detach().cpu().numpy(),
            "rng_identity": np.asarray(hashlib.sha256(
                b"qsafe.mjlab_rng_identity.v1\0" + raw_identity
            ).hexdigest().encode("ascii"), dtype="S64"),
            "ppo_iteration": np.asarray(
                policy_step // (self.num_envs * self.rollout_steps),
                dtype=np.int64),
        }

    def _normal_preview_event(self, environment_id: int) -> dict[str, np.ndarray]:
        count = int(self.count[environment_id].item())
        indices = ordered_ring_indices(count, CAPTURE_RING_STEPS)
        if len(indices) != CAPTURE_RING_STEPS:
            raise RuntimeError("normal preview requires a mature 97-step window")
        device_indices = torch.as_tensor(
            indices, dtype=torch.long, device=self.count.device)

        def cpu(value: torch.Tensor) -> np.ndarray:
            return value[environment_id, device_indices].detach().cpu().numpy()

        return {
            "environment_id": np.asarray(environment_id, dtype=np.int32),
            "episode_id": self.episode_id[
                environment_id].detach().cpu().numpy(),
            "qpos": cpu(self.qpos),
            "qvel": cpu(self.qvel),
            "action_requested": cpu(self.action_requested),
            "q_target": cpu(self.q_target),
            "episode_step": cpu(self.episode_step),
            "policy_step": cpu(self.policy_step),
            "fall_within_96_steps": np.asarray(False),
        }

    @staticmethod
    def _write_npz_no_clobber(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
        np.savez_compressed(temporary, **arrays)
        content = temporary.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite preview archive: {path}") from exc
        temporary.unlink()
        _fsync_dir(path.parent)
        return digest

    def close(self, provenance: Mapping[str, Any]) -> Path:
        if not self.preview_frames:
            raise RuntimeError("parallel PPO preview archive is empty")
        preview_arrays = {
            name: np.stack([frame[name] for frame in self.preview_frames])
            for name in self.preview_frames[0]
        }
        preview_path = self.writer.root / "parallel-preview.npz"
        preview_sha = self._write_npz_no_clobber(preview_path, preview_arrays)
        normal_preview_path = None
        normal_preview_sha = None
        if self.normal_preview is not None:
            normal_preview_path = self.writer.root / "normal-preview.npz"
            normal_preview_sha = self._write_npz_no_clobber(
                normal_preview_path, self.normal_preview)
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
            "parallel_preview": {
                "path": preview_path.name,
                "sha256": preview_sha,
                "environments": self.preview_envs,
                "policy_steps": len(self.preview_frames),
            },
            "normal_preview": (
                None if normal_preview_path is None else {
                    "path": normal_preview_path.name,
                    "sha256": normal_preview_sha,
                    "policy_steps": CAPTURE_RING_STEPS,
                }
            ),
        })
