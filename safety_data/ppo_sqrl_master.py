"""Atomic transition shards for the multi-stage PPO SQRL master dataset."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "qsafe.ppo_sqrl_master.v1"
SHARD_SCHEMA_VERSION = "qsafe.ppo_sqrl_transition_shard.v1"
OBSERVATION_HISTORY_SHAPE = (5, 46)
ACTION_DIM = 12
STAGES = ("early", "boundary", "mature")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    content = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite {path}") from exc
    temporary.unlink()
    _fsync_directory(path.parent)


def validate_transition_arrays(arrays: Mapping[str, np.ndarray]) -> int:
    required = {
        "observation_history_t", "critic_action", "next_observation_history",
        "c_t_plus_1", "terminated", "truncated", "action_requested",
        "action_pre_projection", "absolute_q_target", "action_log_probability",
        "policy_entropy", "action_std", "action_saturation",
        "action_change_rate", "ppo_seed", "collector_stage",
        "collector_checkpoint", "env_id", "episode_id", "vector_step",
        "randomization_identity", "rng_identity", "policy_observation_t",
        "next_policy_observation", "next_action_encoder_bias",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"transition shard is missing fields: {sorted(missing)}")
    count = len(arrays["c_t_plus_1"])
    shapes = {
        "observation_history_t": (count, *OBSERVATION_HISTORY_SHAPE),
        "next_observation_history": (count, *OBSERVATION_HISTORY_SHAPE),
        "critic_action": (count, ACTION_DIM),
        "action_requested": (count, ACTION_DIM),
        "action_pre_projection": (count, ACTION_DIM),
        "absolute_q_target": (count, ACTION_DIM),
        "action_std": (count, ACTION_DIM),
        "action_saturation": (count, ACTION_DIM),
        "next_action_encoder_bias": (count, ACTION_DIM),
    }
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{name} has shape {arrays[name].shape}, expected {shape}")
    one_dimensional = required - set(shapes) - {
        "policy_observation_t", "next_policy_observation",
    }
    for name in one_dimensional:
        if arrays[name].shape != (count,):
            raise ValueError(f"{name} must have shape [{count}]")
    policy_shape = arrays["policy_observation_t"].shape
    if len(policy_shape) != 2 or policy_shape[0] != count or (
            arrays["next_policy_observation"].shape != policy_shape):
        raise ValueError("policy observations must be matching [N,D] arrays")
    numeric = [
        name for name, value in arrays.items()
        if np.issubdtype(value.dtype, np.number)
    ]
    if any(not np.all(np.isfinite(arrays[name])) for name in numeric):
        raise ValueError("transition shard contains non-finite numeric values")
    terminated = arrays["terminated"].astype(bool)
    truncated = arrays["truncated"].astype(bool)
    cost = arrays["c_t_plus_1"].astype(bool)
    if np.any(terminated & truncated) or np.any(cost != terminated):
        raise ValueError("cost must equal first-fall termination and exclude timeout")
    if not np.array_equal(arrays["critic_action"], arrays["absolute_q_target"]):
        raise ValueError("critic_action must be exactly the absolute PD target")
    stages = set(arrays["collector_stage"].astype("U").tolist())
    if not stages.issubset(STAGES) or not stages:
        raise ValueError("transition shard contains an unknown collector stage")
    return count


class TransitionShardWriter:
    """Write immutable fixed-vector-step transition shards."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.shards: list[dict[str, Any]] = []
        self.transition_count = 0

    def write(self, arrays: Mapping[str, np.ndarray]) -> Path:
        copied = {name: np.asarray(value) for name, value in arrays.items()}
        count = validate_transition_arrays(copied)
        path = self.root / f"transitions-{len(self.shards):06d}.npz"
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
        np.savez_compressed(temporary, **copied)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        digest = sha256_file(path)
        self.shards.append({
            "path": path.name,
            "sha256": digest,
            "transition_count": count,
            "fall_count": int(np.asarray(copied["c_t_plus_1"], bool).sum()),
        })
        self.transition_count += count
        return path

    def close(self, provenance: Mapping[str, Any]) -> Path:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "transition_count": self.transition_count,
            "shards": self.shards,
            "critic_action_semantic": (
                "absolute_12d_joint_target_applied_to_pd_for_current_20ms_interval"),
            "cost_semantic": "c_t_plus_1_first_fall_terminal",
            "action_sampling": "stochastic_ppo",
            "post_fall_control": "same_vector_step_reset_without_recovery",
            "provenance": dict(provenance),
        }
        path = self.root / "manifest.json"
        _write_json_no_clobber(path, manifest)
        return path


def validate_master_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get(
            "action_sampling") != "stochastic_ppo" or manifest.get(
                "post_fall_control") != "same_vector_step_reset_without_recovery":
        raise ValueError("invalid PPO SQRL master manifest")
    total = 0
    for shard in manifest.get("shards", []):
        shard_path = path.parent / shard["path"]
        if sha256_file(shard_path) != shard["sha256"]:
            raise ValueError(f"transition shard hash changed: {shard_path}")
        with np.load(shard_path, allow_pickle=False) as loaded:
            count = validate_transition_arrays(loaded)
        if count != shard["transition_count"]:
            raise ValueError("manifest transition count disagrees with shard")
        total += count
    if total != manifest.get("transition_count"):
        raise ValueError("manifest total transition count is wrong")
    return manifest


def split_role(seed: int, environment_id: int, episode_id: int) -> str:
    identity = f"{seed}:{environment_id}:{episode_id}".encode("ascii")
    bucket = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little") % 10
    return "fit" if bucket < 7 else "calibration" if bucket < 9 else "test"
