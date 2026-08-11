#!/usr/bin/env python3
"""Create a claim-audit summary for a validated natural-PPO collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from safety_data.mjlab_natural_falls import MJLAB_TO_TARGET_JOINT
from safety_data.natural_ppo_archive import AGE_BOUNDARIES
from safety_data.natural_ppo_direct_training import load_direct_dataset


QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(value: np.ndarray) -> dict[str, float]:
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    if not len(value) or not np.all(np.isfinite(value)):
        raise ValueError("summary input is empty or non-finite")
    return {
        f"p{int(round(100 * q)):02d}": float(np.quantile(value, q))
        for q in QUANTILES
    }


def _age_counts(policy_step: np.ndarray) -> dict[str, int]:
    age = np.searchsorted(
        np.asarray(AGE_BOUNDARIES, dtype=np.int64),
        np.asarray(policy_step, dtype=np.int64), side="right")
    return {str(index): int(np.sum(age == index)) for index in range(6)}


def _tilt(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion /= np.maximum(
        np.linalg.norm(quaternion, axis=1, keepdims=True), 1e-12)
    w, x, y, z = quaternion.T
    roll = np.arctan2(
        2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return np.maximum(np.abs(roll), np.abs(pitch))


def _feature_summary(
    observation: np.ndarray,
    action: np.ndarray,
    joint_range: np.ndarray,
) -> dict[str, Any]:
    frame = np.asarray(observation, dtype=np.float64)[:, -1]
    joint_position = frame[:, :12]
    joint_velocity = frame[:, 12:24]
    base_angular_velocity = frame[:, 24:27]
    base_linear_velocity = frame[:, 27:30]
    quaternion = frame[:, 30:34]
    width = joint_range[:, 1] - joint_range[:, 0]
    lower_margin = (joint_position - joint_range[:, 0]) / width
    upper_margin = (joint_range[:, 1] - joint_position) / width
    normalized_margin = np.minimum(lower_margin, upper_margin)
    return {
        "maximum_abs_roll_or_pitch_rad": _quantiles(_tilt(quaternion)),
        "base_angular_speed_rad_s": _quantiles(np.linalg.norm(
            base_angular_velocity, axis=1)),
        "base_linear_speed_m_s": _quantiles(np.linalg.norm(
            base_linear_velocity, axis=1)),
        "joint_velocity_l2_rad_s": _quantiles(np.linalg.norm(
            joint_velocity, axis=1)),
        "minimum_normalized_joint_limit_margin": _quantiles(
            np.min(normalized_margin, axis=1)),
        "joint_coordinates_within_5pct_of_limit_fraction": float(np.mean(
            normalized_margin <= 0.05)),
        "requested_action_abs_max": _quantiles(np.max(np.abs(action), axis=1)),
    }


def _load_target_joint_range(model_path: Path) -> np.ndarray:
    import mujoco

    model = mujoco.MjModel.from_binary_path(str(model_path))
    hinge_ids = np.flatnonzero(model.jnt_type != mujoco.mjtJoint.mjJNT_FREE)
    if len(hinge_ids) != 12:
        raise ValueError("compiled target model does not have twelve hinge joints")
    ranges = np.asarray(model.jnt_range[hinge_ids], dtype=np.float64)
    return ranges[np.asarray(MJLAB_TO_TARGET_JOINT, dtype=np.int64)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    archive = run / "natural-falls"
    dataset = load_direct_dataset(args.dataset)
    pair_report_path = args.pairs.with_suffix(".report.json")
    pair_report = json.loads(pair_report_path.read_text(encoding="utf-8"))
    if pair_report.get("pair_file_sha256") != _sha256(args.pairs) or not (
            pair_report.get("one_to_one_matching_complete")):
        raise ValueError("summary requires a complete validated match file")
    manifest_path = archive / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    normal_manifest_path = archive / manifest["provenance"]["normal_manifest"]
    normal_manifest = json.loads(normal_manifest_path.read_text(encoding="utf-8"))
    if pair_report.get("fall_manifest_sha256") != _sha256(manifest_path) or (
            pair_report.get("normal_manifest_sha256") != _sha256(
                normal_manifest_path)):
        raise ValueError("validation report does not bind collection manifests")

    raw_normal_policy_steps = []
    normal_root = normal_manifest_path.parent
    for shard in normal_manifest["shards"]:
        path = normal_root / shard["path"]
        if _sha256(path) != shard["sha256"]:
            raise ValueError("normal shard changed after validation")
        with np.load(path, allow_pickle=False) as arrays:
            raw_normal_policy_steps.append(arrays["policy_step"].copy())

    fall_policy_step = []
    fall_environment = []
    fall_episode_length = []
    retained_trajectory_length = []
    for shard in manifest["shards"]:
        path = archive / shard["path"]
        if _sha256(path) != shard["sha256"]:
            raise ValueError("fall shard changed after validation")
        with np.load(path, allow_pickle=False) as arrays:
            length = arrays["trajectory_length"].astype(np.int64)
            row = np.arange(len(length))
            fall_policy_step.append(arrays["trajectory_policy_step"][row, length - 1])
            fall_environment.append(arrays["environment_id"].copy())
            fall_episode_length.append(
                arrays["trajectory_episode_step"][row, length - 1] + 1)
            retained_trajectory_length.append(length)
    fall_policy_step_array = np.concatenate(fall_policy_step)
    fall_environment_array = np.concatenate(fall_environment)
    fall_episode_length_array = np.concatenate(fall_episode_length)
    retained_trajectory_length_array = np.concatenate(retained_trajectory_length)

    intervals = []
    environment_count = int(manifest["provenance"]["environments"])
    for environment_id in np.unique(fall_environment_array):
        steps = np.sort(fall_policy_step_array[
            fall_environment_array == environment_id])
        if len(steps) > 1:
            intervals.extend((np.diff(steps) / environment_count).tolist())

    arrays = dataset.arrays
    positive = arrays["label"]
    negative = ~positive
    model_path = run / "target-aligned-model.mjb"
    joint_range = _load_target_joint_range(model_path)
    prefall_offsets = {
        str(offset): int(np.sum(
            positive & (arrays["prefall_offset"] == offset)))
        for offset in (1, 2, 4, 8, 16, 32, 64)
    }
    report = {
        "schema_version": "qsafe.natural_ppo_collection_summary.v1",
        "claim_scope": "formal_30m_collection_and_direct_state_risk_dataset",
        "run_manifest_sha256": _sha256(run / "manifest.json"),
        "fall_manifest_sha256": _sha256(manifest_path),
        "normal_manifest_sha256": _sha256(normal_manifest_path),
        "pair_report_sha256": _sha256(pair_report_path),
        "dataset_sha256": dataset.file_sha256,
        "fixed_exposure_policy_env_steps": int(manifest["provenance"][
            "fixed_exposure"]),
        "environment_count": environment_count,
        "recorded_independent_falls": int(len(fall_policy_step_array)),
        "raw_normal_candidates": int(normal_manifest["event_count"]),
        "matched_prefall_normal_pairs": int(np.sum(positive)),
        "prefall_offset_counts": prefall_offsets,
        "fall_event_age_bucket_counts": _age_counts(fall_policy_step_array),
        "raw_normal_age_bucket_counts": _age_counts(np.concatenate(
            raw_normal_policy_steps)),
        "matched_positive_age_bucket_counts": _age_counts(
            arrays["policy_step"][positive]),
        "fall_episode_policy_steps": _quantiles(fall_episode_length_array),
        "retained_prefall_trajectory_policy_steps": _quantiles(
            retained_trajectory_length_array),
        "within_environment_interfall_policy_steps": (
            None if not intervals else _quantiles(np.asarray(intervals))),
        "prefall_state_distribution": _feature_summary(
            arrays["observation_history"][positive],
            arrays["action_requested"][positive], joint_range),
        "matched_normal_state_distribution": _feature_summary(
            arrays["observation_history"][negative],
            arrays["action_requested"][negative], joint_range),
        "runtime_external_force": "verified_zero",
        "fall_counting": "first_terminal_once_then_same_vector_step_reset",
        "recovery_executed_during_ppo": False,
        "production_supervision": "state_risk_only",
        "executed_action_use": "diagnostic_only",
        "objective1_pass": False,
        "phase2_authorized": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
