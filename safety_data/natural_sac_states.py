"""Collect unperturbed native-SAC states for Q_safe calibration and testing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch

from safety_data.policies import load_frozen_droq_policy
from train.config import load_app_config
from train.mujoco_snapshot_env import MujocoSnapshotEnv


HORIZON_POLICY_STEPS = 96


def episode_h96_labels(
    length: int,
    *,
    failed: bool,
    horizon: int = HORIZON_POLICY_STEPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return eligibility, fall label, and steps-to-outcome for one episode."""
    if isinstance(length, bool) or int(length) <= 0 or int(horizon) <= 0:
        raise ValueError("episode length and horizon must be positive integers")
    length = int(length)
    horizon = int(horizon)
    distance = length - np.arange(length, dtype=np.int32)
    label = np.asarray(failed, dtype=bool) & (distance <= horizon)
    eligible = label | (distance > horizon) | (~np.asarray(failed) & (
        distance >= horizon))
    steps_to_outcome = np.where(label, distance, horizon).astype(np.int16)
    return eligible, label, steps_to_outcome


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _publish_no_clobber(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite SAC state file: {destination}") from exc
    temporary.unlink()
    descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def collect_natural_sac_states(
    *,
    actor_checkpoint: str | Path,
    actor_seed: int,
    training_step: int,
    source_seed: int,
    exposure_steps: int,
    config_path: str | Path,
    model_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Run one fixed SAC actor naturally and publish H96 state labels."""
    for name, value in (
        ("actor_seed", actor_seed), ("training_step", training_step),
        ("source_seed", source_seed), ("exposure_steps", exposure_steps),
    ):
        if isinstance(value, bool) or int(value) < (1 if name == "exposure_steps" else 0):
            raise ValueError(f"{name} is invalid")
    output = Path(output).resolve()
    report_path = output.with_suffix(".manifest.json")
    if output.exists() or report_path.exists():
        raise FileExistsError("natural SAC state output was already consumed")
    config_path = Path(config_path).resolve()
    model_path = Path(model_path).resolve()
    actor_checkpoint = Path(actor_checkpoint).resolve()
    robot, train, _ = load_app_config(config_path)
    torch.set_num_threads(1)
    if not np.isclose(float(robot.move_speed), 0.30, atol=1e-12, rtol=0.0):
        raise ValueError("natural SAC calibration requires +0.30 m/s")
    if not np.isclose(
            float(robot.fallen_orientation_rad), 1.047198, atol=1e-12, rtol=0.0):
        raise ValueError("natural SAC fall predicate must use the registered 60 degrees")
    if train.use_action_filter:
        raise ValueError("natural SAC snapshot schema currently requires no action filter")
    policy = load_frozen_droq_policy(
        actor_checkpoint, config_path, observation_dim=robot.obs_dim,
        action_dim=robot.num_joints, training_step=int(training_step), device="cpu")
    env = MujocoSnapshotEnv(
        model_path, robot, policy_frequency=train.control_frequency,
        max_joint_delta=train.max_joint_delta,
        use_action_filter=train.use_action_filter)
    simulator = env.simulator_fingerprint()
    rng = np.random.default_rng(int(source_seed))
    reset_rng = np.random.default_rng(
        np.random.SeedSequence([int(source_seed), 0x52534554]))
    action_rng = np.random.default_rng(
        np.random.SeedSequence([int(source_seed), 0x4143544E]))

    rows: dict[str, list[Any]] = {name: [] for name in (
        "identity", "observation_history", "history_length",
        "integration_state", "previous_action_requested",
        "previous_action_executed", "previous_action_q_target",
        "action_requested", "action_executed", "action_q_target",
        "episode_id", "episode_step", "policy_step", "label",
        "steps_to_outcome", "terminal_kind",
    )}
    del rng  # all randomness has a named, separated stream
    policy_steps = 0
    episode_id = 0
    independent_falls = 0
    timeouts = 0
    with policy.inference_session() as sample_action:
        while policy_steps < int(exposure_steps):
            env.reset_standing(settle_seconds=1.0, rng=reset_rng)
            episode: list[dict[str, Any]] = []
            failed = False
            for episode_step in range(int(train.max_episode_steps)):
                if policy_steps >= int(exposure_steps):
                    break
                if np.any(env.data.xfrc_applied != 0.0):
                    raise RuntimeError("natural SAC rollout encountered external force")
                history = env.record_observation()
                snapshot = env.capture()
                if snapshot.application_state.action_filter_state is not None:
                    raise RuntimeError(
                        "natural SAC snapshot unexpectedly has a filter state")
                action = sample_action(history[-1], action_rng)
                result = env.step(action)
                if np.any(env.data.xfrc_applied != 0.0):
                    raise RuntimeError("natural SAC rollout produced external force")
                raw_identity = np.asarray([
                    int(actor_seed), int(training_step), int(source_seed),
                    int(episode_id), int(episode_step),
                ], dtype=np.uint64).tobytes()
                identity = hashlib.sha256(
                    b"qsafe.natural_sac_state.v1\0" + raw_identity).hexdigest()
                episode.append({
                    "identity": identity,
                    "observation_history": history,
                    "history_length": min(episode_step + 1, 5),
                    "integration_state": snapshot.integration_state,
                    "previous_action_requested": (
                        snapshot.application_state.previous_action_requested),
                    "previous_action_executed": (
                        snapshot.application_state.previous_action_executed),
                    "previous_action_q_target": (
                        snapshot.application_state.previous_action_q_target),
                    "action_requested": result.application.action_requested,
                    "action_executed": result.application.action_executed,
                    "action_q_target": result.application.action_q_target,
                    "episode_id": episode_id,
                    "episode_step": episode_step,
                    "policy_step": policy_steps,
                })
                policy_steps += 1
                if result.failure:
                    failed = True
                    independent_falls += 1
                    break
            if not episode:
                break
            if not failed and len(episode) == int(train.max_episode_steps):
                timeouts += 1
            eligible, label, steps_to_outcome = episode_h96_labels(
                len(episode), failed=failed)
            for index in np.flatnonzero(eligible):
                state = episode[int(index)]
                for name in (
                    "identity", "observation_history", "history_length",
                    "integration_state", "previous_action_requested",
                    "previous_action_executed", "previous_action_q_target",
                    "action_requested", "action_executed", "action_q_target",
                    "episode_id", "episode_step", "policy_step",
                ):
                    rows[name].append(state[name])
                rows["label"].append(bool(label[index]))
                rows["steps_to_outcome"].append(int(steps_to_outcome[index]))
                rows["terminal_kind"].append("fall" if failed else "timeout")
            episode_id += 1

    arrays = {
        "identity": np.asarray(rows["identity"], dtype="S64"),
        "observation_history": np.asarray(
            rows["observation_history"], dtype=np.float32),
        "history_length": np.asarray(rows["history_length"], dtype=np.int8),
        "integration_state": np.asarray(rows["integration_state"], dtype=np.float64),
        "previous_action_requested": np.asarray(
            rows["previous_action_requested"], dtype=np.float32),
        "previous_action_executed": np.asarray(
            rows["previous_action_executed"], dtype=np.float32),
        "previous_action_q_target": np.asarray(
            rows["previous_action_q_target"], dtype=np.float32),
        "action_requested": np.asarray(rows["action_requested"], dtype=np.float32),
        "action_executed": np.asarray(rows["action_executed"], dtype=np.float32),
        "action_q_target": np.asarray(rows["action_q_target"], dtype=np.float32),
        "episode_id": np.asarray(rows["episode_id"], dtype=np.int64),
        "episode_step": np.asarray(rows["episode_step"], dtype=np.int32),
        "policy_step": np.asarray(rows["policy_step"], dtype=np.int64),
        "label": np.asarray(rows["label"], dtype=bool),
        "steps_to_outcome": np.asarray(rows["steps_to_outcome"], dtype=np.int16),
        "terminal_kind": np.asarray(rows["terminal_kind"], dtype="S7"),
    }
    count = len(arrays["label"])
    if count == 0 or len(set(map(bytes, arrays["identity"]))) != count:
        raise RuntimeError("natural SAC collection is empty or has duplicate identities")
    if not np.all(np.isfinite(arrays["observation_history"])) or not np.all(
            np.isfinite(arrays["integration_state"])):
        raise RuntimeError("natural SAC collection contains non-finite states")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    _publish_no_clobber(temporary, output)
    report = {
        "schema_version": "qsafe.natural_sac_states.v1",
        "generator_commit": _git_head(),
        "output_file": output.name,
        "output_sha256": _sha256(output),
        "actor_seed": int(actor_seed),
        "actor_training_step": int(training_step),
        "actor_manifest": policy.manifest(),
        "source_seed": int(source_seed),
        "fixed_exposure_policy_steps": int(exposure_steps),
        "recorded_eligible_states": count,
        "positive_h96_states": int(arrays["label"].sum()),
        "independent_falls": independent_falls,
        "timeouts": timeouts,
        "episodes": episode_id,
        "horizon_policy_steps": HORIZON_POLICY_STEPS,
        "fall_predicate": {
            "minimum_base_height_m": 0.18,
            "maximum_abs_roll_or_pitch_rad": float(robot.fallen_orientation_rad),
        },
        "external_force": "verified_zero",
        "recovery_executed": False,
        "simulator": simulator,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "model_path": str(model_path),
        "objective1_claim_eligible": False,
        "phase2_authorized": False,
    }
    content = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
    temporary_report = report_path.with_name(
        f".{report_path.name}.tmp-{os.getpid()}")
    with temporary_report.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    _publish_no_clobber(temporary_report, report_path)
    return report
