"""Development-only transfer audit from a direct PPO Q_safe to SAC replay."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl.qsafe.network import QSafeNetworkConfig, SelectiveAdvantageQSafe
from safety_data.natural_ppo_direct_training import (
    binary_auc,
    expected_calibration_error,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruct_ordered_histories_and_h96_labels(
    observation: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    *,
    horizon: int = 96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 5-frame histories, H96 fall labels and uncensored mask."""
    observation = np.asarray(observation, dtype=np.float32)
    terminated = np.asarray(terminated, dtype=bool).reshape(-1)
    truncated = np.asarray(truncated, dtype=bool).reshape(-1)
    if observation.ndim != 2 or observation.shape[1] != 46 or len(
            observation) != len(terminated) or terminated.shape != truncated.shape:
        raise ValueError("ordered SAC replay arrays have invalid shapes")
    if horizon <= 0 or not np.all(np.isfinite(observation)) or np.any(
            terminated & truncated):
        raise ValueError("ordered SAC replay arrays have invalid values")

    count = len(observation)
    histories = np.empty((count, 5, 46), dtype=np.float32)
    episode_start = 0
    for index in range(count):
        start = max(episode_start, index - 4)
        frames = observation[start:index + 1]
        histories[index, :5 - len(frames)] = frames[0]
        histories[index, 5 - len(frames):] = frames
        if terminated[index] or truncated[index]:
            episode_start = index + 1

    label = np.zeros(count, dtype=bool)
    eligible = np.zeros(count, dtype=bool)
    next_done = np.full(count + 1, count, dtype=np.int64)
    nearest = count
    done = terminated | truncated
    for index in range(count - 1, -1, -1):
        if done[index]:
            nearest = index
        next_done[index] = nearest
    for index in range(count):
        boundary = int(next_done[index])
        if boundary < min(count, index + horizon):
            if terminated[boundary]:
                label[index] = True
                eligible[index] = True
            # A timeout before H96 is censored rather than a negative label.
        elif index + horizon <= count:
            eligible[index] = True
    return histories, label, eligible


def validate_ordered_replay_continuity(
    observation: np.ndarray,
    next_observation: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    *,
    atol: float = 1e-6,
    max_break_fraction: float = 1e-3,
) -> np.ndarray:
    observation = np.asarray(observation, dtype=np.float32)
    next_observation = np.asarray(next_observation, dtype=np.float32)
    done = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
    if observation.shape != next_observation.shape or observation.ndim != 2:
        raise ValueError("ordered SAC replay observation shapes differ")
    if len(observation) <= 1:
        raise ValueError("ordered SAC replay is too short")
    difference = np.max(np.abs(
        next_observation[:-1] - observation[1:]), axis=1)
    breaks = np.zeros(len(observation), dtype=bool)
    breaks[:-1] = (~done[:-1]) & (difference > atol)
    fraction = float(breaks.sum() / max(1, len(observation) - 1))
    if fraction > max_break_fraction:
        raise ValueError("SAC replay is not in contiguous episode order")
    return breaks


def _predict_state_risk(
    artifact: dict[str, Any], histories: np.ndarray, *, batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    normalization = artifact["normalization"]
    mean = torch.as_tensor(normalization["observation_mean"]).numpy()
    std = torch.as_tensor(normalization["observation_std"]).numpy()
    normalized = ((histories - mean) / std).astype(np.float32)
    config = QSafeNetworkConfig(**artifact["network_config"])
    members = []
    for state_dict in artifact["member_state_dicts"]:
        model = SelectiveAdvantageQSafe(config)
        model.load_state_dict(state_dict)
        model.eval()
        members.append(model)
    probabilities = []
    with torch.inference_mode():
        for model, temperature in zip(
                members, artifact["state_temperatures"], strict=True):
            chunks = []
            for start in range(0, len(normalized), batch_size):
                history = torch.from_numpy(normalized[start:start + batch_size])
                state = model.encode_state(history)
                logit = model.state_risk_head(state).reshape(-1)
                chunks.append(torch.sigmoid(logit / float(temperature)).numpy())
            probabilities.append(np.concatenate(chunks))
    member = np.stack(probabilities)
    return member.mean(axis=0), member.std(axis=0)


def evaluate_direct_qsafe_on_ordered_sac_replay(
    *,
    model_path: str | Path,
    replay_path: str | Path,
    output_path: str | Path,
    batch_size: int = 2048,
) -> dict[str, Any]:
    model_path = Path(model_path)
    replay_path = Path(replay_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("SAC transfer audit output path was already consumed")
    artifact = torch.load(model_path, map_location="cpu", weights_only=False)
    if artifact.get("schema_version") != "qsafe.natural_ppo_state_trigger_model.v3" or (
            artifact.get("production_head") != "state_risk_only"):
        raise ValueError("input is not a natural-PPO state-trigger Q_safe model")
    replay = torch.load(replay_path, map_location="cpu", weights_only=False)
    required = {
        "observation", "next_observation", "terminated", "truncated",
        "num_in_buffer", "current_idx",
    }
    if not isinstance(replay, dict) or not required.issubset(replay):
        raise ValueError("SAC replay checkpoint fields are incomplete")
    count = int(replay["num_in_buffer"])
    if int(replay["current_idx"]) != count:
        raise ValueError("wrapped replay order is not claimable")
    observation = replay["observation"][:count].cpu().numpy()
    next_observation = replay["next_observation"][:count].cpu().numpy()
    terminated = replay["terminated"][:count].cpu().numpy().astype(bool)
    truncated = replay["truncated"][:count].cpu().numpy().astype(bool)
    continuity_breaks = validate_ordered_replay_continuity(
        observation, next_observation, terminated, truncated)
    histories, label, eligible = reconstruct_ordered_histories_and_h96_labels(
        observation, terminated, truncated | continuity_breaks)
    selected = np.flatnonzero(eligible)
    if not np.any(label[selected]) or np.all(label[selected]):
        raise ValueError("eligible SAC transfer set is not binary")
    risk, uncertainty = _predict_state_risk(
        artifact, histories[selected], batch_size=batch_size)
    target = label[selected]
    report = {
        "schema_version": "qsafe.natural_ppo_sac_transfer.v1",
        "development_only": True,
        "claim_eligible": False,
        "model_file_sha256": _sha256(model_path),
        "replay_file_sha256": _sha256(replay_path),
        "replay_transitions": count,
        "continuity_breaks_censored": int(continuity_breaks.sum()),
        "eligible_h96_states": int(len(selected)),
        "positive_prefall_states": int(target.sum()),
        "positive_fraction": float(target.mean()),
        "state_auroc": binary_auc(target, risk),
        "state_ece": expected_calibration_error(target, risk),
        "accuracy_at_0_5": float(np.mean((risk >= 0.5) == target)),
        "mean_predicted_risk_positive": float(risk[target].mean()),
        "mean_predicted_risk_negative": float(risk[~target].mean()),
        "mean_ensemble_uncertainty": float(uncertainty.mean()),
        "history_frames": 5,
        "risk_horizon_policy_steps": 96,
        "timeout_before_horizon_handling": "censored",
        "sac_model_test_consumed": False,
        "objective1_claim_eligible": False,
    }
    rendered = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, output_path)
    except FileExistsError as exc:
        raise FileExistsError(
            "SAC transfer audit output path was already consumed") from exc
    temporary.unlink()
    descriptor = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return report
