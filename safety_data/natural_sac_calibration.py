"""SAC-only calibration for the natural-PPO state-risk trigger.

The PPO archive is allowed to fit representation weights.  This module is the
only path that turns that uncalibrated model into a deployable probability and
uncertainty artifact, and it accepts only the preregistered natural-SAC roles.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.optimize import minimize
import torch

from rl.qsafe.network import QSafeNetworkConfig, SelectiveAdvantageQSafe
from safety_data.natural_ppo_direct_training import (
    binary_auc,
    expected_calibration_error,
)


ROLE_ROSTER = {
    "probability": (47, {25_000: 9401, 50_000: 9402, 100_000: 9403}),
    "uncertainty": (48, {25_000: 9411, 50_000: 9412, 100_000: 9413}),
    "selector": (49, {25_000: 9421, 50_000: 9422, 100_000: 9423}),
}
EXPECTED_EXPOSURE_STEPS = 10_000
EXPECTED_HORIZON_STEPS = 96
EXPECTED_FALL_ANGLE_RAD = 1.047198
EXPECTED_MINIMUM_HEIGHT_M = 0.18


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _publish_no_clobber(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite calibration output: {destination}") from exc
    temporary.unlink()
    descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class NaturalSacRole:
    name: str
    observation_history: np.ndarray
    label: np.ndarray
    source_seed: np.ndarray
    episode_id: np.ndarray
    identities: np.ndarray
    input_files: tuple[dict[str, Any], ...]


def load_natural_sac_role(directory: str | Path, role: str) -> NaturalSacRole:
    """Load and fail-closed validate one frozen natural-SAC role."""
    if role not in ROLE_ROSTER:
        raise ValueError(f"unknown natural-SAC role: {role}")
    directory = Path(directory).resolve()
    actor_seed, source_by_step = ROLE_ROSTER[role]
    observations: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sources: list[np.ndarray] = []
    episodes: list[np.ndarray] = []
    identities: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    generator_commit: str | None = None
    for training_step, source_seed in source_by_step.items():
        stem = f"actor{actor_seed}-age{training_step}-source{source_seed}"
        data_path = directory / f"{stem}.npz"
        manifest_path = directory / f"{stem}.manifest.json"
        if not data_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"missing preregistered {role} source {stem}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest = _sha256(data_path)
        required = {
            "schema_version": "qsafe.natural_sac_states.v1",
            "output_sha256": digest,
            "actor_seed": actor_seed,
            "actor_training_step": training_step,
            "source_seed": source_seed,
            "fixed_exposure_policy_steps": EXPECTED_EXPOSURE_STEPS,
            "horizon_policy_steps": EXPECTED_HORIZON_STEPS,
            "external_force": "verified_zero",
            "recovery_executed": False,
            "phase2_authorized": False,
        }
        for name, expected in required.items():
            if manifest.get(name) != expected:
                raise ValueError(f"{stem} violates frozen field {name}")
        predicate = manifest.get("fall_predicate", {})
        if not math.isclose(float(predicate.get("minimum_base_height_m", math.nan)),
                            EXPECTED_MINIMUM_HEIGHT_M, abs_tol=1e-12) or not math.isclose(
                float(predicate.get("maximum_abs_roll_or_pitch_rad", math.nan)),
                EXPECTED_FALL_ANGLE_RAD, abs_tol=1e-12):
            raise ValueError(f"{stem} uses the wrong fall predicate")
        current_commit = str(manifest.get("generator_commit", ""))
        if len(current_commit) != 40 or (generator_commit is not None and
                                        current_commit != generator_commit):
            raise ValueError("natural-SAC sources do not share one generator commit")
        generator_commit = current_commit
        with np.load(data_path, allow_pickle=False) as loaded:
            observation = loaded["observation_history"].copy()
            label = loaded["label"].astype(bool, copy=True)
            identity = loaded["identity"].copy()
            episode = loaded["episode_id"].astype(np.int64, copy=True)
        count = len(label)
        if observation.shape != (count, 5, 46) or identity.shape != (count,) or (
                episode.shape != (count,)) or not np.all(np.isfinite(observation)):
            raise ValueError(f"{stem} arrays violate the natural-SAC schema")
        if count != int(manifest.get("recorded_eligible_states", -1)) or int(
                label.sum()) != int(manifest.get("positive_h96_states", -1)):
            raise ValueError(f"{stem} manifest counts do not match its arrays")
        observations.append(observation)
        labels.append(label)
        sources.append(np.full(count, source_seed, dtype=np.int64))
        episodes.append(episode)
        identities.append(identity)
        records.append({
            "file": data_path.name,
            "sha256": digest,
            "manifest_sha256": _sha256(manifest_path),
            "actor_seed": actor_seed,
            "training_step": training_step,
            "source_seed": source_seed,
            "states": count,
            "positives": int(label.sum()),
        })
    all_identity = np.concatenate(identities)
    if len(set(map(bytes, all_identity))) != len(all_identity):
        raise ValueError(f"{role} contains duplicate snapshot identities")
    return NaturalSacRole(
        name=role,
        observation_history=np.concatenate(observations),
        label=np.concatenate(labels),
        source_seed=np.concatenate(sources),
        episode_id=np.concatenate(episodes),
        identities=all_identity,
        input_files=tuple(records),
    )


def assert_roles_disjoint(*roles: NaturalSacRole) -> None:
    seen: set[bytes] = set()
    for role in roles:
        current = set(map(bytes, role.identities))
        if seen & current:
            raise ValueError("natural-SAC calibration roles overlap")
        seen.update(current)


def _load_models(artifact: Mapping[str, Any], device: torch.device) -> list[SelectiveAdvantageQSafe]:
    if artifact.get("schema_version") != "qsafe.natural_ppo_state_trigger_model.v3" or (
            artifact.get("temperature_status") != "pending_sac_only_calibration"):
        raise ValueError("input is not an unconsumed natural-PPO state-risk model")
    config = QSafeNetworkConfig(**artifact["network_config"])
    models = []
    for state_dict in artifact["member_state_dicts"]:
        model = SelectiveAdvantageQSafe(config).to(device)
        model.load_state_dict(state_dict, strict=True)
        models.append(model.eval())
    if len(models) != 5:
        raise ValueError("natural state-risk calibration requires five members")
    return models


def _member_logits(models: list[SelectiveAdvantageQSafe], observation: np.ndarray,
                   normalization: Mapping[str, Any], device: torch.device,
                   batch_size: int = 2048) -> np.ndarray:
    mean = np.asarray(normalization["observation_mean"], dtype=np.float32)
    std = np.asarray(normalization["observation_std"], dtype=np.float32)
    normalized = (observation - mean) / std
    tensor = torch.from_numpy(normalized.astype(np.float32, copy=False))
    result = []
    with torch.inference_mode():
        for model in models:
            chunks = []
            for start in range(0, len(tensor), batch_size):
                batch = tensor[start:start + batch_size].to(device)
                state = model.encode_state(batch)
                chunks.append(model.state_risk_head(state).reshape(-1).cpu().numpy())
            result.append(np.concatenate(chunks))
    return np.stack(result).astype(np.float64)


def fit_member_affine_calibration(
    logits: np.ndarray, label: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit monotone per-member Platt scaling to frozen logits.

    A temperature alone cannot correct the large PPO-to-SAC base-rate shift.
    The positive temperature preserves ranking while the intercept corrects
    that shift using only the probability-calibration role.
    """
    logits = np.asarray(logits, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64).reshape(-1)
    if logits.ndim != 2 or logits.shape[1] != len(label) or logits.shape[0] != 5 or (
            not np.all(np.isfinite(logits))) or not np.any(label) or np.all(label):
        raise ValueError("temperature inputs must be finite five-member binary data")
    temperatures = []
    biases = []
    for member_logits in logits:
        def objective(parameters: np.ndarray) -> float:
            scaled = member_logits / math.exp(float(parameters[0])) + float(parameters[1])
            return float(np.mean(np.logaddexp(0.0, scaled) - label * scaled))
        optimum = minimize(
            objective, np.zeros(2, dtype=np.float64), method="L-BFGS-B",
            bounds=((-4.0, 4.0), (-20.0, 20.0)),
        )
        if not optimum.success:
            raise RuntimeError("affine Platt optimization failed")
        temperatures.append(math.exp(float(optimum.x[0])))
        biases.append(float(optimum.x[1]))
    return (np.asarray(temperatures, dtype=np.float64),
            np.asarray(biases, dtype=np.float64))


def finite_sample_upper_residual(residual: np.ndarray, alpha: float = 0.05) -> float:
    """Return the split-conformal one-sided finite-sample residual quantile."""
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    if len(residual) == 0 or not np.all(np.isfinite(residual)) or not 0.0 < alpha < 1.0:
        raise ValueError("invalid conformal residuals")
    rank = min(len(residual), math.ceil((len(residual) + 1) * (1.0 - alpha)))
    return float(np.partition(residual, rank - 1)[rank - 1])


def calibrate_natural_sac_state_risk(
    *, model_path: str | Path, calibration_root: str | Path,
    output_path: str | Path, device: str | None = None,
) -> dict[str, Any]:
    """Freeze SAC temperatures and uncertainty without opening selector/test outcomes."""
    model_path = Path(model_path).resolve()
    calibration_root = Path(calibration_root).resolve()
    output_path = Path(output_path).resolve()
    report_path = output_path.with_suffix(".report.json")
    if output_path.exists() or report_path.exists():
        raise FileExistsError("SAC-calibrated state-risk output was already consumed")
    probability = load_natural_sac_role(calibration_root / "probability", "probability")
    uncertainty = load_natural_sac_role(calibration_root / "uncertainty", "uncertainty")
    assert_roles_disjoint(probability, uncertainty)
    selected_device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    artifact = torch.load(model_path, map_location="cpu", weights_only=False)
    models = _load_models(artifact, selected_device)
    probability_logits = _member_logits(
        models, probability.observation_history, artifact["normalization"], selected_device)
    temperatures, biases = fit_member_affine_calibration(
        probability_logits, probability.label)
    probability_members = 1.0 / (1.0 + np.exp(
        -(probability_logits / temperatures[:, None] + biases[:, None])))
    probability_mean = probability_members.mean(axis=0)
    uncertainty_logits = _member_logits(
        models, uncertainty.observation_history, artifact["normalization"], selected_device)
    uncertainty_members = 1.0 / (1.0 + np.exp(
        -(uncertainty_logits / temperatures[:, None] + biases[:, None])))
    uncertainty_mean = uncertainty_members.mean(axis=0)
    residual_offset = finite_sample_upper_residual(
        uncertainty.label.astype(np.float64) - uncertainty_mean)
    upper = np.clip(uncertainty_mean + residual_offset, 0.0, 1.0)
    calibrated = dict(artifact)
    calibrated.update({
        "schema_version": "qsafe.natural_ppo_state_trigger_model.v5",
        "source_model_sha256": _sha256(model_path),
        "state_temperatures": temperatures.tolist(),
        "state_biases": biases.tolist(),
        "temperature_status": "frozen_sac_only",
        "probability_calibration_method": "per_member_affine_platt_scaling",
        "uncertainty_status": "frozen_sac_only",
        "uncertainty_method": "finite_sample_one_sided_residual_quantile",
        "uncertainty_alpha": 0.05,
        "uncertainty_residual_offset": residual_offset,
        "probability_calibration_inputs": list(probability.input_files),
        "uncertainty_calibration_inputs": list(uncertainty.input_files),
        "selector_calibration_consumed": False,
        "sac_model_test_consumed": False,
        "objective1_claim_eligible": False,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    torch.save(calibrated, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    _publish_no_clobber(temporary, output_path)
    report = {
        "schema_version": "qsafe.natural_sac_state_calibration_report.v2",
        "source_model_sha256": _sha256(model_path),
        "output_model_sha256": _sha256(output_path),
        "state_temperatures": temperatures.tolist(),
        "state_biases": biases.tolist(),
        "probability_calibration_method": "per_member_affine_platt_scaling",
        "probability_samples": len(probability.label),
        "probability_positives": int(probability.label.sum()),
        "probability_auroc": binary_auc(probability.label, probability_mean),
        "probability_ece": expected_calibration_error(probability.label, probability_mean),
        "probability_inputs": list(probability.input_files),
        "uncertainty_samples": len(uncertainty.label),
        "uncertainty_positives": int(uncertainty.label.sum()),
        "uncertainty_residual_offset": residual_offset,
        "uncertainty_empirical_upper_coverage": float(np.mean(
            uncertainty.label.astype(np.float64) <= upper)),
        "uncertainty_inputs": list(uncertainty.input_files),
        "selector_calibration_consumed": False,
        "protected_model_test_consumed": False,
        "objective1_claim_eligible": False,
        "phase2_authorized": False,
    }
    report["report_content_sha256"] = _canonical_sha256(report)
    content = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode("utf-8")
    temporary_report = report_path.with_name(f".{report_path.name}.tmp-{os.getpid()}")
    with temporary_report.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    _publish_no_clobber(temporary_report, report_path)
    return report
