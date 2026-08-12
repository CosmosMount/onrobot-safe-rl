"""Outcome-blind selector sampling and fixed-recovery SAC branching."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from safety_data.native import ReplicaSeedBundle, evaluate_same_state_group
from safety_data.natural_sac_calibration import (
    ROLE_ROSTER,
    load_natural_sac_role,
    predict_calibrated_state_risk,
)
from safety_data.policies import load_frozen_droq_policy
from safety_data.recovery_behaviors import (
    RECOVERY_BEHAVIOR_STEPS,
    RecoveryBehaviorLibrary,
    build_recovery_behavior_library,
)
from train.config import load_app_config
from train.mujoco_snapshot_env import ApplicationState, BranchSnapshot, MujocoSnapshotEnv


ALLOWED_ORIGINAL_INDICES = (4, 5, 6, 7, 8)
RISK_BAND_LOWER_QUANTILES = (0.65, 0.80, 0.90, 0.95)
RISK_BAND_UPPER_QUANTILES = (0.80, 0.90, 0.95, 1.00)
RISK_BAND_NAMES = ("top_20_to_35", "top_10_to_20", "top_5_to_10", "top_0_to_5")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_no_clobber(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite recovery output: {destination}") from exc
    temporary.unlink()


class FixedNonpolicyRecoveryView:
    """Six-candidate view that makes mature-policy options unreachable."""

    def __init__(self, full_library: RecoveryBehaviorLibrary) -> None:
        if not isinstance(full_library, RecoveryBehaviorLibrary):
            raise TypeError("full_library must be the attested recovery library")
        self._library = full_library
        self._behavior_steps = np.asarray(
            [0] + [RECOVERY_BEHAVIOR_STEPS[index]
                   for index in ALLOWED_ORIGINAL_INDICES], dtype=np.int64)

    @property
    def behavior_steps(self) -> np.ndarray:
        return self._behavior_steps.copy()

    def capture_branch_state(self) -> None:
        return None

    def restore_branch_state(self, state: None) -> None:
        if state is not None:
            raise ValueError("fixed nonpolicy recovery state must be None")

    def __call__(self, candidate_index: int, observation_history: np.ndarray,
                 step: int, nominal_action: np.ndarray) -> np.ndarray:
        if not 1 <= int(candidate_index) <= len(ALLOWED_ORIGINAL_INDICES):
            raise ValueError("fixed nonpolicy candidate index must lie in [1,5]")
        return self._library(
            ALLOWED_ORIGINAL_INDICES[int(candidate_index) - 1],
            observation_history, step, nominal_action)

    def preview(self, observation_history: np.ndarray,
                nominal_action: np.ndarray) -> np.ndarray:
        actions = [np.asarray(nominal_action, dtype=np.float32)]
        for local_index in range(1, 6):
            actions.append(self(local_index, observation_history, 0, nominal_action))
        return np.stack(actions).astype(np.float32)

    def manifest(self) -> dict[str, Any]:
        return {
            "full_library_fingerprint_sha256": self._library.fingerprint(),
            "original_k9_indices": list(ALLOWED_ORIGINAL_INDICES),
            "behavior_steps": self._behavior_steps.tolist(),
            "mature_policy_options_executable": False,
        }


def build_selector_branch_plan(
    *, calibration_root: str | Path, calibrated_model: str | Path,
    output: str | Path, samples_per_band: int = 300, device: str | None = None,
) -> dict[str, Any]:
    """Select risk-stratified snapshots without reading branch outcomes."""
    if isinstance(samples_per_band, bool) or int(samples_per_band) <= 0:
        raise ValueError("samples_per_band must be positive")
    calibration_root = Path(calibration_root).resolve()
    calibrated_model = Path(calibrated_model).resolve()
    output = Path(output).resolve()
    report_path = output.with_suffix(".manifest.json")
    if output.exists() or report_path.exists():
        raise FileExistsError("selector branch plan was already published")
    role = load_natural_sac_role(calibration_root / "selector", "selector")
    risk, uncertainty = predict_calibrated_state_risk(
        calibrated_model, role.observation_history, device=device)
    cutoffs = np.quantile(risk, RISK_BAND_LOWER_QUANTILES)
    upper_cutoffs = np.quantile(risk, RISK_BAND_UPPER_QUANTILES)
    row_index = np.empty(len(risk), dtype=np.int64)
    for source_seed in np.unique(role.source_seed):
        selected = np.flatnonzero(role.source_seed == source_seed)
        row_index[selected] = np.arange(len(selected), dtype=np.int64)
    chosen: list[int] = []
    chosen_band: list[int] = []
    populations: list[int] = []
    for band_index, (lower, upper) in enumerate(zip(cutoffs, upper_cutoffs, strict=True)):
        mask = risk >= lower
        if band_index + 1 < len(cutoffs):
            mask &= risk < upper
        else:
            mask &= risk <= upper
        eligible = np.flatnonzero(mask)
        populations.append(len(eligible))
        if len(eligible) < int(samples_per_band):
            raise RuntimeError(f"risk band {band_index} has too few selector states")
        order = sorted(eligible.tolist(), key=lambda index: hashlib.sha256(
            b"qsafe.selector.outcome_blind.v1\0" + bytes(role.identities[index])
        ).digest())
        selected = order[:int(samples_per_band)]
        chosen.extend(selected)
        chosen_band.extend([band_index] * len(selected))
    chosen_array = np.asarray(chosen, dtype=np.int64)
    arrays = {
        "identity": role.identities[chosen_array],
        "source_seed": role.source_seed[chosen_array],
        "row_index": row_index[chosen_array],
        "risk": risk[chosen_array].astype(np.float32),
        "ensemble_std": uncertainty[chosen_array].astype(np.float32),
        "risk_band": np.asarray(chosen_band, dtype=np.int8),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    _publish_no_clobber(temporary, output)
    report = {
        "schema_version": "qsafe.natural_sac_selector_branch_plan.v1",
        "output_file": output.name,
        "output_sha256": _sha256(output),
        "calibrated_model_sha256": _sha256(calibrated_model),
        "selector_inputs": list(role.input_files),
        "selection_inputs": ["snapshot_identity", "calibrated_state_risk"],
        "natural_fall_label_used_for_selection": False,
        "branch_outcome_used_for_selection": False,
        "risk_band_names": list(RISK_BAND_NAMES),
        "risk_band_lower_quantiles": list(RISK_BAND_LOWER_QUANTILES),
        "risk_band_upper_quantiles": list(RISK_BAND_UPPER_QUANTILES),
        "risk_band_lower_cutoffs": cutoffs.tolist(),
        "risk_band_upper_cutoffs": upper_cutoffs.tolist(),
        "risk_band_population": populations,
        "samples_per_band": int(samples_per_band),
        "selected_states": len(chosen_array),
        "phase2_authorized": False,
    }
    content = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
    temporary_report = report_path.with_name(f".{report_path.name}.tmp-{os.getpid()}")
    with temporary_report.open("xb") as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    _publish_no_clobber(temporary_report, report_path)
    return report


class _SessionContinuation:
    def __init__(self, sample_action: Callable[[np.ndarray, np.random.Generator], np.ndarray]):
        self._sample_action = sample_action

    def __call__(self, history: np.ndarray, step: int,
                 rng: np.random.Generator) -> np.ndarray:
        del step
        return self._sample_action(np.asarray(history, dtype=np.float32)[-1], rng)


def _snapshot_from_row(arrays: Any, row: int) -> BranchSnapshot:
    history_length = int(arrays["history_length"][row])
    if not 1 <= history_length <= 5:
        raise ValueError("snapshot history length is outside [1,5]")
    padded = np.asarray(arrays["observation_history"][row], dtype=np.float32)
    return BranchSnapshot(
        np.asarray(arrays["integration_state"][row], dtype=np.float64),
        ApplicationState(
            previous_action_requested=arrays["previous_action_requested"][row],
            previous_action_executed=arrays["previous_action_executed"][row],
            previous_action_q_target=arrays["previous_action_q_target"][row],
            observation_history=padded[-history_length:],
            action_filter_state=None,
        ),
    )


def evaluate_selector_recovery_source(
    *, source_data: str | Path, source_manifest: str | Path,
    branch_plan: str | Path, mature_checkpoint: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Evaluate nominal plus five fixed nonpolicy responses with H96 CRN."""
    source_data = Path(source_data).resolve()
    source_manifest = Path(source_manifest).resolve()
    branch_plan = Path(branch_plan).resolve()
    mature_checkpoint = Path(mature_checkpoint).resolve()
    output = Path(output).resolve()
    report_path = output.with_suffix(".manifest.json")
    if output.exists() or report_path.exists():
        raise FileExistsError("selector recovery branch output was already published")
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if manifest.get("output_sha256") != _sha256(source_data) or (
            manifest.get("external_force") != "verified_zero") or (
            manifest.get("recovery_executed") is not False):
        raise ValueError("source natural-SAC manifest is not eligible")
    source_seed = int(manifest["source_seed"])
    with np.load(branch_plan, allow_pickle=False) as loaded_plan:
        source_mask = loaded_plan["source_seed"] == source_seed
        plan = {name: loaded_plan[name][source_mask].copy() for name in loaded_plan.files}
    if len(plan["identity"]) == 0:
        raise ValueError("branch plan contains no rows for this source")
    robot, train, _ = load_app_config(manifest["config_path"])
    if train.use_action_filter or not np.isclose(robot.move_speed, 0.30) or not np.isclose(
            robot.fallen_orientation_rad, 1.047198):
        raise ValueError("selector branching runtime differs from frozen target")
    torch.set_num_threads(1)
    actor = load_frozen_droq_policy(
        manifest["actor_manifest"]["actor_path"], manifest["config_path"],
        observation_dim=robot.obs_dim, action_dim=robot.num_joints,
        training_step=int(manifest["actor_training_step"]), device="cpu")
    mature = load_frozen_droq_policy(
        mature_checkpoint, manifest["config_path"], observation_dim=robot.obs_dim,
        action_dim=robot.num_joints, training_step=500_000, device="cpu")
    env = MujocoSnapshotEnv(
        manifest["model_path"], robot, policy_frequency=train.control_frequency,
        max_joint_delta=train.max_joint_delta, use_action_filter=False)
    fixed = FixedNonpolicyRecoveryView(
        build_recovery_behavior_library(mature, env.action_applier))
    falls = np.empty((len(plan["identity"]), 6), dtype=bool)
    first_failure = np.empty((len(plan["identity"]), 6), dtype=np.int16)
    with np.load(source_data, allow_pickle=False) as arrays, actor.inference_session() as sample:
        continuation = _SessionContinuation(sample)
        for output_row, source_row in enumerate(plan["row_index"]):
            source_row = int(source_row)
            if bytes(arrays["identity"][source_row]) != bytes(plan["identity"][output_row]):
                raise RuntimeError("branch plan row identity does not match source data")
            snapshot = _snapshot_from_row(arrays, source_row)
            env.restore(snapshot)
            history = env.observation_history()
            nominal = np.asarray(arrays["action_requested"][source_row], dtype=np.float32)
            candidates = fixed.preview(history, nominal)
            seed = int.from_bytes(hashlib.sha256(
                b"qsafe.fixed_recovery.crn.v1\0" + bytes(plan["identity"][output_row])
            ).digest()[:8], "little")
            seeds = ReplicaSeedBundle(
                crn_id=np.asarray([seed], dtype=np.uint64),
                rollout_seed=np.asarray([seed ^ 0x524F4C4C], dtype=np.uint64),
                perturbation_seed=np.asarray([seed ^ 0x50455254], dtype=np.uint64),
            )
            evaluated = evaluate_same_state_group(
                env, snapshot, candidates, seeds, horizon_steps=96,
                continuation_policy=continuation, recovery_program=fixed)
            falls[output_row] = evaluated.fall[:, 0]
            first_failure[output_row] = evaluated.first_failure_step[:, 0]
    arrays_out = dict(plan)
    arrays_out.update({"fall": falls, "first_failure_step": first_failure})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays_out)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    _publish_no_clobber(temporary, output)
    report = {
        "schema_version": "qsafe.natural_sac_fixed_recovery_branches.v1",
        "output_file": output.name,
        "output_sha256": _sha256(output),
        "source_data_sha256": _sha256(source_data),
        "branch_plan_sha256": _sha256(branch_plan),
        "source_seed": source_seed,
        "actor_seed": int(manifest["actor_seed"]),
        "actor_training_step": int(manifest["actor_training_step"]),
        "states": len(falls),
        "horizon_policy_steps": 96,
        "replicas": 1,
        "external_force": "verified_zero",
        "fixed_recovery": fixed.manifest(),
        "candidate_original_k9_indices": [0] + list(ALLOWED_ORIGINAL_INDICES),
        "phase2_authorized": False,
    }
    content = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
    temporary_report = report_path.with_name(f".{report_path.name}.tmp-{os.getpid()}")
    with temporary_report.open("xb") as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    _publish_no_clobber(temporary_report, report_path)
    return report
