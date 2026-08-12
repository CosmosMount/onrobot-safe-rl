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
    binary_auc,
    expected_calibration_error,
    load_natural_sac_role,
    predict_calibrated_state_risk,
)
from safety_data.policies import load_frozen_droq_policy
from safety_data.natural_ppo_recovery_policy import NaturalPpoRecoveryPolicy
from safety_data.recovery_behaviors import (
    RECOVERY_BEHAVIOR_STEPS,
    RecoveryBehaviorLibrary,
    build_recovery_behavior_library,
)
from runtime.inference.actions import qpos_to_action
from runtime.inference.observations import quat_to_euler_xyz
from train.config import load_app_config
from train.mujoco_snapshot_env import ApplicationState, BranchSnapshot, MujocoSnapshotEnv


ALLOWED_ORIGINAL_INDICES = (4, 5, 6, 7, 8)
MATURE_SHORT_DURATIONS = (1, 2, 3, 5, 10)
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


class MatureShortRecoveryView:
    """Development response family motivated by the earlier 2/3-frame oracle."""

    def __init__(self, full_library: RecoveryBehaviorLibrary) -> None:
        if not isinstance(full_library, RecoveryBehaviorLibrary):
            raise TypeError("full_library must be the attested recovery library")
        self._library = full_library

    @property
    def behavior_steps(self) -> np.ndarray:
        return np.asarray([0] + list(MATURE_SHORT_DURATIONS), dtype=np.int64)

    def capture_branch_state(self) -> None:
        return None

    def restore_branch_state(self, state: None) -> None:
        if state is not None:
            raise ValueError("mature short recovery state must be None")

    def __call__(self, candidate_index: int, observation_history: np.ndarray,
                 step: int, nominal_action: np.ndarray) -> np.ndarray:
        if not 1 <= int(candidate_index) <= len(MATURE_SHORT_DURATIONS):
            raise ValueError("mature short candidate index must lie in [1,5]")
        # Full K9 index one is the frozen mature actor.  Its registered L10
        # duration is not consulted because this wrapper owns shorter locks.
        return self._library(1, observation_history, min(int(step), 9), nominal_action)

    def preview(self, observation_history: np.ndarray,
                nominal_action: np.ndarray) -> np.ndarray:
        mature = self(1, observation_history, 0, nominal_action)
        return np.stack([nominal_action] + [mature] * 5).astype(np.float32)

    def manifest(self) -> dict[str, Any]:
        return {
            "full_library_fingerprint_sha256": self._library.fingerprint(),
            "base_original_k9_index": 1,
            "mature_policy_durations": list(MATURE_SHORT_DURATIONS),
        }


class PpoShortRecoveryView:
    """Run the frozen 30M natural-PPO mean for a short persistent response."""

    def __init__(self, policy: NaturalPpoRecoveryPolicy, env: MujocoSnapshotEnv) -> None:
        self._policy = policy
        self._env = env
        self._previous = policy.initial_previous_action(env)

    @property
    def behavior_steps(self) -> np.ndarray:
        return np.asarray([0] + list(MATURE_SHORT_DURATIONS), dtype=np.int64)

    def capture_branch_state(self) -> np.ndarray:
        return self._previous.copy()

    def restore_branch_state(self, state: np.ndarray) -> None:
        value = np.asarray(state, dtype=np.float32)
        if value.shape != (12,):
            raise ValueError("PPO recovery branch state must be 12D")
        self._previous = value.copy()

    def __call__(self, candidate_index: int, observation_history: np.ndarray,
                 step: int, nominal_action: np.ndarray) -> np.ndarray:
        del observation_history, step, nominal_action
        action = self._policy.action(self._env, self._previous)
        self._previous = action.copy()
        return action

    def preview(self, observation_history: np.ndarray,
                nominal_action: np.ndarray) -> np.ndarray:
        del observation_history
        action = self._policy.action(self._env, self._previous)
        return np.stack([nominal_action] + [action] * 5).astype(np.float32)

    def manifest(self) -> dict[str, Any]:
        return self._policy.manifest() | {
            "ppo_policy_durations": list(MATURE_SHORT_DURATIONS),
        }


_ATTITUDE_OPTIONS = (
    # (angle gain, gyro gain, duration)
    (0.35, 0.00, 5), (0.55, 0.00, 5), (0.75, 0.00, 5),
    (0.35, 0.00, 10), (0.55, 0.00, 10), (0.75, 0.00, 10),
    (0.55, 0.08, 10), (0.75, 0.12, 10),
)


class AttitudeFeedbackRecoveryView:
    """Deployable posture feedback using only corrected observation fields."""

    @property
    def behavior_steps(self) -> np.ndarray:
        return np.asarray([0] + [item[2] for item in _ATTITUDE_OPTIONS], dtype=np.int64)

    def capture_branch_state(self) -> None:
        return None

    def restore_branch_state(self, state: None) -> None:
        if state is not None:
            raise ValueError("attitude recovery is stateless")

    def __call__(self, candidate_index: int, observation_history: np.ndarray,
                 step: int, nominal_action: np.ndarray) -> np.ndarray:
        del step, nominal_action
        if not 1 <= int(candidate_index) <= len(_ATTITUDE_OPTIONS):
            raise ValueError("attitude candidate index must lie in [1,8]")
        angle_gain, gyro_gain, _ = _ATTITUDE_OPTIONS[int(candidate_index) - 1]
        newest = np.asarray(observation_history, dtype=np.float32)[-1]
        roll, pitch, _ = quat_to_euler_xyz(newest[30:34])
        gyro = newest[24:27]
        roll_signal = float(roll + gyro_gain * gyro[0])
        pitch_signal = float(pitch + gyro_gain * gyro[1])
        target = np.asarray([0.05, 0.70, -1.40] * 4, dtype=np.float32)
        # FR, FL, RR, RL. Positive roll raises the left side, so extend the
        # right legs; positive pitch raises the front, so extend the rear.
        side = np.asarray([-1.0, 1.0, -1.0, 1.0], dtype=np.float32)
        fore = np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.float32)
        extension = np.clip(
            -angle_gain * (side * roll_signal + fore * pitch_signal),
            -0.45, 0.45)
        target[1::3] -= 0.45 * extension
        target[2::3] += 0.90 * extension
        previous_target = newest[34:46]
        target = previous_target + np.clip(target - previous_target, -0.08, 0.08)
        return np.clip(qpos_to_action(
            target, init_qpos=np.asarray([0.05, 0.70, -1.40] * 4),
            action_offset=np.asarray([0.2, 0.4, 0.4] * 4)), -1.0, 1.0)

    def preview(self, observation_history: np.ndarray,
                nominal_action: np.ndarray) -> np.ndarray:
        return np.stack([nominal_action] + [
            self(index, observation_history, 0, nominal_action)
            for index in range(1, 9)
        ]).astype(np.float32)

    def manifest(self) -> dict[str, Any]:
        return {
            "kind": "corrected_observation_attitude_posture_feedback",
            "options": [list(item) for item in _ATTITUDE_OPTIONS],
            "privileged_state_used": False,
            "maximum_qtarget_delta_per_step_rad": 0.08,
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
    candidate_set: str = "fixed_nonpolicy",
    ppo_checkpoint: str | Path | None = None,
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
    full_library = build_recovery_behavior_library(mature, env.action_applier)
    if candidate_set == "fixed_nonpolicy":
        recovery: Any = FixedNonpolicyRecoveryView(full_library)
        original_indices = [0] + list(ALLOWED_ORIGINAL_INDICES)
    elif candidate_set == "full_k9_development":
        recovery = full_library
        original_indices = list(range(9))
    elif candidate_set == "mature_short_development":
        recovery = MatureShortRecoveryView(full_library)
        original_indices = [0, 101, 102, 103, 105, 110]
    elif candidate_set == "ppo_short_development":
        if ppo_checkpoint is None:
            raise ValueError("ppo_short_development requires ppo_checkpoint")
        recovery = PpoShortRecoveryView(
            NaturalPpoRecoveryPolicy(ppo_checkpoint), env)
        original_indices = [0, 201, 202, 203, 205, 210]
    elif candidate_set == "attitude_feedback_development":
        recovery = AttitudeFeedbackRecoveryView()
        original_indices = [0] + list(range(301, 309))
    else:
        raise ValueError("unknown recovery candidate_set")
    falls = np.empty((len(plan["identity"]), len(original_indices)), dtype=bool)
    first_failure = np.empty((len(plan["identity"]), len(original_indices)), dtype=np.int16)
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
            candidates = (
                recovery.preview(history, nominal)
                if isinstance(recovery, (FixedNonpolicyRecoveryView,
                                         MatureShortRecoveryView,
                                         PpoShortRecoveryView,
                                         AttitudeFeedbackRecoveryView))
                else recovery.preview_candidates(history, nominal)
            )
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
                continuation_policy=continuation, recovery_program=recovery)
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
        "candidate_set": candidate_set,
        "fixed_recovery": (
            recovery.manifest() if isinstance(
                recovery, (FixedNonpolicyRecoveryView, MatureShortRecoveryView,
                           PpoShortRecoveryView,
                           AttitudeFeedbackRecoveryView))
            else {
                "full_library_fingerprint_sha256": recovery.fingerprint(),
                "original_k9_indices": list(range(9)),
                "behavior_steps": recovery.behavior_steps.tolist(),
                "mature_policy_options_executable": True,
            }
        ),
        "candidate_original_k9_indices": original_indices,
        "phase2_authorized": False,
    }
    content = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
    temporary_report = report_path.with_name(f".{report_path.name}.tmp-{os.getpid()}")
    with temporary_report.open("xb") as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    _publish_no_clobber(temporary_report, report_path)
    return report


def _bootstrap_paired_lcb(delta: np.ndarray, *, seed: int) -> tuple[float, float, float]:
    delta = np.asarray(delta, dtype=np.float64).reshape(-1)
    if len(delta) == 0:
        raise ValueError("paired bootstrap requires states")
    rng = np.random.default_rng(seed)
    values = np.empty(50_000, dtype=np.float64)
    for start in range(0, len(values), 500):
        end = min(start + 500, len(values))
        indices = rng.integers(0, len(delta), size=(end - start, len(delta)))
        values[start:end] = delta[indices].mean(axis=1)
    low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
    return float(low), float(median), float(high)


def freeze_selector_recovery(
    *, calibrated_model: str | Path, branch_plan: str | Path,
    branch_files: list[str | Path], output_model: str | Path,
) -> dict[str, Any]:
    """Freeze the passing trigger/response pair before model-test access."""
    calibrated_model = Path(calibrated_model).resolve()
    branch_plan = Path(branch_plan).resolve()
    output_model = Path(output_model).resolve()
    report_path = output_model.with_suffix(".selector-report.json")
    if output_model.exists() or report_path.exists():
        raise FileExistsError("frozen selector output was already published")
    plan_manifest_path = branch_plan.with_suffix(".manifest.json")
    plan_manifest = json.loads(plan_manifest_path.read_text(encoding="utf-8"))
    if plan_manifest.get("output_sha256") != _sha256(branch_plan) or (
            plan_manifest.get("calibrated_model_sha256") != _sha256(calibrated_model)):
        raise ValueError("branch plan is not bound to the calibrated model")
    loaded_arrays: dict[str, list[np.ndarray]] = {
        name: [] for name in ("identity", "risk", "ensemble_std", "risk_band", "fall")}
    branch_records = []
    seen_sources: set[int] = set()
    for raw_path in branch_files:
        path = Path(raw_path).resolve()
        manifest_path = path.with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("output_sha256") != _sha256(path) or (
                manifest.get("branch_plan_sha256") != _sha256(branch_plan)) or (
                manifest.get("candidate_original_k9_indices") != [0, 4, 5, 6, 7, 8]) or (
                manifest.get("external_force") != "verified_zero"):
            raise ValueError("fixed recovery branch manifest is invalid")
        source = int(manifest["source_seed"])
        if source in seen_sources:
            raise ValueError("duplicate fixed recovery source")
        seen_sources.add(source)
        with np.load(path, allow_pickle=False) as loaded:
            for name in loaded_arrays:
                loaded_arrays[name].append(loaded[name].copy())
        branch_records.append({
            "file": path.name, "sha256": _sha256(path),
            "manifest_sha256": _sha256(manifest_path), "source_seed": source,
            "states": int(manifest["states"]),
        })
    if seen_sources != {9421, 9422, 9423}:
        raise ValueError("selector recovery requires all preregistered sources")
    arrays = {name: np.concatenate(values) for name, values in loaded_arrays.items()}
    if len(arrays["identity"]) != int(plan_manifest["selected_states"]) or len(
            set(map(bytes, arrays["identity"]))) != len(arrays["identity"]):
        raise ValueError("selector recovery outputs are incomplete or duplicated")
    fall = arrays["fall"].astype(np.float64)
    if fall.shape != (len(arrays["identity"]), 6):
        raise ValueError("selector recovery fall array must have six candidates")
    max_std = 0.20
    searches = []
    passing = []
    durations = [0, 10, 10, 25, 25, 25]
    original_indices = [0, 4, 5, 6, 7, 8]
    for lowest_band, quantile in ((3, 0.95), (2, 0.90), (1, 0.80), (0, 0.65)):
        selected = (arrays["risk_band"] >= lowest_band) & (
            arrays["ensemble_std"] <= max_std)
        for local_index in range(1, 6):
            delta = fall[selected, 0] - fall[selected, local_index]
            ci = _bootstrap_paired_lcb(
                delta, seed=20260812 + 100 * lowest_band + local_index)
            row = {
                "trigger_risk_quantile": quantile,
                "trigger_risk_threshold": float(plan_manifest[
                    "risk_band_lower_cutoffs"][lowest_band]),
                "ensemble_std_max": max_std,
                "sample_states": int(selected.sum()),
                "candidate_local_index": local_index,
                "candidate_original_k9_index": original_indices[local_index],
                "candidate_duration_steps": durations[local_index],
                "nominal_fall_rate": float(fall[selected, 0].mean()),
                "recovery_fall_rate": float(fall[selected, local_index].mean()),
                "absolute_reduction": float(delta.mean()),
                "absolute_reduction_ci95": list(ci),
                "improved": int(np.sum(delta > 0)),
                "worsened": int(np.sum(delta < 0)),
            }
            row["calibration_gate"] = bool(
                row["absolute_reduction"] >= 0.03 and ci[0] > 0.0)
            searches.append(row)
            if row["calibration_gate"]:
                passing.append(row)
    if not passing:
        raise RuntimeError("no fixed nonpolicy recovery passes the calibration gate")
    chosen = sorted(passing, key=lambda row: (
        row["recovery_fall_rate"], row["candidate_duration_steps"],
        row["candidate_original_k9_index"], -row["absolute_reduction"],
    ))[0]
    base_intervention_fraction = 1.0 - float(chosen["trigger_risk_quantile"])
    sampled_band = arrays["risk_band"] >= {
        0.95: 3, 0.90: 2, 0.80: 1, 0.65: 0
    }[float(chosen["trigger_risk_quantile"])]
    abstention_acceptance = float(np.mean(
        arrays["ensemble_std"][sampled_band] <= max_std))
    selector = {
        "risk_threshold": chosen["trigger_risk_threshold"],
        "risk_quantile": chosen["trigger_risk_quantile"],
        "ensemble_std_max": max_std,
        "recovery_original_k9_index": chosen["candidate_original_k9_index"],
        "recovery_duration_steps": chosen["candidate_duration_steps"],
        "estimated_intervention_rate": base_intervention_fraction * abstention_acceptance,
        "reselection_during_option": False,
        "decision_frequency": "every_policy_step_when_no_option_active",
    }
    artifact = torch.load(calibrated_model, map_location="cpu", weights_only=False)
    if artifact.get("schema_version") != "qsafe.natural_ppo_state_trigger_model.v5" or (
            artifact.get("selector_calibration_consumed") is not False):
        raise ValueError("input model is not ready for selector freezing")
    frozen = dict(artifact)
    frozen.update({
        "schema_version": "qsafe.natural_ppo_state_trigger_model.v6",
        "source_calibrated_model_sha256": _sha256(calibrated_model),
        "selector_status": "frozen_sac_only",
        "selector": selector,
        "selector_branch_plan_sha256": _sha256(branch_plan),
        "selector_calibration_branch_inputs": branch_records,
        "selector_calibration_consumed": True,
        "sac_model_test_consumed": False,
        "objective1_claim_eligible": False,
    })
    output_model.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_model.with_name(f".{output_model.name}.tmp-{os.getpid()}")
    torch.save(frozen, temporary)
    with temporary.open("rb") as stream: os.fsync(stream.fileno())
    _publish_no_clobber(temporary, output_model)
    report = {
        "schema_version": "qsafe.natural_sac_selector_freeze_report.v1",
        "source_calibrated_model_sha256": _sha256(calibrated_model),
        "output_model_sha256": _sha256(output_model),
        "branch_plan_sha256": _sha256(branch_plan),
        "branch_inputs": branch_records,
        "search": searches,
        "chosen": chosen,
        "selector": selector,
        "calibration_pass": True,
        "protected_model_test_consumed": False,
        "objective1_claim_eligible": False,
        "phase2_authorized": False,
    }
    content = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
    temporary_report = report_path.with_name(f".{report_path.name}.tmp-{os.getpid()}")
    with temporary_report.open("xb") as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    _publish_no_clobber(temporary_report, report_path)
    return report


def build_protected_model_test_plan(
    *, source_directory: str | Path, frozen_model: str | Path,
    output: str | Path, target_states: int = 1200, device: str | None = None,
) -> dict[str, Any]:
    """Freeze a risk-enriched but outcome-blind H96-nonoverlapping test plan."""
    source_directory = Path(source_directory).resolve()
    frozen_model = Path(frozen_model).resolve()
    output = Path(output).resolve()
    report_path = output.with_suffix(".manifest.json")
    if output.exists() or report_path.exists():
        raise FileExistsError("protected model-test plan was already published")
    artifact = torch.load(frozen_model, map_location="cpu", weights_only=False)
    if artifact.get("schema_version") != "qsafe.natural_ppo_state_trigger_model.v6" or (
            artifact.get("selector_status") != "frozen_sac_only") or (
            artifact.get("sac_model_test_consumed") is not False):
        raise ValueError("protected planning requires a frozen unconsumed v6 model")
    expected = []
    for actor_seed in (43, 44, 45, 46):
        for checkpoint_index, training_step in enumerate((25_000, 50_000, 100_000), 1):
            source_seed = 9500 + 10 * (actor_seed - 43) + checkpoint_index
            expected.append((actor_seed, training_step, source_seed))
    observations = []
    source_records = []
    sources: list[tuple[Path, dict[str, Any], dict[str, np.ndarray]]] = []
    generator_commit = None
    for actor_seed, training_step, source_seed in expected:
        stem = f"actor{actor_seed}-age{training_step}-source{source_seed}"
        data_path = source_directory / f"{stem}.npz"
        manifest_path = source_directory / f"{stem}.manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {
            "output_sha256": _sha256(data_path), "actor_seed": actor_seed,
            "actor_training_step": training_step, "source_seed": source_seed,
            "fixed_exposure_policy_steps": 15_000,
            "external_force": "verified_zero", "recovery_executed": False,
            "phase2_authorized": False,
        }
        if any(manifest.get(name) != value for name, value in required.items()):
            raise ValueError(f"protected source {stem} violates the frozen roster")
        if generator_commit is None:
            generator_commit = manifest["generator_commit"]
        elif generator_commit != manifest["generator_commit"]:
            raise ValueError("protected sources use different generator commits")
        with np.load(data_path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name].copy() for name in (
                "identity", "observation_history", "episode_id", "episode_step")}
        observations.append(arrays["observation_history"])
        sources.append((data_path, manifest, arrays))
        source_records.append({
            "file": data_path.name, "sha256": _sha256(data_path),
            "manifest_sha256": _sha256(manifest_path), "actor_seed": actor_seed,
            "training_step": training_step, "source_seed": source_seed,
            "states": len(arrays["identity"]),
        })
    all_risk, all_std = predict_calibrated_state_risk(
        frozen_model, np.concatenate(observations), device=device)
    selector = artifact["selector"]
    candidate_rows = []
    offset = 0
    for (_, manifest, arrays), observation in zip(sources, observations, strict=True):
        count = len(observation)
        risk = all_risk[offset:offset + count]
        std = all_std[offset:offset + count]
        offset += count
        for episode_id in np.unique(arrays["episode_id"]):
            episode = np.flatnonzero(arrays["episode_id"] == episode_id)
            blocks = arrays["episode_step"][episode] // 97
            for block in np.unique(blocks):
                rows = episode[blocks == block]
                row = int(rows[np.argmax(risk[rows])])
                candidate_rows.append({
                    "identity": arrays["identity"][row],
                    "source_seed": int(manifest["source_seed"]),
                    "row_index": row,
                    "risk": float(risk[row]), "ensemble_std": float(std[row]),
                })
    if len(candidate_rows) < int(target_states):
        raise RuntimeError("protected sources cannot supply 1200 nonoverlapping units")
    # Risk enrichment is fixed before any H96 label or branch outcome is read.
    chosen = sorted(candidate_rows, key=lambda row: (
        -row["risk"], hashlib.sha256(bytes(row["identity"])).digest()))[:int(target_states)]
    qsafe = np.asarray([
        row["risk"] >= float(selector["risk_threshold"]) and
        row["ensemble_std"] <= float(selector["ensemble_std_max"])
        for row in chosen], dtype=bool)
    intervention_count = int(qsafe.sum())
    placebo_order = sorted(range(len(chosen)), key=lambda index: hashlib.sha256(
        b"qsafe.protected_matched_placebo.v1\0" + bytes(chosen[index]["identity"])
    ).digest())
    placebo = np.zeros(len(chosen), dtype=bool)
    placebo[placebo_order[:intervention_count]] = True
    arrays_out = {
        "identity": np.asarray([row["identity"] for row in chosen], dtype="S64"),
        "source_seed": np.asarray([row["source_seed"] for row in chosen], dtype=np.int64),
        "row_index": np.asarray([row["row_index"] for row in chosen], dtype=np.int64),
        "risk": np.asarray([row["risk"] for row in chosen], dtype=np.float32),
        "ensemble_std": np.asarray([row["ensemble_std"] for row in chosen], dtype=np.float32),
        "qsafe_intervene": qsafe,
        "placebo_intervene": placebo,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays_out)
    with temporary.open("rb") as stream: os.fsync(stream.fileno())
    _publish_no_clobber(temporary, output)
    report = {
        "schema_version": "qsafe.natural_sac_protected_model_test_plan.v1",
        "output_file": output.name, "output_sha256": _sha256(output),
        "frozen_model_sha256": _sha256(frozen_model),
        "source_inputs": source_records, "candidate_nonoverlap_policy_steps": 97,
        "candidate_choice_within_block": "maximum_frozen_risk",
        "cohort_choice": "top_frozen_risk_then_identity_hash",
        "natural_fall_label_used": False, "branch_outcome_used": False,
        "independent_snapshots": len(chosen),
        "qsafe_interventions": intervention_count,
        "placebo_interventions": int(placebo.sum()),
        "intervention_rate": float(qsafe.mean()),
        "selector": selector, "protected_outcomes_opened": False,
        "objective1_claim_eligible": False, "phase2_authorized": False,
    }
    content = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
    temporary_report = report_path.with_name(f".{report_path.name}.tmp-{os.getpid()}")
    with temporary_report.open("xb") as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    _publish_no_clobber(temporary_report, report_path)
    return report


def summarize_protected_model_test(
    *, frozen_model: str | Path, model_test_plan: str | Path,
    source_directory: str | Path, branch_files: list[str | Path],
    output_model: str | Path,
) -> dict[str, Any]:
    """Consume protected outcomes once and publish an explicit pass/fail."""
    frozen_model = Path(frozen_model).resolve()
    model_test_plan = Path(model_test_plan).resolve()
    source_directory = Path(source_directory).resolve()
    output_model = Path(output_model).resolve()
    report_path = output_model.with_suffix(".model-test-report.json")
    if output_model.exists() or report_path.exists():
        raise FileExistsError("protected model-test output was already published")
    artifact = torch.load(frozen_model, map_location="cpu", weights_only=False)
    if artifact.get("schema_version") != "qsafe.natural_ppo_state_trigger_model.v6" or (
            artifact.get("sac_model_test_consumed") is not False):
        raise ValueError("protected outcomes require an unconsumed frozen v6 model")
    plan_manifest_path = model_test_plan.with_suffix(".manifest.json")
    plan_manifest = json.loads(plan_manifest_path.read_text(encoding="utf-8"))
    if plan_manifest.get("output_sha256") != _sha256(model_test_plan) or (
            plan_manifest.get("frozen_model_sha256") != _sha256(frozen_model)) or (
            plan_manifest.get("protected_outcomes_opened") is not False):
        raise ValueError("protected plan is not bound to the frozen model")
    with np.load(model_test_plan, allow_pickle=False) as loaded:
        plan = {name: loaded[name].copy() for name in loaded.files}
    outcome_by_identity: dict[bytes, np.ndarray] = {}
    branch_records = []
    for raw_path in branch_files:
        path = Path(raw_path).resolve()
        manifest_path = path.with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("output_sha256") != _sha256(path) or (
                manifest.get("branch_plan_sha256") != _sha256(model_test_plan)):
            raise ValueError("protected branch output is not bound to its plan")
        with np.load(path, allow_pickle=False) as loaded:
            for identity, falls in zip(loaded["identity"], loaded["fall"], strict=True):
                key = bytes(identity)
                if key in outcome_by_identity:
                    raise ValueError("duplicate protected branch identity")
                outcome_by_identity[key] = np.asarray(falls, dtype=bool).copy()
        branch_records.append({
            "file": path.name, "sha256": _sha256(path),
            "manifest_sha256": _sha256(manifest_path),
            "source_seed": int(manifest["source_seed"]),
            "states": int(manifest["states"]),
        })
    if len(branch_records) != 12 or len(outcome_by_identity) != len(plan["identity"]):
        raise ValueError("protected model-test branches are incomplete")
    fall = np.stack([outcome_by_identity[bytes(value)] for value in plan["identity"]])
    if fall.shape != (len(plan["identity"]), 6):
        raise ValueError("protected model-test requires nominal plus fixed K9 4-8")
    nominal = fall[:, 0]
    qsafe = np.where(plan["qsafe_intervene"], fall[:, 1], nominal)
    placebo = np.where(plan["placebo_intervene"], fall[:, 1], nominal)
    qsafe_delta = nominal.astype(float) - qsafe.astype(float)
    placebo_delta = placebo.astype(float) - qsafe.astype(float)
    qsafe_ci = _bootstrap_paired_lcb(qsafe_delta, seed=20260813)
    placebo_ci = _bootstrap_paired_lcb(placebo_delta, seed=20260814)
    # Natural H96 labels are opened only here, after every model and selector
    # field has been frozen and every branch has been published.
    labels = np.empty(len(plan["identity"]), dtype=bool)
    for source_seed in np.unique(plan["source_seed"]):
        record = next(item for item in plan_manifest["source_inputs"]
                      if int(item["source_seed"]) == int(source_seed))
        path = source_directory / record["file"]
        if _sha256(path) != record["sha256"]:
            raise ValueError("protected natural-label source changed")
        selected = np.flatnonzero(plan["source_seed"] == source_seed)
        with np.load(path, allow_pickle=False) as loaded:
            labels[selected] = loaded["label"][plan["row_index"][selected]]
            if not np.array_equal(
                    loaded["identity"][plan["row_index"][selected]],
                    plan["identity"][selected]):
                raise ValueError("protected natural-label identities differ")
    auc = binary_auc(labels, plan["risk"])
    ece = expected_calibration_error(labels, plan["risk"])
    response_mask = plan["qsafe_intervene"]
    response_delta = fall[response_mask, 0].astype(float) - fall[
        response_mask, 1].astype(float)
    response_ci = _bootstrap_paired_lcb(response_delta, seed=20260815)
    checks = {
        "minimum_independent_snapshots": len(nominal) >= 1200,
        "natural_state_auroc_at_least_0.60": auc >= 0.60,
        "ece_at_most_0.08": ece <= 0.08,
        "intervention_rate_at_most_0.35": float(plan["qsafe_intervene"].mean()) <= 0.35,
        "trigger_response_reduction_at_least_0.03": float(response_delta.mean()) >= 0.03,
        "trigger_response_reduction_lcb_positive": response_ci[0] > 0.0,
        "overall_reduction_positive": float(qsafe_delta.mean()) > 0.0,
        "overall_reduction_lcb_positive": qsafe_ci[0] > 0.0,
        "better_than_placebo_lcb_positive": placebo_ci[0] > 0.0,
    }
    model_test_pass = all(checks.values())
    report = {
        "schema_version": "qsafe.natural_sac_protected_model_test_report.v1",
        "frozen_model_sha256": _sha256(frozen_model),
        "model_test_plan_sha256": _sha256(model_test_plan),
        "branch_inputs": branch_records,
        "independent_snapshots": len(nominal),
        "natural_positive_labels": int(labels.sum()),
        "natural_state_auroc": auc, "natural_state_ece": ece,
        "interventions": int(plan["qsafe_intervene"].sum()),
        "intervention_rate": float(plan["qsafe_intervene"].mean()),
        "nominal_fall_rate": float(nominal.mean()),
        "qsafe_fall_rate": float(qsafe.mean()),
        "placebo_fall_rate": float(placebo.mean()),
        "overall_absolute_reduction": float(qsafe_delta.mean()),
        "overall_absolute_reduction_ci95": list(qsafe_ci),
        "qsafe_vs_placebo_reduction": float(placebo_delta.mean()),
        "qsafe_vs_placebo_reduction_ci95": list(placebo_ci),
        "trigger_states": int(response_mask.sum()),
        "trigger_nominal_fall_rate": float(fall[response_mask, 0].mean()),
        "trigger_recovery_fall_rate": float(fall[response_mask, 1].mean()),
        "trigger_response_reduction": float(response_delta.mean()),
        "trigger_response_reduction_ci95": list(response_ci),
        "checks": checks, "model_test_pass": model_test_pass,
        "failure_classification": None if model_test_pass else (
            "fixed_nonpolicy_response_not_cross_seed_generalizable"),
        "objective1_claim_eligible": False, "phase2_authorized": False,
    }
    consumed = dict(artifact)
    consumed.update({
        "schema_version": "qsafe.natural_ppo_state_trigger_model.v7",
        "source_frozen_model_sha256": _sha256(frozen_model),
        "sac_model_test_consumed": True,
        "protected_model_test_report": report,
        "model_test_pass": model_test_pass,
        "objective1_claim_eligible": False,
    })
    output_model.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_model.with_name(f".{output_model.name}.tmp-{os.getpid()}")
    torch.save(consumed, temporary)
    with temporary.open("rb") as stream: os.fsync(stream.fileno())
    _publish_no_clobber(temporary, output_model)
    report["output_model_sha256"] = _sha256(output_model)
    content = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
    temporary_report = report_path.with_name(f".{report_path.name}.tmp-{os.getpid()}")
    with temporary_report.open("xb") as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    _publish_no_clobber(temporary_report, report_path)
    return report
