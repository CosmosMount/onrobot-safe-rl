"""Fail-closed K9 selector with a fingerprinted calibration bundle."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
import re
from typing import Any, Mapping

import numpy as np

from rl.qsafe.recovery_program import (
    RECOVERY_PROGRAM_CANDIDATE_COUNT,
    RECOVERY_PROGRAM_NOMINAL_INDEX,
)


RECOVERY_SELECTOR_BUNDLE_SCHEMA_VERSION = "qsafe.recovery_selector_bundle.v1"
RECOVERY_SELECTOR_ENSEMBLE_STD_DDOF = 0
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

_FROZEN_SELECTOR_GRID = {
    "nominal_risk_lcb_trigger": (0.10, 0.20, 0.30, 0.40, 0.50),
    "min_benefit_lcb": (0.00, 0.02, 0.05, 0.08, 0.12),
    "max_risk_ucb": (0.25, 0.40, 0.55, 0.70),
}
_FROZEN_SELECTOR_SUPPORT = {
    "max_epistemic_std": 0.20,
    "max_action_delta_rms": 0.50,
    "max_q_target_delta_rms": 0.25,
}

_CANDIDATE_CHOICE_SEMANTICS = {
    "candidate_axis": "canonical_ordered_K9_recovery_programs",
    "nominal_index": RECOVERY_PROGRAM_NOMINAL_INDEX,
    "prediction_contract": "finite_member_probabilities_M_at_least_2_by_K9",
    "score_definitions": {
        "risk_mean": "arithmetic_mean_across_ensemble_members",
        "risk_std": "population_standard_deviation_ddof_0",
        "benefit_mean": (
            "arithmetic_mean_of_member_nominal_risk_minus_candidate_risk"),
        "risk_ucb": "clip_risk_mean_plus_signed_risk_upper_to_0_1",
        "benefit_lcb": (
            "clip_benefit_mean_minus_signed_benefit_lower_to_minus1_1"),
        "nominal_risk_lcb": (
            "clip_nominal_risk_mean_minus_signed_nominal_lower_to_0_1"),
        "requested_action_rms_from_nominal": (
            "sqrt_mean_square_delta_over_12_requested_components"),
        "q_target_rms_from_nominal": (
            "sqrt_mean_square_delta_over_12_q_target_components"),
    },
    "nominal_trigger_rule": (
        "nominal_risk_lcb_greater_than_or_equal_trigger"),
    "eligibility_rules": {
        "candidate_mask": "required_true",
        "benefit_lcb": "strictly_greater_than_min_benefit_lcb",
        "risk_ucb": "less_than_or_equal_max_risk_ucb",
        "ensemble_probability_std": (
            "less_than_or_equal_max_epistemic_std"),
        "requested_action_rms_from_nominal": (
            "less_than_or_equal_max_action_delta_rms"),
        "q_target_rms_from_nominal": (
            "less_than_or_equal_max_q_target_delta_rms"),
    },
    "nominal_candidate_eligibility": "forced_false",
    "executed_action_slew_gate": "forbidden",
    "selection_order": [
        "lowest_risk_ucb",
        "highest_benefit_lcb",
        "lowest_canonical_candidate_index",
    ],
    "state_below_trigger": "select_nominal",
    "no_eligible_candidate": "select_nominal",
    "selected_identity": "persistent_K9_recovery_program_index",
    "task_q_gate": "forbidden",
    "reward_q_gate": "forbidden",
    "frozen_selector_grid": {
        name: list(values) for name, values in _FROZEN_SELECTOR_GRID.items()
    },
    "frozen_support_limits": dict(_FROZEN_SELECTOR_SUPPORT),
}


def recovery_candidate_choice_semantics() -> dict[str, Any]:
    """Return a detached copy of the immutable V4 choice semantics."""
    return copy.deepcopy(_CANDIDATE_CHOICE_SEMANTICS)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("selector bundle must be canonical JSON") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _checked_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class RecoveryConformalOffsets:
    """Frozen signed one-sided residual quantiles from uncertainty calibration.

    Finite-sample conformal order statistics can be negative.  The two option
    vectors keep candidate zero at exactly zero because its trigger has the
    separate signed ``nominal_lower`` offset.
    """

    nominal_lower: float
    risk_upper: np.ndarray
    benefit_lower: np.ndarray
    calibration_report_sha256: str

    def validated(self) -> "RecoveryConformalOffsets":
        nominal = _finite_signed(self.nominal_lower, "nominal_lower")
        risk = _offset_vector(self.risk_upper, "risk_upper")
        benefit = _offset_vector(self.benefit_lower, "benefit_lower")
        if risk[RECOVERY_PROGRAM_NOMINAL_INDEX] != 0.0 or benefit[
                RECOVERY_PROGRAM_NOMINAL_INDEX] != 0.0:
            raise ValueError("nominal conformal option offsets must equal zero")
        digest = _checked_sha256(
            self.calibration_report_sha256,
            "calibration_report_sha256",
        )
        return RecoveryConformalOffsets(
            nominal_lower=nominal,
            risk_upper=_readonly(risk),
            benefit_lower=_readonly(benefit),
            calibration_report_sha256=digest,
        )


@dataclass(frozen=True)
class RecoverySelectorConfig:
    nominal_risk_lcb_trigger: float
    min_benefit_lcb: float
    max_risk_ucb: float
    max_epistemic_std: float
    max_action_delta_rms: float
    max_q_target_delta_rms: float

    def validated(self) -> "RecoverySelectorConfig":
        values = {
            name: _finite_nonnegative(getattr(self, name), name)
            for name in self.__dataclass_fields__
        }
        for name in (
            "nominal_risk_lcb_trigger",
            "min_benefit_lcb",
            "max_risk_ucb",
            "max_epistemic_std",
        ):
            if values[name] > 1.0:
                raise ValueError(f"{name} must not exceed one")
        return RecoverySelectorConfig(**values)


def _offsets_manifest(offsets: RecoveryConformalOffsets) -> dict[str, Any]:
    return {
        "nominal_lower": offsets.nominal_lower,
        "risk_upper": offsets.risk_upper.tolist(),
        "benefit_lower": offsets.benefit_lower.tolist(),
        "calibration_report_sha256": offsets.calibration_report_sha256,
    }


def _config_manifest(config: RecoverySelectorConfig) -> dict[str, Any]:
    return {
        name: getattr(config, name)
        for name in config.__dataclass_fields__
    }


def _validate_frozen_bundle_config(
    config: RecoverySelectorConfig,
) -> RecoverySelectorConfig:
    checked = config.validated()
    for name, allowed in _FROZEN_SELECTOR_GRID.items():
        if getattr(checked, name) not in allowed:
            raise ValueError(
                f"{name} is not one of the preregistered selector-grid values")
    for name, expected in _FROZEN_SELECTOR_SUPPORT.items():
        if getattr(checked, name) != expected:
            raise ValueError(
                f"{name} differs from the preregistered selector support")
    return checked


def _bundle_payload(
    offsets: RecoveryConformalOffsets,
    selector_config: RecoverySelectorConfig,
    *,
    probability_calibration_report_sha256: str,
    uncertainty_calibration_report_sha256: str,
    selector_search_report_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": RECOVERY_SELECTOR_BUNDLE_SCHEMA_VERSION,
        "offsets": _offsets_manifest(offsets),
        "selector_config": _config_manifest(selector_config),
        "calibration_and_search_report_sha256": {
            "probability_calibration": probability_calibration_report_sha256,
            "uncertainty_calibration": uncertainty_calibration_report_sha256,
            "selector_search": selector_search_report_sha256,
        },
        "candidate_choice_semantics": recovery_candidate_choice_semantics(),
        "ensemble_std_ddof": RECOVERY_SELECTOR_ENSEMBLE_STD_DDOF,
    }


@dataclass(frozen=True)
class RecoverySelectorBundle:
    """Canonical immutable selector inputs and their provenance binding."""

    offsets: RecoveryConformalOffsets
    selector_config: RecoverySelectorConfig
    probability_calibration_report_sha256: str
    uncertainty_calibration_report_sha256: str
    selector_search_report_sha256: str
    ensemble_std_ddof: int
    bundle_sha256: str

    @classmethod
    def create(
        cls,
        *,
        offsets: RecoveryConformalOffsets,
        selector_config: RecoverySelectorConfig,
        probability_calibration_report_sha256: str,
        uncertainty_calibration_report_sha256: str,
        selector_search_report_sha256: str,
    ) -> "RecoverySelectorBundle":
        if not isinstance(offsets, RecoveryConformalOffsets):
            raise TypeError("offsets must be RecoveryConformalOffsets")
        if not isinstance(selector_config, RecoverySelectorConfig):
            raise TypeError("selector_config must be RecoverySelectorConfig")
        checked_offsets = offsets.validated()
        checked_config = _validate_frozen_bundle_config(selector_config)
        probability_hash = _checked_sha256(
            probability_calibration_report_sha256,
            "probability_calibration_report_sha256",
        )
        uncertainty_hash = _checked_sha256(
            uncertainty_calibration_report_sha256,
            "uncertainty_calibration_report_sha256",
        )
        selector_hash = _checked_sha256(
            selector_search_report_sha256,
            "selector_search_report_sha256",
        )
        if uncertainty_hash != checked_offsets.calibration_report_sha256:
            raise ValueError(
                "offsets must bind the uncertainty calibration report")
        payload = _bundle_payload(
            checked_offsets,
            checked_config,
            probability_calibration_report_sha256=probability_hash,
            uncertainty_calibration_report_sha256=uncertainty_hash,
            selector_search_report_sha256=selector_hash,
        )
        return cls(
            offsets=checked_offsets,
            selector_config=checked_config,
            probability_calibration_report_sha256=probability_hash,
            uncertainty_calibration_report_sha256=uncertainty_hash,
            selector_search_report_sha256=selector_hash,
            ensemble_std_ddof=RECOVERY_SELECTOR_ENSEMBLE_STD_DDOF,
            bundle_sha256=_canonical_sha256(payload),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecoverySelectorBundle":
        if not isinstance(value, Mapping):
            raise TypeError("selector bundle serialization must be a mapping")
        expected_keys = {
            "schema_version",
            "offsets",
            "selector_config",
            "calibration_and_search_report_sha256",
            "candidate_choice_semantics",
            "ensemble_std_ddof",
            "bundle_sha256",
        }
        if set(value) != expected_keys:
            raise ValueError("selector bundle serialization fields are not exact")
        if value.get("schema_version") != RECOVERY_SELECTOR_BUNDLE_SCHEMA_VERSION:
            raise ValueError("selector bundle schema version has drifted")
        offsets = value.get("offsets")
        config = value.get("selector_config")
        reports = value.get("calibration_and_search_report_sha256")
        if not isinstance(offsets, Mapping) or set(offsets) != {
            "nominal_lower", "risk_upper", "benefit_lower",
            "calibration_report_sha256",
        }:
            raise ValueError("selector bundle offsets fields are not exact")
        if not isinstance(config, Mapping) or set(config) != {
            "nominal_risk_lcb_trigger", "min_benefit_lcb", "max_risk_ucb",
            "max_epistemic_std", "max_action_delta_rms",
            "max_q_target_delta_rms",
        }:
            raise ValueError("selector bundle config fields are not exact")
        if not isinstance(reports, Mapping) or set(reports) != {
            "probability_calibration", "uncertainty_calibration",
            "selector_search",
        }:
            raise ValueError("selector bundle report hashes are not exact")
        if value.get("candidate_choice_semantics") != (
                recovery_candidate_choice_semantics()):
            raise ValueError("selector candidate-choice semantics have drifted")
        ddof = value.get("ensemble_std_ddof")
        if isinstance(ddof, (bool, np.bool_)) or ddof != (
                RECOVERY_SELECTOR_ENSEMBLE_STD_DDOF):
            raise ValueError("selector ensemble standard-deviation ddof must be zero")
        result = cls.create(
            offsets=RecoveryConformalOffsets(
                nominal_lower=offsets.get("nominal_lower"),
                risk_upper=np.asarray(offsets.get("risk_upper"), dtype=np.float64),
                benefit_lower=np.asarray(
                    offsets.get("benefit_lower"), dtype=np.float64),
                calibration_report_sha256=offsets.get(
                    "calibration_report_sha256"),
            ),
            selector_config=RecoverySelectorConfig(**dict(config)),
            probability_calibration_report_sha256=reports.get(
                "probability_calibration"),
            uncertainty_calibration_report_sha256=reports.get(
                "uncertainty_calibration"),
            selector_search_report_sha256=reports.get("selector_search"),
        )
        if result.bundle_sha256 != _checked_sha256(
                value.get("bundle_sha256"), "bundle_sha256") or (
                result.to_dict() != dict(value)):
            raise ValueError("selector bundle canonical serialization/hash mismatch")
        return result

    def validated(self) -> "RecoverySelectorBundle":
        return RecoverySelectorBundle.from_dict(self.to_dict())

    @property
    def candidate_choice_semantics(self) -> dict[str, Any]:
        return recovery_candidate_choice_semantics()

    def to_dict(self) -> dict[str, Any]:
        if not isinstance(self.offsets, RecoveryConformalOffsets):
            raise TypeError("offsets must be RecoveryConformalOffsets")
        if not isinstance(self.selector_config, RecoverySelectorConfig):
            raise TypeError("selector_config must be RecoverySelectorConfig")
        checked_offsets = self.offsets.validated()
        checked_config = _validate_frozen_bundle_config(self.selector_config)
        probability_hash = _checked_sha256(
            self.probability_calibration_report_sha256,
            "probability_calibration_report_sha256",
        )
        uncertainty_hash = _checked_sha256(
            self.uncertainty_calibration_report_sha256,
            "uncertainty_calibration_report_sha256",
        )
        selector_hash = _checked_sha256(
            self.selector_search_report_sha256,
            "selector_search_report_sha256",
        )
        if uncertainty_hash != checked_offsets.calibration_report_sha256:
            raise ValueError(
                "offsets must bind the uncertainty calibration report")
        if isinstance(self.ensemble_std_ddof, (bool, np.bool_)) or (
                self.ensemble_std_ddof
                != RECOVERY_SELECTOR_ENSEMBLE_STD_DDOF):
            raise ValueError("selector ensemble standard-deviation ddof must be zero")
        payload = _bundle_payload(
            checked_offsets,
            checked_config,
            probability_calibration_report_sha256=probability_hash,
            uncertainty_calibration_report_sha256=uncertainty_hash,
            selector_search_report_sha256=selector_hash,
        )
        digest = _checked_sha256(self.bundle_sha256, "bundle_sha256")
        if digest != _canonical_sha256(payload):
            raise ValueError("selector bundle SHA-256 mismatch")
        return payload | {"bundle_sha256": digest}


@dataclass(frozen=True)
class RecoverySelection:
    selected_index: int
    intervened: bool
    reason: str
    nominal_risk_lcb: float
    risk_mean: np.ndarray
    risk_std: np.ndarray
    risk_ucb: np.ndarray
    benefit_mean: np.ndarray
    benefit_lcb: np.ndarray
    action_delta_rms: np.ndarray
    q_target_delta_rms: np.ndarray
    eligible: np.ndarray

    @property
    def requested_action_delta_rms(self) -> np.ndarray:
        return self.action_delta_rms


def _finite_signed(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_nonnegative(value: Any, name: str) -> float:
    result = _finite_signed(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    result.setflags(write=False)
    return result


def _offset_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (RECOVERY_PROGRAM_CANDIDATE_COUNT,) or not np.all(
            np.isfinite(array)):
        raise ValueError(f"{name} must be a finite signed K9 vector")
    return array.copy()


def _action(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (RECOVERY_PROGRAM_CANDIDATE_COUNT, 12) or not np.all(
            np.isfinite(array)):
        raise ValueError(f"{name} must be a finite [9,12] array")
    return array


def _mask(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (RECOVERY_PROGRAM_CANDIDATE_COUNT,) or (
            array.dtype != np.dtype(np.bool_)):
        raise ValueError("candidate_mask must be boolean shape [9]")
    if not np.all(array):
        raise ValueError("candidate_mask must keep every locked K9 program valid")
    return array.astype(bool, copy=True)


def _rms_from_nominal(value: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(value - value[:1]), axis=1))


def select_recovery_program(
    member_risk: np.ndarray,
    *,
    candidate_requested: np.ndarray,
    candidate_executed: np.ndarray,
    candidate_q_target: np.ndarray,
    candidate_mask: np.ndarray,
    offsets: RecoveryConformalOffsets,
    config: RecoverySelectorConfig,
) -> RecoverySelection:
    """Select one K9 program without a task-Q or reward-Q gate.

    The pure function keeps explicit offsets/config inputs so the locked
    selector grid can be evaluated before its search-report hash exists.  The
    deployable inference boundary accepts only a completed selector bundle.
    """
    risk = np.asarray(member_risk, dtype=np.float64)
    if risk.ndim != 2 or risk.shape[1] != RECOVERY_PROGRAM_CANDIDATE_COUNT or (
            risk.shape[0] < 2):
        raise ValueError("member_risk must have shape [M>=2,9]")
    if not np.all(np.isfinite(risk)) or np.any((risk < 0.0) | (risk > 1.0)):
        raise ValueError("member_risk must contain finite probabilities")
    requested = _action(candidate_requested, "candidate_requested")
    # Executed actions remain part of the model descriptor and the runtime
    # proof, but are explicitly not a selector slew gate.
    _action(candidate_executed, "candidate_executed")
    target = _action(candidate_q_target, "candidate_q_target")
    support = _mask(candidate_mask)
    if not isinstance(offsets, RecoveryConformalOffsets):
        raise TypeError("offsets must be RecoveryConformalOffsets")
    if not isinstance(config, RecoverySelectorConfig):
        raise TypeError("config must be RecoverySelectorConfig")
    checked_offsets = offsets.validated()
    checked_config = config.validated()

    risk_mean = risk.mean(axis=0)
    risk_std = risk.std(axis=0, ddof=RECOVERY_SELECTOR_ENSEMBLE_STD_DDOF)
    benefit_mean = (risk[:, :1] - risk).mean(axis=0)
    risk_ucb = np.clip(
        risk_mean + checked_offsets.risk_upper, 0.0, 1.0)
    benefit_lcb = np.clip(
        benefit_mean - checked_offsets.benefit_lower, -1.0, 1.0)
    nominal_lcb = float(np.clip(
        risk_mean[0] - checked_offsets.nominal_lower, 0.0, 1.0))
    requested_delta = _rms_from_nominal(requested)
    target_delta = _rms_from_nominal(target)
    eligible = (
        support
        & (benefit_lcb > checked_config.min_benefit_lcb)
        & (risk_ucb <= checked_config.max_risk_ucb)
        & (risk_std <= checked_config.max_epistemic_std)
        & (requested_delta <= checked_config.max_action_delta_rms)
        & (target_delta <= checked_config.max_q_target_delta_rms)
    )
    eligible[0] = False

    reason = "selected"
    selected = 0
    if nominal_lcb < checked_config.nominal_risk_lcb_trigger:
        reason = "state_below_trigger"
    else:
        indices = np.flatnonzero(eligible)
        if len(indices) == 0:
            reason = "no_eligible"
        else:
            order = np.lexsort((
                indices,
                -benefit_lcb[indices],
                risk_ucb[indices],
            ))
            selected = int(indices[order[0]])
    return RecoverySelection(
        selected_index=selected,
        intervened=selected != 0,
        reason=reason,
        nominal_risk_lcb=nominal_lcb,
        risk_mean=_readonly(risk_mean),
        risk_std=_readonly(risk_std),
        risk_ucb=_readonly(risk_ucb),
        benefit_mean=_readonly(benefit_mean),
        benefit_lcb=_readonly(benefit_lcb),
        action_delta_rms=_readonly(requested_delta),
        q_target_delta_rms=_readonly(target_delta),
        eligible=_readonly(eligible),
    )


__all__ = [
    "RECOVERY_SELECTOR_BUNDLE_SCHEMA_VERSION",
    "RECOVERY_SELECTOR_ENSEMBLE_STD_DDOF",
    "RecoveryConformalOffsets",
    "RecoverySelection",
    "RecoverySelectorBundle",
    "RecoverySelectorConfig",
    "recovery_candidate_choice_semantics",
    "select_recovery_program",
]
