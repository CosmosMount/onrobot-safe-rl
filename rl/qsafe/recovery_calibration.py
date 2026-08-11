"""Outcome-file-agnostic Stage-B calibration and selector statistics.

This module deliberately accepts in-memory sufficient statistics only.  It has
no path handling and no dataset loading surface, which keeps model-test opening
and consumption in the dedicated Stage-B workflow.  The functions here encode
the frozen K9 signed-conformal, selector-grid, and hierarchical-bootstrap
semantics used by the state-dependent recovery protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import re
from typing import Any, Literal, Mapping

import numpy as np
import torch

from rl.qsafe.network import QSafeEnsemble
from rl.qsafe.recovery_program import (
    RECOVERY_PROGRAM_CANDIDATE_COUNT,
    RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM,
    RECOVERY_PROGRAM_NOMINAL_INDEX,
    RECOVERY_PROGRAM_VIEW,
)
from rl.qsafe.recovery_selector import (
    RecoveryConformalOffsets,
    RecoverySelectorConfig,
    select_recovery_program,
)


SIGNED_CONFORMAL_SCHEMA_VERSION = "qsafe.recovery_signed_conformal.v1"
SELECTOR_SEARCH_SCHEMA_VERSION = "qsafe.recovery_selector_search.v1"
STAGE_B_ENSEMBLE_MEMBERS = 5
STAGE_B_SELECTOR_BOOTSTRAP_REPLICATES = 50_000
STAGE_B_SELECTOR_BOOTSTRAP_SEED = 20_260_811
STAGE_B_RISK_FAMILYWISE_ALPHA = 0.05
STAGE_B_BENEFIT_FAMILYWISE_ALPHA = 0.05
STAGE_B_NONNOMINAL_OPTIONS = 8
STAGE_B_PER_OPTION_ALPHA = 0.00625
STAGE_B_NOMINAL_ALPHA = 0.05

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SELECTOR_TRIGGER_GRID = (0.10, 0.20, 0.30, 0.40, 0.50)
_SELECTOR_BENEFIT_GRID = (0.00, 0.02, 0.05, 0.08, 0.12)
_SELECTOR_RISK_GRID = (0.25, 0.40, 0.55, 0.70)
_SELECTOR_MAX_STD = 0.20
_SELECTOR_MAX_REQUESTED_RMS = 0.50
_SELECTOR_MAX_QTARGET_RMS = 0.25
_BOOTSTRAP_CHUNK_SIZE = 128


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be canonical finite JSON") from exc


def canonical_sha256(value: Mapping[str, Any]) -> str:
    """Return the SHA-256 of the repository's canonical JSON encoding."""
    if not isinstance(value, Mapping):
        raise TypeError("canonical hash input must be a mapping")
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def execution_lock_sha256(value: str | Mapping[str, Any]) -> str:
    """Resolve either a precomputed digest or an in-memory execution lock."""
    if isinstance(value, str):
        if _HEX64.fullmatch(value) is None:
            raise ValueError("execution lock must be a lowercase SHA-256")
        return value
    if not isinstance(value, Mapping):
        raise TypeError("execution lock must be a SHA-256 or mapping")
    return canonical_sha256(value)


def _readonly(value: Any, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _member_risk(value: Any) -> np.ndarray:
    risk = np.asarray(value, dtype=np.float64)
    if risk.ndim != 3 or risk.shape[1:] == (0, 0) or risk.shape[1] != (
            STAGE_B_ENSEMBLE_MEMBERS) or risk.shape[2] != (
                RECOVERY_PROGRAM_CANDIDATE_COUNT):
        raise ValueError("member_risk must have shape [G,5,9]")
    if not np.all(np.isfinite(risk)) or np.any((risk < 0.0) | (risk > 1.0)):
        raise ValueError("member_risk must contain finite probabilities")
    return risk.copy()


def _empirical_risk(value: Any, groups: int) -> np.ndarray:
    risk = np.asarray(value, dtype=np.float64)
    if risk.shape != (groups, RECOVERY_PROGRAM_CANDIDATE_COUNT):
        raise ValueError("empirical_risk must have shape [G,9]")
    if not np.all(np.isfinite(risk)) or np.any((risk < 0.0) | (risk > 1.0)):
        raise ValueError("empirical_risk must contain finite probabilities")
    return risk.copy()


def _all_k9_mask(value: Any, groups: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (groups, RECOVERY_PROGRAM_CANDIDATE_COUNT) or (
            raw.dtype.kind not in "biu") or not np.all(
                np.isin(raw, (0, 1, False, True))):
        raise ValueError("candidate_mask must be binary shape [G,9]")
    mask = raw.astype(bool)
    if not np.all(mask):
        raise ValueError("Stage-B requires every locked K9 candidate valid")
    return mask.copy()


def _action_array(value: Any, groups: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (groups, RECOVERY_PROGRAM_CANDIDATE_COUNT, 12) or (
            not np.all(np.isfinite(array))):
        raise ValueError(f"{name} must be finite shape [G,9,12]")
    return array.copy()


@torch.no_grad()
def predict_recovery_member_risk(
    trained: Any,
    view: Any,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Return calibrated member probabilities in canonical ``[G,5,9]`` order.

    The helper mirrors the provenance checks of ensemble-mean inference but
    retains all five calibrated members for conformal and selector statistics.
    It reads no outcome tensor from ``view``.
    """
    batch_size = _positive_int(batch_size, "batch_size")
    required = (
        "ensemble", "normalization", "command_vx", "privileged_dim",
        "action_view", "action_dim", "recovery_program_binding",
        "recovery_program_feature_manifest",
        "recovery_program_feature_contract_sha256",
        "recovery_library_fingerprint_sha256",
    )
    if any(not hasattr(trained, name) for name in required):
        raise TypeError("trained ensemble is missing recovery provenance")
    if not isinstance(trained.ensemble, QSafeEnsemble) or len(
            trained.ensemble.members) != STAGE_B_ENSEMBLE_MEMBERS:
        raise ValueError("Stage-B requires exactly five Q_safe members")
    if trained.normalization is None or not view.normalization.equivalent_to(
            trained.normalization):
        raise ValueError("prediction view must use train-fitted normalization")
    if trained.action_view != RECOVERY_PROGRAM_VIEW or (
            view.action_view != RECOVERY_PROGRAM_VIEW) or trained.action_dim != (
                RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM) or view.action_dim != (
                    RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM):
        raise ValueError("prediction requires the frozen 82D recovery view")
    if view.privileged_dim != trained.privileged_dim:
        raise ValueError("prediction and trained feature views differ")
    if abs(float(view.command_vx) - float(trained.command_vx)) > 1e-6:
        raise ValueError("prediction and trained command speeds differ")
    for name in (
        "recovery_program_binding",
        "recovery_program_feature_manifest",
        "recovery_program_feature_contract_sha256",
        "recovery_library_fingerprint_sha256",
    ):
        if getattr(view, name) != getattr(trained, name):
            raise ValueError("prediction and training recovery contracts differ")
    groups = _positive_int(view.group_count, "view.group_count")
    _all_k9_mask(view.mask, groups)

    ensemble = trained.ensemble.to(device).eval()
    result = np.full(
        (groups, STAGE_B_ENSEMBLE_MEMBERS,
         RECOVERY_PROGRAM_CANDIDATE_COUNT),
        np.nan,
        dtype=np.float32,
    )
    indices = np.asarray(view.all_indices(), dtype=np.int64)
    if not np.array_equal(indices, np.arange(groups, dtype=np.int64)):
        raise ValueError("prediction view indices must be canonical group order")
    for start in range(0, groups, batch_size):
        selected = indices[start:start + batch_size]
        batch = view.batch(selected, device)
        prediction = ensemble.predict(
            batch.observation_history,
            batch.nominal_action,
            batch.candidate_action,
            batch.privileged_state,
        )
        member = prediction.member_risk.detach().cpu().numpy()
        if member.shape != (
                STAGE_B_ENSEMBLE_MEMBERS, len(selected),
                RECOVERY_PROGRAM_CANDIDATE_COUNT):
            raise ValueError("Q_safe member prediction shape drifted")
        result[selected] = np.transpose(member, (1, 0, 2)).astype(
            np.float32, copy=False)
    if not np.all(np.isfinite(result)) or np.any((result < 0.0) | (result > 1.0)):
        raise ValueError("Q_safe produced invalid member probabilities")
    return _readonly(result)


@dataclass(frozen=True)
class ConformalOrderStatistic:
    value: float
    one_based_rank: int
    sample_count: int
    alpha: float


def finite_sample_upper_order_statistic(
    scores: Any,
    *,
    alpha: float,
) -> ConformalOrderStatistic:
    """Return ``min(n, ceil((n+1)*(1-alpha)))`` using stable ordering."""
    array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("conformal scores must be nonempty and finite")
    if isinstance(alpha, (bool, np.bool_)) or not math.isfinite(float(alpha)) or (
            not 0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must lie strictly between zero and one")
    alpha = float(alpha)
    rank = min(len(array), math.ceil((len(array) + 1) * (1.0 - alpha)))
    # Stable ordering is explicit even though equal scores have equal values;
    # this prevents future payload-bearing tie handling from becoming random.
    order = np.argsort(array, kind="stable")
    return ConformalOrderStatistic(
        value=float(array[order[rank - 1]]),
        one_based_rank=int(rank),
        sample_count=int(len(array)),
        alpha=alpha,
    )


@dataclass(frozen=True)
class SignedConformalCalibration:
    nominal_lower: float
    risk_upper: np.ndarray
    benefit_lower: np.ndarray
    group_count: int
    option_rank: int
    nominal_rank: int
    execution_lock_sha256: str

    def __post_init__(self) -> None:
        group_count = _positive_int(self.group_count, "group_count")
        option_rank = _positive_int(self.option_rank, "option_rank")
        nominal_rank = _positive_int(self.nominal_rank, "nominal_rank")
        if option_rank > group_count or nominal_rank > group_count:
            raise ValueError("conformal ranks must not exceed group_count")
        if not math.isfinite(float(self.nominal_lower)):
            raise ValueError("nominal_lower must be finite and signed")
        risk = np.asarray(self.risk_upper, dtype=np.float64)
        benefit = np.asarray(self.benefit_lower, dtype=np.float64)
        if risk.shape != (RECOVERY_PROGRAM_CANDIDATE_COUNT,) or benefit.shape != (
                RECOVERY_PROGRAM_CANDIDATE_COUNT,) or not np.all(
                    np.isfinite(risk)) or not np.all(np.isfinite(benefit)):
            raise ValueError("conformal option offsets must be finite K9 vectors")
        if risk[RECOVERY_PROGRAM_NOMINAL_INDEX] != 0.0 or benefit[
                RECOVERY_PROGRAM_NOMINAL_INDEX] != 0.0:
            raise ValueError("nominal option offsets must equal zero")
        lock = execution_lock_sha256(self.execution_lock_sha256)
        object.__setattr__(self, "risk_upper", _readonly(risk))
        object.__setattr__(self, "benefit_lower", _readonly(benefit))
        object.__setattr__(self, "execution_lock_sha256", lock)

    def report_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SIGNED_CONFORMAL_SCHEMA_VERSION,
            "execution_lock_sha256": self.execution_lock_sha256,
            "group_count": self.group_count,
            "candidate_order": "canonical_K9_nominal_then_eight_nonnominal",
            "families": {
                "risk_upper": {
                    "familywise_alpha": STAGE_B_RISK_FAMILYWISE_ALPHA,
                    "nonnominal_options": STAGE_B_NONNOMINAL_OPTIONS,
                    "per_option_alpha": STAGE_B_PER_OPTION_ALPHA,
                    "one_based_rank": self.option_rank,
                    "score": "empirical_risk_k_minus_predicted_risk_k",
                },
                "benefit_lower": {
                    "familywise_alpha": STAGE_B_BENEFIT_FAMILYWISE_ALPHA,
                    "nonnominal_options": STAGE_B_NONNOMINAL_OPTIONS,
                    "per_option_alpha": STAGE_B_PER_OPTION_ALPHA,
                    "one_based_rank": self.option_rank,
                    "score": (
                        "predicted_benefit_k_minus_empirical_benefit_k"),
                },
                "nominal_trigger_lower": {
                    "marginal_alpha": STAGE_B_NOMINAL_ALPHA,
                    "one_based_rank": self.nominal_rank,
                    "score": (
                        "predicted_nominal_risk_minus_empirical_nominal_risk"),
                },
            },
            "coverage_claims": {
                "cross_family_joint_correction": "none",
                "joint_sixteen_bound_coverage": "forbidden",
                "selector_intersection_joint_coverage": "forbidden",
            },
            "rank_rule": "one_based_min_n_ceil_n_plus_1_times_1_minus_alpha",
            "tie_order": "stable_input_order_no_randomization",
            "signed_offsets_no_zero_truncation": True,
            "offsets": {
                "nominal_lower": float(self.nominal_lower),
                "risk_upper": self.risk_upper.tolist(),
                "benefit_lower": self.benefit_lower.tolist(),
            },
        }

    @property
    def report_sha256(self) -> str:
        return canonical_sha256(self.report_payload())

    @property
    def offsets(self) -> RecoveryConformalOffsets:
        return RecoveryConformalOffsets(
            nominal_lower=float(self.nominal_lower),
            risk_upper=self.risk_upper,
            benefit_lower=self.benefit_lower,
            calibration_report_sha256=self.report_sha256,
        ).validated()

    def to_report(self) -> dict[str, Any]:
        result = self.report_payload()
        result["report_sha256"] = self.report_sha256
        return result


def fit_signed_recovery_conformal(
    member_risk: Any,
    empirical_risk: Any,
    *,
    candidate_mask: Any,
    execution_lock: str | Mapping[str, Any],
    expected_group_count: int | None = None,
) -> SignedConformalCalibration:
    """Fit the two separate eight-option families and nominal trigger bound."""
    member = _member_risk(member_risk)
    groups = member.shape[0]
    if expected_group_count is not None and groups != _positive_int(
            expected_group_count, "expected_group_count"):
        raise ValueError("uncertainty-calibration group count differs from lock")
    empirical = _empirical_risk(empirical_risk, groups)
    _all_k9_mask(candidate_mask, groups)
    predicted = member.mean(axis=1)
    predicted_benefit = predicted[:, :1] - predicted
    empirical_benefit = empirical[:, :1] - empirical

    risk_upper = np.zeros(RECOVERY_PROGRAM_CANDIDATE_COUNT, dtype=np.float64)
    benefit_lower = np.zeros(
        RECOVERY_PROGRAM_CANDIDATE_COUNT, dtype=np.float64)
    option_rank: int | None = None
    for option in range(1, RECOVERY_PROGRAM_CANDIDATE_COUNT):
        risk_stat = finite_sample_upper_order_statistic(
            empirical[:, option] - predicted[:, option],
            alpha=STAGE_B_PER_OPTION_ALPHA,
        )
        benefit_stat = finite_sample_upper_order_statistic(
            predicted_benefit[:, option] - empirical_benefit[:, option],
            alpha=STAGE_B_PER_OPTION_ALPHA,
        )
        if risk_stat.one_based_rank != benefit_stat.one_based_rank or (
                option_rank is not None and option_rank != risk_stat.one_based_rank):
            raise AssertionError("equal-size conformal option ranks diverged")
        option_rank = risk_stat.one_based_rank
        risk_upper[option] = risk_stat.value
        benefit_lower[option] = benefit_stat.value

    nominal_stat = finite_sample_upper_order_statistic(
        predicted[:, RECOVERY_PROGRAM_NOMINAL_INDEX]
        - empirical[:, RECOVERY_PROGRAM_NOMINAL_INDEX],
        alpha=STAGE_B_NOMINAL_ALPHA,
    )
    assert option_rank is not None
    return SignedConformalCalibration(
        nominal_lower=nominal_stat.value,
        risk_upper=risk_upper,
        benefit_lower=benefit_lower,
        group_count=groups,
        option_rank=option_rank,
        nominal_rank=nominal_stat.one_based_rank,
        execution_lock_sha256=execution_lock_sha256(execution_lock),
    )


def recovery_selector_grid() -> tuple[RecoverySelectorConfig, ...]:
    """Return the exact machine-listed 5 x 5 x 4 Stage-B grid."""
    result = tuple(
        RecoverySelectorConfig(
            nominal_risk_lcb_trigger=trigger,
            min_benefit_lcb=benefit,
            max_risk_ucb=max_risk,
            max_epistemic_std=_SELECTOR_MAX_STD,
            max_action_delta_rms=_SELECTOR_MAX_REQUESTED_RMS,
            max_q_target_delta_rms=_SELECTOR_MAX_QTARGET_RMS,
        )
        for trigger, benefit, max_risk in itertools.product(
            _SELECTOR_TRIGGER_GRID,
            _SELECTOR_BENEFIT_GRID,
            _SELECTOR_RISK_GRID,
        )
    )
    if len(result) != 100:
        raise AssertionError("frozen selector grid is not exactly 100 points")
    return result


def _config_payload(config: RecoverySelectorConfig) -> dict[str, float]:
    checked = config.validated()
    return {
        name: float(getattr(checked, name))
        for name in checked.__dataclass_fields__
    }


def _cluster_labels(value: Any, groups: int, name: str) -> np.ndarray:
    raw = np.asarray(value).reshape(-1)
    if raw.shape != (groups,):
        raise ValueError(f"{name} must have shape [G]")
    labels = raw.astype(str)
    if np.any(labels == ""):
        raise ValueError(f"{name} must contain nonempty labels")
    return labels


def _actor_labels(value: Any, groups: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (groups,) or raw.dtype.kind not in "iu" or np.any(raw < 0):
        raise ValueError("actor_training_seed must be nonnegative integer [G]")
    if np.any(raw.astype(np.uint64) > np.iinfo(np.int64).max):
        raise ValueError("actor_training_seed exceeds int64")
    return raw.astype(np.int64, copy=True)


def _hierarchical_cluster_means(
    values: np.ndarray,
    actor_training_seed: np.ndarray,
    source_seed: np.ndarray | None,
    inner_cluster_id: np.ndarray,
) -> tuple[
    np.ndarray,
    tuple[tuple[np.ndarray, ...], ...],
    np.ndarray,
]:
    actors = np.unique(actor_training_seed)
    per_actor: list[tuple[np.ndarray, ...]] = []
    for actor in actors:
        actor_mask = actor_training_seed == actor
        sources = (
            np.asarray(["__single_source__"])
            if source_seed is None else np.unique(source_seed[actor_mask])
        )
        actor_sources: list[np.ndarray] = []
        for source in sources:
            source_mask = (
                actor_mask if source_seed is None
                else actor_mask & (source_seed == source)
            )
            inner = np.unique(inner_cluster_id[source_mask])
            if len(inner) == 0:
                raise ValueError("every actor/source requires an inner cluster")
            actor_sources.append(np.stack([
                values[source_mask & (inner_cluster_id == cluster)].mean(axis=0)
                for cluster in inner
            ]))
        per_actor.append(tuple(actor_sources))
    point = np.stack([
        np.stack([source.mean(axis=0) for source in actor]).mean(axis=0)
        for actor in per_actor
    ]).mean(axis=0)
    return actors, tuple(per_actor), point


@dataclass(frozen=True)
class HierarchicalBootstrapResult:
    point_estimate: np.ndarray
    replicates: np.ndarray
    actor_count: int
    source_counts: tuple[int, ...]
    inner_cluster_counts: tuple[tuple[int, ...], ...]
    seed: int
    inner_unit: Literal["source", "trajectory"]

    def __post_init__(self) -> None:
        point = np.asarray(self.point_estimate, dtype=np.float64)
        replicates = np.asarray(self.replicates, dtype=np.float64)
        if point.ndim != 1 or replicates.ndim != 2 or replicates.shape[1] != (
                len(point)) or len(replicates) == 0 or not np.all(
                    np.isfinite(point)) or not np.all(np.isfinite(replicates)):
            raise ValueError("bootstrap arrays must be finite [J] and [B,J]")
        actor_count = _positive_int(self.actor_count, "actor_count")
        if len(self.source_counts) != actor_count or len(
                self.inner_cluster_counts) != actor_count:
            raise ValueError("bootstrap source metadata are invalid")
        for actor_index in range(actor_count):
            source_count = _positive_int(
                self.source_counts[actor_index], "source_count")
            if len(self.inner_cluster_counts[actor_index]) != source_count or any(
                    _positive_int(value, "inner_cluster_count") <= 0
                    for value in self.inner_cluster_counts[actor_index]):
                raise ValueError("bootstrap inner-cluster counts are invalid")
        if self.inner_unit not in ("source", "trajectory"):
            raise ValueError("inner_unit must be source or trajectory")
        object.__setattr__(self, "point_estimate", _readonly(point))
        object.__setattr__(self, "replicates", _readonly(replicates))


def hierarchical_bootstrap_means(
    values: Any,
    *,
    actor_training_seed: Any,
    source_seed: Any | None = None,
    inner_cluster_id: Any,
    replicates: int,
    seed: int,
    inner_unit: Literal["source", "trajectory"],
) -> HierarchicalBootstrapResult:
    """Actor-outer, source-or-trajectory-inner cluster bootstrap.

    Every inner cluster is reduced to one macro value before resampling.  Actor
    identities are sampled with replacement.  When ``source_seed`` is given,
    every registered source stratum of a sampled actor is retained and receives
    its own independent within-source trajectory resample; point and bootstrap
    statistics weight actor, then source, then complete group equally.  Without
    ``source_seed``, each actor has one implicit source.  The generator is
    exactly ``numpy.random.PCG64``.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0 or (
            not np.all(np.isfinite(array))):
        raise ValueError("values must be finite shape [G,J]")
    groups = array.shape[0]
    actors = _actor_labels(actor_training_seed, groups)
    source = (
        None if source_seed is None
        else _cluster_labels(source_seed, groups, "source_seed")
    )
    inner = _cluster_labels(inner_cluster_id, groups, "inner_cluster_id")
    if inner_unit not in ("source", "trajectory"):
        raise ValueError("inner_unit must be source or trajectory")
    count = _positive_int(replicates, "replicates")
    seed = _nonnegative_int(seed, "seed")
    if source is not None:
        for label in np.unique(source):
            if len(np.unique(actors[source == label])) != 1:
                raise ValueError("each source_seed must belong to exactly one actor")
    unique_actors, per_actor, point = _hierarchical_cluster_means(
        array, actors, source, inner)
    actor_count = len(unique_actors)
    rng = np.random.Generator(np.random.PCG64(seed))
    bootstrap = np.empty((count, array.shape[1]), dtype=np.float64)

    for start in range(0, count, _BOOTSTRAP_CHUNK_SIZE):
        stop = min(start + _BOOTSTRAP_CHUNK_SIZE, count)
        size = stop - start
        outer = rng.integers(0, actor_count, size=(size, actor_count))
        slot_values = np.empty(
            (size, actor_count, array.shape[1]), dtype=np.float64)
        for slot in range(actor_count):
            for actor_index, cluster_means in enumerate(per_actor):
                rows = np.flatnonzero(outer[:, slot] == actor_index)
                if len(rows) == 0:
                    continue
                source_values = np.empty(
                    (len(rows), len(cluster_means), array.shape[1]),
                    dtype=np.float64,
                )
                for source_index, source_means in enumerate(cluster_means):
                    inner_count = len(source_means)
                    draw = rng.integers(
                        0, inner_count, size=(len(rows), inner_count))
                    source_values[:, source_index] = source_means[draw].mean(
                        axis=1)
                slot_values[rows, slot] = source_values.mean(axis=1)
        bootstrap[start:stop] = slot_values.mean(axis=1)
    return HierarchicalBootstrapResult(
        point_estimate=point,
        replicates=bootstrap,
        actor_count=actor_count,
        source_counts=tuple(len(value) for value in per_actor),
        inner_cluster_counts=tuple(
            tuple(len(source_means) for source_means in actor)
            for actor in per_actor
        ),
        seed=seed,
        inner_unit=inner_unit,
    )


@dataclass(frozen=True)
class SimultaneousLowerBand:
    point_estimate: np.ndarray
    lower_bound: np.ndarray
    common_critical_value: float
    bootstrap_replicates: np.ndarray
    quantile: float
    seed: int
    inner_unit: Literal["source", "trajectory"]

    def __post_init__(self) -> None:
        point = np.asarray(self.point_estimate, dtype=np.float64)
        lower = np.asarray(self.lower_bound, dtype=np.float64)
        bootstrap = np.asarray(self.bootstrap_replicates, dtype=np.float64)
        if point.ndim != 1 or lower.shape != point.shape or bootstrap.ndim != 2 or (
                bootstrap.shape[1] != len(point)) or not np.all(
                    np.isfinite(point)) or not np.all(np.isfinite(lower)) or (
                        not np.all(np.isfinite(bootstrap))):
            raise ValueError("simultaneous-band arrays have invalid shapes")
        if not math.isfinite(float(self.common_critical_value)):
            raise ValueError("simultaneous critical value must be finite")
        object.__setattr__(self, "point_estimate", _readonly(point))
        object.__setattr__(self, "lower_bound", _readonly(lower))
        object.__setattr__(self, "bootstrap_replicates", _readonly(bootstrap))


def simultaneous_one_sided_lower_band(
    values: Any,
    *,
    actor_training_seed: Any,
    source_seed: Any | None = None,
    inner_cluster_id: Any,
    replicates: int = STAGE_B_SELECTOR_BOOTSTRAP_REPLICATES,
    seed: int = STAGE_B_SELECTOR_BOOTSTRAP_SEED,
    inner_unit: Literal["source", "trajectory"] = "trajectory",
    quantile: float = 0.95,
) -> SimultaneousLowerBand:
    """Construct the nonstudentized max-centered-error one-sided band."""
    if isinstance(quantile, (bool, np.bool_)) or not math.isfinite(
            float(quantile)) or not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must lie strictly between zero and one")
    result = hierarchical_bootstrap_means(
        values,
        actor_training_seed=actor_training_seed,
        source_seed=source_seed,
        inner_cluster_id=inner_cluster_id,
        replicates=replicates,
        seed=seed,
        inner_unit=inner_unit,
    )
    max_centered_error = np.max(
        result.point_estimate[None, :] - result.replicates, axis=1)
    critical = float(np.quantile(
        max_centered_error, float(quantile), method="linear"))
    return SimultaneousLowerBand(
        point_estimate=result.point_estimate,
        lower_bound=result.point_estimate - critical,
        common_critical_value=critical,
        bootstrap_replicates=result.replicates,
        quantile=float(quantile),
        seed=result.seed,
        inner_unit=result.inner_unit,
    )


@dataclass(frozen=True)
class RecoverySelectorGridRow:
    grid_index: int
    config: RecoverySelectorConfig
    absolute_fall_reduction: float
    simultaneous_lcb: float
    intervention_rate: float
    feasible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_index": self.grid_index,
            "selector_config": _config_payload(self.config),
            "absolute_fall_reduction": float(self.absolute_fall_reduction),
            "simultaneous_one_sided_95_lcb": float(self.simultaneous_lcb),
            "intervention_rate": float(self.intervention_rate),
            "feasible": bool(self.feasible),
        }


@dataclass(frozen=True)
class RecoverySelectorSearchResult:
    rows: tuple[RecoverySelectorGridRow, ...]
    selected_grid_index: int | None
    common_critical_value: float
    bootstrap_replicates: int
    bootstrap_seed: int
    bootstrap_inner_unit: Literal["source", "trajectory"]
    bootstrap_middle_unit: str
    execution_lock_sha256: str

    def __post_init__(self) -> None:
        if len(self.rows) != 100 or tuple(row.grid_index for row in self.rows) != (
                tuple(range(100))):
            raise ValueError("selector search must contain the exact 100 rows")
        if self.selected_grid_index is not None and (
                self.selected_grid_index < 0 or self.selected_grid_index >= 100 or
                not self.rows[self.selected_grid_index].feasible):
            raise ValueError("selected grid index is not feasible")
        if not math.isfinite(float(self.common_critical_value)):
            raise ValueError("selector critical value must be finite")
        _positive_int(self.bootstrap_replicates, "bootstrap_replicates")
        _nonnegative_int(self.bootstrap_seed, "bootstrap_seed")
        if self.bootstrap_inner_unit not in ("source", "trajectory"):
            raise ValueError("bootstrap_inner_unit must be source or trajectory")
        if self.bootstrap_middle_unit not in (
                "none_single_implicit_source",
                "retain_every_registered_source_stratum",
        ):
            raise ValueError("bootstrap_middle_unit is invalid")
        object.__setattr__(
            self, "execution_lock_sha256",
            execution_lock_sha256(self.execution_lock_sha256))

    @property
    def selected_config(self) -> RecoverySelectorConfig | None:
        if self.selected_grid_index is None:
            return None
        return self.rows[self.selected_grid_index].config

    def report_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SELECTOR_SEARCH_SCHEMA_VERSION,
            "execution_lock_sha256": self.execution_lock_sha256,
            "grid_points_exact": 100,
            "grid_order": (
                "trigger_outer_then_benefit_then_max_risk_machine_order"),
            "candidate_comparators": {
                "nominal_risk_lcb_trigger": "greater_than_or_equal",
                "minimum_benefit_lcb": "strictly_greater_than",
                "maximum_candidate_risk_ucb": "less_than_or_equal",
                "maximum_ensemble_probability_std": "less_than_or_equal",
                "maximum_first_requested_action_rms_delta": (
                    "less_than_or_equal"),
                "maximum_first_qtarget_rms_delta": "less_than_or_equal",
            },
            "candidate_choice_order": [
                "lowest_risk_ucb",
                "largest_benefit_lcb",
                "locked_candidate_index",
            ],
            "bootstrap": {
                "replicates": self.bootstrap_replicates,
                "seed": self.bootstrap_seed,
                "rng_bit_generator": "numpy_PCG64",
                "outer_unit": "actor_training_seed",
                "middle_unit": self.bootstrap_middle_unit,
                "inner_unit": self.bootstrap_inner_unit,
                "quantile_method": "linear",
                "one_sided_quantile": 0.95,
                "simultaneous_band": (
                    "max_centered_error_across_complete_grid_nonstudentized"),
                "common_critical_value": float(self.common_critical_value),
            },
            "feasibility": {
                "minimum_absolute_reduction": 0.03,
                "simultaneous_lcb_strictly_positive": True,
                "maximum_intervention_rate": 0.35,
            },
            "choice_order": [
                "largest_simultaneous_lcb",
                "lower_intervention_rate",
                "machine_grid_order",
            ],
            "selected_grid_index": self.selected_grid_index,
            "rows": [row.to_dict() for row in self.rows],
        }

    @property
    def report_sha256(self) -> str:
        return canonical_sha256(self.report_payload())

    def to_report(self) -> dict[str, Any]:
        result = self.report_payload()
        result["report_sha256"] = self.report_sha256
        return result


def search_recovery_selector_grid(
    member_risk: Any,
    empirical_risk: Any,
    *,
    candidate_requested: Any,
    candidate_executed: Any,
    candidate_q_target: Any,
    candidate_mask: Any,
    offsets: RecoveryConformalOffsets,
    actor_training_seed: Any,
    source_seed: Any,
    inner_cluster_id: Any,
    execution_lock: str | Mapping[str, Any],
    bootstrap_replicates: int = STAGE_B_SELECTOR_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = STAGE_B_SELECTOR_BOOTSTRAP_SEED,
    bootstrap_inner_unit: Literal["source", "trajectory"] = "trajectory",
    expected_group_count: int | None = None,
) -> RecoverySelectorSearchResult:
    """Evaluate all 100 selectors and apply the simultaneous search gate."""
    member = _member_risk(member_risk)
    groups = member.shape[0]
    if expected_group_count is not None and groups != _positive_int(
            expected_group_count, "expected_group_count"):
        raise ValueError("selector-calibration group count differs from lock")
    empirical = _empirical_risk(empirical_risk, groups)
    requested = _action_array(candidate_requested, groups, "candidate_requested")
    executed = _action_array(candidate_executed, groups, "candidate_executed")
    q_target = _action_array(candidate_q_target, groups, "candidate_q_target")
    mask = _all_k9_mask(candidate_mask, groups)
    if not isinstance(offsets, RecoveryConformalOffsets):
        raise TypeError("offsets must be RecoveryConformalOffsets")
    checked_offsets = offsets.validated()
    actors = _actor_labels(actor_training_seed, groups)
    sources = _cluster_labels(source_seed, groups, "source_seed")
    inner = _cluster_labels(inner_cluster_id, groups, "inner_cluster_id")
    grid = recovery_selector_grid()

    selected = np.zeros((groups, len(grid)), dtype=np.int64)
    for group in range(groups):
        for grid_index, config in enumerate(grid):
            decision = select_recovery_program(
                member[group],
                candidate_requested=requested[group],
                candidate_executed=executed[group],
                candidate_q_target=q_target[group],
                candidate_mask=mask[group],
                offsets=checked_offsets,
                config=config,
            )
            selected[group, grid_index] = decision.selected_index
    reduction = (
        empirical[:, :1]
        - np.take_along_axis(empirical, selected, axis=1)
    )
    intervention = (selected != RECOVERY_PROGRAM_NOMINAL_INDEX).astype(
        np.float64)
    band = simultaneous_one_sided_lower_band(
        reduction,
        actor_training_seed=actors,
        source_seed=sources,
        inner_cluster_id=inner,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
        inner_unit=bootstrap_inner_unit,
        quantile=0.95,
    )
    _, _, intervention_point = _hierarchical_cluster_means(
        intervention, actors, sources, inner)

    rows = tuple(
        RecoverySelectorGridRow(
            grid_index=index,
            config=config,
            absolute_fall_reduction=float(band.point_estimate[index]),
            simultaneous_lcb=float(band.lower_bound[index]),
            intervention_rate=float(intervention_point[index]),
            feasible=bool(
                band.point_estimate[index] >= 0.03
                and band.lower_bound[index] > 0.0
                and intervention_point[index] <= 0.35
            ),
        )
        for index, config in enumerate(grid)
    )
    feasible = [row for row in rows if row.feasible]
    chosen = None
    if feasible:
        chosen = min(
            feasible,
            key=lambda row: (
                -row.simultaneous_lcb,
                row.intervention_rate,
                row.grid_index,
            ),
        ).grid_index
    return RecoverySelectorSearchResult(
        rows=rows,
        selected_grid_index=chosen,
        common_critical_value=band.common_critical_value,
        bootstrap_replicates=_positive_int(
            bootstrap_replicates, "bootstrap_replicates"),
        bootstrap_seed=_nonnegative_int(bootstrap_seed, "bootstrap_seed"),
        bootstrap_inner_unit=bootstrap_inner_unit,
        bootstrap_middle_unit="retain_every_registered_source_stratum",
        execution_lock_sha256=execution_lock_sha256(execution_lock),
    )


__all__ = [
    "ConformalOrderStatistic",
    "HierarchicalBootstrapResult",
    "RecoverySelectorGridRow",
    "RecoverySelectorSearchResult",
    "SELECTOR_SEARCH_SCHEMA_VERSION",
    "SIGNED_CONFORMAL_SCHEMA_VERSION",
    "STAGE_B_SELECTOR_BOOTSTRAP_REPLICATES",
    "STAGE_B_SELECTOR_BOOTSTRAP_SEED",
    "SignedConformalCalibration",
    "SimultaneousLowerBand",
    "canonical_sha256",
    "execution_lock_sha256",
    "finite_sample_upper_order_statistic",
    "fit_signed_recovery_conformal",
    "hierarchical_bootstrap_means",
    "predict_recovery_member_risk",
    "recovery_selector_grid",
    "search_recovery_selector_grid",
    "simultaneous_one_sided_lower_band",
]
