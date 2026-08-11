"""Frozen one-shot Stage-B model-test statistics for persistent recovery.

This module is intentionally path-free.  It accepts already validated in-memory
arrays and immutable selector/placebo bundles; the dedicated evaluator owns the
irreversible evidence-consumption boundary.  Keeping the numerical code pure
also makes it possible to test the exact bootstrap with a small replicate count
without adding a production CLI override.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np

from rl.qsafe.recovery_placebo import (
    MatchedRandomPlaceboBundle,
    select_matched_random_placebo,
)
from rl.qsafe.recovery_program import (
    RECOVERY_PROGRAM_BEHAVIOR_STEPS,
    RECOVERY_PROGRAM_CANDIDATE_COUNT,
    RECOVERY_PROGRAM_NOMINAL_INDEX,
)
from rl.qsafe.recovery_selector import (
    RecoverySelectorBundle,
    select_recovery_program,
)


MODEL_TEST_STATISTICS_SCHEMA_VERSION = (
    "qsafe.state_dependent_recovery_v5.stage_b.model_test_statistics.v1")
STAGE_B_MODEL_TEST_BOOTSTRAP_REPLICATES = 50_000
STAGE_B_MODEL_TEST_BOOTSTRAP_SEED = 20_260_812
STAGE_B_MODEL_TEST_ENSEMBLE_MEMBERS = 5
STAGE_B_MODEL_TEST_ECE_BINS = 10
STAGE_B_STRONG_PAIR_GAP = 0.25

MIN_PAIR_ACCURACY = 0.60
MIN_PAIR_ACCURACY_LCB = 0.55
MIN_STRONG_PAIR_ACCURACY = 0.62
MIN_TOP1_REDUCTION = 0.05
MIN_TOP1_REDUCTION_LCB = 0.03
MIN_SELECTOR_REDUCTION = 0.03
MAX_SELECTOR_INTERVENTION_RATE = 0.35
MIN_ORACLE_GAP_CAPTURE = 0.25
MAX_ECE = 0.08

_BOOTSTRAP_CHUNK_SIZE = 128


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


def _readonly(value: Any, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _integer_vector(value: Any, groups: int, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (groups,) or raw.dtype.kind not in "iu" or np.any(raw < 0):
        raise ValueError(f"{name} must be a nonnegative integer vector [G]")
    if np.any(raw.astype(np.uint64) > np.iinfo(np.int64).max):
        raise ValueError(f"{name} exceeds int64")
    return raw.astype(np.int64, copy=True)


def _text_vector(value: Any, groups: int, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (groups,) or raw.dtype.kind not in "US":
        raise ValueError(f"{name} must be a non-object text vector [G]")
    result = raw.astype(str, copy=True)
    if np.any(result == ""):
        raise ValueError(f"{name} must contain nonempty values")
    return result


def _member_risk(value: Any) -> np.ndarray:
    risk = np.asarray(value, dtype=np.float64)
    if risk.ndim != 3 or risk.shape[1:] != (
            STAGE_B_MODEL_TEST_ENSEMBLE_MEMBERS,
            RECOVERY_PROGRAM_CANDIDATE_COUNT):
        raise ValueError("member_risk must have shape [G,5,9]")
    if len(risk) == 0 or not np.all(np.isfinite(risk)) or np.any(
            (risk < 0.0) | (risk > 1.0)):
        raise ValueError("member_risk must contain finite probabilities")
    return risk.copy()


def _fall_array(value: Any, groups: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 3 or raw.shape[:2] != (
            groups, RECOVERY_PROGRAM_CANDIDATE_COUNT) or raw.shape[2] < 1 or (
                raw.dtype.kind not in "biu") or not np.all(
                    np.isin(raw, (0, 1, False, True))):
        raise ValueError("fall must be binary shape [G,9,R>=1]")
    return raw.astype(bool, copy=True)


def _action(value: Any, groups: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (
            groups, RECOVERY_PROGRAM_CANDIDATE_COUNT, 12) or not np.all(
                np.isfinite(array)):
        raise ValueError(f"{name} must be finite shape [G,9,12]")
    return array.copy()


def _candidate_mask(value: Any, groups: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (groups, RECOVERY_PROGRAM_CANDIDATE_COUNT) or (
            raw.dtype.kind not in "biu") or not np.all(
                np.isin(raw, (0, 1, False, True))):
        raise ValueError("candidate_mask must be binary shape [G,9]")
    mask = raw.astype(bool, copy=True)
    if not np.all(mask):
        raise ValueError("Stage-B Model-Test requires every locked K9 option")
    return mask


def _behavior_steps(value: Any, groups: int) -> np.ndarray:
    raw = np.asarray(value)
    expected = np.asarray(RECOVERY_PROGRAM_BEHAVIOR_STEPS, dtype=np.int64)
    if raw.shape == (RECOVERY_PROGRAM_CANDIDATE_COUNT,):
        raw = np.broadcast_to(raw, (groups, RECOVERY_PROGRAM_CANDIDATE_COUNT))
    if raw.shape != (groups, RECOVERY_PROGRAM_CANDIDATE_COUNT) or (
            raw.dtype.kind not in "iu") or not np.array_equal(
                raw.astype(np.int64), np.broadcast_to(expected, raw.shape)):
        raise ValueError("candidate_behavior_steps differ from frozen K9 order")
    return raw.astype(np.int64, copy=True)


def _pair_scores(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    minimum_gap: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return group-macro accuracy over all empirical non-tied pairs."""
    scores = np.full(len(target), np.nan, dtype=np.float64)
    counts = np.zeros(len(target), dtype=np.int64)
    for group in range(len(target)):
        correct: list[float] = []
        for left in range(RECOVERY_PROGRAM_CANDIDATE_COUNT):
            for right in range(left + 1, RECOVERY_PROGRAM_CANDIDATE_COUNT):
                target_delta = float(target[group, left] - target[group, right])
                if target_delta == 0.0 or abs(target_delta) < minimum_gap:
                    continue
                predicted_delta = float(
                    prediction[group, left] - prediction[group, right])
                correct.append(
                    0.5 if predicted_delta == 0.0 else float(
                        np.sign(target_delta) == np.sign(predicted_delta)))
        if correct:
            scores[group] = float(np.mean(correct))
            counts[group] = len(correct)
    return scores, counts


@dataclass(frozen=True)
class _Hierarchy:
    actors: np.ndarray
    sources_by_actor: tuple[tuple[np.ndarray, ...], ...]
    source_labels_by_actor: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "actors", _readonly(self.actors, np.int64))


def _hierarchy(
    actor_training_seed: np.ndarray,
    source_seed: np.ndarray,
    trajectory_id: np.ndarray,
) -> _Hierarchy:
    actors = np.unique(actor_training_seed)
    sources_by_actor: list[tuple[np.ndarray, ...]] = []
    labels_by_actor: list[tuple[int, ...]] = []
    observed_sources: set[int] = set()
    for actor in actors:
        actor_mask = actor_training_seed == actor
        sources = np.unique(source_seed[actor_mask])
        rows: list[np.ndarray] = []
        labels: list[int] = []
        for source in sources:
            if int(source) in observed_sources:
                raise ValueError("each source_seed must belong to exactly one actor")
            observed_sources.add(int(source))
            selected = np.flatnonzero(actor_mask & (source_seed == source))
            if len(selected) == 0:
                raise AssertionError("registered actor/source stratum is empty")
            if len(np.unique(trajectory_id[selected])) != len(selected):
                raise ValueError(
                    "Stage-B Model-Test requires one state per complete trajectory")
            rows.append(selected)
            labels.append(int(source))
        sources_by_actor.append(tuple(rows))
        labels_by_actor.append(tuple(labels))
    return _Hierarchy(
        actors=actors,
        sources_by_actor=tuple(sources_by_actor),
        source_labels_by_actor=tuple(labels_by_actor),
    )


def _source_then_actor_point(
    values: np.ndarray,
    hierarchy: _Hierarchy,
    *,
    allow_missing: bool = False,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    actor_values: list[np.ndarray] = []
    for sources in hierarchy.sources_by_actor:
        source_values = np.stack([
            (np.nanmean(array[rows], axis=0)
             if allow_missing else np.mean(array[rows], axis=0))
            for rows in sources
        ])
        actor_values.append(
            np.nanmean(source_values, axis=0)
            if allow_missing else np.mean(source_values, axis=0))
    result = np.stack(actor_values)
    return (
        np.nanmean(result, axis=0)
        if allow_missing else np.mean(result, axis=0))


@dataclass(frozen=True)
class ModelTestBootstrap:
    point: np.ndarray
    replicates: np.ndarray
    pair_point: float | None
    pair_replicates: np.ndarray
    seed: int

    def __post_init__(self) -> None:
        point = np.asarray(self.point, dtype=np.float64)
        replicates = np.asarray(self.replicates, dtype=np.float64)
        pair = np.asarray(self.pair_replicates, dtype=np.float64)
        if point.ndim != 1 or replicates.ndim != 2 or replicates.shape[1] != (
                len(point)) or len(replicates) == 0 or pair.shape != (
                    len(replicates),) or not np.all(np.isfinite(point)) or (
                        not np.all(np.isfinite(replicates))):
            raise ValueError("model-test bootstrap arrays are invalid")
        if self.pair_point is not None and not math.isfinite(self.pair_point):
            raise ValueError("pair_point must be finite or None")
        object.__setattr__(self, "point", _readonly(point))
        object.__setattr__(self, "replicates", _readonly(replicates))
        object.__setattr__(self, "pair_replicates", _readonly(pair))


def hierarchical_model_test_bootstrap(
    values: Any,
    pair_score: Any,
    *,
    actor_training_seed: Any,
    source_seed: Any,
    trajectory_id: Any,
    replicates: int,
    seed: int,
) -> ModelTestBootstrap:
    """Actor-outer bootstrap retaining sources and resampling trajectories.

    Actor identities are drawn with replacement.  For every sampled actor slot,
    each registered source stratum is retained, and complete trajectories are
    drawn with replacement within that source.  The point and every replicate
    therefore have equal actor, then equal source, then equal group weight.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0 or not (
            np.all(np.isfinite(array))):
        raise ValueError("values must be finite shape [G,J]")
    groups = len(array)
    pair = np.asarray(pair_score, dtype=np.float64)
    if pair.shape != (groups,) or np.any(np.isinf(pair)):
        raise ValueError("pair_score must have shape [G] with finite/NaN values")
    actors = _integer_vector(actor_training_seed, groups, "actor_training_seed")
    sources = _integer_vector(source_seed, groups, "source_seed")
    trajectory = _text_vector(trajectory_id, groups, "trajectory_id")
    hierarchy = _hierarchy(actors, sources, trajectory)
    count = _positive_int(replicates, "replicates")
    seed = _nonnegative_int(seed, "seed")
    actor_count = len(hierarchy.actors)
    point = _source_then_actor_point(array, hierarchy)
    pair_value = _source_then_actor_point(
        pair, hierarchy, allow_missing=True)[0]
    pair_point = None if not np.isfinite(pair_value) else float(pair_value)

    rng = np.random.Generator(np.random.PCG64(seed))
    bootstrap = np.empty((count, array.shape[1]), dtype=np.float64)
    pair_bootstrap = np.full(count, np.nan, dtype=np.float64)
    for start in range(0, count, _BOOTSTRAP_CHUNK_SIZE):
        stop = min(start + _BOOTSTRAP_CHUNK_SIZE, count)
        size = stop - start
        outer = rng.integers(0, actor_count, size=(size, actor_count))
        slot_values = np.empty(
            (size, actor_count, array.shape[1]), dtype=np.float64)
        slot_pairs = np.full((size, actor_count), np.nan, dtype=np.float64)
        for slot in range(actor_count):
            for actor_index, source_rows in enumerate(
                    hierarchy.sources_by_actor):
                output_rows = np.flatnonzero(outer[:, slot] == actor_index)
                if len(output_rows) == 0:
                    continue
                sampled_sources = np.empty(
                    (len(output_rows), len(source_rows), array.shape[1]),
                    dtype=np.float64,
                )
                sampled_pair_sources = np.full(
                    (len(output_rows), len(source_rows)), np.nan,
                    dtype=np.float64)
                for source_index, rows in enumerate(source_rows):
                    draw = rng.integers(
                        0, len(rows), size=(len(output_rows), len(rows)))
                    sampled = rows[draw]
                    sampled_sources[:, source_index] = array[sampled].mean(axis=1)
                    pair_values = pair[sampled]
                    valid = np.isfinite(pair_values)
                    denominator = valid.sum(axis=1)
                    numerator = np.where(valid, pair_values, 0.0).sum(axis=1)
                    nonempty = denominator > 0
                    sampled_pair_sources[nonempty, source_index] = (
                        numerator[nonempty] / denominator[nonempty])
                slot_values[output_rows, slot] = sampled_sources.mean(axis=1)
                finite_source = np.isfinite(sampled_pair_sources)
                denominator = finite_source.sum(axis=1)
                numerator = np.where(
                    finite_source, sampled_pair_sources, 0.0).sum(axis=1)
                nonempty = denominator > 0
                slot_pairs[output_rows[nonempty], slot] = (
                    numerator[nonempty] / denominator[nonempty])
        bootstrap[start:stop] = slot_values.mean(axis=1)
        finite_slot = np.isfinite(slot_pairs)
        denominator = finite_slot.sum(axis=1)
        numerator = np.where(finite_slot, slot_pairs, 0.0).sum(axis=1)
        nonempty = denominator > 0
        pair_bootstrap[start:stop][nonempty] = (
            numerator[nonempty] / denominator[nonempty])
    return ModelTestBootstrap(
        point=point,
        replicates=bootstrap,
        pair_point=pair_point,
        pair_replicates=pair_bootstrap,
        seed=seed,
    )


def _equal_hierarchy_group_weights(
    groups: int,
    hierarchy: _Hierarchy,
) -> np.ndarray:
    weights = np.zeros(groups, dtype=np.float64)
    actor_mass = 1.0 / len(hierarchy.actors)
    for sources in hierarchy.sources_by_actor:
        source_mass = actor_mass / len(sources)
        for rows in sources:
            weights[rows] = source_mass / len(rows)
    if not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("hierarchical group weights do not sum to one")
    return weights


def stable_equal_mass_ece(
    prediction: Any,
    target: Any,
    sample_weight: Any,
    *,
    bins: int = STAGE_B_MODEL_TEST_ECE_BINS,
) -> dict[str, Any]:
    """Stable tie-invariant equal-weight-mass calibration error."""
    bins = _positive_int(bins, "bins")
    predicted = np.asarray(prediction, dtype=np.float64).reshape(-1)
    observed = np.asarray(target, dtype=np.float64).reshape(-1)
    weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if len(predicted) == 0 or not (
            len(predicted) == len(observed) == len(weight)) or not np.all(
                np.isfinite(predicted)) or not np.all(np.isfinite(observed)) or (
                    not np.all(np.isfinite(weight))) or np.any(weight <= 0.0):
        raise ValueError("ECE inputs must be nonempty, finite, and equally sized")
    if np.any((predicted < 0.0) | (predicted > 1.0)) or np.any(
            (observed < 0.0) | (observed > 1.0)):
        raise ValueError("ECE predictions and targets must lie in [0,1]")
    order = np.argsort(predicted, kind="stable")
    predicted = predicted[order]
    observed = observed[order]
    weight = weight[order]
    unique, inverse = np.unique(predicted, return_inverse=True)
    tied_weight = np.bincount(inverse, weights=weight, minlength=len(unique))
    tied_target = np.bincount(
        inverse, weights=weight * observed, minlength=len(unique)) / tied_weight
    total = float(tied_weight.sum())
    target_mass = total / bins
    bin_weight = np.zeros(bins, dtype=np.float64)
    bin_prediction = np.zeros(bins, dtype=np.float64)
    bin_target = np.zeros(bins, dtype=np.float64)
    sample = 0
    remaining = float(tied_weight[0])
    for bin_index in range(bins):
        desired = total - bin_weight.sum() if bin_index == bins - 1 else target_mass
        while desired > 1e-15 * total and sample < len(unique):
            portion = min(remaining, desired)
            bin_weight[bin_index] += portion
            bin_prediction[bin_index] += portion * unique[sample]
            bin_target[bin_index] += portion * tied_target[sample]
            desired -= portion
            remaining -= portion
            if remaining <= 1e-15 * total:
                sample += 1
                if sample < len(unique):
                    remaining = float(tied_weight[sample])
    confidence = bin_prediction / bin_weight
    frequency = bin_target / bin_weight
    mass = bin_weight / total
    ece = float(np.sum(mass * np.abs(confidence - frequency)))
    return {
        "bins": bins,
        "method": "stable_equal_weight_mass_tie_blocks_fractionally_split",
        "ece": ece,
        "mass_fraction": mass.tolist(),
        "mean_prediction": confidence.tolist(),
        "empirical_frequency": frequency.tolist(),
    }


def _subgroup_effects(
    values: np.ndarray,
    *,
    actor_training_seed: np.ndarray,
    source_seed: np.ndarray,
    checkpoint_step: np.ndarray,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for label, grouping in (
        ("checkpoint_age", checkpoint_step),
        ("actor_training_seed", actor_training_seed),
        ("source_seed", source_seed),
    ):
        entries: dict[str, float] = {}
        for value in np.unique(grouping):
            selected = grouping == value
            local_actors = actor_training_seed[selected]
            local_sources = source_seed[selected]
            # Trajectories are unique at this boundary, so stable local row IDs
            # are sufficient for the same actor->source->group point estimator.
            local_trajectory = np.asarray([
                f"subgroup-{index}" for index in np.flatnonzero(selected)])
            hierarchy = _hierarchy(
                local_actors, local_sources, local_trajectory)
            entries[str(int(value))] = float(
                _source_then_actor_point(values[selected], hierarchy)[0])
        result[label] = entries
    return result


def _strictly_positive_subgroups(
    top1: Mapping[str, Mapping[str, float]],
    selector: Mapping[str, Mapping[str, float]],
) -> bool:
    return all(
        value > 0.0
        for collection in (top1, selector)
        for entries in collection.values()
        for value in entries.values()
    )


def _quantile_or_none(
    value: np.ndarray,
    quantile: float,
) -> float | None:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        return None
    return float(np.quantile(array, quantile, method="linear"))


def evaluate_stage_b_model_test(
    *,
    member_risk: Any,
    fall: Any,
    candidate_requested: Any,
    candidate_executed: Any,
    candidate_q_target: Any,
    candidate_mask: Any,
    candidate_behavior_steps: Any,
    actor_training_seed: Any,
    source_seed: Any,
    checkpoint_step: Any,
    trajectory_fingerprint_sha256: Any,
    group_id: Any,
    group_fingerprint_sha256: Any,
    selector_bundle: RecoverySelectorBundle,
    placebo_bundle: MatchedRandomPlaceboBundle,
    bootstrap_replicates: int = STAGE_B_MODEL_TEST_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = STAGE_B_MODEL_TEST_BOOTSTRAP_SEED,
    production_contract: bool = True,
) -> dict[str, Any]:
    """Evaluate the immutable Model-Test without fitting or selecting anything."""
    member = _member_risk(member_risk)
    groups = len(member)
    outcomes = _fall_array(fall, groups)
    requested = _action(candidate_requested, groups, "candidate_requested")
    executed = _action(candidate_executed, groups, "candidate_executed")
    q_target = _action(candidate_q_target, groups, "candidate_q_target")
    mask = _candidate_mask(candidate_mask, groups)
    behavior_steps = _behavior_steps(candidate_behavior_steps, groups)
    actors = _integer_vector(
        actor_training_seed, groups, "actor_training_seed")
    sources = _integer_vector(source_seed, groups, "source_seed")
    checkpoints = _integer_vector(checkpoint_step, groups, "checkpoint_step")
    trajectories = _text_vector(
        trajectory_fingerprint_sha256,
        groups,
        "trajectory_fingerprint_sha256",
    )
    group_ids = _text_vector(group_id, groups, "group_id")
    fingerprints = _text_vector(
        group_fingerprint_sha256, groups, "group_fingerprint_sha256")
    if any(len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value)
           for value in trajectories):
        raise ValueError(
            "trajectory fingerprints must be lowercase SHA-256")
    if any(len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value)
           for value in fingerprints):
        raise ValueError("group fingerprints must be lowercase SHA-256")
    if any(len(np.unique(value)) != groups for value in (
            trajectories, group_ids, fingerprints)):
        raise ValueError(
            "one state per complete trajectory and unique group/state identities "
            "are required")
    if not isinstance(selector_bundle, RecoverySelectorBundle):
        raise TypeError("selector_bundle must be RecoverySelectorBundle")
    selector = selector_bundle.validated()
    if not isinstance(placebo_bundle, MatchedRandomPlaceboBundle):
        raise TypeError("placebo_bundle must be MatchedRandomPlaceboBundle")
    placebo = MatchedRandomPlaceboBundle.from_dict(placebo_bundle.to_dict())
    if placebo.selector_bundle_sha256 != selector.bundle_sha256 or (
            placebo.selector_config != selector.selector_config):
        raise ValueError("placebo bundle is not bound to the frozen selector")
    if type(production_contract) is not bool:
        raise TypeError("production_contract must be boolean")
    if production_contract:
        if bootstrap_replicates != STAGE_B_MODEL_TEST_BOOTSTRAP_REPLICATES or (
                bootstrap_seed != STAGE_B_MODEL_TEST_BOOTSTRAP_SEED):
            raise ValueError(
                "frozen production Model-Test bootstrap contract drifted")
        expected_source_assignment = {
            **{8700 + actor - 52: (actor, 25_000)
               for actor in range(53, 57)},
            **{8710 + actor - 52: (actor, 50_000)
               for actor in range(53, 57)},
            **{8720 + actor - 52: (actor, 100_000)
               for actor in range(53, 57)},
        }
        if groups != 768 or outcomes.shape[2] != 64 or set(
                map(int, np.unique(actors))) != set(range(53, 57)) or set(
                    map(int, np.unique(sources))) != set(
                        expected_source_assignment) or set(
                            map(int, np.unique(checkpoints))) != {
                                25_000, 50_000, 100_000}:
            raise ValueError("frozen production Model-Test cohort identity drifted")
        for source, (actor, checkpoint) in expected_source_assignment.items():
            selected = sources == source
            if int(np.count_nonzero(selected)) != 64 or not np.all(
                    actors[selected] == actor) or not np.all(
                        checkpoints[selected] == checkpoint):
                raise ValueError(
                    "frozen actor/source/checkpoint assignment drifted")

    # Canonicalize every downstream tie and PCG64 draw to the frozen identity
    # order.  No outcome participates in this ordering.
    order = np.lexsort((group_ids, trajectories, sources, actors))
    member = member[order]
    outcomes = outcomes[order]
    requested = requested[order]
    executed = executed[order]
    q_target = q_target[order]
    mask = mask[order]
    behavior_steps = behavior_steps[order]
    actors = actors[order]
    sources = sources[order]
    checkpoints = checkpoints[order]
    trajectories = trajectories[order]
    group_ids = group_ids[order]
    fingerprints = fingerprints[order]
    hierarchy = _hierarchy(actors, sources, trajectories)
    for actor_index, actor in enumerate(hierarchy.actors):
        source_rows = hierarchy.sources_by_actor[actor_index]
        observed_ages = {
            int(checkpoints[rows][0]) for rows in source_rows
        }
        if any(len(np.unique(checkpoints[rows])) != 1 for rows in source_rows):
            raise ValueError("each source stratum must have exactly one checkpoint age")
        if observed_ages != set(np.unique(checkpoints)):
            raise ValueError("every actor must retain every registered age stratum")

    empirical = outcomes.mean(axis=2, dtype=np.float64)
    prediction = member.mean(axis=1)
    top1_index = np.argmin(prediction, axis=1)
    selector_index = np.zeros(groups, dtype=np.int64)
    selector_nominal_lcb = np.zeros(groups, dtype=np.float64)
    placebo_index = np.zeros(groups, dtype=np.int64)
    for group in range(groups):
        decision = select_recovery_program(
            member[group],
            candidate_requested=requested[group],
            candidate_executed=executed[group],
            candidate_q_target=q_target[group],
            candidate_mask=mask[group],
            offsets=selector.offsets,
            config=selector.selector_config,
        )
        selector_index[group] = decision.selected_index
        selector_nominal_lcb[group] = decision.nominal_risk_lcb
        requested_distance = np.sqrt(np.mean(np.square(
            requested[group] - requested[group, :1]), axis=1))
        qtarget_distance = np.sqrt(np.mean(np.square(
            q_target[group] - q_target[group, :1]), axis=1))
        placebo_support = (
            mask[group]
            & (requested_distance <= selector.selector_config.max_action_delta_rms)
            & (qtarget_distance <= selector.selector_config.max_q_target_delta_rms)
        )
        placebo_support[RECOVERY_PROGRAM_NOMINAL_INDEX] = True
        placebo_decision = select_matched_random_placebo(
            placebo,
            nominal_risk_lcb=decision.nominal_risk_lcb,
            candidate_support_mask=placebo_support,
            candidate_duration_steps=behavior_steps[group],
            first_action_distance=requested_distance,
            source_seed=int(sources[group]),
            group_fingerprint_sha256=str(fingerprints[group]),
            draw_index=0,
        )
        placebo_index[group] = placebo_decision.selected_index

    rows = np.arange(groups)
    nominal = empirical[:, RECOVERY_PROGRAM_NOMINAL_INDEX]
    oracle_index = np.argmin(empirical, axis=1)
    top1_reduction = nominal - empirical[rows, top1_index]
    selector_reduction = nominal - empirical[rows, selector_index]
    selector_intervention = (
        selector_index != RECOVERY_PROGRAM_NOMINAL_INDEX).astype(np.float64)
    placebo_reduction = nominal - empirical[rows, placebo_index]
    placebo_intervention = (
        placebo_index != RECOVERY_PROGRAM_NOMINAL_INDEX).astype(np.float64)
    oracle_reduction = nominal - empirical[rows, oracle_index]
    pair_score, pair_count = _pair_scores(
        empirical, prediction, minimum_gap=np.nextafter(0.0, 1.0))
    strong_score, strong_count = _pair_scores(
        empirical, prediction, minimum_gap=STAGE_B_STRONG_PAIR_GAP)

    bootstrap = hierarchical_model_test_bootstrap(
        np.column_stack((
            top1_reduction,
            selector_reduction,
            selector_intervention,
            placebo_reduction,
            placebo_intervention,
            oracle_reduction,
        )),
        pair_score,
        actor_training_seed=actors,
        source_seed=sources,
        trajectory_id=trajectories,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    top1_lcb = _quantile_or_none(bootstrap.replicates[:, 0], 0.05)
    selector_lcb = _quantile_or_none(bootstrap.replicates[:, 1], 0.05)
    pair_lcb = _quantile_or_none(bootstrap.pair_replicates, 0.025)
    strong_point_raw = _source_then_actor_point(
        strong_score, hierarchy, allow_missing=True)[0]
    strong_point = (
        None if not np.isfinite(strong_point_raw) else float(strong_point_raw))
    top1_point = float(bootstrap.point[0])
    selector_point = float(bootstrap.point[1])
    selector_intervention_point = float(bootstrap.point[2])
    oracle_point = float(bootstrap.point[5])
    oracle_capture = (
        None if oracle_point <= 0.0 else float(top1_point / oracle_point))

    group_weight = _equal_hierarchy_group_weights(groups, hierarchy)
    candidate_weight = np.repeat(
        group_weight[:, None] / RECOVERY_PROGRAM_CANDIDATE_COUNT,
        RECOVERY_PROGRAM_CANDIDATE_COUNT,
        axis=1,
    )
    calibration = stable_equal_mass_ece(
        prediction,
        empirical,
        candidate_weight,
        bins=STAGE_B_MODEL_TEST_ECE_BINS,
    )
    top1_subgroups = _subgroup_effects(
        top1_reduction,
        actor_training_seed=actors,
        source_seed=sources,
        checkpoint_step=checkpoints,
    )
    selector_subgroups = _subgroup_effects(
        selector_reduction,
        actor_training_seed=actors,
        source_seed=sources,
        checkpoint_step=checkpoints,
    )

    gates = {
        "pair_accuracy_point_gte_0_60": (
            bootstrap.pair_point is not None
            and bootstrap.pair_point >= MIN_PAIR_ACCURACY),
        "pair_accuracy_q025_lcb_gte_0_55": (
            pair_lcb is not None and pair_lcb >= MIN_PAIR_ACCURACY_LCB),
        "strong_pair_accuracy_gap_gte_0_25_gte_0_62": (
            strong_point is not None
            and int(strong_count.sum()) > 0
            and strong_point >= MIN_STRONG_PAIR_ACCURACY),
        "top1_reduction_point_gte_0_05": top1_point >= MIN_TOP1_REDUCTION,
        "top1_reduction_q05_lcb_gte_0_03": (
            top1_lcb is not None and top1_lcb >= MIN_TOP1_REDUCTION_LCB),
        "selector_reduction_point_gte_0_03": (
            selector_point >= MIN_SELECTOR_REDUCTION),
        "selector_reduction_q05_lcb_strictly_positive": (
            selector_lcb is not None and selector_lcb > 0.0),
        "selector_intervention_rate_lte_0_35": (
            selector_intervention_point <= MAX_SELECTOR_INTERVENTION_RATE),
        "top1_and_selector_all_directional_subgroups_strictly_positive": (
            _strictly_positive_subgroups(
                top1_subgroups, selector_subgroups)),
        "oracle_opportunity_strictly_positive": oracle_point > 0.0,
        "oracle_gap_capture_gte_0_25": (
            oracle_capture is not None
            and oracle_capture >= MIN_ORACLE_GAP_CAPTURE),
        "stable_equal_mass_ece_lte_0_08": calibration["ece"] <= MAX_ECE,
        "placebo_fit_frozen_and_eligible": bool(placebo.fit_metrics.eligible),
    }
    statistic_pass = all(gates.values())
    order_payload = {
        "actor_training_seed": actors.tolist(),
        "source_seed": sources.tolist(),
        "trajectory_fingerprint_sha256": trajectories.tolist(),
        "group_id": group_ids.tolist(),
        "group_fingerprint_sha256": fingerprints.tolist(),
    }
    report: dict[str, Any] = {
        "schema_version": MODEL_TEST_STATISTICS_SCHEMA_VERSION,
        "cohort": {
            "groups": groups,
            "candidates": RECOVERY_PROGRAM_CANDIDATE_COUNT,
            "replicas": int(outcomes.shape[2]),
            "actor_training_seeds": [
                int(value) for value in np.unique(actors)],
            "source_seeds": [int(value) for value in np.unique(sources)],
            "checkpoint_steps": [
                int(value) for value in np.unique(checkpoints)],
            "unique_trajectory_groups": int(len(np.unique(trajectories))),
            "stable_identity_order_sha256": _canonical_sha256(order_payload),
            "stable_identity_order": (
                "actor_training_seed_source_seed_trajectory_fingerprint_"
                "group_id_candidate_index"),
        },
        "bootstrap": {
            "replicates": _positive_int(
                bootstrap_replicates, "bootstrap_replicates"),
            "seed": _nonnegative_int(bootstrap_seed, "bootstrap_seed"),
            "rng_bit_generator": "numpy_PCG64",
            "outer_unit": "actor_training_seed_with_replacement",
            "middle_unit": "retain_all_registered_source_strata",
            "inner_unit": (
                "complete_trajectory_groups_within_source_with_replacement"),
            "weighting": (
                "equal_actor_then_equal_source_then_equal_complete_group"),
            "quantile_method": "linear",
        },
        "metrics": {
            "pair_accuracy_group_macro": bootstrap.pair_point,
            "pair_accuracy_q025_lcb": pair_lcb,
            "pair_groups": int(np.isfinite(pair_score).sum()),
            "pair_comparisons": int(pair_count.sum()),
            "strong_pair_gap_inclusive": STAGE_B_STRONG_PAIR_GAP,
            "strong_pair_accuracy_group_macro": strong_point,
            "strong_pair_groups": int(np.isfinite(strong_score).sum()),
            "strong_pair_comparisons": int(strong_count.sum()),
            "top1_absolute_fall_reduction": top1_point,
            "top1_reduction_q05_lcb": top1_lcb,
            "frozen_selector_absolute_fall_reduction": selector_point,
            "frozen_selector_reduction_q05_lcb": selector_lcb,
            "frozen_selector_intervention_rate": selector_intervention_point,
            "oracle_absolute_fall_reduction": oracle_point,
            "oracle_gap_capture": oracle_capture,
            "placebo_absolute_fall_reduction_diagnostic": float(
                bootstrap.point[3]),
            "placebo_intervention_rate_diagnostic": float(bootstrap.point[4]),
            "ece": calibration,
        },
        "directional_subgroups": {
            "top1_absolute_fall_reduction": top1_subgroups,
            "frozen_selector_absolute_fall_reduction": selector_subgroups,
        },
        "gates": gates,
        "pass": statistic_pass,
        "model_or_threshold_updates_from_model_test": False,
    }
    report["statistics_sha256"] = _canonical_sha256(report)
    return report


__all__ = [
    "MAX_ECE",
    "MAX_SELECTOR_INTERVENTION_RATE",
    "MIN_ORACLE_GAP_CAPTURE",
    "MIN_PAIR_ACCURACY",
    "MIN_PAIR_ACCURACY_LCB",
    "MIN_SELECTOR_REDUCTION",
    "MIN_STRONG_PAIR_ACCURACY",
    "MIN_TOP1_REDUCTION",
    "MIN_TOP1_REDUCTION_LCB",
    "MODEL_TEST_STATISTICS_SCHEMA_VERSION",
    "ModelTestBootstrap",
    "STAGE_B_MODEL_TEST_BOOTSTRAP_REPLICATES",
    "STAGE_B_MODEL_TEST_BOOTSTRAP_SEED",
    "evaluate_stage_b_model_test",
    "hierarchical_model_test_bootstrap",
    "stable_equal_mass_ece",
]
