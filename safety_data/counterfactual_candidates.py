"""Outcome-blind stochastic candidate diversification for counterfactual Q_safe."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTION_SCALE = np.asarray([0.2, 0.4, 0.4] * 4, dtype=np.float32)
DUPLICATE_DISTANCE = 0.025
PROPOSALS = 64
SELECTIONS_PER_BIN = 5
PERCENTILES = (0.10, 0.30, 0.50, 0.70, 0.90)
BIN_NAMES = ("near", "medium", "far")


class InsufficientCandidateDiversity(RuntimeError):
    """The complete state group must be discarded without changing K."""


def normalized_physical_distance(
    first: np.ndarray, second: np.ndarray,
) -> np.ndarray:
    first = np.asarray(first, np.float32)
    second = np.asarray(second, np.float32)
    if first.shape[-1] != 12 or second.shape[-1] != 12:
        raise ValueError("physical critic actions must end in 12 dimensions")
    return np.sqrt(np.mean(np.square((first - second) / ACTION_SCALE), axis=-1))


@dataclass(frozen=True)
class CandidateSelection:
    proposal_indices: np.ndarray
    critic_actions: np.ndarray
    distance: np.ndarray
    distance_bin: np.ndarray


def _percentile_indices(size: int) -> list[int]:
    if size < SELECTIONS_PER_BIN:
        raise InsufficientCandidateDiversity("candidate distance bin has fewer than 5 actions")
    # Nearest-rank positions; deterministic conflict resolution walks toward
    # larger physical distance as preregistered.
    used: set[int] = set()
    result = []
    for fraction in PERCENTILES:
        desired = int(np.floor(fraction * (size - 1) + 0.5))
        chosen = desired
        while chosen in used and chosen + 1 < size:
            chosen += 1
        if chosen in used:
            chosen = desired - 1
            while chosen in used and chosen >= 0:
                chosen -= 1
        if chosen < 0 or chosen in used:
            raise InsufficientCandidateDiversity("percentile selection has no unused action")
        used.add(chosen)
        result.append(chosen)
    return result


def select_diverse_candidates(
    nominal_critic_action: np.ndarray,
    proposal_critic_actions: np.ndarray,
) -> CandidateSelection:
    """Select 5 near, 5 medium and 5 far actions without outcome access."""
    nominal = np.asarray(nominal_critic_action, np.float32)
    proposals = np.asarray(proposal_critic_actions, np.float32)
    if nominal.shape != (12,) or proposals.shape != (PROPOSALS, 12):
        raise ValueError("candidate selection requires nominal [12] and proposals [64,12]")
    if not np.all(np.isfinite(nominal)) or not np.all(np.isfinite(proposals)):
        raise ValueError("candidate actions must be finite")

    retained: list[int] = []
    for index, proposal in enumerate(proposals):
        references = np.concatenate((nominal[None], proposals[retained]), axis=0)
        if np.all(normalized_physical_distance(references, proposal) >= DUPLICATE_DISTANCE):
            retained.append(index)
    if len(retained) < 15:
        raise InsufficientCandidateDiversity("fewer than 15 unique stochastic proposals")

    retained_indices = np.asarray(retained, np.int16)
    distances = normalized_physical_distance(proposals[retained_indices], nominal)
    # Stable sort makes the proposal index the deterministic distance tie-break.
    order = np.argsort(distances, kind="stable")
    ranked = retained_indices[order]
    bins = np.array_split(ranked, 3)
    selected_indices: list[int] = []
    selected_bins: list[str] = []
    for name, values in zip(BIN_NAMES, bins, strict=True):
        positions = _percentile_indices(len(values))
        selected_indices.extend(int(values[position]) for position in positions)
        selected_bins.extend([name] * SELECTIONS_PER_BIN)
    chosen = np.asarray(selected_indices, np.int16)
    return CandidateSelection(
        proposal_indices=chosen,
        critic_actions=proposals[chosen].copy(),
        distance=normalized_physical_distance(proposals[chosen], nominal).astype(np.float32),
        distance_bin=np.asarray(selected_bins, "U6"),
    )

