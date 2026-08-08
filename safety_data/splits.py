"""Trajectory-atomic nested subsets for Q_safe learning curves."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Integral
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class NestedSubset:
    requested_groups: int
    actual_groups: int
    trajectory_count: int
    indices: np.ndarray

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices, dtype=np.int64).reshape(-1).copy()
        indices.setflags(write=False)
        object.__setattr__(self, "indices", indices)


def nested_trajectory_subsets(
    trajectory_id: np.ndarray,
    requested_groups: Sequence[int],
    *,
    seed: int,
) -> list[NestedSubset]:
    """Build monotone subsets without cutting a source trajectory."""
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    trajectory = np.asarray(trajectory_id).astype(str).reshape(-1)
    if any(isinstance(value, bool) or not isinstance(value, Integral)
           for value in requested_groups):
        raise ValueError("requested group counts must be integers")
    requested = [int(value) for value in requested_groups]
    if not requested or any(value <= 0 for value in requested) or requested != sorted(
            set(requested)):
        raise ValueError("requested group counts must be unique, positive and sorted")
    if len(trajectory) == 0:
        raise ValueError("trajectory_id is empty")
    if np.any(trajectory == ""):
        raise ValueError("trajectory_id contains an empty cluster name")
    if requested[-1] > len(trajectory):
        raise ValueError(
            "requested group count exceeds the available trajectory-atomic data")
    unique = np.unique(trajectory)

    def order_key(name: str) -> str:
        return hashlib.sha256(f"{seed}\0{name}".encode("utf-8")).hexdigest()

    ordered = sorted(unique.tolist(), key=order_key)
    members = {name: np.flatnonzero(trajectory == name) for name in ordered}
    selected: list[np.ndarray] = []
    selected_groups = 0
    next_trajectory = 0
    subsets: list[NestedSubset] = []
    for target in requested:
        previous_trajectory_count = next_trajectory
        while selected_groups < target and next_trajectory < len(ordered):
            indices = members[ordered[next_trajectory]]
            selected.append(indices)
            selected_groups += len(indices)
            next_trajectory += 1
        if next_trajectory == previous_trajectory_count:
            raise ValueError(
                "requested learning-curve point adds no trajectory because the "
                "previous atomic subset already overshot it")
        combined = np.sort(np.concatenate(selected)).astype(np.int64)
        subsets.append(NestedSubset(
            requested_groups=target,
            actual_groups=len(combined),
            trajectory_count=next_trajectory,
            indices=combined,
        ))
    return subsets
