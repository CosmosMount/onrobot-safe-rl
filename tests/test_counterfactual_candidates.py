from __future__ import annotations

import numpy as np
import pytest

from safety_data.counterfactual_candidates import (
    ACTION_SCALE,
    InsufficientCandidateDiversity,
    normalized_physical_distance,
    select_diverse_candidates,
)


def _proposals() -> np.ndarray:
    # Strictly increasing normalized distance, with enough values in every bin.
    distance = np.linspace(0.03, 1.92, 64, dtype=np.float32)
    return distance[:, None] * ACTION_SCALE[None]


def test_normalized_physical_rms_distance() -> None:
    assert normalized_physical_distance(np.zeros(12), ACTION_SCALE) == pytest.approx(1.0)


def test_selection_is_deterministic_and_has_five_per_rank_third() -> None:
    first = select_diverse_candidates(np.zeros(12), _proposals())
    second = select_diverse_candidates(np.zeros(12), _proposals())
    np.testing.assert_array_equal(first.proposal_indices, second.proposal_indices)
    assert list(first.distance_bin).count("near") == 5
    assert list(first.distance_bin).count("medium") == 5
    assert list(first.distance_bin).count("far") == 5
    assert np.all(np.diff(first.distance[:5]) > 0)
    assert first.distance[:5].max() < first.distance[5:10].min()
    assert first.distance[5:10].max() < first.distance[10:].min()


def test_near_duplicates_are_removed_and_short_bin_fails_closed() -> None:
    proposals = np.zeros((64, 12), np.float32)
    with pytest.raises(InsufficientCandidateDiversity):
        select_diverse_candidates(np.zeros(12), proposals)


def test_selection_api_has_no_outcome_argument() -> None:
    import inspect
    assert tuple(inspect.signature(select_diverse_candidates).parameters) == (
        "nominal_critic_action", "proposal_critic_actions")
