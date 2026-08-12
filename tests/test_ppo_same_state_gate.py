import numpy as np
import pytest

from safety_data.ppo_same_state_gate import (
    independent_oracle, selector_outcome, stable_state_indices,
    state_bootstrap_lcb, summarize_selector,
)


def test_oracle_discovery_is_independent_of_evaluation_replicas():
    fall = np.ones((2, 3, 8), dtype=bool)
    fall[0, 1, :4] = False
    fall[0, 2, 4:] = False
    choice, outcome = independent_oracle(fall)
    assert choice.tolist() == [1, 0]
    assert outcome.tolist() == [1.0, 1.0]


def test_selector_and_state_group_bootstrap():
    fall = np.ones((20, 2, 8), dtype=bool)
    fall[:, 1, 4:] = False
    choice = np.ones(20, dtype=np.int16)
    assert np.all(selector_outcome(fall, choice) == 0)
    summary = summarize_selector(fall, choice, bootstrap_seed=7)
    assert summary["fall_reduction"] == 1
    assert summary["fall_reduction_lcb95"] == 1


def test_identity_selection_is_outcome_blind_and_validated():
    ids = np.asarray([b"a", b"b", b"c"], dtype="S1")
    assert np.array_equal(stable_state_indices(ids, 2), stable_state_indices(ids, 2))
    with pytest.raises(ValueError):
        stable_state_indices(ids, 4)
    with pytest.raises(ValueError):
        state_bootstrap_lcb(np.ones(2), np.ones(3))
