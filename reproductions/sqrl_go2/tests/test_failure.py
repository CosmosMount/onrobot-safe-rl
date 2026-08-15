import pytest

from reproductions.sqrl_go2.env.failure import failure_cost


def test_inversion_debounce_is_not_a_failure_until_canonical_termination():
    assert failure_cost(
        {"fallen": True, "inverted": True},
        terminated=False, truncated=False) == 0.0
    assert failure_cost(
        {"fallen": True, "inverted": True},
        terminated=True, truncated=False) == 1.0


def test_termination_without_fall_predicate_fails_closed():
    with pytest.raises(ValueError, match="lacks"):
        failure_cost(
            {"fallen": False, "inverted": False},
            terminated=True, truncated=False)
