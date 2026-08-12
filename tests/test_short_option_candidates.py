import numpy as np
import pytest

from safety_data.short_option_candidates import (
    BETA, apply_closed_loop_residual, option_candidate_layout,
    select_farthest_residuals,
)


def test_farthest_selection_is_deterministic_and_outcome_blind() -> None:
    nominal = np.zeros(12, np.float32)
    proposals = np.zeros((64, 12), np.float32)
    for index in range(64):
        proposals[index, index % 12] = 0.01 * (index + 1)
    first = select_farthest_residuals(nominal, proposals)
    second = select_farthest_residuals(nominal, proposals)
    np.testing.assert_array_equal(first.proposal_indices, second.proposal_indices)
    assert len(set(first.proposal_indices.tolist())) == 5
    np.testing.assert_allclose(first.selected_targets - nominal, first.residuals)


def test_fixed_layout_and_beta_schedules() -> None:
    duration, direction = option_candidate_layout(np.ones((5, 12), np.float32))
    np.testing.assert_array_equal(duration, [0] + [1] * 5 + [4] * 5 + [8] * 5)
    np.testing.assert_array_equal(direction, [-1] + list(range(5)) * 3)
    np.testing.assert_allclose(BETA[4], [1, .75, .5, .25])
    np.testing.assert_allclose(BETA[8], [1, .875, .75, .625, .5, .375, .25, .125])


def test_closed_loop_projection_reports_joint_saturation() -> None:
    target, saturated = apply_closed_loop_residual(
        np.zeros(12), np.ones(12), duration=4, option_step=1,
        joint_lower=np.full(12, -0.5), joint_upper=np.full(12, 0.5))
    np.testing.assert_allclose(target, .5)
    assert saturated.all()
    with pytest.raises(ValueError):
        apply_closed_loop_residual(
            np.zeros(12), np.ones(12), duration=4, option_step=4,
            joint_lower=np.full(12, -1), joint_upper=np.full(12, 1))
