from __future__ import annotations

import unittest

import numpy as np

from runtime.inference.actions import ActionApplier, ActionFilterButter


def _applier(*, filtered: bool = True) -> ActionApplier:
    init_qpos = np.asarray(
        [0.0, 0.8, -1.5] * 4,
        dtype=np.float32,
    )
    return ActionApplier(
        init_qpos=init_qpos,
        action_offset=np.asarray([0.4, 0.5, 0.6] * 4, dtype=np.float32),
        joint_min=np.asarray([-0.8, -0.2, -2.7] * 4, dtype=np.float32),
        joint_max=np.asarray([0.8, 1.8, -0.4] * 4, dtype=np.float32),
        max_joint_delta=np.asarray([0.08, 0.10, 0.12] * 4, dtype=np.float32),
        action_filter=(
            ActionFilterButter(12, sampling_rate=50.0, highcut=4.0, order=2)
            if filtered
            else None
        ),
    )


def _assert_filter_state_equal(
    testcase: unittest.TestCase,
    left,
    right,
) -> None:
    testcase.assertIsNot(left, right)
    np.testing.assert_array_equal(left.x_history, right.x_history)
    np.testing.assert_array_equal(left.y_history, right.y_history)


class ActionPreviewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.applier = _applier()
        self.current_qpos = self.applier.init_qpos + np.asarray(
            [0.02, -0.03, 0.01] * 4,
            dtype=np.float32,
        )
        self.applier.reset_filter()
        # Make the baseline non-trivial so a missing restore is observable.
        self.applier.project(
            np.asarray([0.15, -0.10, 0.05] * 4, dtype=np.float32),
            self.current_qpos,
        )

    def test_preview_is_side_effect_free_and_matches_independent_projection(self):
        actions = np.asarray([
            [0.00, 0.00, 0.00] * 4,
            [0.20, -0.15, 0.10] * 4,
            [-0.25, 0.20, -0.05] * 4,
        ], dtype=np.float32)
        action_filter = self.applier.action_filter
        assert action_filter is not None
        baseline = action_filter.capture_state()

        previews = self.applier.preview_many(actions, self.current_qpos)

        _assert_filter_state_equal(
            self, baseline, action_filter.capture_state())
        self.assertEqual(len(previews), len(actions))
        for index, action in enumerate(actions):
            action_filter.restore_state(baseline)
            expected = self.applier.project(action, self.current_qpos)
            np.testing.assert_array_equal(
                previews[index].action_requested, expected.action_requested)
            np.testing.assert_array_equal(
                previews[index].action_executed, expected.action_executed)
            np.testing.assert_array_equal(
                previews[index].action_q_target, expected.action_q_target)
        action_filter.restore_state(baseline)

    def test_candidate_order_does_not_change_candidate_projection(self):
        actions = np.asarray([
            [0.05, 0.10, -0.15] * 4,
            [-0.30, 0.25, 0.20] * 4,
            [0.40, -0.35, 0.30] * 4,
        ], dtype=np.float32)
        forward = self.applier.preview_many(actions, self.current_qpos)
        reverse = self.applier.preview_many(actions[::-1], self.current_qpos)
        for forward_projection, reverse_projection in zip(
            forward, reversed(reverse), strict=True,
        ):
            np.testing.assert_array_equal(
                forward_projection.action_executed,
                reverse_projection.action_executed,
            )
            np.testing.assert_array_equal(
                forward_projection.action_q_target,
                reverse_projection.action_q_target,
            )

    def test_filter_state_is_restored_when_a_later_candidate_raises(self):
        actions = np.asarray([
            [0.10, 0.00, -0.10] * 4,
            [np.nan, 0.00, 0.00] * 4,
        ], dtype=np.float32)
        action_filter = self.applier.action_filter
        assert action_filter is not None
        baseline = action_filter.capture_state()

        with self.assertRaisesRegex(ValueError, "finite"):
            self.applier.preview_many(actions, self.current_qpos)

        _assert_filter_state_equal(
            self, baseline, action_filter.capture_state())

    def test_unfiltered_preview_has_same_projection_contract(self):
        applier = _applier(filtered=False)
        actions = np.asarray([
            np.zeros(12, dtype=np.float32),
            np.full(12, 0.2, dtype=np.float32),
        ])
        previews = applier.preview_many(actions, self.current_qpos)
        direct = tuple(
            applier.project(action, self.current_qpos) for action in actions)
        for preview, expected in zip(previews, direct, strict=True):
            np.testing.assert_array_equal(
                preview.action_q_target, expected.action_q_target)

    def test_preview_requires_an_explicit_candidate_axis(self):
        with self.assertRaisesRegex(ValueError, r"\[candidates"):
            self.applier.preview_many(
                np.zeros(12, dtype=np.float32), self.current_qpos)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            self.applier.preview_many(
                np.empty((0, 12), dtype=np.float32), self.current_qpos)


if __name__ == "__main__":
    unittest.main()
