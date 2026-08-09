from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import unittest

import numpy as np

from safety_data.native import evaluate_same_state_group


@dataclass(frozen=True)
class _Cfg:
    num_joints: int = 2


class _FakeSnapshotEnv:
    """Small deterministic backend that exposes evaluator call ordering."""

    def __init__(self, events: list[tuple] | None = None):
        self.cfg = _Cfg()
        self.events = [] if events is None else events
        self.base_history = np.arange(30, dtype=np.float32).reshape(5, 6)
        self.branch_actions: list[list[np.ndarray]] = []
        self.record_count = 0
        self.history_view_count = 0
        self._branch_step = 0
        self._history = self.base_history.copy()

    def restore(self, snapshot: object) -> None:
        del snapshot
        self._branch_step = 0
        self._history = self.base_history.copy()
        self.branch_actions.append([])
        self.events.append(("restore", len(self.branch_actions) - 1))

    def observation_history(self) -> np.ndarray:
        self.history_view_count += 1
        self.events.append(("history", self._branch_step))
        return self._history.copy()

    def record_observation(self) -> np.ndarray:
        self.record_count += 1
        self._history = self.base_history + np.float32(self._branch_step)
        self.events.append(("record", self._branch_step))
        return self._history.copy()

    def step(self, action: np.ndarray) -> SimpleNamespace:
        value = np.asarray(action, dtype=np.float32).copy()
        self.events.append(("step", self._branch_step, value.copy()))
        self.branch_actions[-1].append(value)
        self._branch_step += 1
        application = SimpleNamespace(
            action_requested=value.copy(),
            action_executed=(0.5 * value).astype(np.float32),
            action_q_target=(2.0 + value).astype(np.float32),
        )
        return SimpleNamespace(
            application=application,
            failure=False,
            tilt_rad=0.1 * self._branch_step,
            height_m=1.0 - 0.1 * self._branch_step,
        )


class _StatefulContinuation:
    def __init__(self, events: list[tuple], initial_state: int = 11):
        self.events = events
        self.initial_state = initial_state
        self.state = initial_state
        self.calls: list[dict[str, object]] = []

    def capture_branch_state(self) -> int:
        return self.state

    def restore_branch_state(self, state: int) -> None:
        self.state = state

    def __call__(
        self,
        observation_history: np.ndarray,
        step: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        before = self.state
        self.state += 1
        draw = int(rng.integers(0, 101))
        value = np.asarray(
            [0.001 * draw + 0.0001 * self.state,
             -0.001 * draw - 0.0001 * self.state],
            dtype=np.float32,
        )
        self.events.append(("continuation", step))
        self.calls.append({
            "step": step,
            "state_before": before,
            "history": observation_history.copy(),
            "action": value.copy(),
        })
        return value


class _StatefulRecoveryProgram:
    def __init__(
        self,
        events: list[tuple],
        candidates: np.ndarray,
        behavior_steps: np.ndarray,
        *,
        initial_state: int = 5,
    ):
        self.events = events
        self.candidates = np.asarray(candidates, dtype=np.float32)
        self.behavior_steps = np.asarray(behavior_steps)
        self.initial_state = initial_state
        self.state = initial_state
        self.calls: list[dict[str, object]] = []

    def capture_branch_state(self) -> int:
        return self.state

    def restore_branch_state(self, state: int) -> None:
        self.state = state

    def __call__(
        self,
        candidate_index: int,
        observation_history: np.ndarray,
        step: int,
        nominal_action: np.ndarray,
    ) -> np.ndarray:
        before = self.state
        self.state += 1
        if step == 0:
            value = self.candidates[candidate_index].copy()
        else:
            value = np.asarray([
                0.10 * candidate_index + 0.01 * step + 0.001 * self.state,
                -0.10 * candidate_index - 0.01 * step - 0.001 * self.state,
            ], dtype=np.float32)
        self.events.append(("recovery", candidate_index, step))
        self.calls.append({
            "candidate": candidate_index,
            "step": step,
            "state_before": before,
            "history": observation_history.copy(),
            "nominal_action": np.asarray(nominal_action).copy(),
            "action": value.copy(),
        })
        return value


class NativeClosedLoopRecoveryTest(unittest.TestCase):
    def test_closed_loop_order_duration_crn_state_and_outputs(self):
        events: list[tuple] = []
        env = _FakeSnapshotEnv(events)
        candidates = np.asarray([
            [0.10, -0.10],
            [0.40, 0.30],
            [-0.40, -0.30],
        ], dtype=np.float32)
        continuation = _StatefulContinuation(events)
        recovery = _StatefulRecoveryProgram(
            events, candidates, np.asarray([0, 2, 3], dtype=np.int64))

        result = evaluate_same_state_group(
            env,  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            candidates,
            np.asarray([101, 102], dtype=np.int64),
            horizon_steps=4,
            continuation_policy=continuation,
            recovery_program=recovery,
        )

        # Candidate zero is never sent to the recovery program.  Step zero for
        # every branch also consumes neither a continuation call nor a rollout
        # RNG: only H-1 continuation calls exist per K/R branch.
        self.assertEqual(len(continuation.calls), 3 * 2 * 3)
        self.assertTrue(all(call["step"] > 0 for call in continuation.calls))
        self.assertTrue(all(
            call["candidate"] > 0 for call in recovery.calls))
        self.assertEqual(
            [(call["candidate"], call["step"]) for call in recovery.calls],
            [(1, 0), (1, 1), (1, 0), (1, 1),
             (2, 0), (2, 1), (2, 2),
             (2, 0), (2, 1), (2, 2)],
        )

        # Step zero reads the existing snapshot history without appending it.
        # Subsequent H-1 steps append exactly once, regardless of overrides.
        self.assertEqual(env.history_view_count, 2 * 2)
        self.assertEqual(env.record_count, 3 * 2 * 3)
        for call in recovery.calls:
            if call["step"] == 0:
                np.testing.assert_array_equal(call["history"], env.base_history)

        # Continuation state and RNG are reset per branch, hence every
        # candidate sees the exact same paired nominal action stream.
        continuation_actions = np.asarray([
            call["action"] for call in continuation.calls
        ]).reshape(3, 2, 3, 2)
        np.testing.assert_array_equal(
            continuation_actions,
            np.broadcast_to(continuation_actions[0:1], (3, 2, 3, 2)),
        )
        self.assertTrue(all(
            continuation.calls[offset]["state_before"]
            == continuation.initial_state
            for offset in range(0, len(continuation.calls), 3)))

        # The recovery state is likewise reset for every replica branch.
        recovery_state_sequences = (
            [call["state_before"] for call in recovery.calls[0:2]],
            [call["state_before"] for call in recovery.calls[2:4]],
            [call["state_before"] for call in recovery.calls[4:7]],
            [call["state_before"] for call in recovery.calls[7:10]],
        )
        self.assertEqual(recovery_state_sequences, (
            [5, 6], [5, 6], [5, 6, 7], [5, 6, 7]))
        self.assertEqual(continuation.state, continuation.initial_state)
        self.assertEqual(recovery.state, recovery.initial_state)

        # Every active post-zero recovery call occurs after continuation and
        # immediately before the corresponding environment step.
        for index, event in enumerate(events):
            if event[0] == "recovery" and event[2] > 0:
                self.assertEqual(events[index - 1][0], "continuation")
                self.assertEqual(events[index + 1][0], "step")
        for index, event in enumerate(events):
            if event[0] == "recovery" and event[2] == 0:
                self.assertEqual(events[index - 1][0], "history")
                self.assertEqual(events[index + 1][0], "step")

        # The final restore adds one empty branch after the six evaluated
        # branches.  Recovery overrides only while step < behavior_steps[k].
        self.assertEqual(len(env.branch_actions), 3 * 2 + 1)
        self.assertEqual(env.branch_actions[-1], [])
        actions = np.asarray(env.branch_actions[:-1]).reshape(3, 2, 4, 2)
        np.testing.assert_array_equal(
            actions[:, :, 0],
            np.broadcast_to(candidates[:, None, :], (3, 2, 2)),
        )
        np.testing.assert_array_equal(actions[0, :, 1:], continuation_actions[0])
        np.testing.assert_array_equal(actions[1, :, 2:], continuation_actions[1, :, 1:])
        np.testing.assert_array_equal(actions[2, :, 3:], continuation_actions[2, :, 2:])
        self.assertFalse(np.array_equal(actions[1, :, 1], continuation_actions[1, :, 0]))
        self.assertFalse(np.array_equal(actions[2, :, 1:3], continuation_actions[2, :, 0:2]))

        np.testing.assert_array_equal(result.candidate_requested, candidates)
        np.testing.assert_array_equal(
            result.candidate_executed, (0.5 * candidates).astype(np.float32))
        np.testing.assert_array_equal(
            result.candidate_q_target, (2.0 + candidates).astype(np.float32))
        self.assertFalse(np.any(result.fall))
        np.testing.assert_array_equal(result.first_failure_step, 5)
        np.testing.assert_allclose(result.max_tilt_rad, 0.4, rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(result.min_height_m, 0.6, rtol=0.0, atol=1e-7)

    def test_step_zero_preview_mismatch_fails_before_branch_step(self):
        events: list[tuple] = []
        env = _FakeSnapshotEnv(events)
        candidates = np.asarray([[0.0, 0.0], [0.4, -0.4]], dtype=np.float32)
        continuation = _StatefulContinuation(events)

        class MismatchProgram(_StatefulRecoveryProgram):
            def __call__(self, candidate_index, history, step, nominal_action):
                value = super().__call__(
                    candidate_index, history, step, nominal_action)
                if step == 0:
                    value = value.copy()
                    value[0] += np.float32(0.01)
                return value

        recovery = MismatchProgram(
            events, candidates, np.asarray([0, 1], dtype=np.int64))

        with self.assertRaisesRegex(RuntimeError, "step-zero action differs"):
            evaluate_same_state_group(
                env,  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                candidates,
                np.asarray([55], dtype=np.int64),
                horizon_steps=2,
                continuation_policy=continuation,
                recovery_program=recovery,
            )

        # Branch 0 completed nominally.  Branch 1 was restored and inspected,
        # but mismatch detection happened before its first env.step.  The last
        # empty branch is the unconditional final snapshot restore.
        self.assertEqual(len(env.branch_actions), 3)
        self.assertEqual(len(env.branch_actions[0]), 2)
        self.assertEqual(env.branch_actions[1], [])
        self.assertEqual(env.branch_actions[2], [])
        self.assertEqual(continuation.state, continuation.initial_state)
        self.assertEqual(recovery.state, recovery.initial_state)

    def test_recovery_program_validates_durations_and_exclusivity(self):
        candidates = np.asarray([[0.0, 0.0], [0.2, -0.2]], dtype=np.float32)

        cases = (
            (np.asarray([0.0, 1.0]), "one-dimensional integer"),
            (np.asarray([[0, 1]], dtype=np.int64), "one-dimensional integer"),
            (np.asarray([0], dtype=np.int64), "one-dimensional integer"),
            (np.asarray([1, 1], dtype=np.int64), "nominal candidate"),
            (np.asarray([0, 0], dtype=np.int64), r"\[1, H\]"),
            (np.asarray([0, 3], dtype=np.int64), r"\[1, H\]"),
        )
        for behavior_steps, expected in cases:
            with self.subTest(behavior_steps=behavior_steps), self.assertRaisesRegex(
                    ValueError, expected):
                events: list[tuple] = []
                evaluate_same_state_group(
                    _FakeSnapshotEnv(events),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    candidates,
                    np.asarray([1], dtype=np.int64),
                    horizon_steps=2,
                    continuation_policy=_StatefulContinuation(events),
                    recovery_program=_StatefulRecoveryProgram(
                        events, candidates, behavior_steps),
                )

        events = []
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            evaluate_same_state_group(
                _FakeSnapshotEnv(events),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                candidates,
                np.asarray([1], dtype=np.int64),
                horizon_steps=2,
                continuation_policy=_StatefulContinuation(events),
                option_steps=np.asarray([1, 1], dtype=np.int64),
                recovery_program=_StatefulRecoveryProgram(
                    events, candidates, np.asarray([0, 2], dtype=np.int64)),
            )

    def test_recovery_program_rejects_malformed_actions_before_step(self):
        candidates = np.asarray([[0.0, 0.0], [0.2, -0.2]], dtype=np.float32)
        cases = (
            (np.asarray([[0.2, -0.2]], dtype=np.float32), "shape"),
            (np.asarray([np.nan, -0.2], dtype=np.float32), "finite"),
            (np.asarray([1.1, -0.2], dtype=np.float32), "normalized"),
            (np.asarray(["bad", "action"]), "numeric"),
        )

        for malformed, expected in cases:
            with self.subTest(expected=expected):
                events: list[tuple] = []
                env = _FakeSnapshotEnv(events)

                class MalformedProgram(_StatefulRecoveryProgram):
                    def __call__(self, candidate_index, history, step, nominal_action):
                        del candidate_index, history, step, nominal_action
                        self.state += 1
                        return malformed

                recovery = MalformedProgram(
                    events, candidates, np.asarray([0, 1], dtype=np.int64))
                with self.assertRaisesRegex(ValueError, expected):
                    evaluate_same_state_group(
                        env,  # type: ignore[arg-type]
                        object(),  # type: ignore[arg-type]
                        candidates,
                        np.asarray([3], dtype=np.int64),
                        horizon_steps=1,
                        continuation_policy=_StatefulContinuation(events),
                        recovery_program=recovery,
                    )
                self.assertEqual(env.branch_actions[1], [])
                self.assertEqual(recovery.state, recovery.initial_state)


if __name__ == "__main__":
    unittest.main()
