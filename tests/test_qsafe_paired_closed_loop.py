from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from safety_data.paired_closed_loop import (
    ClosedLoopRollout,
    PairedClosedLoopOutcome,
    ShieldStepDecision,
    evaluate_paired_snapshot,
    summarize_paired_closed_loop,
)
from train.mujoco_snapshot_env import ApplicationState, BranchSnapshot


def _snapshot() -> BranchSnapshot:
    return BranchSnapshot(
        integration_state=np.asarray([0.0, 1.0], dtype=np.float64),
        application_state=ApplicationState(
            previous_action_requested=np.zeros(12, dtype=np.float32),
            previous_action_executed=np.zeros(12, dtype=np.float32),
            previous_action_q_target=np.zeros(12, dtype=np.float32),
            observation_history=np.zeros((5, 46), dtype=np.float32),
            action_filter_state=None,
        ),
    )


class _FakeEnv:
    def __init__(self):
        self.restore_count = 0
        self.step_count = 0

    def restore(self, snapshot):
        self.restore_count += 1
        self.step_count = 0

    def measurement(self):
        return SimpleNamespace(
            failure=False, tilt_rad=0.05, height_m=0.4)

    def observation_history(self):
        return np.zeros((5, 46), dtype=np.float32)

    def record_observation(self):
        history = np.zeros((5, 46), dtype=np.float32)
        history[-1, 0] = self.step_count
        return history

    def step(self, action):
        self.step_count += 1
        failure = bool(float(np.mean(action)) > 0.2)
        return SimpleNamespace(
            failure=failure,
            tilt_rad=1.1 if failure else 0.1,
            height_m=0.15 if failure else 0.38,
        )


class _FakeActor:
    training_step = 500_000

    def __init__(self):
        self.draws = []

    def manifest(self):
        return {}

    def sample_action(self, observation, rng):
        self.draws.append(float(rng.normal()))
        return np.full(12, 0.8, dtype=np.float32)

    def deterministic_action(self, observation):
        return np.full(12, 0.8, dtype=np.float32)


class _FakeShield:
    def __init__(self):
        self.nominal_seen = []

    def decide(
        self, env, observation_history, nominal_action, *, pair_seed, step,
    ):
        self.nominal_seen.append(np.asarray(nominal_action).copy())
        return ShieldStepDecision(
            selected_action=np.zeros(12, dtype=np.float32),
            selected_index=1,
            intervened=True,
            reason="selected",
            requested_delta_rms=0.8,
            q_target_delta_rms=0.2,
            nominal_risk_lcb=0.8,
            selected_risk_ucb=0.1,
            selected_benefit_lcb=0.5,
        )


class _RecordingDisturbance:
    policy_steps = (8, 16)

    def __init__(self):
        self.draws = []

    def __call__(self, env, step, rng):
        self.draws.append((step, float(rng.normal())))


def _rollout(fall: bool, *, interventions: int = 0) -> ClosedLoopRollout:
    return ClosedLoopRollout(
        fall=fall,
        first_failure_step=1 if fall else 33,
        steps_executed=1 if fall else 32,
        max_tilt_rad=1.1 if fall else 0.1,
        min_height_m=0.15 if fall else 0.38,
        interventions=interventions,
        no_eligible_steps=0,
        requested_delta_rms_sum=float(interventions) * 0.2,
        selection_reasons={"selected": interventions} if interventions else {},
    )


class PairedClosedLoopTest(unittest.TestCase):
    def test_exact_snapshot_arms_share_rng_and_are_order_invariant(self):
        snapshot = _snapshot()
        first_env = _FakeEnv()
        first_actor = _FakeActor()
        first_shield = _FakeShield()
        first_disturbance = _RecordingDisturbance()
        first = evaluate_paired_snapshot(
            first_env,
            snapshot,
            actor=first_actor,
            shield=first_shield,
            pair_id="pair-1",
            trajectory_id="trajectory-1",
            source_seed=101,
            pair_seed=991,
            horizon_steps=32,
            disturbance_program=first_disturbance,
        )

        self.assertTrue(first.nominal.fall)
        self.assertFalse(first.shield.fall)
        self.assertEqual(first.fall_reduction, 1)
        self.assertEqual(first.shield.interventions, 32)
        # Baseline step zero and shield step zero receive identical actor noise.
        self.assertEqual(first_actor.draws[0], first_actor.draws[1])
        # Likewise for the disturbance stream at the common step zero call.
        self.assertEqual(first_disturbance.draws[0], first_disturbance.draws[1])
        self.assertEqual(first_env.restore_count, 3)

        second = evaluate_paired_snapshot(
            _FakeEnv(),
            snapshot,
            actor=_FakeActor(),
            shield=_FakeShield(),
            pair_id="pair-1",
            trajectory_id="trajectory-1",
            source_seed=101,
            pair_seed=991,
            horizon_steps=32,
            disturbance_program=_RecordingDisturbance(),
            arm_order=("shield", "nominal"),
        )
        self.assertEqual(first.nominal.to_dict(), second.nominal.to_dict())
        self.assertEqual(first.shield.to_dict(), second.shield.to_dict())

    def test_gate_uses_unique_pair_trajectory_and_snapshot_units(self):
        outcomes = []
        for index in range(1000):
            nominal_fall = index < 400
            shield_fall = index < 200
            outcomes.append(PairedClosedLoopOutcome(
                pair_id=f"pair-{index}",
                state_hash=f"state-{index}",
                trajectory_id=f"trajectory-{index}",
                source_seed=7301 + index % 3,
                pair_seed=900_000 + index,
                horizon_steps=32,
                nominal=_rollout(nominal_fall),
                shield=_rollout(shield_fall, interventions=1),
            ))
        summary = summarize_paired_closed_loop(
            outcomes, bootstrap_replicates=2000, bootstrap_seed=41)

        self.assertTrue(summary.paired_closed_loop_gate)
        self.assertEqual(summary.nominal_falls, 400)
        self.assertEqual(summary.shield_falls, 200)
        self.assertAlmostEqual(summary.absolute_fall_reduction, 0.20)
        self.assertGreater(summary.absolute_fall_reduction_ci95[0], 0.0)
        self.assertEqual(summary.improved_pairs, 200)
        self.assertEqual(summary.worsened_pairs, 0)
        self.assertEqual(summary.source_seeds, (7301, 7302, 7303))

        duplicate = outcomes.copy()
        duplicate[-1] = replace(
            duplicate[-1], trajectory_id=duplicate[0].trajectory_id)
        with self.assertRaisesRegex(ValueError, "duplicate trajectory_id"):
            summarize_paired_closed_loop(
                duplicate, bootstrap_replicates=10)

    def test_gate_cannot_pass_with_small_or_wrong_horizon_table(self):
        records = [PairedClosedLoopOutcome(
            pair_id=f"pair-{index}",
            state_hash=f"state-{index}",
            trajectory_id=f"trajectory-{index}",
            source_seed=3,
            pair_seed=100 + index,
            horizon_steps=32,
            nominal=_rollout(True),
            shield=_rollout(False, interventions=1),
        ) for index in range(10)]
        summary = summarize_paired_closed_loop(
            records, bootstrap_replicates=100, bootstrap_seed=9)
        self.assertFalse(summary.paired_closed_loop_gate)
        self.assertFalse(summary.gate_checks["independent_pairs"])

        wrong_horizon = [PairedClosedLoopOutcome(
            pair_id=item.pair_id,
            state_hash=item.state_hash,
            trajectory_id=item.trajectory_id,
            source_seed=item.source_seed,
            pair_seed=item.pair_seed,
            horizon_steps=16,
            nominal=item.nominal,
            shield=replace(
                item.shield,
                first_failure_step=17,
                steps_executed=16,
            ),
        ) for item in records]
        with self.assertRaisesRegex(ValueError, "H32"):
            summarize_paired_closed_loop(
                wrong_horizon, bootstrap_replicates=10)

    def test_step_zero_disturbance_is_rejected_before_rollout(self):
        disturbance = _RecordingDisturbance()
        disturbance.policy_steps = (0,)
        with self.assertRaisesRegex(ValueError, "step-zero"):
            evaluate_paired_snapshot(
                _FakeEnv(),
                _snapshot(),
                actor=_FakeActor(),
                shield=_FakeShield(),
                pair_id="pair",
                trajectory_id="trajectory",
                source_seed=1,
                pair_seed=2,
                disturbance_program=disturbance,
            )


if __name__ == "__main__":
    unittest.main()
