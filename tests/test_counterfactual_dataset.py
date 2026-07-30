import tempfile
import unittest
from pathlib import Path

import numpy as np

from learner.counterfactual_dataset import (
    BranchMeasurement,
    BranchSnapshot,
    evaluate_snapshot_candidates,
    load_counterfactual_artifact,
    make_candidate_actions,
    merge_counterfactual_artifacts,
    save_counterfactual_artifact,
)
from train.config import load_app_config


class FakeBackend:
    def __init__(self):
        self.state = 0.0
        self.restored = []

    def restore_state(self, state):
        self.state = float(state[0])
        self.restored.append(self.state)

    def observation(self, previous_action, previous_executed_action,
                    command_speed):
        return np.asarray([
            self.state, previous_action[0], command_speed], np.float32)

    def step_action(self, action):
        self.state += float(action[0])
        failure = abs(self.state) >= 1.5
        return BranchMeasurement(
            failure=failure,
            near_failure=abs(self.state) >= 1.0,
            base_tilt_rad=abs(self.state),
            base_height_m=0.3 - 0.05 * abs(self.state),
            contact_count=int(abs(self.state) >= 0.5),
            undesired_contact_count=int(failure),
            max_contact_force=10.0 * abs(self.state),
        )


class CounterfactualDatasetTest(unittest.TestCase):
    def test_every_candidate_restores_identical_snapshot(self):
        backend = FakeBackend()
        snapshot = BranchSnapshot(
            simulator_state=np.asarray([0.25]),
            observation=np.asarray([0.25, 0.0, 0.5], np.float32),
            previous_action=np.asarray([0.0], np.float32),
            previous_executed_action=np.asarray([0.0], np.float32),
            command_speed=0.5,
        )
        candidates = [
            ('nominal', np.asarray([0.25], np.float32)),
            ('nominal_delta', np.asarray([1.0], np.float32)),
            ('previous', np.asarray([0.0], np.float32)),
            ('contracted_previous', np.asarray([0.0], np.float32)),
        ]
        records = evaluate_snapshot_candidates(
            backend, snapshot, candidates,
            lambda observation: np.asarray([0.25], np.float32),
            snapshot_index=7, horizons=(2, 4))
        self.assertEqual(backend.restored, [0.25] * len(candidates))
        self.assertEqual(records[0].snapshot_index, 7)
        self.assertFalse(records[0].outcomes[2].failure)
        self.assertTrue(records[1].outcomes[2].failure)
        self.assertEqual(records[1].outcomes[2].time_to_failure, 2)
        self.assertEqual(records[1].nominal_safety_improvement[2], -1.0)

    def test_candidates_and_artifact_round_trip(self):
        rng = np.random.default_rng(3)
        candidates = make_candidate_actions(
            np.zeros(2, np.float32), np.ones(2, np.float32),
            rng=rng, perturbation_count=3)
        self.assertEqual(
            [family for family, _ in candidates],
            ['nominal', 'nominal_delta', 'nominal_delta', 'nominal_delta',
             'previous', 'contracted_previous'])

        backend = FakeBackend()
        snapshot = BranchSnapshot(
            simulator_state=np.asarray([0.0]),
            observation=np.zeros(3, np.float32),
            previous_action=np.zeros(1, np.float32),
            previous_executed_action=np.zeros(1, np.float32),
            command_speed=0.5,
        )
        records = evaluate_snapshot_candidates(
            backend, snapshot,
            [('nominal', np.zeros(1, np.float32))],
            lambda observation: np.zeros(1, np.float32),
            snapshot_index=0, horizons=(2,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'branches.pkl'
            save_counterfactual_artifact(
                path, snapshots=[snapshot], branches=records,
                metadata={'seed': 3})
            payload = load_counterfactual_artifact(path)
        self.assertEqual(payload['metadata']['seed'], 3)
        self.assertEqual(len(payload['snapshots']), 1)
        self.assertEqual(len(payload['branches']), 1)

    def test_merge_remaps_snapshot_and_episode_ids(self):
        backend = FakeBackend()
        artifacts = []
        for speed in (0.30, 0.35):
            snapshot = BranchSnapshot(
                simulator_state=np.asarray([0.0]),
                observation=np.asarray([0.0, 0.0, speed], np.float32),
                previous_action=np.zeros(1, np.float32),
                previous_executed_action=np.zeros(1, np.float32),
                command_speed=speed,
                episode_id=0,
            )
            branches = evaluate_snapshot_candidates(
                backend, snapshot,
                [('nominal', np.zeros(1, np.float32))],
                lambda observation: np.zeros(1, np.float32),
                snapshot_index=0, horizons=(2,))
            artifacts.append({
                'format': 'counterfactual_branch_v1',
                'metadata': {'command_speed': speed},
                'snapshots': [snapshot],
                'branches': branches,
            })
        merged = merge_counterfactual_artifacts(artifacts)
        self.assertEqual(
            [item.snapshot_index for item in merged['branches']], [0, 1])
        self.assertEqual(
            [item.episode_id for item in merged['snapshots']], [0, 1])
        self.assertEqual(
            [item.command_speed for item in merged['snapshots']], [0.30, 0.35])

    def test_mujoco_integration_state_restore_is_deterministic(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest('optional mujoco package is not installed')
        from train.mujoco_branch import MujocoBranchBackend

        root = Path(__file__).resolve().parents[1]
        cfg, _, _ = load_app_config(root / 'config/go2.yaml')
        backend = MujocoBranchBackend(
            root / 'mjcf/robot/go2.xml', cfg,
            policy_frequency=20.0)
        backend.reset_standing(settle_seconds=0.1)
        initial = backend.capture_state()
        action = np.zeros(cfg.num_joints, np.float32)
        backend.step_action(action)
        first = backend.capture_state()
        backend.restore_state(initial)
        backend.step_action(action)
        second = backend.capture_state()
        np.testing.assert_array_equal(first, second)


if __name__ == '__main__':
    unittest.main()
