from __future__ import annotations

import unittest

import numpy as np

from jaxrl.agents.safety_critic import SafetyCritic
from learner.branch_supervision import (BranchSupervisionDataset,
                                        conformal_upper_offset,
                                        mine_selected_false_safe,
                                        split_branch_episodes,
                                        split_branch_episodes_three_way,
                                        split_branch_snapshots)
from learner.counterfactual_dataset import CandidateBranch, HorizonOutcome


def _branch(snapshot, index, *, failure, time_to_failure=-1):
    outcome = HorizonOutcome(
        horizon=8, failure=failure, near_failure=failure,
        time_to_failure=time_to_failure,
        max_tilt_rad=0.5 if failure else 0.1,
        min_base_height_m=0.1 if failure else 0.3,
        max_contact_count=4, max_undesired_contact_count=int(failure),
        max_contact_force=10.0 if failure else 1.0)
    action = np.asarray([0.8 if failure else -0.8], np.float32)
    return CandidateBranch(
        snapshot_index=snapshot, candidate_index=index,
        candidate_family='nominal' if index == 0 else 'nominal_delta',
        observation=np.asarray([0.0], np.float32),
        action=action, nominal_action=np.asarray([0.8], np.float32),
        previous_action=np.zeros(1, np.float32), command_speed=0.5,
        action_distance=float(index), outcomes={8: outcome})


class BranchSupervisionTest(unittest.TestCase):
    def test_snapshot_split_is_disjoint(self):
        branches = []
        for snapshot in range(10):
            branches.extend([
                _branch(snapshot, 0, failure=True, time_to_failure=3),
                _branch(snapshot, 1, failure=False),
            ])
        train, validation, validation_ids = split_branch_snapshots(
            branches, validation_fraction=0.2, seed=7)
        train_ids = {item.snapshot_index for item in train}
        val_ids = {item.snapshot_index for item in validation}
        self.assertFalse(train_ids.intersection(val_ids))
        self.assertEqual(val_ids, set(validation_ids))
        self.assertEqual(len(val_ids), 2)

    def test_episode_split_keeps_adjacent_snapshots_together(self):
        branches = []
        snapshot_episode_ids = {}
        for snapshot in range(8):
            snapshot_episode_ids[snapshot] = snapshot // 2
            branches.extend([
                _branch(snapshot, 0, failure=True, time_to_failure=3),
                _branch(snapshot, 1, failure=False),
            ])
        train, validation, _, validation_episodes = split_branch_episodes(
            branches, snapshot_episode_ids,
            validation_fraction=0.25, seed=5)
        train_episodes = {
            snapshot_episode_ids[item.snapshot_index] for item in train}
        val_episodes = {
            snapshot_episode_ids[item.snapshot_index]
            for item in validation}
        self.assertFalse(train_episodes.intersection(val_episodes))
        self.assertEqual(val_episodes, set(validation_episodes))

    def test_three_way_episode_split_is_disjoint_and_reproducible(self):
        branches = []
        snapshot_episode_ids = {}
        for snapshot in range(20):
            snapshot_episode_ids[snapshot] = snapshot // 2
            branches.extend([
                _branch(snapshot, 0, failure=True, time_to_failure=3),
                _branch(snapshot, 1, failure=False),
            ])
        split = split_branch_episodes_three_way(
            branches, snapshot_episode_ids, seed=11)
        train, calibration, validation, manifest = split
        episode_sets = [
            {snapshot_episode_ids[item.snapshot_index] for item in subset}
            for subset in (train, calibration, validation)
        ]
        self.assertFalse(episode_sets[0].intersection(episode_sets[1]))
        self.assertFalse(episode_sets[0].intersection(episode_sets[2]))
        self.assertFalse(episode_sets[1].intersection(episode_sets[2]))
        self.assertEqual(sum(map(len, episode_sets)), 10)
        repeated = split_branch_episodes_three_way(
            branches, snapshot_episode_ids, seed=11)
        self.assertEqual(manifest['fingerprint'], repeated[3]['fingerprint'])

    def test_counterfactual_update_learns_riskier_action_order(self):
        branches = []
        for snapshot in range(4):
            branches.extend([
                _branch(snapshot, 0, failure=True, time_to_failure=2),
                _branch(snapshot, 1, failure=False),
            ])
        dataset = BranchSupervisionDataset(branches, horizon=8, seed=4)
        safety = SafetyCritic.create(
            3, 1, 1, hidden_dims=(16, 16), learning_rate=3e-3)
        observations = np.zeros((1, 1), np.float32)
        riskier = np.full((1, 1), 0.8, np.float32)
        safer = np.full((1, 1), -0.8, np.float32)
        before = (
            safety.predict_logits(observations, riskier)
            - safety.predict_logits(observations, safer))[0]
        for _ in range(60):
            safety, _ = SafetyCritic.update_counterfactual(
                safety, dataset.sample(16, 16),
                ranking_weight=1.0, ranking_margin=0.2)
        after = (
            safety.predict_logits(observations, riskier)
            - safety.predict_logits(observations, safer))[0]
        self.assertGreater(after, before)
        self.assertGreater(after, 0.2)

    def test_hard_negative_mining_finds_selector_false_safe(self):
        branches = [
            _branch(0, 0, failure=True, time_to_failure=2),
            _branch(0, 1, failure=False),
            _branch(1, 0, failure=False),
            _branch(1, 1, failure=True, time_to_failure=4),
        ]
        hard, stats = mine_selected_false_safe(
            branches, [0.1, 0.05, 0.1, 0.05],
            horizon=8, epsilon=0.2,
            support=[True, True, True, True])
        self.assertEqual(hard, {(0, 0)})
        self.assertEqual(stats['hard_negative_count'], 1.0)
        self.assertEqual(stats['hard_negative_selected_count'], 2.0)

    def test_hard_negative_sampling_upweights_mined_failures(self):
        branches = []
        for snapshot in range(4):
            branches.extend([
                _branch(snapshot, 0, failure=True, time_to_failure=2),
                _branch(snapshot, 1, failure=False),
            ])
        dataset = BranchSupervisionDataset(
            branches, horizon=8, seed=4,
            hard_negative_keys={(0, 0)})
        batch = dataset.sample(
            16, 16, hard_negative_fraction=1.0,
            hard_negative_weight=7.0)
        self.assertEqual(int(np.sum(batch['point_weights'] == 7.0)), 8)
        self.assertEqual(int(np.sum(batch['point_weights'] == 1.0)), 8)

    def test_conformal_upper_offset_has_requested_empirical_coverage(self):
        labels = np.asarray([1.0, 1.0, 0.0, 0.0])
        scores = np.asarray([0.2, 0.8, 0.1, 0.2])
        offset = conformal_upper_offset(labels, scores, alpha=0.25)
        covered = labels <= np.clip(scores + offset, 0.0, 1.0)
        self.assertGreaterEqual(float(np.mean(covered)), 0.75)
        self.assertGreaterEqual(offset, 0.0)


if __name__ == '__main__':
    unittest.main()
