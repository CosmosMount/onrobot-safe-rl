from __future__ import annotations

import unittest

import numpy as np

from learner.counterfactual_dataset import CandidateBranch, HorizonOutcome
from learner.episode_bootstrap import (
    bootstrap_episode_branches,
    split_episode_roles,
)


def _branch(snapshot, candidate):
    outcome = HorizonOutcome(
        horizon=16, failure=bool(candidate), near_failure=bool(candidate),
        time_to_failure=4 if candidate else -1,
        max_tilt_rad=0.5, min_base_height_m=0.2,
        max_contact_count=4, max_undesired_contact_count=0,
        max_contact_force=1.0)
    return CandidateBranch(
        snapshot_index=snapshot, candidate_index=candidate,
        candidate_family='nominal' if candidate == 0 else 'nominal_delta',
        observation=np.zeros(2, np.float32),
        action=np.asarray([candidate], np.float32),
        nominal_action=np.zeros(1, np.float32),
        previous_action=np.zeros(1, np.float32),
        command_speed=0.5, action_distance=float(candidate),
        outcomes={16: outcome})


class EpisodeBootstrapTest(unittest.TestCase):
    def test_four_way_split_is_disjoint_and_complete(self):
        roles = split_episode_roles(
            list(range(100)), seed=9)
        sets = {key: set(value) for key, value in roles.items()}
        self.assertEqual(
            set().union(*sets.values()), set(range(100)))
        for left, left_values in sets.items():
            for right, right_values in sets.items():
                if left < right:
                    self.assertFalse(
                        left_values.intersection(right_values))
        self.assertEqual(len(sets['validation']), 20)
        self.assertEqual(len(sets['temperature']), 10)
        self.assertEqual(len(sets['conformal']), 10)
        self.assertEqual(len(sets['fit']), 60)

    def test_bootstrap_remaps_duplicate_snapshot_groups(self):
        branches = []
        snapshot_episode_ids = {}
        for episode in range(8):
            snapshot_episode_ids[episode] = episode
            branches.extend([
                _branch(episode, 0), _branch(episode, 1)])
        bootstrapped, metadata = bootstrap_episode_branches(
            branches, snapshot_episode_ids, list(range(8)), seed=4)
        self.assertEqual(len(bootstrapped), 16)
        self.assertEqual(metadata['bootstrapped_snapshots'], 8)
        grouped = {}
        for item in bootstrapped:
            grouped.setdefault(item.snapshot_index, []).append(item)
        self.assertEqual(len(grouped), 8)
        self.assertTrue(all(len(items) == 2 for items in grouped.values()))
        self.assertLess(metadata['unique_episode_count'], 8)


if __name__ == '__main__':
    unittest.main()
