import unittest

import numpy as np

from scripts.diagnose_qsafe_capacity import (
    Episode,
    _rank_metrics,
    prior_correct,
    reconstruct_episodes,
    split_episodes,
)


def _item(*, done=False, label=0.0):
    return {
        'dones': done,
        'future_failure_labels': label,
    }


class QSafeDiagnosticsTest(unittest.TestCase):
    def test_episode_reconstruction_preserves_terminal_boundaries(self):
        episodes = reconstruct_episodes([
            _item(), _item(done=True, label=1.0),
            _item(), _item(), _item(done=True),
            _item(),
        ], speed=0.6)
        self.assertEqual([len(ep.items) for ep in episodes], [2, 3, 1])
        self.assertEqual([ep.failed for ep in episodes], [True, False, False])

    def test_split_keeps_episode_objects_disjoint(self):
        episodes = [
            Episode(0.6, index, (_item(done=True, label=float(index % 2)),))
            for index in range(10)
        ]
        train, val = split_episodes(
            episodes, val_fraction=0.2, seed=7)
        self.assertFalse({ep.index for ep in train}
                         & {ep.index for ep in val})
        self.assertEqual(len(train) + len(val), len(episodes))
        self.assertTrue(any(ep.failed for ep in val))
        self.assertTrue(any(not ep.failed for ep in val))

    def test_prior_correction_preserves_ranking_and_improves_prior(self):
        scores = np.asarray([0.1, 0.4, 0.6, 0.9])
        corrected = prior_correct(scores, natural_prior=0.1)
        self.assertTrue(np.all(np.diff(corrected) > 0.0))
        self.assertLess(float(np.mean(corrected)), float(np.mean(scores)))

    def test_metrics_include_calibration_and_ranking(self):
        metrics = _rank_metrics(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0.1, 0.2, 0.8, 0.9]))
        self.assertAlmostEqual(metrics['auroc'], 1.0)
        self.assertAlmostEqual(metrics['average_precision'], 1.0)
        self.assertLess(metrics['brier'], 0.05)
        self.assertIn('ece_10bin', metrics)


if __name__ == '__main__':
    unittest.main()
