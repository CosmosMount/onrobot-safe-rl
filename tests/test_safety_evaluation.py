from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from train.safety_evaluation import (SafetyEvalRecord, analyze_records,
                                     write_evaluation_artifacts)


def _record(step: int, q: float, *, unsafe: bool = False,
            boundary: bool = False, intervention: bool = False):
    return SafetyEvalRecord(
        episode=0, step=step, q_safe=q, unsafe=unsafe,
        boundary=boundary, intervention=intervention,
        termination_reason=2 if unsafe else 0, reward=1.0)


class SafetyEvaluationTest(unittest.TestCase):

    def test_pre_failure_rise_passes_gate(self):
        records = [
            _record(0, 0.05), _record(1, 0.08), _record(2, 0.12),
            _record(3, 0.55, boundary=True),
            _record(4, 0.78, boundary=True),
            _record(5, 0.96, unsafe=True, intervention=True),
        ]
        report = analyze_records(
            records, horizon=2, min_auc=0.7, min_warning_delta=0.2)
        self.assertTrue(report['gate']['ready_for_shield'])
        self.assertEqual(report['failures'], 1)
        self.assertEqual(report['fall_rate_per_episode'], 1.0)
        self.assertEqual(report['average_episode_length'], 6.0)
        self.assertEqual(report['future_failure_positive_steps'], 3)
        self.assertGreater(report['q_safe_auroc'], 0.9)
        self.assertEqual(records[3].time_to_failure, 2)

    def test_no_failures_blocks_gate_and_writes_artifacts(self):
        records = [_record(0, 0.1), _record(1, 0.2)]
        report = analyze_records(records, horizon=3)
        self.assertFalse(report['gate']['ready_for_shield'])
        self.assertFalse(
            report['gate']['has_positive_and_negative_samples'])
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_evaluation_artifacts(records, report, tmp)
            self.assertTrue(all(path.exists() for path in paths.values()))
            loaded = json.loads(Path(paths['report']).read_text())
            self.assertEqual(loaded['num_steps'], 2)
            self.assertIn('<svg', Path(paths['figure']).read_text())


if __name__ == '__main__':
    unittest.main()
