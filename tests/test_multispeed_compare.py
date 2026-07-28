from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

from scripts.run_multispeed_sqrl_sac_compare import (
    _completed_ft_row, _load_rows, _write_json)


class MultispeedCompareTest(unittest.TestCase):

    def test_json_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows.json'
            rows = [{'algo': 'sqrl', 'ft_speed': 0.4}]
            _write_json(path, rows)
            self.assertEqual(_load_rows(path), rows)

    def test_completed_ft_is_rehydrated_without_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ft_dir = root / 'ft'
            ft_dir.mkdir()
            checkpoint = ft_dir / 'training_snapshot_000000016000.pkl'
            with checkpoint.open('wb') as stream:
                pickle.dump({'step': 16000, 'metadata': {}}, stream)
            log = root / 'ft.log'
            log.write_text(
                '[step 15900] rolling n=1000 forward_vel=0.45 '
                'dx=0 upright=1 action_sat=0 falls=7 loop_hz=10 '
                'action_hz=10\n', encoding='utf-8')
            row = _completed_ft_row(
                ft_dir=ft_dir, log_path=log, algo='sqrl', speed=0.4,
                start_step=12000, target_step=16000)
            self.assertIsNotNone(row)
            self.assertEqual(row['falls_total_end'], 7)
            self.assertTrue(row['reused_completed_run'])

    def test_incomplete_ft_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ft_dir = root / 'ft'
            ft_dir.mkdir()
            checkpoint = ft_dir / 'training_snapshot_000000012927.pkl'
            with checkpoint.open('wb') as stream:
                pickle.dump({'step': 12927, 'metadata': {}}, stream)
            log = root / 'ft.log'
            log.write_text('', encoding='utf-8')
            row = _completed_ft_row(
                ft_dir=ft_dir, log_path=log, algo='sqrl', speed=0.5,
                start_step=12000, target_step=16000)
            self.assertIsNone(row)


if __name__ == '__main__':
    unittest.main()
