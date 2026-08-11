from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from safety_data.mjlab_capacity_authorization import compile_capacity_authorization


def _report(envs: int, throughput: float, seconds: float) -> dict:
    return {
        "schema_version": "qsafe.mjlab_go2_capacity.v1",
        "envs": envs,
        "pass": True,
        "elapsed_seconds": seconds,
        "generator_commit": "a" * 40,
        "generator_worktree_clean_at_launch": True,
        "external_force_nonzero": False,
        "push_event_present": False,
        "nonfinite": False,
        "gpu_sampling_error": None,
        "peak_total_gpu_memory_mib": 3000,
        "memory_growth_mib": 0.0,
        "policy_env_steps_per_second": throughput,
        "mean_gpu_utilization_percent": 90.0,
        "target_alignment": {"contract_sha256": "b" * 64},
        "versions": {"mujoco": "3.5.0", "warp": "1.12.0"},
    }


class MjlabCapacityAuthorizationTest(unittest.TestCase):
    def _paths(self, root: Path, values: list[dict]) -> list[Path]:
        paths = []
        for index, value in enumerate(values):
            path = root / f"report-{index}.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            paths.append(path)
        return paths

    def test_selects_2048_and_authorizes_exact_2000_training_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ladder = [
                _report(256, 16000.0, 301.0),
                _report(512, 28000.0, 302.0),
                _report(1024, 44000.0, 303.0),
                _report(2048, 58000.0, 304.0),
            ]
            paths = self._paths(root, ladder)
            stability = root / "stability.json"
            stability.write_text(json.dumps(
                _report(2048, 57900.0, 1801.0)), encoding="utf-8")
            result = compile_capacity_authorization(paths, stability)
            self.assertTrue(result["authorized"])
            self.assertEqual(result["selected_capacity_envs"], 2048)
            self.assertEqual(result["production_envs"], 2000)
            self.assertEqual(len(result["upgrade_gains"]), 3)

    def test_stops_at_first_upgrade_below_fifteen_percent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ladder = [
                _report(256, 16000.0, 301.0),
                _report(512, 28000.0, 301.0),
                _report(1024, 30000.0, 301.0),
                _report(2048, 50000.0, 301.0),
            ]
            paths = self._paths(root, ladder)
            stability = root / "stability.json"
            stability.write_text(json.dumps(
                _report(512, 27900.0, 1801.0)), encoding="utf-8")
            result = compile_capacity_authorization(
                paths, stability, production_envs=512)
            self.assertEqual(result["selected_capacity_envs"], 512)

    def test_rejects_dirty_or_mismatched_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ladder = [
                _report(256, 16000.0, 301.0),
                _report(512, 28000.0, 301.0),
                _report(1024, 44000.0, 301.0),
                _report(2048, 58000.0, 301.0),
            ]
            ladder[2]["generator_worktree_clean_at_launch"] = False
            paths = self._paths(root, ladder)
            stability = root / "stability.json"
            stability.write_text(json.dumps(
                _report(2048, 58000.0, 1801.0)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dirty"):
                compile_capacity_authorization(paths, stability)


if __name__ == "__main__":
    unittest.main()
