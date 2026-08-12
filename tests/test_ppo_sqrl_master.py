from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from safety_data.ppo_sqrl_master import (
    TransitionShardWriter,
    split_role,
    validate_master_manifest,
    validate_transition_arrays,
)


def arrays(count: int = 4) -> dict[str, np.ndarray]:
    action = np.arange(count * 12, dtype=np.float32).reshape(count, 12)
    return {
        "observation_history_t": np.zeros((count, 5, 46), np.float32),
        "critic_action": action.copy(),
        "next_observation_history": np.ones((count, 5, 46), np.float32),
        "c_t_plus_1": np.asarray([False, True, False, False])[:count],
        "terminated": np.asarray([False, True, False, False])[:count],
        "truncated": np.asarray([False, False, True, False])[:count],
        "action_requested": np.zeros((count, 12), np.float32),
        "action_pre_projection": np.zeros((count, 12), np.float32),
        "absolute_q_target": action.copy(),
        "action_log_probability": np.zeros(count, np.float32),
        "policy_entropy": np.ones(count, np.float32),
        "action_std": np.ones((count, 12), np.float32),
        "action_saturation": np.zeros((count, 12), bool),
        "action_change_rate": np.zeros(count, np.float32),
        "ppo_seed": np.full(count, 137, np.int32),
        "collector_stage": np.full(count, "boundary", "U8"),
        "collector_checkpoint": np.full(count, "model_19.pt", "U32"),
        "env_id": np.arange(count, dtype=np.int32),
        "episode_id": np.zeros(count, np.int64),
        "vector_step": np.arange(count, dtype=np.int64),
        "randomization_identity": np.arange(count, dtype=np.uint64),
        "rng_identity": np.arange(count, dtype=np.uint64) + 10,
        "policy_observation_t": np.zeros((count, 48), np.float32),
        "next_policy_observation": np.ones((count, 48), np.float32),
        "next_action_encoder_bias": np.zeros((count, 12), np.float32),
    }


class PpoSqrlMasterTest(unittest.TestCase):
    def test_atomic_shard_round_trip_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = TransitionShardWriter(Path(directory) / "master")
            writer.write(arrays())
            manifest = writer.close({"seed": 137})
            loaded = validate_master_manifest(manifest)
            self.assertEqual(loaded["transition_count"], 4)
            shard = manifest.parent / loaded["shards"][0]["path"]
            shard.write_bytes(shard.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                validate_master_manifest(manifest)

    def test_action_and_cost_semantics_fail_closed(self):
        value = arrays()
        value["critic_action"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "absolute PD target"):
            validate_transition_arrays(value)
        value = arrays()
        value["c_t_plus_1"][0] = True
        with self.assertRaisesRegex(ValueError, "cost must equal"):
            validate_transition_arrays(value)

    def test_episode_role_is_deterministic(self):
        self.assertEqual(split_role(137, 4, 9), split_role(137, 4, 9))
        self.assertIn(split_role(137, 4, 9), {"fit", "calibration", "test"})


if __name__ == "__main__":
    unittest.main()
