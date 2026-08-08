from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from safety_data.candidates import EvidenceCandidateConfig
from safety_data.collector import (
    GaussianImpulseSchedule,
    NativeCollectionConfig,
    collect_native_groups,
)
from safety_data.policies import load_frozen_droq_policy
from train.config import load_app_config
from train.mujoco_snapshot_env import MujocoSnapshotEnv


MODEL = Path(
    "/home/xyz/code/unitree_mujoco/unitree_robots/go2/scene_empty.xml")
CONFIG = Path("config/go2_50hz_sqrl_paper_sac_pretrain.yaml")
CHECKPOINT = Path(
    "saved/experiments/sqrl_paper/seed42/pretrain_sac_async_v1/"
    "step_000000500000/agent")


class NativeGroupCollectionIntegrationTest(unittest.TestCase):
    def test_real_mjcf_actor_to_grouped_schema(self):
        try:
            import mujoco  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("mujoco is unavailable") from exc
        if not MODEL.exists() or not (CHECKPOINT / "actor.pt").exists():
            raise unittest.SkipTest("real MJCF or development checkpoint is unavailable")
        robot, train, _ = load_app_config(CONFIG)
        policy = load_frozen_droq_policy(
            CHECKPOINT,
            CONFIG,
            observation_dim=robot.obs_dim,
            action_dim=robot.num_joints,
            device="cpu",
        )
        env = MujocoSnapshotEnv(
            MODEL,
            robot,
            policy_frequency=train.control_frequency,
            max_joint_delta=train.max_joint_delta,
            use_action_filter=train.use_action_filter,
        )
        result = collect_native_groups(
            env=env,
            source_policy=policy,
            continuation_policy=policy,
            candidate_config=EvidenceCandidateConfig(),
            branch_disturbance=GaussianImpulseSchedule(
                policy_steps=(1,), linear_std_mps=0.1,
                angular_std_radps=0.2),
            config=NativeCollectionConfig(
                split="development_integration_test",
                target_groups=1,
                source_seed=8801,
                policy_training_seed=42,
                horizon_steps=2,
                replicas=2,
                natural_acceptance_probability=1.0,
                max_episode_steps=2,
                max_groups_per_trajectory=1,
                max_source_steps=2,
                settle_seconds=0.02,
            ),
            generator_commit="integration-test",
        )
        report = result.dataset.validate()
        self.assertEqual(report["groups"], 1)
        self.assertEqual(report["max_candidates"], 16)
        self.assertEqual(report["replicas"], 2)
        self.assertGreaterEqual(report["min_valid_candidates_per_group"], 8)
        self.assertEqual(
            result.dataset.manifest["collection_protocol"]
            ["replica_seed_contract"],
            "explicit_three_stream_v1",
        )
        protocol = result.dataset.manifest["collection_protocol"]
        self.assertEqual(protocol["profile_name"], "native_poc_v1")
        self.assertEqual(
            protocol["scope"], "development_boundary_mechanism_only")
        self.assertIn("does not replace natural closed-loop", protocol["evidence_limit"])
        self.assertTrue(
            protocol["source_and_continuation_fingerprints_match"])
        self.assertEqual(
            result.dataset["sampling_stratum"].tolist(), ["random_accept"])
        self.assertEqual(
            result.dataset["acceptance_probability"].tolist(), [1.0])
        self.assertFalse(np.array_equal(
            result.dataset["rollout_seed"],
            result.dataset["perturbation_seed"],
        ))
        result.privileged.validate(result.dataset)


if __name__ == "__main__":
    unittest.main()
