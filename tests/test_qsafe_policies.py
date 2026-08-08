from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from rl.agents.inference import build_inference_policy
from safety_data.paths import ProtectedEvidencePathError
from safety_data.policies import load_frozen_droq_policy
from train.config import load_app_config


class FrozenDroQPolicyTest(unittest.TestCase):
    observation_dim = 6
    action_dim = 3

    def _fixture(self, directory: Path) -> tuple[Path, Path]:
        config = directory / "droq.yaml"
        config.write_text(
            """
reward_profile: upstream
train:
  agent: droq
  control_frequency: 50.0
  droq:
    device_type: cpu
    buffer_device_type: cpu
    hidden_dims: [16, 16]
""".strip() + "\n",
            encoding="utf-8",
        )
        _, _, cfg = load_app_config(path=config)
        cfg.device_type = "cpu"
        torch.manual_seed(818)
        source = build_inference_policy(
            self.observation_dim, self.action_dim, cfg)
        actor = directory / "actor.pt"
        torch.save({
            "network_state_dict": source.actor.state_dict(),
            # An optimizer payload proves that the loader can ignore it.  It
            # is data-only so weights_only=True can still read the fixture.
            "optimizer_state_dict": {"sentinel": torch.tensor([91])},
            "update_step": 17,
        }, actor)
        return actor, config

    def _load(self, actor: Path, config: Path):
        return load_frozen_droq_policy(
            actor,
            config,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            training_step=123,
            device="cpu",
        )

    def test_deterministic_and_seeded_stochastic_actions(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            actor, config = self._fixture(Path(raw_directory))
            policy = self._load(actor, config)
            observation = np.linspace(
                -0.4, 0.5, self.observation_dim, dtype=np.float32)

            deterministic_a = policy.deterministic_action(observation)
            deterministic_b = policy.deterministic_action(observation)
            np.testing.assert_array_equal(deterministic_a, deterministic_b)

            sample_a = policy.sample_action(
                observation, np.random.default_rng(7001))
            sample_b = policy.sample_action(
                observation, np.random.default_rng(7001))
            sample_c = policy.sample_action(
                observation, np.random.default_rng(7002))
            np.testing.assert_array_equal(sample_a, sample_b)
            self.assertFalse(np.array_equal(sample_a, sample_c))

            history = np.stack([observation * 0.0, observation])
            continuation = policy(
                history, 1, np.random.default_rng(7001))
            np.testing.assert_array_equal(sample_a, continuation)

    def test_sampling_restores_global_torch_rng(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            actor, config = self._fixture(Path(raw_directory))
            torch.manual_seed(99101)
            state_before_load = torch.random.get_rng_state().clone()
            policy = self._load(actor, config)
            torch.testing.assert_close(
                torch.random.get_rng_state(), state_before_load,
                rtol=0, atol=0)
            observation = np.zeros(self.observation_dim, dtype=np.float32)
            torch.manual_seed(99031)
            state_before = torch.random.get_rng_state().clone()

            policy.sample_action(observation, np.random.default_rng(82))

            torch.testing.assert_close(
                torch.random.get_rng_state(), state_before, rtol=0, atol=0)

    def test_sampling_consumes_one_explicit_numpy_seed_only(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            actor, config = self._fixture(Path(raw_directory))
            policy = self._load(actor, config)
            observation = np.zeros(self.observation_dim, dtype=np.float32)
            actual_rng = np.random.default_rng(8201)
            expected_rng = np.random.default_rng(8201)

            policy.sample_action(observation, actual_rng)
            expected_rng.integers(0, np.iinfo(np.int64).max)

            np.testing.assert_array_equal(
                actual_rng.integers(0, 2**31, size=8),
                expected_rng.integers(0, 2**31, size=8),
            )

    def test_manifest_records_hashes_steps_config_and_is_immutable(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            actor, config = self._fixture(Path(raw_directory))
            policy = self._load(actor, config)
            manifest = policy.manifest()
            self.assertEqual(manifest["training_step"], 123)
            self.assertEqual(manifest["actor_update_step"], 17)
            self.assertEqual(manifest["load_contract"],
                             "actor_pt_network_state_dict_only")
            self.assertEqual(manifest["device"], "cpu")
            self.assertEqual(manifest["hidden_dims"], [16, 16])
            self.assertEqual(len(manifest["actor_sha256"]), 64)
            self.assertEqual(len(manifest["actor_state_dict_sha256"]), 64)
            self.assertEqual(
                policy.actor_state_dict_sha256,
                manifest["actor_state_dict_sha256"],
            )
            self.assertEqual(len(manifest["config_sha256"]), 64)
            self.assertEqual(
                len(manifest["resolved_agent_config_sha256"]), 64)
            self.assertEqual(
                len(manifest["resolved_train_config_sha256"]), 64)
            self.assertEqual(policy.fingerprint(),
                             manifest["policy_fingerprint_sha256"])
            self.assertEqual(
                policy.checkpoint_fingerprint(),
                manifest["checkpoint_fingerprint_sha256"],
            )
            self.assertEqual(len(policy.checkpoint_fingerprint()), 64)
            json.dumps(manifest)

            manifest["training_step"] = -1
            self.assertEqual(policy.manifest()["training_step"], 123)

    def test_policy_fingerprint_tracks_loaded_weights_not_checkpoint_extras(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            actor, config = self._fixture(root)
            first = self._load(actor, config)
            checkpoint = torch.load(actor, map_location="cpu", weights_only=True)
            repacked_directory = root / "repacked"
            repacked_directory.mkdir()
            repacked_actor = repacked_directory / "actor.pt"
            torch.save({
                "network_state_dict": checkpoint["network_state_dict"],
                "optimizer_state_dict": {
                    "ignored_repacked_sentinel": torch.tensor([12345])},
                "update_step": checkpoint["update_step"],
            }, repacked_actor)
            second = self._load(repacked_actor, config)

            self.assertNotEqual(first.actor_sha256, second.actor_sha256)
            self.assertEqual(
                first.actor_state_dict_sha256,
                second.actor_state_dict_sha256,
            )
            self.assertEqual(first.fingerprint(), second.fingerprint())
            self.assertNotEqual(
                first.checkpoint_fingerprint(),
                second.checkpoint_fingerprint(),
            )

    def test_policy_fingerprint_is_invariant_to_hardware_only_config(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            actor, config = self._fixture(root)
            alternate = root / "droq_cuda_source.yaml"
            alternate.write_text(
                config.read_text(encoding="utf-8").replace(
                    "device_type: cpu", "device_type: cuda"),
                encoding="utf-8",
            )

            first = self._load(actor, config)
            second = self._load(actor, alternate)

            self.assertEqual(first.fingerprint(), second.fingerprint())
            self.assertNotEqual(
                first.checkpoint_fingerprint(),
                second.checkpoint_fingerprint(),
            )
            self.assertEqual(first.manifest()["device"], "cpu")
            self.assertEqual(second.manifest()["device"], "cpu")

    def test_training_step_path_is_inferred_and_disagreement_fails(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            actor, config = self._fixture(root)
            agent = root / "step_000000000321" / "agent"
            agent.mkdir(parents=True)
            relocated = agent / "actor.pt"
            relocated.write_bytes(actor.read_bytes())
            policy = load_frozen_droq_policy(
                agent,
                config,
                observation_dim=self.observation_dim,
                action_dim=self.action_dim,
            )
            self.assertEqual(policy.training_step, 321)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                load_frozen_droq_policy(
                    relocated,
                    config,
                    observation_dim=self.observation_dim,
                    action_dim=self.action_dim,
                    training_step=322,
                )

    def test_protected_paths_fail_before_loading(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            actor, config = self._fixture(root)
            with self.assertRaises(ProtectedEvidencePathError):
                load_frozen_droq_policy(
                    root / "formal_confirmation" / "actor.pt",
                    config,
                    observation_dim=self.observation_dim,
                    action_dim=self.action_dim,
                    training_step=1,
                )
            with self.assertRaises(ProtectedEvidencePathError):
                load_frozen_droq_policy(
                    actor,
                    root / "sealed_inputs" / "config.yaml",
                    observation_dim=self.observation_dim,
                    action_dim=self.action_dim,
                    training_step=1,
                )


class RealFrozenDroQPolicyTest(unittest.TestCase):
    actor = Path(
        "saved/experiments/sqrl_paper/seed42/pretrain_sac_async_v1/"
        "step_000000500000/agent/actor.pt")
    config = Path("config/go2_50hz_sqrl_paper_sac_pretrain.yaml")

    @unittest.skipUnless(actor.is_file(), "development DroQ checkpoint absent")
    def test_real_checkpoint_loads_on_cpu_without_training_agent(self):
        policy = load_frozen_droq_policy(
            self.actor,
            self.config,
            observation_dim=46,
            action_dim=12,
            device="cpu",
        )
        observation = np.zeros(46, dtype=np.float32)
        action = policy.deterministic_action(observation)
        self.assertEqual(action.shape, (12,))
        self.assertEqual(policy.training_step, 500_000)
        self.assertEqual(policy.manifest()["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
