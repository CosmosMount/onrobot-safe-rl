from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from rl.agents.droq.network import DroQEnsembleCritic
from safety_data.paths import ProtectedEvidencePathError
from safety_data.reward_q import (
    REWARD_Q_ACTION_SEMANTICS,
    REWARD_Q_AGGREGATION,
    load_frozen_droq_reward_q,
)


class FrozenDroQRewardQTest(unittest.TestCase):
    observation_dim = 6
    action_dim = 3
    hidden_dims = [16, 16]
    num_qs = 3

    def _fixture(
        self,
        directory: Path,
    ) -> tuple[Path, Path, DroQEnsembleCritic]:
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
    num_qs: 3
    critic_dropout_rate: 0.1
    critic_layer_norm: true
""".strip() + "\n",
            encoding="utf-8",
        )
        torch.manual_seed(5818)
        source = DroQEnsembleCritic(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims,
            num_qs=self.num_qs,
            dropout_rate=0.1,
            use_layer_norm=True,
        ).eval()
        critic = directory / "critic.pt"
        torch.save({
            "network_state_dict": source.state_dict(),
            # The data-only sentinel proves optimizer state is never installed.
            "optimizer_state_dict": {"sentinel": torch.tensor([9102])},
            "scheduler_state_dict": None,
            "update_step": 17,
        }, critic)
        return critic, config, source

    def _load(self, critic: Path, config: Path):
        return load_frozen_droq_reward_q(
            critic,
            config,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            training_step=123,
            device="cpu",
        )

    def test_conservative_values_match_pointwise_ensemble_minimum(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            critic, config, source = self._fixture(Path(raw_directory))
            reward_q = self._load(critic, config)
            observation = np.linspace(
                -0.4, 0.5, self.observation_dim, dtype=np.float32)
            actions = np.asarray([
                [-0.8, 0.2, 0.1],
                [0.0, -0.3, 0.7],
                [1.0, -1.0, 0.5],
                [0.2, 0.3, -0.4],
            ], dtype=np.float32)

            with torch.no_grad():
                expected_per_critic, _ = source(
                    torch.as_tensor(observation).reshape(1, -1).expand(4, -1),
                    torch.as_tensor(actions),
                    training=False,
                )
            expected = expected_per_critic.numpy()
            evaluation = reward_q.evaluate(
                observation, actions, include_per_critic=True)

            self.assertEqual(evaluation.aggregation, REWARD_Q_AGGREGATION)
            self.assertEqual(evaluation.per_critic.shape, (3, 4))
            np.testing.assert_allclose(
                evaluation.per_critic, expected, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(
                evaluation.conservative, np.min(expected, axis=0),
                rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(
                reward_q(observation, actions), evaluation.conservative)

            without_heads = reward_q.evaluate(observation, actions)
            self.assertIsNone(without_heads.per_critic)

    def test_loading_and_evaluation_restore_global_rng_state(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            critic, config, _ = self._fixture(Path(raw_directory))
            torch.manual_seed(99101)
            torch_before_load = torch.random.get_rng_state().clone()
            np.random.seed(12091)
            numpy_before_load = np.random.get_state()

            reward_q = self._load(critic, config)

            torch.testing.assert_close(
                torch.random.get_rng_state(), torch_before_load,
                rtol=0, atol=0)
            numpy_after_load = np.random.get_state()
            self.assertEqual(numpy_before_load[0], numpy_after_load[0])
            np.testing.assert_array_equal(
                numpy_before_load[1], numpy_after_load[1])
            self.assertEqual(numpy_before_load[2:], numpy_after_load[2:])

            torch.manual_seed(99031)
            torch_before_evaluation = torch.random.get_rng_state().clone()
            numpy_before_evaluation = np.random.get_state()
            reward_q.conservative_values(
                np.zeros(self.observation_dim, dtype=np.float32),
                np.zeros((5, self.action_dim), dtype=np.float32),
            )
            torch.testing.assert_close(
                torch.random.get_rng_state(), torch_before_evaluation,
                rtol=0, atol=0)
            numpy_after_evaluation = np.random.get_state()
            self.assertEqual(
                numpy_before_evaluation[0], numpy_after_evaluation[0])
            np.testing.assert_array_equal(
                numpy_before_evaluation[1], numpy_after_evaluation[1])
            self.assertEqual(
                numpy_before_evaluation[2:], numpy_after_evaluation[2:])

    def test_manifest_records_contract_hashes_and_is_immutable(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            critic, config, _ = self._fixture(Path(raw_directory))
            reward_q = self._load(critic, config)
            manifest = reward_q.manifest()

            self.assertEqual(manifest["training_step"], 123)
            self.assertEqual(manifest["critic_update_step"], 17)
            self.assertEqual(
                manifest["load_contract"],
                "critic_pt_network_state_dict_only")
            self.assertEqual(manifest["device"], "cpu")
            self.assertEqual(manifest["hidden_dims"], [16, 16])
            self.assertEqual(manifest["num_qs"], 3)
            self.assertEqual(
                manifest["ensemble_aggregation"], REWARD_Q_AGGREGATION)
            self.assertEqual(
                manifest["action_semantics"], REWARD_Q_ACTION_SEMANTICS)
            self.assertEqual(len(manifest["critic_sha256"]), 64)
            self.assertEqual(len(manifest["critic_state_dict_sha256"]), 64)
            self.assertEqual(
                reward_q.critic_state_dict_sha256,
                manifest["critic_state_dict_sha256"])
            self.assertEqual(len(manifest["config_sha256"]), 64)
            self.assertEqual(
                len(manifest["resolved_agent_config_sha256"]), 64)
            self.assertEqual(
                len(manifest["resolved_train_config_sha256"]), 64)
            self.assertEqual(
                reward_q.fingerprint(),
                manifest["reward_q_fingerprint_sha256"])
            self.assertEqual(
                reward_q.checkpoint_fingerprint(),
                manifest["checkpoint_fingerprint_sha256"])
            self.assertEqual(reward_q.num_qs, 3)
            self.assertEqual(reward_q.device, torch.device("cpu"))
            json.dumps(manifest)

            manifest["training_step"] = -1
            self.assertEqual(reward_q.manifest()["training_step"], 123)

    def test_loaded_weight_fingerprint_ignores_checkpoint_extras(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            critic, config, _ = self._fixture(root)
            first = self._load(critic, config)
            checkpoint = torch.load(
                critic, map_location="cpu", weights_only=True)
            repacked_directory = root / "repacked"
            repacked_directory.mkdir()
            repacked = repacked_directory / "critic.pt"
            torch.save({
                "network_state_dict": checkpoint["network_state_dict"],
                "optimizer_state_dict": {
                    "ignored_repacked_sentinel": torch.tensor([12345])},
                "update_step": checkpoint["update_step"],
            }, repacked)
            second = self._load(repacked, config)

            self.assertNotEqual(first.critic_sha256, second.critic_sha256)
            self.assertEqual(
                first.critic_state_dict_sha256,
                second.critic_state_dict_sha256)
            self.assertEqual(first.fingerprint(), second.fingerprint())
            self.assertNotEqual(
                first.checkpoint_fingerprint(),
                second.checkpoint_fingerprint())

    def test_training_step_is_inferred_and_disagreement_fails(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            critic, config, _ = self._fixture(root)
            agent = root / "step_000000000321" / "agent"
            agent.mkdir(parents=True)
            relocated = agent / "critic.pt"
            relocated.write_bytes(critic.read_bytes())

            reward_q = load_frozen_droq_reward_q(
                agent,
                config,
                observation_dim=self.observation_dim,
                action_dim=self.action_dim,
            )
            self.assertEqual(reward_q.training_step, 321)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                load_frozen_droq_reward_q(
                    relocated,
                    config,
                    observation_dim=self.observation_dim,
                    action_dim=self.action_dim,
                    training_step=322,
                )

    def test_input_shapes_values_and_device_fail_closed(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            critic, config, _ = self._fixture(Path(raw_directory))
            reward_q = self._load(critic, config)
            observation = np.zeros(self.observation_dim, dtype=np.float32)
            actions = np.zeros((2, self.action_dim), dtype=np.float32)

            with self.assertRaisesRegex(ValueError, "observation must have"):
                reward_q(observation[:-1], actions)
            invalid_observation = observation.copy()
            invalid_observation[0] = np.nan
            with self.assertRaisesRegex(ValueError, "observation must be finite"):
                reward_q(invalid_observation, actions)
            with self.assertRaisesRegex(ValueError, "shape"):
                reward_q(observation, actions[0])
            with self.assertRaisesRegex(ValueError, "at least one"):
                reward_q(observation, actions[:0])
            invalid_actions = actions.copy()
            invalid_actions[0, 0] = np.inf
            with self.assertRaisesRegex(ValueError, "must be finite"):
                reward_q(observation, invalid_actions)
            invalid_actions[0, 0] = 1.0001
            with self.assertRaisesRegex(ValueError, r"\[-1, 1\]"):
                reward_q(observation, invalid_actions)
            with self.assertRaisesRegex(TypeError, "include_per_critic"):
                reward_q.evaluate(
                    observation, actions, include_per_critic=1)  # type: ignore[arg-type]
            with self.assertRaisesRegex(ValueError, "CPU.*index"):
                load_frozen_droq_reward_q(
                    critic, config,
                    observation_dim=self.observation_dim,
                    action_dim=self.action_dim,
                    training_step=123,
                    device="cpu:1")
            with self.assertRaisesRegex(ValueError, "CPU or CUDA"):
                load_frozen_droq_reward_q(
                    critic, config,
                    observation_dim=self.observation_dim,
                    action_dim=self.action_dim,
                    training_step=123,
                    device="meta")

    def test_architecture_mismatch_and_protected_paths_fail_closed(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            critic, config, _ = self._fixture(root)
            mismatch = root / "mismatch.yaml"
            mismatch.write_text(
                config.read_text(encoding="utf-8").replace(
                    "hidden_dims: [16, 16]", "hidden_dims: [8, 8]"),
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "architecture"):
                load_frozen_droq_reward_q(
                    critic, mismatch,
                    observation_dim=self.observation_dim,
                    action_dim=self.action_dim,
                    training_step=123)
            with self.assertRaises(ProtectedEvidencePathError):
                load_frozen_droq_reward_q(
                    root / "formal_confirmation" / "critic.pt",
                    config,
                    observation_dim=self.observation_dim,
                    action_dim=self.action_dim,
                    training_step=1)
            with self.assertRaises(ProtectedEvidencePathError):
                load_frozen_droq_reward_q(
                    critic,
                    root / "sealed_inputs" / "config.yaml",
                    observation_dim=self.observation_dim,
                    action_dim=self.action_dim,
                    training_step=1)


class RealFrozenDroQRewardQTest(unittest.TestCase):
    critic = Path(
        "saved/experiments/sqrl_paper/seed42/pretrain_sac_async_v1/"
        "step_000000500000/agent/critic.pt")
    config = Path("config/go2_50hz_sqrl_paper_sac_pretrain.yaml")

    @unittest.skipUnless(critic.is_file(), "development DroQ checkpoint absent")
    def test_real_500k_checkpoint_runs_on_cpu_without_training_agent(self):
        torch.manual_seed(80192)
        rng_before = torch.random.get_rng_state().clone()
        reward_q = load_frozen_droq_reward_q(
            self.critic,
            self.config,
            observation_dim=46,
            action_dim=12,
            device="cpu",
        )
        values = reward_q.conservative_values(
            np.zeros(46, dtype=np.float32),
            np.stack([
                np.zeros(12, dtype=np.float32),
                np.full(12, 0.25, dtype=np.float32),
                np.full(12, -0.25, dtype=np.float32),
            ]),
        )

        torch.testing.assert_close(
            torch.random.get_rng_state(), rng_before, rtol=0, atol=0)
        self.assertEqual(values.shape, (3,))
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertEqual(reward_q.training_step, 500_000)
        self.assertEqual(reward_q.num_qs, 2)
        self.assertEqual(reward_q.manifest()["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
