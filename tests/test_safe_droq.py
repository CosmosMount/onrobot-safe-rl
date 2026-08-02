from __future__ import annotations

import tempfile
import unittest

import numpy as np
import torch
import gymnasium as gym

from rl.agents import create_agent
from rl.agents.safe_droq.replay import SafetyReplay
from train.config import load_app_config
from train.env import Go2Env
from train.main import _parse_args


def _transition(step: int, *, terminated: bool = False, truncated: bool = False):
    return {
        "observation": np.asarray([[step, 0.0]], dtype=np.float32),
        "action": np.asarray([[0.1]], dtype=np.float32),
        "reward": np.asarray([0.0], dtype=np.float32),
        "terminated": np.asarray([terminated], dtype=np.float32),
        "truncated": np.asarray([truncated], dtype=np.float32),
        "next_observation": np.asarray([[step + 1, 0.0]], dtype=np.float32),
        "unsafe_label": np.asarray([terminated], dtype=np.float32),
        "near_failure_label": np.asarray([step >= 3], dtype=np.float32),
        "replay_repeat_index": np.asarray([0], dtype=np.int32),
    }


class SafetyReplayTest(unittest.TestCase):

    def _replay(self):
        return SafetyReplay(
            capacity=100,
            min_length=1,
            batch_size=8,
            failure_horizon=3,
            device=torch.device("cpu"),
            seed=42)

    def test_marks_only_failure_and_preceding_horizon(self):
        replay = self._replay()
        for step in range(6):
            replay.add_batch(_transition(step, terminated=step == 5))
        self.assertEqual(len(replay), 6)
        self.assertEqual(replay.positive_count, 3)
        labels = [
            int(item["future_failure"]) for item in replay._items]
        self.assertEqual(labels, [0, 0, 0, 1, 1, 1])

    def test_terminal_replay_repeats_do_not_duplicate_safety_data(self):
        replay = self._replay()
        transition = _transition(0, terminated=True)
        for repeat in range(4):
            transition["replay_repeat_index"] = np.asarray(
                [repeat], dtype=np.int32)
            replay.add_batch(transition)
        self.assertEqual(len(replay), 1)
        self.assertEqual(replay.positive_count, 1)

    def test_balanced_batch_contains_both_classes(self):
        replay = self._replay()
        for step in range(6):
            replay.add_batch(_transition(step, terminated=step == 5))
        batch = replay.sample()
        self.assertEqual(tuple(batch["observation"].shape), (8, 2))
        self.assertEqual(float(batch["future_failure"].sum()), 4.0)

    def test_round_trip_preserves_pending_episode(self):
        replay = self._replay()
        replay.add_batch(_transition(0))
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/replay.pt"
            replay.save(path)
            restored = self._replay()
            restored.load(path)
        self.assertEqual(len(restored._episode), 1)


class SafeDroQConfigTest(unittest.TestCase):

    def test_cli_accepts_explicit_paired_seed(self):
        args = _parse_args(["--seed", "46"])
        self.assertEqual(args.seed, 46)

    def test_50hz_overlay_supports_baseline_and_safety_agent(self):
        _, safe_train, safe_cfg = load_app_config(
            path="config/go2_50hz_safe.yaml",
            agent="safe_droq")
        _, sac_train, sac_cfg = load_app_config(
            path="config/go2_50hz_safe.yaml",
            agent="droq")
        self.assertEqual(safe_train.control_frequency, 50.0)
        self.assertEqual(sac_train.control_frequency, 50.0)
        self.assertEqual(safe_cfg.agent_type, "safe_droq")
        self.assertEqual(sac_cfg.agent_type, "droq")
        self.assertEqual(safe_cfg.safety_mode, "logging")

    def test_adaptive_gated_overlay_delays_and_ramps_masking(self):
        _, _, cfg = load_app_config(
            path="config/go2_50hz_safe_adaptive_gated.yaml",
            agent="safe_droq")
        self.assertEqual(cfg.safety_activation_step, 2000)
        self.assertEqual(cfg.safety_masking_ramp_steps, 2000)
        self.assertAlmostEqual(cfg.safety_max_action_rms, 0.35)
        self.assertAlmostEqual(cfg.safety_reward_q_margin, 10.0)
        self.assertAlmostEqual(cfg.safety_min_risk_improvement, 0.05)

        agent = self._cpu_agent()
        agent._cfg.safety_activation_step = 2000
        agent._cfg.safety_masking_ramp_steps = 2000
        self.assertEqual(agent._masking_progress(1999), 0.0)
        self.assertAlmostEqual(agent._masking_progress(2999), 0.5)
        self.assertEqual(agent._masking_progress(3999), 1.0)

        _, _, strict = load_app_config(
            path="config/go2_50hz_safe_adaptive_gated_v2.yaml",
            agent="safe_droq")
        self.assertEqual(strict.safety_activation_step, 5000)
        self.assertAlmostEqual(strict.safety_max_action_rms, 0.15)
        self.assertAlmostEqual(strict.safety_reward_q_margin, 0.0)

        _, _, local = load_app_config(
            path="config/go2_50hz_safe_adaptive_gated_v3.yaml",
            agent="safe_droq")
        self.assertEqual(local.safety_update_period, 25)
        self.assertTrue(local.safety_contract_candidates)
        self.assertAlmostEqual(local.safety_min_risk_improvement, 0.02)
        self.assertAlmostEqual(local.safety_max_action_rms, 0.15)

    def _cpu_agent(self, *, frozen: bool = False):
        _, _, cfg = load_app_config(
            path="config/go2_50hz_safe.yaml",
            agent="safe_droq")
        cfg.device_type = "cpu"
        cfg.buffer_device_type = "cpu"
        cfg.hidden_dims = [16, 16]
        cfg.safety_hidden_dims = [16, 16]
        cfg.num_qs = 2
        cfg.num_min_qs = 1
        cfg.sample_batch_size = 4
        cfg.buffer_min_length = 4
        cfg.safety_batch_size = 4
        cfg.safety_buffer_min_length = 1
        cfg.freeze_safety_critic = frozen
        return create_agent(
            gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
            gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            {},
            cfg,
        )

    def test_frozen_safety_critic_ignores_replay_and_updates(self):
        agent = self._cpu_agent(frozen=True)
        before = {
            key: value.detach().clone()
            for key, value in agent._safety_critic.network.state_dict().items()
        }
        for step in range(4):
            agent.process_transition(
                _transition(step, terminated=step == 3))
            self.assertEqual(agent._update_safety(), {})
        self.assertEqual(len(agent._safety_replay), 0)
        for key, value in agent._safety_critic.network.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_seeded_exploration_nominals_match(self):
        robot, train, _ = load_app_config(
            path="config/go2_50hz_safe.yaml",
            agent="droq")
        first = Go2Env(None, robot, train.control_frequency, 400, seed=42)
        second = Go2Env(None, robot, train.control_frequency, 400, seed=42)
        for _ in range(10):
            np.testing.assert_array_equal(
                first.sample_action(), second.sample_action())


if __name__ == "__main__":
    unittest.main()
