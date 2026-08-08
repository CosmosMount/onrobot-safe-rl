from __future__ import annotations

import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch

from rl.agents import create_agent
from rl.agents.inference import build_inference_policy
from rl.agents.paper_sqrl.inference import PaperSQRLInferencePolicy
from rl.agents.paper_sqrl.replay import RecentTrajectoryReplay
from runtime.inference.__main__ import PolicyInferenceRuntime
from train.config import load_app_config
from train.env import Go2Env


def transition(step: int, *, terminated: bool = False,
               truncated: bool = False, unsafe: bool = False):
    return {
        "observation": np.asarray([[step, 0.0]], dtype=np.float32),
        "action": np.asarray([[0.1]], dtype=np.float32),
        "reward": np.asarray([0.0], dtype=np.float32),
        "terminated": np.asarray([terminated], dtype=np.float32),
        "truncated": np.asarray([truncated], dtype=np.float32),
        "next_observation": np.asarray([[step + 1, 0.0]], dtype=np.float32),
        "unsafe_label": np.asarray([unsafe], dtype=np.float32),
        "replay_repeat_index": np.asarray([0], dtype=np.int32),
    }


class RecentTrajectoryReplayTest(unittest.TestCase):
    def test_retains_latest_complete_trajectories_without_relabeling(self):
        replay = RecentTrajectoryReplay(
            max_trajectories=2, min_transitions=1, batch_size=8,
            device=torch.device("cpu"), seed=42)
        for episode in range(3):
            replay.add_batch(transition(0))
            replay.add_batch(transition(
                1, terminated=episode == 2, truncated=episode != 2,
                unsafe=episode == 2))
        self.assertEqual(replay.trajectory_count, 2)
        self.assertEqual(len(replay), 4)
        self.assertEqual(replay.failure_count, 1)

    def test_incomplete_episode_round_trip(self):
        replay = RecentTrajectoryReplay(
            max_trajectories=2, min_transitions=1, batch_size=2,
            device=torch.device("cpu"), seed=42)
        replay.add_batch(transition(0))
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/replay.pt"
            replay.save(path)
            restored = RecentTrajectoryReplay(
                max_trajectories=2, min_transitions=1, batch_size=2,
                device=torch.device("cpu"), seed=0)
            restored.load(path)
        self.assertEqual(len(restored._current), 1)


class PaperSQRLAgentTest(unittest.TestCase):
    def _agent(self, phase: str = "pretrain"):
        path = (
            "config/go2_50hz_sqrl_paper_pretrain.yaml"
            if phase == "pretrain"
            else "config/go2_50hz_sqrl_paper_finetune.yaml")
        _, _, cfg = load_app_config(path=path)
        cfg.device_type = "cpu"
        cfg.buffer_device_type = "cpu"
        cfg.hidden_dims = [16, 16]
        cfg.safety_hidden_dims = [16, 16]
        cfg.num_qs = 2
        cfg.num_min_qs = 1
        cfg.buffer_min_length = 1
        cfg.sample_batch_size = 2
        cfg.safety_replay_min_transitions = 1
        cfg.safety_batch_size = 2
        cfg.safety_num_candidates = 4
        cfg.safety_boundary_pool_multiplier = 2
        cfg.pretrain_task_steps_per_cycle = 2
        cfg.pretrain_safety_episodes_per_cycle = 1
        return create_agent(
            gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
            gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            {}, cfg)

    def test_pretrain_alternates_task_and_constrained_trajectory(self):
        agent = self._agent()
        agent.process_transition(transition(0))
        self.assertEqual(agent._collection_phase, "task")
        agent.process_transition(transition(1, truncated=True))
        self.assertEqual(agent._collection_phase, "safety")
        reward_size = len(agent._replay_buffer)
        agent.process_transition(transition(2))
        agent.process_transition(transition(3, terminated=True, unsafe=True))
        self.assertEqual(agent._collection_phase, "task")
        self.assertEqual(len(agent._replay_buffer), reward_size)
        self.assertEqual(agent._safety_replay.trajectory_count, 1)
        self.assertGreater(agent._pending_safety_updates, 0)

    def test_finetune_always_constrains_actions(self):
        agent = self._agent("finetune")
        action = agent.sample_actions(
            0, {"next_observation": np.zeros((1, 2), dtype=np.float32)},
            training=True)
        self.assertEqual(np.asarray(action).shape, (1, 1))
        self.assertEqual(agent.get_metrics()["safety/active"], 1.0)

    def test_paper_configs_match_minitaur_protocol(self):
        robot, train, cfg = load_app_config(
            path="config/go2_50hz_sqrl_paper_pretrain.yaml")
        self.assertAlmostEqual(robot.move_speed, 0.30)
        self.assertEqual(train.max_steps, 500_000)
        self.assertEqual(train.max_episode_steps, 500)
        self.assertEqual(train.utd_ratio, 1)
        self.assertEqual(cfg.sqrl_phase, "pretrain")
        self.assertEqual(cfg.num_qs, 2)
        self.assertEqual(cfg.num_min_qs, 2)
        self.assertEqual(cfg.critic_dropout_rate, 0.0)
        self.assertFalse(cfg.critic_layer_norm)
        self.assertAlmostEqual(cfg.safety_gamma, 0.70)
        self.assertEqual(list(cfg.safety_hidden_dims), [256, 256])
        self.assertEqual(cfg.safety_updates_per_cycle, 1)

        sac_robot, sac_train, sac_cfg = load_app_config(
            path="config/go2_50hz_sqrl_paper_sac_pretrain.yaml")
        self.assertAlmostEqual(sac_robot.move_speed, 0.30)
        self.assertEqual(sac_train.max_steps, 500_000)
        self.assertTrue(sac_train.async_collection)
        self.assertEqual(sac_train.utd_ratio, 1)
        self.assertEqual(sac_cfg.agent_type, "droq")
        self.assertEqual(sac_cfg.num_qs, 2)
        self.assertEqual(sac_cfg.num_min_qs, 2)
        self.assertEqual(sac_cfg.critic_dropout_rate, 0.0)
        self.assertFalse(sac_cfg.critic_layer_norm)

        target_robot, target_train, target_cfg = load_app_config(
            path="config/go2_50hz_sqrl_paper_sac_finetune.yaml")
        self.assertAlmostEqual(target_robot.move_speed, 0.40)
        self.assertEqual(target_train.max_steps, 500_000)
        self.assertEqual(target_cfg.agent_type, "droq")

    def test_inference_copy_has_independent_weights_and_exact_schedule(self):
        agent = self._agent()
        policy = PaperSQRLInferencePolicy(2, 1, agent.cfg)
        policy.load_snapshot(
            agent.export_inference_snapshot(snapshot_version=3))
        self.assertEqual(policy.snapshot_version, 3)
        for source, copied in zip(
                agent._actor.network.parameters(), policy.actor.parameters()):
            self.assertTrue(torch.equal(source.cpu(), copied.cpu()))
            self.assertNotEqual(source.data_ptr(), copied.data_ptr())

        first = policy.decide(
            np.zeros(2, dtype=np.float32), training=False)
        self.assertEqual(first.metadata["phase"], "task")
        self.assertFalse(first.metadata["active"])
        policy.observe_transition(
            policy_step=True, terminated=False, truncated=False)
        policy.observe_transition(
            policy_step=True, terminated=False, truncated=True)
        self.assertEqual(policy.phase, "safety")
        constrained = policy.decide(
            np.zeros(2, dtype=np.float32), training=False)
        self.assertTrue(constrained.metadata["active"])
        event = policy.observe_transition(
            policy_step=True, terminated=True, truncated=False)
        self.assertTrue(event["safety_trajectory_complete"])
        self.assertEqual(policy.phase, "task")

    def test_inference_rejects_stale_weight_snapshot(self):
        agent = self._agent()
        policy = PaperSQRLInferencePolicy(2, 1, agent.cfg)
        policy.load_snapshot(
            agent.export_inference_snapshot(snapshot_version=2))
        with self.assertRaisesRegex(ValueError, "backwards"):
            policy.load_snapshot(
                agent.export_inference_snapshot(snapshot_version=1))

    def test_sac_inference_copy_needs_no_safety_critic(self):
        _, _, cfg = load_app_config(
            path="config/go2_50hz_sqrl_paper_sac_pretrain.yaml")
        cfg.device_type = "cpu"
        cfg.buffer_device_type = "cpu"
        cfg.hidden_dims = [16, 16]
        cfg.num_qs = 2
        cfg.buffer_max_length = 10
        cfg.buffer_min_length = 1
        cfg.sample_batch_size = 2
        agent = create_agent(
            gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
            gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            {}, cfg)
        payload = agent.export_inference_snapshot(snapshot_version=7)
        self.assertNotIn("safety_critic_state_dict", payload)
        policy = build_inference_policy(2, 1, cfg)
        policy.load_snapshot(payload)
        decision = policy.decide(
            np.zeros(2, dtype=np.float32), training=False)
        self.assertFalse(decision.metadata.get("active", False))
        self.assertEqual(decision.action_requested.shape, (1,))
        np.testing.assert_array_equal(
            decision.action_nominal, decision.action_requested)

    def test_checkpoint_restores_sqrl_collection_schedule(self):
        agent = self._agent()
        agent._collection_phase = "safety"
        agent._task_steps_in_cycle = 123
        agent._safety_episodes_in_cycle = 2
        agent._pending_safety_updates = 1
        with tempfile.TemporaryDirectory() as directory:
            agent.save(directory)
            restored = self._agent()
            restored.load(directory)
        self.assertEqual(restored._collection_phase, "safety")
        self.assertEqual(restored._task_steps_in_cycle, 123)
        self.assertEqual(restored._safety_episodes_in_cycle, 2)
        self.assertEqual(restored._pending_safety_updates, 1)
        policy = PaperSQRLInferencePolicy(2, 1, restored.cfg)
        policy.load_snapshot(
            restored.export_inference_snapshot(snapshot_version=9))
        self.assertEqual(policy.phase, "safety")
        self.assertEqual(policy.task_steps, 123)


class RuntimeHandshakeTest(unittest.TestCase):
    def test_env_clear_sends_an_explicit_runtime_command(self):
        sent = []

        class Sender:
            def send(self, message):
                sent.append(message)

        env = Go2Env.__new__(Go2Env)
        env._action_tx = Sender()
        env.clear_action()
        self.assertEqual(sent, [{"command": "clear"}])

    def test_runtime_clear_command_resets_stale_episode_counter(self):
        class Receiver:
            def recv_latest(self):
                return {"command": "clear"}

        runtime = PolicyInferenceRuntime.__new__(PolicyInferenceRuntime)
        runtime._action_rx = Receiver()
        runtime._policy_action_cleared = True
        runtime._step_count = 497
        runtime._receive_action()
        self.assertEqual(runtime._step_count, 0)

    def test_env_reset_waits_for_runtime_counter_acknowledgement(self):
        class Sender:
            def __init__(self):
                self.messages = []

            def wait_ready(self):
                pass

            def send(self, message):
                self.messages.append(message)

        class Receiver:
            def __init__(self):
                self.messages = iter([
                    {"observation": np.asarray([5.0], dtype=np.float32),
                     "info": {"step_count": 273}},
                    {"observation": np.asarray([1.0], dtype=np.float32),
                     "info": {"step_count": 1}},
                ])

            def bind(self):
                pass

            def recv(self, timeout=None):
                return next(self.messages)

        env = Go2Env.__new__(Go2Env)
        env._action_tx = Sender()
        env._state_rx = Receiver()
        env.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(1,), dtype=np.float32)
        observation = env.reset()
        np.testing.assert_array_equal(observation, [1.0])
        self.assertEqual(env._action_tx.messages, [{"command": "clear"}])


if __name__ == "__main__":
    unittest.main()
