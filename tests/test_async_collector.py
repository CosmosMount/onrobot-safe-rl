from __future__ import annotations

import unittest
import queue
import threading
import time
from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from rl.agents import create_agent
from rl.agents.paper_sqrl.inference import SQRLActionDecision
from rl.agents.paper_sqrl.inference import export_inference_weights
from runtime.inference.transport import SharedMemoryReceiver, SharedMemoryRingQueue
from train.async_collector import PaperSQRLCollectorCore, run_async_collector
from train.async_loop import _manifest_lineage
from train.config import load_app_config
from train.ordered_runtime import RuntimeEnvelope


class AsyncManifestLineageTest(unittest.TestCase):
    def test_resume_preserves_origin_hashes_and_records_resume_hashes(self):
        existing = {
            "initial_actor_hash": "origin-actor",
            "initial_reward_critic_hash": "origin-critic",
            "initial_safety_critic_hash": "origin-safety",
        }
        current = {
            "actor_hash": "resume-actor",
            "reward_critic_hash": "resume-critic",
            "safety_critic_hash": "resume-safety",
        }
        lineage = _manifest_lineage(existing, current, 25000)
        self.assertEqual(lineage["initial_actor_hash"], "origin-actor")
        self.assertEqual(lineage["resume_actor_hash"], "resume-actor")

    def test_fresh_run_uses_current_hashes_as_origin(self):
        current = {
            "actor_hash": "actor",
            "reward_critic_hash": "critic",
            "safety_critic_hash": "safety",
        }
        lineage = _manifest_lineage({}, current, 0)
        self.assertEqual(lineage["initial_actor_hash"], "actor")
        self.assertNotIn("resume_actor_hash", lineage)


class FakePolicy:
    def __init__(self):
        self.phase = "task"
        self.observed = []

    def decide(self, observation, *, training, nominal=None):
        action = (np.asarray(nominal, dtype=np.float32) if nominal is not None
                  else np.asarray([0.25], dtype=np.float32))
        return SQRLActionDecision(
            action, self.phase, self.phase == "safety", False, False,
            1.0, None, None)

    def observe_transition(self, **kwargs):
        self.observed.append(kwargs)
        return {"safety_trajectory_complete": False}


def envelope(step, action_id, *, policy_step=True, terminated=False):
    return RuntimeEnvelope(
        runtime_step_id=step,
        episode_id=1,
        episode_step=step,
        applied_action_id=action_id,
        observation=np.asarray([float(step), 0.0], dtype=np.float32),
        reward=1.0,
        done=terminated,
        info={
            "policy_step": policy_step,
            "terminated": terminated,
            "truncated": False,
            "fallen": terminated,
        },
    )


class AsyncCollectorCoreTest(unittest.TestCase):
    def setUp(self):
        self.policy = FakePolicy()
        self.core = PaperSQRLCollectorCore(
            self.policy,
            gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            start_training=1, explore_action_scale=0.5, seed=42)

    def test_transition_uses_observation_and_action_that_were_sent(self):
        observation = np.asarray([0.0, 0.0], dtype=np.float32)
        decision = self.core.next_decision(observation)
        self.core.remember_action(7, observation, decision)
        item = self.core.accept_state(envelope(9, 7))
        self.assertIsNotNone(item)
        transition = item["transition"]
        np.testing.assert_array_equal(
            transition["observation"], observation[None, ...])
        np.testing.assert_array_equal(
            transition["action"], decision.action[None, ...])
        np.testing.assert_array_equal(
            transition["next_observation"], [[9.0, 0.0]])
        self.assertEqual(int(transition["applied_action_id"][0]), 7)

    def test_supervisor_tick_is_not_inserted_into_policy_replay(self):
        item = self.core.accept_state(
            envelope(1, -1, policy_step=False))
        self.assertIsNone(item)
        self.assertEqual(self.core.policy_steps, 0)
        self.assertEqual(self.policy.observed, [])

    def test_missing_action_attribution_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "missing"):
            self.core.accept_state(envelope(1, 3))

    def test_terminal_is_emitted_once_and_labeled_unsafe(self):
        decision = self.core.next_decision(np.zeros(2, dtype=np.float32))
        self.core.remember_action(0, np.zeros(2, dtype=np.float32), decision)
        item = self.core.accept_state(envelope(1, 0, terminated=True))
        self.assertEqual(float(item["transition"]["terminated"][0]), 1.0)
        self.assertEqual(float(item["transition"]["unsafe_label"][0]), 1.0)
        self.assertEqual(len(self.policy.observed), 1)

    def test_repeated_runtime_action_is_attributed_to_successive_states(self):
        decision = self.core.next_decision(np.zeros(2, dtype=np.float32))
        self.core.remember_action(4, np.zeros(2, dtype=np.float32), decision)
        first = self.core.accept_state(envelope(1, 4))
        second = self.core.accept_state(envelope(2, 4))
        np.testing.assert_array_equal(
            first["transition"]["observation"], [[0.0, 0.0]])
        np.testing.assert_array_equal(
            second["transition"]["observation"], [[1.0, 0.0]])
        self.assertEqual(self.core.repeated_action_steps, 1)


class AsyncCollectorTransportIntegrationTest(unittest.TestCase):
    def test_slow_learner_does_not_change_order_or_terminal_boundary(self):
        suffix = str(time.time_ns())
        action_key = f"async-action-{suffix}"
        state_key = f"async-state-{suffix}"
        action_rx = SharedMemoryReceiver(action_key)
        action_rx.bind()
        state_tx = SharedMemoryRingQueue(
            f"{state_key}.ordered", capacity=2048, slot_size=16 * 1024)
        state_tx.create()
        _, train_cfg, cfg = load_app_config(
            path="config/go2_50hz_sqrl_paper_pretrain.yaml")
        cfg.device_type = "cpu"
        cfg.buffer_device_type = "cpu"
        cfg.hidden_dims = [16, 16]
        cfg.safety_hidden_dims = [16, 16]
        cfg.num_qs = 2
        cfg.num_min_qs = 1
        cfg.buffer_max_length = 100
        cfg.buffer_min_length = 1
        cfg.sample_batch_size = 2
        cfg.safety_num_candidates = 4
        cfg.safety_boundary_pool_multiplier = 2
        train_cfg.start_training = 0
        train_cfg.explore_action_scale = 0.5
        robot = SimpleNamespace(
            runtime_action_shm=action_key, runtime_state_shm=state_key)
        obs_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(2,), dtype=np.float32)
        action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(1,), dtype=np.float32)
        agent = create_agent(obs_space, action_space, {}, cfg)
        transitions = queue.Queue()
        weights = queue.Queue()
        controls = queue.Queue()
        weights.put(export_inference_weights(agent, version=0))
        thread = threading.Thread(
            target=run_async_collector,
            kwargs={
                "robot_cfg": robot,
                "agent_cfg": cfg,
                "train_cfg": train_cfg,
                "observation_dim": 2,
                "action_dim": 1,
                "action_space": action_space,
                "transition_queue": transitions,
                "weight_queue": weights,
                "control_queue": controls,
            }, daemon=True)
        thread.start()

        def write(step, action_id=-1, *, episode_id=1,
                  policy=False, truncated=False):
            state_tx.write({
                "observation": np.asarray([step, 0], dtype=np.float32),
                "reward": 1.0 if policy else 0.0,
                "done": truncated,
                "info": {
                    "runtime_step_id": step,
                    "episode_id": episode_id,
                    "episode_step": step - 2 if policy else 0,
                    "applied_action_id": action_id,
                    "policy_step": policy,
                    "terminated": False,
                    "truncated": truncated,
                },
            })

        try:
            write(1)
            clear = action_rx.recv(timeout=1.0)
            self.assertEqual(clear, {"command": "clear"})
            write(2, episode_id=2)
            for step in range(3, 503):
                command = action_rx.recv(timeout=1.0)
                write(step, int(command["action_id"]), policy=True,
                      episode_id=2, truncated=step == 502)
                # The consumer intentionally does not read its transition
                # queue here, modelling a stalled learner for 500 ticks.
            items = [transitions.get(timeout=1.0) for _ in range(500)]
            ids = [int(item["transition"]["runtime_step_id"][0])
                   for item in items]
            self.assertEqual(ids, list(range(3, 503)))
            self.assertTrue(bool(items[-1]["transition"]["truncated"][0]))
            self.assertEqual(
                int(items[-1]["transition"]["episode_step"][0]), 500)
        finally:
            controls.put("stop")
            write(503, episode_id=2)
            thread.join(timeout=2.0)
            action_rx.close(unlink=True)
            state_tx.close(unlink=True)
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
