from __future__ import annotations

import unittest
import uuid

import numpy as np

from runtime.inference.transport import (
    SharedMemoryReceiver,
    SharedMemoryRingQueue,
)
from train.ordered_runtime import OrderedRuntimeChannel, RuntimeProtocolError


class OrderedRuntimeProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        suffix = uuid.uuid4().hex
        self.action_key = f"test-action-{suffix}"
        self.state_key = f"test-state-{suffix}"
        self.action_rx = SharedMemoryReceiver(self.action_key)
        self.action_rx.bind()
        self.state_tx = SharedMemoryRingQueue(
            f"{self.state_key}.ordered", capacity=1024, slot_size=4096)
        self.state_tx.create()
        self.channel = OrderedRuntimeChannel(
            self.action_key, self.state_key, capacity=1024, slot_size=4096)
        self.channel.connect(timeout=1.0)

    def tearDown(self) -> None:
        self.channel.close()
        self.action_rx.close(unlink=True)
        self.state_tx.close(unlink=True)

    @staticmethod
    def message(step: int, *, action_id: int, done: bool = False) -> dict:
        return {
            "observation": np.asarray([step], dtype=np.float32),
            "reward": 0.0,
            "done": done,
            "info": {
                "runtime_step_id": step,
                "episode_id": 1,
                "episode_step": step,
                "applied_action_id": action_id,
                "terminated": done,
                "truncated": False,
            },
        }

    def test_action_id_is_transmitted_and_state_is_attributed(self) -> None:
        action_id = self.channel.send_action(np.zeros(12, dtype=np.float32))
        command = self.action_rx.recv(timeout=1.0)
        self.assertEqual(command["action_id"], action_id)
        self.state_tx.write(self.message(1, action_id=action_id))
        received = self.channel.recv(timeout=1.0)
        self.assertEqual(received.runtime_step_id, 1)
        self.assertEqual(received.applied_action_id, action_id)

    def test_complete_500_step_episode_retains_terminal(self) -> None:
        action_id = self.channel.send_action(np.zeros(12, dtype=np.float32))
        # Model a learner that is stalled for the entire episode: all states
        # are queued first, then consumed. None may be overwritten.
        for step in range(1, 501):
            self.state_tx.write(self.message(
                step, action_id=action_id, done=step == 500))
        received = [self.channel.recv(timeout=1.0) for _ in range(500)]
        self.assertEqual([item.runtime_step_id for item in received],
                         list(range(1, 501)))
        self.assertFalse(any(item.done for item in received[:-1]))
        self.assertTrue(received[-1].done)
        self.assertEqual(received[-1].episode_step, 500)
        self.assertEqual(self.channel.queue_depth, 0)

    def test_runtime_step_gap_fails_closed(self) -> None:
        action_id = self.channel.send_action(np.zeros(12, dtype=np.float32))
        self.state_tx.write(self.message(1, action_id=action_id))
        self.state_tx.write(self.message(3, action_id=action_id))
        self.channel.recv(timeout=1.0)
        with self.assertRaisesRegex(RuntimeProtocolError, "gap/reordering"):
            self.channel.recv(timeout=1.0)

    def test_unknown_future_action_fails_closed(self) -> None:
        self.channel.send_action(np.zeros(12, dtype=np.float32))
        self.state_tx.write(self.message(1, action_id=9))
        with self.assertRaisesRegex(RuntimeProtocolError, "not sent"):
            self.channel.recv(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
