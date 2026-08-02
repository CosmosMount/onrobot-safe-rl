from __future__ import annotations

import unittest
import uuid
import threading
import time

import runtime.inference.transport as transport

from runtime.inference.transport import (
    SharedMemoryQueueFull,
    SharedMemoryRingQueue,
)


class SharedMemoryRingQueueTest(unittest.TestCase):
    def setUp(self):
        self.key = f"test_ordered_transitions_{uuid.uuid4().hex}"
        self.producer = SharedMemoryRingQueue(
            self.key, capacity=4, slot_size=1024)
        self.producer.create()
        self.consumer = SharedMemoryRingQueue(
            self.key, capacity=4, slot_size=1024)
        self.consumer.open()

    def tearDown(self):
        self.consumer.close()
        self.producer.close(unlink=True)

    def test_preserves_order_and_terminal_message(self):
        messages = [
            {"step_id": 1, "done": False},
            {"step_id": 2, "done": True, "terminated": True},
            {"step_id": 3, "done": False},
        ]
        for message in messages:
            self.producer.write(message)
        self.assertEqual(self.consumer.depth(), 3)
        self.assertEqual(
            [self.consumer.read() for _ in messages], messages)
        self.assertIsNone(self.consumer.read())
        self.assertEqual(self.producer.depth(), 0)

    def test_full_queue_raises_instead_of_overwriting(self):
        for step_id in range(4):
            self.producer.write({"step_id": step_id})
        with self.assertRaises(SharedMemoryQueueFull):
            self.producer.write({"step_id": 4, "done": True})
        self.assertEqual(
            [self.consumer.read()["step_id"] for _ in range(4)],
            [0, 1, 2, 3],
        )

    def test_wraparound_remains_ordered(self):
        for step_id in range(12):
            self.producer.write({"step_id": step_id})
            self.assertEqual(self.consumer.read()["step_id"], step_id)

    def test_geometry_mismatch_is_rejected(self):
        incompatible = SharedMemoryRingQueue(
            self.key, capacity=8, slot_size=1024)
        with self.assertRaises(ValueError):
            incompatible.open()

    def test_transient_mixed_cursor_read_is_retried(self):
        self.producer.write({"value": 1})
        self.assertEqual(self.consumer.read(), {"value": 1})
        assert self.producer._shm is not None
        transport._store_u64(
            self.producer._shm.buf, transport._QUEUE_WRITE_OFFSET, 0)

        def restore():
            time.sleep(0.002)
            assert self.producer._shm is not None
            transport._store_u64(
                self.producer._shm.buf,
                transport._QUEUE_WRITE_OFFSET, 1)

        thread = threading.Thread(target=restore)
        thread.start()
        self.assertEqual(self.consumer.depth(), 0)
        thread.join()


if __name__ == "__main__":
    unittest.main()
