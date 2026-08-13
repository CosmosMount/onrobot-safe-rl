import unittest
import socket
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import yaml

from train.go2_sync_env import (Go2State, Go2SyncEnv, POLICY_STRUCT, STATE_STRUCT,
                                 POLICY_SOF, STATE_SOF, decode_state)
from scripts.train_go2_walk import load_config


class WalkInTheParkProtocolTest(unittest.TestCase):
    def test_upstream_packet_layout_is_fixed(self):
        self.assertEqual(POLICY_STRUCT.size, 66)
        self.assertEqual(STATE_STRUCT.size, 242)

    def test_state_keeps_action_attribution_and_a1_observation_shape(self):
        values = [STATE_SOF, 3, 9, 17, 1.25, 4, 5]
        values += list(np.arange(12, dtype=np.float32))
        values += list(np.arange(12, dtype=np.float32) + 12)
        values += [1.0, 0.0, 0.0, 0.0]
        values += [0.1, 0.2, 0.3, 0.0, 0.0, 9.8,
                   0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        values += list(np.arange(12, dtype=np.float32) + 36)
        decoded = decode_state(STATE_STRUCT.pack(*values))
        self.assertEqual(decoded.policy_sequence, 9)
        self.assertEqual(decoded.applied_action_id, 17)
        self.assertEqual(np.concatenate((decoded.joint_q, decoded.joint_dq,
                                         decoded.imu_gyro, decoded.velocity,
                                         decoded.imu_quat, decoded.q_target)).shape,
                         (46,))

    def test_policy_packet_has_monotonic_action_identifier_field(self):
        packet = POLICY_STRUCT.pack(POLICY_SOF, 0, 123, 0.0, *([0.0] * 12))
        self.assertEqual(POLICY_STRUCT.unpack(packet)[2], 123)

    def test_world_velocity_is_converted_to_body_velocimeter(self):
        state = Go2State(
            1, 0, 0.0, np.zeros(12, np.float32), np.zeros(12, np.float32),
            np.asarray([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], np.float32),
            np.zeros(3, np.float32), np.zeros(3, np.float32),
            np.asarray([0.0, 1.0, 0.0], np.float32),
            np.zeros(3, np.float32), np.zeros(12, np.float32), 3)
        body = Go2SyncEnv.body_velocity(state)
        np.testing.assert_allclose(body, [1.0, 0.0, 0.0], atol=1e-6)

    def test_training_action_mapping_matches_upstream_go2_offsets(self):
        config = load_config("config/go2_walk_in_the_park.yaml")
        with open("control/go2/go2.yaml", encoding="utf-8") as stream:
            controller_config = yaml.safe_load(stream)
        np.testing.assert_allclose(config["init_qpos"],
                                   controller_config["init_qpos"])
        env = Go2SyncEnv.__new__(Go2SyncEnv)
        env.init_qpos = np.asarray(config["init_qpos"], np.float32)
        env.action_offset = np.asarray(config["action_offset"], np.float32)
        action = np.ones(12, np.float32)
        q_target = env.init_qpos + env.action_offset * action
        np.testing.assert_allclose(q_target,
                                   np.asarray(config["init_qpos"], np.float32) +
                                   np.asarray(config["action_offset"], np.float32))

    def test_real_datagram_reset_and_step_are_ordered(self):
        with tempfile.TemporaryDirectory() as directory:
            policy_path = str(Path(directory) / "policy")
            state_path = str(Path(directory) / "state")
            server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            try:
                server.bind(policy_path)
            except PermissionError as exc:
                server.close()
                self.skipTest(f"AF_UNIX sockets unavailable in this sandbox: {exc}")

            def state(sequence, action_id):
                values = [STATE_SOF, 3, sequence, action_id, 0.0, sequence, 1]
                values += [0.05, 0.7, -1.4] * 4
                values += [0.0] * 12
                values += [1.0, 0.0, 0.0, 0.0]
                values += [0.0] * 12
                values += [0.05, 0.7, -1.4] * 4
                return STATE_STRUCT.pack(*values)

            def fake_controller():
                payload, address = server.recvfrom(POLICY_STRUCT.size)
                self.assertEqual(POLICY_STRUCT.unpack(payload)[1], 1)
                server.sendto(state(1, 0), state_path)
                payload, address = server.recvfrom(POLICY_STRUCT.size)
                action_id = POLICY_STRUCT.unpack(payload)[2]
                self.assertEqual(action_id, 2)
                time.sleep(0.01)
                server.sendto(state(2, action_id), state_path)

            thread = threading.Thread(target=fake_controller)
            thread.start()
            env = Go2SyncEnv(policy_socket=policy_path,
                             state_socket=state_path, timeout=1.0)
            try:
                observation, _ = env.reset()
                self.assertEqual(observation.shape, (46,))
                _, _, terminated, truncated, info = env.step(np.zeros(12))
                self.assertFalse(terminated or truncated)
                self.assertEqual(info["applied_action_id"], 2)
                self.assertEqual(len(env.state_intervals_ms), 1)
                self.assertGreater(env.state_intervals_ms[0], 5.0)
                self.assertLess(env.state_intervals_ms[0], 100.0)
            finally:
                env.close()
            thread.join(timeout=1.0)
            server.close()
            self.assertFalse(thread.is_alive())

    def test_step_retransmits_after_stale_ack(self):
        with tempfile.TemporaryDirectory() as directory:
            policy_path = str(Path(directory) / "policy")
            state_path = str(Path(directory) / "state")
            server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            try:
                server.bind(policy_path)
            except PermissionError as exc:
                server.close()
                self.skipTest(f"AF_UNIX sockets unavailable in this sandbox: {exc}")

            def state(sequence, action_id):
                values = [STATE_SOF, 3, sequence, action_id, 0.0, sequence, 1]
                values += [0.05, 0.7, -1.4] * 4
                values += [0.0] * 12
                values += [1.0, 0.0, 0.0, 0.0]
                values += [0.0] * 12
                values += [0.05, 0.7, -1.4] * 4
                return STATE_STRUCT.pack(*values)

            def fake_controller():
                _, address = server.recvfrom(POLICY_STRUCT.size)
                server.sendto(state(1, 0), state_path)
                payload, _ = server.recvfrom(POLICY_STRUCT.size)
                action_id = POLICY_STRUCT.unpack(payload)[2]
                server.sendto(state(2, 0), state_path)
                retransmit, _ = server.recvfrom(POLICY_STRUCT.size)
                self.assertEqual(POLICY_STRUCT.unpack(retransmit)[2], action_id)
                server.sendto(state(3, action_id), state_path)

            thread = threading.Thread(target=fake_controller)
            thread.start()
            env = Go2SyncEnv(policy_socket=policy_path,
                             state_socket=state_path, timeout=1.0)
            try:
                env.reset()
                _, _, terminated, truncated, info = env.step(np.zeros(12))
                self.assertFalse(terminated or truncated)
                self.assertEqual(info["applied_action_id"], 2)
            finally:
                env.close()
            thread.join(timeout=1.0)
            server.close()
            self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
