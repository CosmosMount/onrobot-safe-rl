from __future__ import annotations

import unittest

import numpy as np

from learner.checkpoint import agent_state_hash
from scripts.compose_qsafe_transfer_checkpoint import (
    compose_qsafe_transfer_payload,
)


def _empty_replay_state():
    return {
        'dataset_dict': {},
        'size': 0,
        'capacity': 10,
        'insert_index': 0,
    }


class P16QsafeTransferTest(unittest.TestCase):

    def test_only_qsafe_crosses_source_target_boundary(self):
        target_agent = {'actor': {'w': np.asarray([1.0, 2.0])}}
        source_agent = {'actor': {'w': np.asarray([9.0, 8.0])}}
        target_replay = _empty_replay_state()
        target_safety_replay = {'recent': {'items': []}}
        target = {
            'agent_state': target_agent,
            'replay_buffer_state': target_replay,
            'safety_replay_state': target_safety_replay,
            'step': 0,
            'metadata': {'seed': 42},
        }
        source_qsafe = {'critic': {'params': {'w': np.asarray([3.0])}}}
        source = {
            'agent_state': source_agent,
            'safety_critic_state': source_qsafe,
            'replay_buffer_state': {'forbidden': True},
            'step': 100,
        }
        payload = compose_qsafe_transfer_payload(
            target, source,
            target_checkpoint='/tmp/target.pkl',
            source_checkpoint='/tmp/source.pkl')

        self.assertIs(payload['agent_state'], target_agent)
        self.assertIs(payload['replay_buffer_state'], target_replay)
        self.assertIs(payload['safety_replay_state'], target_safety_replay)
        self.assertIs(payload['safety_critic_state'], source_qsafe)
        self.assertEqual(payload['step'], 0)
        metadata = payload['metadata']
        self.assertEqual(metadata['protocol'], 'P16')
        self.assertFalse(metadata['actor_transferred'])
        self.assertFalse(metadata['reward_critic_transferred'])
        self.assertFalse(metadata['reward_replay_transferred'])
        self.assertTrue(metadata['safety_critic_transferred'])
        self.assertEqual(
            metadata['target_initial_agent_hash'],
            agent_state_hash(target_agent))
        self.assertEqual(
            metadata['source_agent_hash'],
            agent_state_hash(source_agent))
        self.assertNotEqual(
            metadata['target_initial_agent_hash'],
            metadata['source_agent_hash'])

    def test_rejects_nonzero_target(self):
        target = {
            'agent_state': {'x': np.asarray([1])},
            'replay_buffer_state': {},
            'safety_replay_state': {},
            'step': 1,
        }
        source = {
            'agent_state': {'x': np.asarray([2])},
            'safety_critic_state': {},
        }
        with self.assertRaisesRegex(RuntimeError, 'step 0'):
            compose_qsafe_transfer_payload(
                target, source,
                target_checkpoint='target.pkl',
                source_checkpoint='source.pkl')


if __name__ == '__main__':
    unittest.main()
