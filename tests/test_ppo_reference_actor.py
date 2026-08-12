from __future__ import annotations

import unittest

import numpy as np

from safety_data.ppo_reference_actor import sac_observation_to_ppo_actor_observation


class PpoReferenceActorTest(unittest.TestCase):
    def test_sac_mapping_has_locked_shape_and_command(self):
        observation = np.zeros((2, 46), np.float32)
        observation[:, 30] = 1.0
        value = sac_observation_to_ppo_actor_observation(
            observation, episode_step=np.asarray([0, 15]))
        self.assertEqual(value.shape, (2, 47))
        np.testing.assert_allclose(value[:, 6:9], [[0.3, 0, 0], [0.3, 0, 0]])
        np.testing.assert_allclose(value[0, 9:11], [0, 1], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
