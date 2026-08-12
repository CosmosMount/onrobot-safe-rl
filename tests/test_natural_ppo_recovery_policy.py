import unittest

import numpy as np

from safety_data.natural_ppo_recovery_policy import NaturalPpoRecoveryPolicy


class NaturalPpoRecoveryPolicyTest(unittest.TestCase):
    def test_final_production_checkpoint_loads_and_is_finite(self):
        policy = NaturalPpoRecoveryPolicy(
            "saved/qsafe_development/natural_ppo/production-30m-seed137-v1/model_119.pt")
        self.assertEqual(policy.iteration, 119)
        self.assertEqual(len(policy.checkpoint_sha256), 64)
        observation = np.zeros(47, dtype=np.float32)
        normalized = (observation - policy.mean) / (policy.std + 1e-2)
        import torch
        with torch.inference_mode():
            action = policy.model(torch.from_numpy(normalized)).numpy()
        self.assertEqual(action.shape, (12,))
        self.assertTrue(np.all(np.isfinite(action)))


if __name__ == "__main__":
    unittest.main()
