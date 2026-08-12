import unittest

import numpy as np

from safety_data.natural_sac_calibration import (
    assert_roles_disjoint,
    finite_sample_upper_residual,
    fit_member_affine_calibration,
    NaturalSacRole,
)


class NaturalSacCalibrationTest(unittest.TestCase):
    def test_temperature_fit_reduces_overconfidence(self):
        label = np.asarray([0, 0, 0, 1, 1, 1], dtype=bool)
        base = np.asarray([-8.0, -4.0, 2.0, -2.0, 4.0, 8.0])
        logits = np.stack([base] * 5)
        temperatures, biases = fit_member_affine_calibration(logits, label)
        self.assertEqual(temperatures.shape, (5,))
        self.assertEqual(biases.shape, (5,))
        self.assertTrue(np.all(temperatures > 1.0))

    def test_finite_sample_quantile_uses_ceiling_rank(self):
        residual = np.arange(19, dtype=np.float64)
        self.assertEqual(finite_sample_upper_residual(residual, alpha=0.10), 17.0)
        residual = np.arange(20, dtype=np.float64)
        self.assertEqual(finite_sample_upper_residual(residual, alpha=0.10), 18.0)

    def test_roles_must_be_identity_disjoint(self):
        def role(name, identity):
            return NaturalSacRole(
                name=name,
                observation_history=np.zeros((1, 5, 46), dtype=np.float32),
                label=np.zeros(1, dtype=bool), source_seed=np.zeros(1),
                episode_id=np.zeros(1), identities=np.asarray([identity], dtype="S64"),
                input_files=(),
            )
        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_roles_disjoint(role("a", b"same"), role("b", b"same"))


if __name__ == "__main__":
    unittest.main()
