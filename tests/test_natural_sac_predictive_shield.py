import unittest

import numpy as np

from safety_data.natural_sac_predictive_shield import _rng


class NaturalSacPredictiveShieldTest(unittest.TestCase):
    def test_rng_stream_is_reproducible_and_separated(self):
        a = _rng(b"state", 1, 2, 3).normal(size=8)
        b = _rng(b"state", 1, 2, 3).normal(size=8)
        c = _rng(b"state", 1, 2, 4).normal(size=8)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))


if __name__ == "__main__":
    unittest.main()
