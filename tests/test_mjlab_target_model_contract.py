from __future__ import annotations

import unittest

from safety_data.mjlab_target_model_contract import _plain_name


class MjlabTargetModelContractTest(unittest.TestCase):
    def test_namespace_is_removed_only_from_mjlab_names(self):
        self.assertEqual(_plain_name("robot/FR_hip_joint"), "FR_hip_joint")
        self.assertEqual(_plain_name("FR_hip_joint"), "FR_hip_joint")
        self.assertIsNone(_plain_name(None))


if __name__ == "__main__":
    unittest.main()
