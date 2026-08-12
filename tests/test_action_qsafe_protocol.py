from __future__ import annotations

import unittest

from safety_data.action_qsafe_protocol import (
    action_qsafe_protocol_sha256,
    load_action_qsafe_protocol,
)


class ActionQsafeProtocolTest(unittest.TestCase):
    def test_superseded_protocol_is_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "identity is invalid"):
            load_action_qsafe_protocol()
        with self.assertRaisesRegex(ValueError, "identity is invalid"):
            action_qsafe_protocol_sha256()


if __name__ == "__main__":
    unittest.main()
