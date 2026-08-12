from __future__ import annotations

import numpy as np
import pytest

from safety_data.counterfactual_replica_extension import (
    INVERSE_PERMUTATION, PERMUTATION, reconstruct_frozen_candidates,
    verify_frozen_candidate_identity,
)


def test_frozen_candidates_only_invert_joint_order() -> None:
    requested_internal = np.arange(16 * 12, dtype=np.float32).reshape(16, 12)
    absolute_internal = requested_internal + 1000
    frozen = reconstruct_frozen_candidates(
        requested_internal[:, PERMUTATION], absolute_internal[:, PERMUTATION])
    np.testing.assert_array_equal(frozen["raw_internal"], requested_internal)
    np.testing.assert_array_equal(frozen["absolute_internal"], absolute_internal)
    np.testing.assert_array_equal(
        frozen["critic_action"][:, INVERSE_PERMUTATION], absolute_internal)


def test_candidate_identity_or_physical_action_change_fails() -> None:
    identity = np.asarray([[b"a", b"b"]])
    action = np.zeros((1, 2, 12), np.float32)
    verify_frozen_candidate_identity(identity, identity.copy(), action, action.copy())
    changed = action.copy(); changed[0, 0, 0] = 1
    with pytest.raises(RuntimeError, match="physical candidate"):
        verify_frozen_candidate_identity(identity, identity, action, changed)
    with pytest.raises(RuntimeError, match="candidate identity"):
        verify_frozen_candidate_identity(identity, np.asarray([[b"a", b"c"]]),
                                         action, action)
