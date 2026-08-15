from __future__ import annotations

import pytest

from reproductions.ppo_sqrl_go2.checkpoint import verify_pretrain_lineage


def test_seed_specific_pretrain_lineage():
    payload = {"metadata": {
        "phase": "pretrain", "seed": 12,
        "actor_sha256": "actor", "safety_sha256": "safety"}}
    verify_pretrain_lineage(
        payload, seed=12, actor_hash="actor", safety_hash="safety")
    with pytest.raises(ValueError, match="seed-specific"):
        verify_pretrain_lineage(payload, seed=13)
