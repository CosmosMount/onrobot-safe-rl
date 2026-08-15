from __future__ import annotations

import json
from pathlib import Path

import pytest

from reproductions.ppo_sqrl_go2.protocol import (
    Protocol, create_protocol_lock, reserve_output_directory,
    target_branch_order, verify_protocol_lock,
)


def test_fixed_geometry_and_branch_order():
    protocol = Protocol()
    assert protocol.pretrain_iterations == 150
    assert protocol.target_iterations == 40
    assert target_branch_order(10) == ("ppo_transfer", "ppo_safe")
    assert target_branch_order(11) == ("ppo_safe", "ppo_transfer")


def test_lock_detects_hash_drift_and_refuses_overwrite(tmp_path: Path):
    source = tmp_path / "source.py"
    source.write_text("first", encoding="utf-8")
    lock = tmp_path / "lock.json"
    create_protocol_lock(lock, protocol_id="test", files=[source])
    assert verify_protocol_lock(lock)["protocol_id"] == "test"
    with pytest.raises(FileExistsError):
        create_protocol_lock(lock, protocol_id="test", files=[source])
    source.write_text("second", encoding="utf-8")
    with pytest.raises(ValueError, match="locked file changed"):
        verify_protocol_lock(lock)


def test_lock_detects_external_hash_drift(tmp_path: Path):
    source = tmp_path / "source.py"
    external = tmp_path / "model.xml"
    source.write_text("source", encoding="utf-8")
    external.write_text("model-v1", encoding="utf-8")
    from reproductions.ppo_sqrl_go2.protocol import sha256_file
    lock = tmp_path / "lock.json"
    create_protocol_lock(
        lock, protocol_id="external", files=[source],
        external_hashes={str(external): sha256_file(external)})
    external.write_text("model-v2", encoding="utf-8")
    with pytest.raises(ValueError, match="external file changed"):
        verify_protocol_lock(lock)


def test_output_reservation_is_no_clobber(tmp_path: Path):
    reserve_output_directory(tmp_path / "run")
    with pytest.raises(FileExistsError):
        reserve_output_directory(tmp_path / "run")
