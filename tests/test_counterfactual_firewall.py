from __future__ import annotations

import json
from pathlib import Path

import pytest

from safety_data.counterfactual_firewall import (
    ProtectedEvidenceError,
    assert_development_artifact,
    load_identity_denylist,
    reject_denied_identities,
)


def test_protected_paths_and_symlink_aliases_fail_before_read(tmp_path: Path) -> None:
    protected = tmp_path / "counterfactual-protected"
    protected.mkdir()
    target = protected / "outcomes.npz"
    target.write_bytes(b"must-not-open")
    alias = tmp_path / "alias"
    alias.symlink_to(protected, target_is_directory=True)
    with pytest.raises(ProtectedEvidenceError):
        assert_development_artifact(target)
    with pytest.raises(ProtectedEvidenceError):
        assert_development_artifact(alias / "outcomes.npz")


def test_round_one_identity_is_rejected(tmp_path: Path) -> None:
    identity = "a" * 64
    path = tmp_path / "denylist.json"
    path.write_text(json.dumps({
        "schema_version": "qsafe.counterfactual_identity_denylist.v1",
        "identities": [identity],
    }))
    denylist = load_identity_denylist(path)
    reject_denied_identities(["b" * 64], denylist)
    with pytest.raises(ProtectedEvidenceError):
        reject_denied_identities([identity.encode()], denylist)
