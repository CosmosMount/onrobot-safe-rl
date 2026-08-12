"""One-way evidence firewall for the counterfactual Q_safe experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


PROTECTED_COMPONENTS = frozenset({
    "protected", "protected-outcomes", "round-one-protected",
    "counterfactual-protected",
})


class ProtectedEvidenceError(RuntimeError):
    """Raised before development code can inspect protected evidence."""


def _components(path: Path) -> set[str]:
    return {part.casefold() for part in path.parts}


def assert_development_artifact(path: str | Path) -> Path:
    """Reject lexical and resolved aliases of every protected directory."""
    candidate = Path(path)
    for inspected in (candidate.absolute(), candidate.resolve(strict=False)):
        if _components(inspected) & PROTECTED_COMPONENTS:
            raise ProtectedEvidenceError(
                f"development tooling cannot access protected evidence: {path}")
    return candidate


def load_identity_denylist(path: str | Path) -> frozenset[str]:
    candidate = assert_development_artifact(path)
    payload = json.loads(candidate.read_text())
    if payload.get("schema_version") != "qsafe.counterfactual_identity_denylist.v1":
        raise ValueError("unsupported counterfactual identity denylist")
    identities = payload.get("identities")
    if not isinstance(identities, list) or len(identities) != len(set(identities)):
        raise ValueError("counterfactual identity denylist is malformed")
    if any(not isinstance(identity, str) or len(identity) != 64
           for identity in identities):
        raise ValueError("counterfactual identity denylist contains invalid identity")
    return frozenset(identities)


def reject_denied_identities(
    identities: Iterable[str | bytes], denylist: frozenset[str],
) -> None:
    normalized = {
        value.decode("ascii") if isinstance(value, bytes) else str(value)
        for value in identities
    }
    overlap = normalized & denylist
    if overlap:
        example = min(overlap)
        raise ProtectedEvidenceError(
            f"development cohort contains sealed round-one identity {example}")

