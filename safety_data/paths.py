"""Fail-closed path checks for development-only Q_safe tooling."""

from __future__ import annotations

from pathlib import Path
import re


_PROTECTED_COMPONENT = re.compile(r"^(sealed|formal)", re.IGNORECASE)


class ProtectedEvidencePathError(PermissionError):
    """Raised before development tooling can open a protected evidence path."""


def _checked_paths(path: str | Path) -> tuple[Path, ...]:
    supplied = Path(path).expanduser()
    # Resolve existing symlinks as well as the lexical spelling.  For a new
    # output, strict=False still resolves every existing parent component.
    return supplied.absolute(), supplied.resolve(strict=False)


def assert_development_path(path: str | Path) -> Path:
    """Return a resolved path or reject protected Formal Test components.

    There is intentionally no ``allow_protected`` escape hatch in this
    package.  Confirmation data must be evaluated by a separately reviewed
    one-shot harness, not by development/audit commands.
    """
    for candidate in _checked_paths(path):
        for component in candidate.parts:
            if _PROTECTED_COMPONENT.match(component.casefold()):
                raise ProtectedEvidencePathError(
                    f"protected evidence path component {component!r} in {candidate}")
    return Path(path).expanduser().resolve(strict=False)
