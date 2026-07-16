"""Config helpers for the P2 layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (isinstance(value, dict) and isinstance(merged.get(key), dict)):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_layered_config(*paths: str | Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in paths:
        out = deep_merge(out, load_yaml(path))
    return out
