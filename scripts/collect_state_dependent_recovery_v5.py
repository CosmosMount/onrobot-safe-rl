#!/usr/bin/env python3
"""Collect one preregistered V5 Stage-A source shard exactly once.

This entrypoint reuses the reviewed V3 simulator/behavior collector while
injecting the immutable V5 protocol identity, split names, and a new RNG hash
domain.  It never performs discovery analysis or starts model training.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sys
from typing import Iterator

from safety_data.closed_loop_recovery_collector import (
    ClosedLoopRecoveryCollectionConfig as _BaseCollectionConfig,
)
from safety_data.state_dependent_recovery_v5 import (
    COLLECTION_PROTOCOL_VERSION,
    DATASET_SPLIT_PREFIX,
    PROTOCOL_NAME,
    PROTOCOL_PATH,
    SEED_ALGORITHM,
    SEED_DOMAIN,
    SEED_ROLE_TAGS,
    SOURCE_SEEDS,
    load_state_dependent_recovery_v5_protocol,
)
import scripts.collect_closed_loop_recovery_triage as _v3_collector


def _load_protocol() -> dict:
    protocol = load_state_dependent_recovery_v5_protocol()
    policy_config = protocol.get("policy_config")
    if not isinstance(policy_config, dict):
        raise ValueError("V5 policy config contract is missing")
    raw_path = Path(str(policy_config.get("path", "")))
    config_path = (
        raw_path if raw_path.is_absolute()
        else PROTOCOL_PATH.parents[1] / raw_path
    )
    expected_config_sha256 = str(policy_config.get("config_sha256", ""))
    if _v3_collector._file_sha256(config_path) != expected_config_sha256:
        raise ValueError(
            "V5 policy config raw SHA-256 differs from the protocol")
    seeds = [
        int(seed)
        for policy in protocol["early_task_policies"]
        for seed in policy["source_seeds"]
    ]
    if seeds != list(SOURCE_SEEDS):
        raise ValueError("V5 source seed order drifted")
    return protocol


def _v5_collection_config(**kwargs: object) -> _BaseCollectionConfig:
    return _BaseCollectionConfig(
        **kwargs,
        seed_domain=SEED_DOMAIN,
        seed_role_tags=SEED_ROLE_TAGS,
        seed_algorithm=SEED_ALGORITHM,
        dataset_split_prefix=DATASET_SPLIT_PREFIX,
        collection_protocol_version=COLLECTION_PROTOCOL_VERSION,
        trajectory_id_prefix="state-dependent-recovery-v5-stage-a",
        explicit_filter_settings_in_action_contract=True,
    )


def _require_absent_v5_root_for_preflight(
    argv: list[str] | None = None,
) -> None:
    """Make a successful V5 preflight prove the canonical root was absent."""
    arguments = sys.argv[1:] if argv is None else argv
    if "--preflight-only" not in arguments:
        return
    protocol = _load_protocol()
    root = Path(os.path.abspath(
        PROTOCOL_PATH.parents[1] / str(protocol["collection"]["artifact_root"])
    ))
    if os.path.lexists(os.fspath(root)):
        raise RuntimeError(
            "V5 preflight requires the canonical artifact root to be absent")


@contextmanager
def _v5_entrypoint_binding() -> Iterator[None]:
    """Temporarily bind V5 constants without changing V3 default behavior."""
    names = {
        "_PROTOCOL_PATH": PROTOCOL_PATH,
        "_PROTOCOL_NAME": PROTOCOL_NAME,
        "_load_protocol": _load_protocol,
        "ClosedLoopRecoveryCollectionConfig": _v5_collection_config,
    }
    original = {name: getattr(_v3_collector, name) for name in names}
    try:
        for name, value in names.items():
            setattr(_v3_collector, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(_v3_collector, name, value)


def main() -> int:
    arguments = sys.argv[1:]
    _require_absent_v5_root_for_preflight(arguments)
    try:
        with _v5_entrypoint_binding():
            result = _v3_collector.main()
    finally:
        # A failing delegated preflight is not allowed to leave a V5 root
        # behind either.  Preserve the delegated exception when no root was
        # created, but fail explicitly on any side effect.
        _require_absent_v5_root_for_preflight(arguments)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
