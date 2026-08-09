#!/usr/bin/env python3
"""Collect one preregistered V4 Stage-A source shard exactly once.

This entrypoint reuses the reviewed V3 simulator/behavior collector while
injecting the immutable V4 protocol identity, split names, and a new RNG hash
domain.  It never performs discovery analysis or starts model training.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from safety_data.closed_loop_recovery_collector import (
    ClosedLoopRecoveryCollectionConfig as _BaseCollectionConfig,
)
from safety_data.state_dependent_recovery_v4 import (
    PROTOCOL_NAME,
    PROTOCOL_PATH,
    SEED_ALGORITHM,
    SEED_DOMAIN,
    SEED_ROLE_TAGS,
    SOURCE_SEEDS,
    load_state_dependent_recovery_v4_protocol,
)
import scripts.collect_closed_loop_recovery_triage as _v3_collector


def _load_protocol() -> dict:
    protocol = load_state_dependent_recovery_v4_protocol()
    policy_config = protocol.get("policy_config")
    if not isinstance(policy_config, dict):
        raise ValueError("V4 policy config contract is missing")
    raw_path = Path(str(policy_config.get("path", "")))
    config_path = (
        raw_path if raw_path.is_absolute()
        else PROTOCOL_PATH.parents[1] / raw_path
    )
    expected_config_sha256 = str(policy_config.get("config_sha256", ""))
    if _v3_collector._file_sha256(config_path) != expected_config_sha256:
        raise ValueError(
            "V4 policy config raw SHA-256 differs from the protocol")
    seeds = [
        int(seed)
        for policy in protocol["early_task_policies"]
        for seed in policy["source_seeds"]
    ]
    if seeds != list(SOURCE_SEEDS):
        raise ValueError("V4 source seed order drifted")
    return protocol


def _v4_collection_config(**kwargs: object) -> _BaseCollectionConfig:
    return _BaseCollectionConfig(
        **kwargs,
        seed_domain=SEED_DOMAIN,
        seed_role_tags=SEED_ROLE_TAGS,
        seed_algorithm=SEED_ALGORITHM,
        dataset_split_prefix="state_dependent_recovery_v4_stage_a",
        collection_protocol_version=(
            "qsafe.state_dependent_recovery.collection.v4_stage_a"),
        trajectory_id_prefix="state-dependent-recovery-v4-stage-a",
        explicit_filter_settings_in_action_contract=True,
    )


@contextmanager
def _v4_entrypoint_binding() -> Iterator[None]:
    """Temporarily bind V4 constants without changing V3 default behavior."""
    names = {
        "_PROTOCOL_PATH": PROTOCOL_PATH,
        "_PROTOCOL_NAME": PROTOCOL_NAME,
        "_load_protocol": _load_protocol,
        "ClosedLoopRecoveryCollectionConfig": _v4_collection_config,
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
    with _v4_entrypoint_binding():
        return _v3_collector.main()


if __name__ == "__main__":
    raise SystemExit(main())
