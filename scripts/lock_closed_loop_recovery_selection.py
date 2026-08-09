#!/usr/bin/env python3
"""Create the single v3 discovery selection lock on canonical artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from safety_data.closed_loop_recovery_triage import (
    create_selection_lock,
    validate_collection_readiness,
)
from safety_data.paths import (
    assert_development_path,
    require_v3_audit_consumed_or_safe_input,
)
from scripts.collect_closed_loop_recovery_triage import (
    _PROTOCOL_PATH,
    _file_sha256,
    _load_protocol,
)
from scripts.merge_grouped_qsafe_shards import _clean_git_commit


_ROOT = Path(__file__).resolve().parents[1]


def _canonical_paths(protocol: dict) -> dict[str, object]:
    collection = protocol["collection"]
    root = assert_development_path(
        require_v3_audit_consumed_or_safe_input(
            _ROOT / str(collection["artifact_root"])))
    seeds = list(map(
        int, protocol["triage_gates"]["data"]["required_source_seeds"]))
    return {
        "root": root,
        "admission": root / str(collection["admission_deployable_filename"]),
        "discovery": root / str(collection["discovery_filename"]),
        "reports": [
            root / str(collection[
                "collection_report_shard_filename_template"]).format(
                    source_seed=seed)
            for seed in seeds
        ],
        "selection_lock": root / str(collection["selection_lock_filename"]),
    }


def main() -> int:
    protocol = _load_protocol()
    paths = _canonical_paths(protocol)
    commit = _clean_git_commit()
    readiness = validate_collection_readiness(
        protocol=protocol,
        collection_report_paths=paths["reports"],
    )
    if readiness["generator_commit"] != commit:
        raise RuntimeError(
            "selection must run from the exact clean collection commit")
    if readiness["protocol_file_sha256"] != _file_sha256(_PROTOCOL_PATH):
        raise RuntimeError(
            "collection reports differ from the canonical protocol file")
    result = create_selection_lock(
        protocol=protocol,
        admission_path=paths["admission"],
        discovery_path=paths["discovery"],
        collection_report_paths=paths["reports"],
        selection_lock_path=paths["selection_lock"],
    )
    if _clean_git_commit() != commit:
        raise RuntimeError("worktree changed while creating the selection lock")
    print(json.dumps({
        "selection_lock": str(paths["selection_lock"]),
        "selection_lock_sha256": result["selection_lock_sha256"],
        "selected_global_candidate": result["selected_global_candidate"],
        "discovery_informativeness": result["data_gate"][
            "discovery_informativeness"],
        "data_gate_pass": result["data_gate"]["pass"],
        "audit_authorized": result["audit_authorized"],
        "merge_readiness_sha256": result["merge_readiness_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
