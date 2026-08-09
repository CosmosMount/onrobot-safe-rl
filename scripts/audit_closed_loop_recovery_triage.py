#!/usr/bin/env python3
"""Irreversibly consume and evaluate the locked v3 audit shards exactly once."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat

from safety_data.closed_loop_recovery_triage import consume_and_evaluate_audit
from safety_data.paths import (
    assert_development_path,
    require_v3_audit_consumed_or_safe_input,
)
from scripts.collect_closed_loop_recovery_triage import (
    _PROTOCOL_PATH,
    _file_sha256,
    _load_protocol,
)
from scripts.collect_native_grouped_qsafe import (
    _prepare_staged_outputs,
    _publish_staged_outputs,
)
from scripts.merge_grouped_qsafe_shards import _clean_git_commit


_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-lock-sha256", required=True,
        help="Exact SHA-256 printed by the one-shot discovery lock command",
    )
    args = parser.parse_args()
    protocol = _load_protocol()
    collection = protocol["collection"]
    # Preserve every canonical final component lexically.  Resolving an output
    # first would turn a dangling symlink at the report name into its missing
    # target and defeat the no-clobber check before irreversible consumption.
    root = Path(os.path.abspath(
        _ROOT / str(collection["artifact_root"])))
    checked_root = assert_development_path(
        require_v3_audit_consumed_or_safe_input(root))
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise RuntimeError("canonical v3 artifact root is missing") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(
            root_metadata.st_mode) or checked_root != root:
        raise RuntimeError(
            "canonical v3 artifact root must be a real unsymlinked directory")
    seeds = list(map(
        int, protocol["triage_gates"]["data"]["required_source_seeds"]))
    lock_path = root / str(collection["selection_lock_filename"])
    consumed_path = root / str(collection["audit_consumed_filename"])
    report_path = root / str(collection["triage_report_filename"])
    audit_paths = [
        root / str(collection["audit_shard_filename_template"]).format(
            source_seed=seed)
        for seed in seeds
    ]
    if os.path.lexists(os.fspath(report_path)):
        raise FileExistsError("refusing to overwrite an existing v3 triage report")
    if os.path.lexists(os.fspath(consumed_path)):
        raise RuntimeError("v3 audit has already been consumed or reserved")
    consumed_path = assert_development_path(
        require_v3_audit_consumed_or_safe_input(consumed_path))
    report_path = assert_development_path(
        require_v3_audit_consumed_or_safe_input(report_path))
    commit = _clean_git_commit()
    protocol_file_sha256 = _file_sha256(_PROTOCOL_PATH)

    # Allocate the same-filesystem staging inode before the irreversible audit
    # marker.  This catches a missing/unwritable report directory without
    # consuming the one-shot audit for a preventable publication failure.
    staged = _prepare_staged_outputs((report_path,))
    staging = staged[0][0]
    try:
        report = consume_and_evaluate_audit(
            protocol=protocol,
            selection_lock_path=lock_path,
            expected_selection_lock_sha256=args.selection_lock_sha256,
            audit_paths=audit_paths,
            audit_consumed_path=consumed_path,
            expected_generator_commit=commit,
            expected_protocol_file_sha256=protocol_file_sha256,
        )
        report["analysis_commit"] = commit
        report["analysis_worktree_clean"] = True
        staging.write_text(
            json.dumps(
                report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if _clean_git_commit() != commit:
            raise RuntimeError("worktree changed during one-shot v3 audit")
        _publish_staged_outputs(staged)
    finally:
        staging.unlink(missing_ok=True)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
