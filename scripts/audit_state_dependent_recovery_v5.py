#!/usr/bin/env python3
"""Irreversibly consume and evaluate the V5 Stage-A audit exactly once."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat

from safety_data.state_dependent_recovery_v5 import (
    PROTOCOL_PATH,
    SOURCE_SEEDS,
    consume_and_evaluate_state_dependent_audit,
    load_state_dependent_recovery_v5_protocol,
)
from scripts.collect_closed_loop_recovery_triage import _file_sha256
from scripts.collect_native_grouped_qsafe import (
    _prepare_staged_outputs,
    _publish_staged_outputs,
)
from scripts.merge_grouped_qsafe_shards import _clean_git_commit


_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-lock-sha256",
        required=True,
        help="Exact SHA-256 printed by the V5 discovery lock operation",
    )
    args = parser.parse_args()
    protocol = load_state_dependent_recovery_v5_protocol()
    collection = protocol["collection"]
    root = Path(os.path.abspath(_ROOT / str(collection["artifact_root"])))
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise RuntimeError("canonical V5 artifact root is missing") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(
            "canonical V5 artifact root must be a real directory")

    lock_path = root / str(collection["selection_lock_filename"])
    consumed_path = root / str(collection["audit_consumed_filename"])
    report_path = root / str(collection["triage_report_filename"])
    # These are only lexical Path objects.  Their final components must not be
    # stat'ed, hashed, opened, or resolved before the consumed marker exists.
    audit_paths = [
        root / str(collection["audit_shard_filename_template"]).format(
            source_seed=seed)
        for seed in SOURCE_SEEDS
    ]
    if os.path.lexists(os.fspath(report_path)):
        raise FileExistsError(
            "refusing to overwrite an existing V5 Stage-A report")
    if os.path.lexists(os.fspath(consumed_path)):
        raise RuntimeError("V5 audit has already been consumed or reserved")

    commit = _clean_git_commit()
    protocol_file_sha256 = _file_sha256(PROTOCOL_PATH)
    # Allocate a same-filesystem report staging inode before consuming audit;
    # successful evaluation publishes it last with no replacement.
    staged = _prepare_staged_outputs((report_path,))
    staging = staged[0][0]
    try:
        report = consume_and_evaluate_state_dependent_audit(
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
            raise RuntimeError("worktree changed during one-shot V5 audit")
        _publish_staged_outputs(staged)
    finally:
        staging.unlink(missing_ok=True)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
