#!/usr/bin/env python3
"""Merge six locked v3 admission ledgers without opening D/A outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import yaml

from safety_data.closed_loop_recovery_collector import (
    AdmissionLedger,
    AdmissionPrivilegedView,
    canonical_protocol_sha256,
    merge_admission_ledgers,
    merge_admission_privileged_views,
)
from safety_data.paths import (
    assert_development_path,
    require_v3_audit_consumed_or_safe_input,
    workflow_evidence_read_scope,
)
from safety_data.closed_loop_recovery_triage import (
    validate_closed_loop_recovery_protocol,
    validate_collection_readiness,
)
from scripts.collect_native_grouped_qsafe import (
    _git_commit,
    _prepare_staged_outputs,
    _publish_staged_outputs,
    _sha256,
)


_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL_PATH = (
    _ROOT / "config" / "qsafe_closed_loop_recovery_triage_v3.yaml")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lexical_absolute(path: str | Path) -> Path:
    """Normalize a path without resolving or opening its final component."""
    return Path(os.path.abspath(os.fspath(path)))


def _admission_output_paths(protocol: dict, root: Path) -> tuple[Path, ...]:
    """Preserve canonical output names through the no-clobber probe."""
    lexical_outputs = (
        root / str(protocol["collection"]["admission_deployable_filename"]),
        root / str(protocol["collection"]["admission_privileged_filename"]),
        root / str(protocol["collection"]["admission_merge_report_filename"]),
    )
    existing = [
        path for path in lexical_outputs if os.path.lexists(os.fspath(path))]
    if existing:
        raise FileExistsError(f"refusing to overwrite admission merge: {existing}")
    outputs = tuple(
        assert_development_path(
            require_v3_audit_consumed_or_safe_input(path))
        for path in lexical_outputs)
    if outputs != lexical_outputs:
        raise ValueError("v3 admission outputs resolve outside artifact root")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("admission_shards", nargs=6)
    parser.add_argument("--privileged-shards", nargs=6, required=True)
    args = parser.parse_args()
    protocol_path = assert_development_path(
        require_v3_audit_consumed_or_safe_input(_PROTOCOL_PATH))
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol.get(
            "protocol_name") != "objective1_closed_loop_recovery_triage_v3":
        raise ValueError("wrong protocol for v3 admission merge")
    validate_closed_loop_recovery_protocol(protocol)
    protocol_sha256 = _file_sha256(protocol_path)
    protocol_contract_sha256 = canonical_protocol_sha256(protocol)
    lexical_root = _lexical_absolute(
        _ROOT / str(protocol["collection"]["artifact_root"]))
    root = assert_development_path(
        require_v3_audit_consumed_or_safe_input(lexical_root))
    if root != lexical_root:
        raise ValueError(
            "v3 admission artifact root must be a real unsymlinked directory")
    required_seeds = list(map(
        int, protocol["triage_gates"]["data"]["required_source_seeds"]))
    expected_admission = [
        _lexical_absolute(root / f"source-{seed}.admission.npz")
        for seed in required_seeds
    ]
    expected_privileged = [
        _lexical_absolute(root / f"source-{seed}.admission.privileged.npz")
        for seed in required_seeds
    ]
    expected_reports = [
        _lexical_absolute(root / str(protocol["collection"][
            "collection_report_shard_filename_template"]).format(
                source_seed=seed))
        for seed in required_seeds
    ]
    if list(map(_lexical_absolute, args.admission_shards)) != expected_admission:
        raise ValueError(
            "admission shards must be the six canonical source paths in "
            "preregistered order")
    if list(map(_lexical_absolute, args.privileged_shards)) != expected_privileged:
        raise ValueError(
            "privileged admission shards must be the six canonical source "
            "paths in preregistered order")
    for label, paths in (
        ("admission shard", expected_admission),
        ("privileged admission shard", expected_privileged),
        ("collection report", expected_reports),
    ):
        for path in paths:
            checked = require_v3_audit_consumed_or_safe_input(path)
            if checked.parent != root:
                raise ValueError(f"v3 {label} is outside artifact root")
    outputs = _admission_output_paths(protocol, root)
    readiness = validate_collection_readiness(
        protocol=protocol,
        collection_report_paths=expected_reports,
    )
    commit = _git_commit()
    if readiness["generator_commit"] != commit:
        raise ValueError(
            "v3 completion reports were generated by a different commit")
    ledgers = []
    for path in args.admission_shards:
        with workflow_evidence_read_scope(
                workflow=protocol["protocol_name"],
                role="admission",
                path=path):
            ledgers.append(AdmissionLedger.load(path))
    views = []
    for path, ledger in zip(args.privileged_shards, ledgers, strict=True):
        with workflow_evidence_read_scope(
                workflow=protocol["protocol_name"],
                role="admission_privileged",
                path=path):
            views.append(AdmissionPrivilegedView.load(path, ledger=ledger))
    for ordinal, (ledger, view, admission_commitment,
                  privileged_commitment) in enumerate(zip(
            ledgers,
            views,
            readiness["role_commitments"]["admission"],
            readiness["role_commitments"]["admission_privileged"],
            strict=True,
    )):
        if ledger.manifest["content_sha256"] != admission_commitment[
                "content_sha256"] or _sha256(ledger.path) != admission_commitment[
                    "file_sha256"]:
            raise ValueError(
                f"admission leaf {ordinal} differs from completion report")
        if view.manifest["content_sha256"] != privileged_commitment[
                "content_sha256"] or _sha256(view.path) != privileged_commitment[
                    "file_sha256"]:
            raise ValueError(
                "privileged admission leaf differs from completion report at "
                f"ordinal {ordinal}")
    observed_seeds = [int(ledger.manifest["source_seed"]) for ledger in ledgers]
    if observed_seeds != required_seeds:
        raise ValueError(
            "admission shards must follow the preregistered source-seed order")
    if any(ledger.manifest.get("protocol_sha256") != protocol_sha256
           for ledger in ledgers):
        raise ValueError("admission shard protocol hash differs from current v3")
    if any(ledger.manifest.get(
            "protocol_contract_sha256") != protocol_contract_sha256
           for ledger in ledgers):
        raise ValueError(
            "admission shard protocol semantics differ from current v3")
    if any(ledger.manifest.get("generator_commit") != commit
           for ledger in ledgers):
        raise ValueError("admission shard generator commit differs from merge commit")
    source_step = {
        int(seed): int(policy["training_step"])
        for policy in protocol["early_task_policies"]
        for seed in policy["source_seeds"]
    }
    groups_per_seed = int(protocol["collection"]["groups_per_source_seed"])
    max_proposals = int(protocol["collection"][
        "max_proposals_per_source_seed"])
    for ledger, seed in zip(ledgers, required_seeds, strict=True):
        validation = ledger.validate()
        accepted = np.asarray(ledger["accepted"], dtype=bool)
        if validation["accepted"] != groups_per_seed:
            raise ValueError(
                f"admission shard {seed} must contain exactly "
                f"{groups_per_seed} accepted groups")
        if validation["proposals"] > max_proposals:
            raise ValueError(f"admission shard {seed} exceeds proposal cap")
        if int(ledger.manifest["policy_training_step"]) != source_step[seed]:
            raise ValueError(
                f"admission shard {seed} has the wrong early-policy age")
        if len(np.unique(np.asarray(
                ledger["trajectory_id"])[accepted].astype(str))) != (
                    groups_per_seed):
            raise ValueError(
                f"admission shard {seed} reuses an accepted source trajectory")
    merged = merge_admission_ledgers(ledgers)
    merged_view = merge_admission_privileged_views(views, ledgers, merged)
    expected_groups = int(protocol["collection"]["total_groups"])
    if merged.validate()["accepted"] != expected_groups:
        raise ValueError("merged admission accepted count differs from protocol")

    staged = _prepare_staged_outputs(outputs)
    ledger_staging, privileged_staging, report_staging = (
        pair[0] for pair in staged)
    try:
        merged.save(ledger_staging)
        merged_view.save(privileged_staging, merged)
        with workflow_evidence_read_scope(
                workflow=protocol["protocol_name"],
                role="admission",
                path=ledger_staging):
            persisted = AdmissionLedger.load(ledger_staging)
        with workflow_evidence_read_scope(
                workflow=protocol["protocol_name"],
                role="admission_privileged",
                path=privileged_staging):
            persisted_view = AdmissionPrivilegedView.load(
                privileged_staging, ledger=persisted)
        report = {
            "schema_version": "qsafe.closed_loop_admission_merge_report.v3",
            "protocol_file_sha256": protocol_sha256,
            "protocol_contract_sha256": protocol_contract_sha256,
            "merge_commit": commit,
            "collection_readiness_sha256": readiness["readiness_sha256"],
            "source_seed_order": observed_seeds,
            "inputs": [
                {
                    "path": str(ledger.path),
                    "file_sha256": _sha256(ledger.path),
                    "content_sha256": ledger.manifest["content_sha256"],
                    "proposals": ledger.validate()["proposals"],
                    "accepted": ledger.validate()["accepted"],
                }
                for ledger in ledgers
            ],
            "output": str(outputs[0]),
            "output_file_sha256": _sha256(ledger_staging),
            "output_content_sha256": persisted.manifest["content_sha256"],
            "privileged_output": str(outputs[1]),
            "privileged_file_sha256": _sha256(privileged_staging),
            "privileged_content_sha256": persisted_view.manifest[
                "content_sha256"],
            "validation": persisted.validate(),
            "privileged_validation": persisted_view.validate(persisted),
            "candidate_outcomes_opened": False,
            "audit_opened": False,
            "model_training_authorized": False,
            "phase2_authorized": False,
        }
        report_staging.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        if _git_commit() != commit:
            raise RuntimeError("worktree changed during admission merge")
        _publish_staged_outputs(staged)
    finally:
        for staging, _ in staged:
            staging.unlink(missing_ok=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
