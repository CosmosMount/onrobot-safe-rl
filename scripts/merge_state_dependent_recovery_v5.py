#!/usr/bin/env python3
"""Merge or discovery-lock canonical V5 Stage-A artifacts.

Operations are deliberately path-free: every input and output is derived from
the immutable protocol.  ``admission`` never opens candidate outcomes;
``discovery`` never opens audit outcomes; ``lock`` creates the one selection
lock only after both report-last merges pass.  ``resume-denied-report`` closes
the report-last crash window from an already hash-bound audit-denied lock and
has no audit-path input.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from safety_data.closed_loop_recovery_collector import (
    AdmissionLedger,
    AdmissionPrivilegedView,
    canonical_protocol_sha256,
    merge_admission_ledgers,
    merge_admission_privileged_views,
)
from safety_data.closed_loop_recovery_triage import _artifact_path
from safety_data.merge import merge_grouped_shards, merge_privileged_shards
from safety_data.paths import workflow_evidence_read_scope
from safety_data.schema import GroupedBranchDataset, PrivilegedBranchView
from safety_data.state_dependent_recovery_v5 import (
    PROTOCOL_NAME,
    PROTOCOL_PATH,
    SOURCE_SEEDS,
    create_state_dependent_selection_lock,
    load_state_dependent_recovery_v5_protocol,
    resume_state_dependent_discovery_failure_report,
    validate_v5_outcome_manifest,
    validate_state_dependent_collection_readiness,
)
from scripts.collect_closed_loop_recovery_triage import _file_sha256
from scripts.merge_grouped_qsafe_shards import (
    _clean_git_commit,
    _publish_no_clobber,
    _sha256,
    _staging_path,
    _v3_exact_discovery_gate,
)


_ROOT = Path(__file__).resolve().parents[1]


def _paths(protocol: dict[str, Any]) -> dict[str, Any]:
    collection = protocol["collection"]
    root = Path(os.path.abspath(_ROOT / str(collection["artifact_root"])))

    def leaves(key: str) -> list[Path]:
        template = str(collection[key])
        return [root / template.format(source_seed=seed) for seed in SOURCE_SEEDS]

    result = {
        "root": root,
        "reports": leaves("collection_report_shard_filename_template"),
        "admission_leaves": leaves("admission_shard_filename_template"),
        "admission_privileged_leaves": leaves(
            "admission_privileged_shard_filename_template"),
        "discovery_leaves": leaves("discovery_shard_filename_template"),
        "discovery_privileged_leaves": leaves(
            "discovery_privileged_shard_filename_template"),
    }
    for name, key in (
        ("admission", "admission_deployable_filename"),
        ("admission_privileged", "admission_privileged_filename"),
        ("admission_report", "admission_merge_report_filename"),
        ("discovery", "discovery_filename"),
        ("discovery_privileged", "discovery_privileged_filename"),
        ("discovery_report", "discovery_merge_report_filename"),
        ("selection_lock", "selection_lock_filename"),
    ):
        result[name] = root / str(collection[key])
    return result


def _canonical_non_audit(
    path: Path,
    *,
    protocol: dict[str, Any],
    filename: str,
    name: str,
) -> Path:
    return _artifact_path(
        path,
        protocol=protocol,
        expected_filename=filename,
        name=name,
    )


def _require_free(outputs: list[Path], operation: str) -> None:
    occupied = [path for path in outputs if os.path.lexists(os.fspath(path))]
    if occupied:
        raise FileExistsError(
            f"refusing to overwrite V5 {operation} outputs: {occupied}")


def _merge_admission(
    *,
    protocol: dict[str, Any],
    paths: dict[str, Any],
    readiness: dict[str, Any],
    commit: str,
) -> dict[str, Any]:
    outputs = [
        _canonical_non_audit(
            paths["admission"], protocol=protocol,
            filename=protocol["collection"]["admission_deployable_filename"],
            name="merged admission output"),
        _canonical_non_audit(
            paths["admission_privileged"], protocol=protocol,
            filename=protocol["collection"]["admission_privileged_filename"],
            name="merged admission privileged output"),
        _canonical_non_audit(
            paths["admission_report"], protocol=protocol,
            filename=protocol["collection"]["admission_merge_report_filename"],
            name="admission merge report"),
    ]
    _require_free(outputs, "admission merge")
    commitments = readiness["role_commitments"]
    ledgers: list[AdmissionLedger] = []
    views: list[AdmissionPrivilegedView] = []
    for ordinal, (seed, path, privileged_path) in enumerate(zip(
            SOURCE_SEEDS,
            paths["admission_leaves"],
            paths["admission_privileged_leaves"],
            strict=True,
    )):
        admission_commitment = commitments["admission"][ordinal]
        privileged_commitment = commitments["admission_privileged"][ordinal]
        if Path(admission_commitment["path"]) != path or Path(
                privileged_commitment["path"]) != privileged_path:
            raise ValueError("admission readiness path order drifted")
        with workflow_evidence_read_scope(
                workflow=PROTOCOL_NAME,
                role="admission",
                path=path):
            ledger = AdmissionLedger.load(path)
        with workflow_evidence_read_scope(
                workflow=PROTOCOL_NAME,
                role="admission_privileged",
                path=privileged_path):
            view = AdmissionPrivilegedView.load(
                privileged_path, ledger=ledger)
        if _sha256(path) != admission_commitment["file_sha256"] or (
                ledger.manifest["content_sha256"] !=
                admission_commitment["content_sha256"]):
            raise ValueError(f"admission leaf {seed} differs from readiness")
        if _sha256(privileged_path) != privileged_commitment[
                "file_sha256"] or view.manifest["content_sha256"] != (
                    privileged_commitment["content_sha256"]):
            raise ValueError(
                f"admission privileged leaf {seed} differs from readiness")
        if ledger.manifest.get("source_seed") != seed or ledger.manifest.get(
                "generator_commit") != commit:
            raise ValueError("admission leaf source/commit drifted")
        ledgers.append(ledger)
        views.append(view)
    merged = merge_admission_ledgers(ledgers)
    merged_view = merge_admission_privileged_views(views, ledgers, merged)
    if merged.validate()["accepted"] != 384:
        raise ValueError("merged V5 admission must contain 384 accepted groups")

    staged = [_staging_path(path) for path in outputs]
    try:
        merged.save(staged[0])
        merged_view.save(staged[1], merged)
        with workflow_evidence_read_scope(
                workflow=PROTOCOL_NAME,
                role="admission",
                path=staged[0]):
            persisted = AdmissionLedger.load(staged[0])
        with workflow_evidence_read_scope(
                workflow=PROTOCOL_NAME,
                role="admission_privileged",
                path=staged[1]):
            persisted_view = AdmissionPrivilegedView.load(
                staged[1], ledger=persisted)
        report = {
            "schema_version": "qsafe.closed_loop_admission_merge_report.v3",
            "protocol_file_sha256": _file_sha256(PROTOCOL_PATH),
            "protocol_contract_sha256": canonical_protocol_sha256(protocol),
            "merge_commit": commit,
            "collection_readiness_sha256": readiness["readiness_sha256"],
            "source_seed_order": list(SOURCE_SEEDS),
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
            "output_file_sha256": _sha256(staged[0]),
            "output_content_sha256": persisted.manifest["content_sha256"],
            "privileged_output": str(outputs[1]),
            "privileged_file_sha256": _sha256(staged[1]),
            "privileged_content_sha256": persisted_view.manifest[
                "content_sha256"],
            "validation": persisted.validate(),
            "privileged_validation": persisted_view.validate(persisted),
            "candidate_outcomes_opened": False,
            "audit_opened": False,
            "model_training_authorized": False,
            "phase2_authorized": False,
        }
        staged[2].write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if _clean_git_commit() != commit:
            raise RuntimeError("worktree changed during V5 admission merge")
        _publish_no_clobber(list(zip(staged, outputs, strict=True)))
    finally:
        for path in staged:
            path.unlink(missing_ok=True)
    return report


def _require_exact_v5_discovery_rng_split(
    manifest: dict[str, Any],
) -> None:
    validate_v5_outcome_manifest(manifest, "discovery")


def _merge_discovery(
    *,
    protocol: dict[str, Any],
    paths: dict[str, Any],
    readiness: dict[str, Any],
    commit: str,
) -> dict[str, Any]:
    outputs = [
        _canonical_non_audit(
            paths["discovery"], protocol=protocol,
            filename=protocol["collection"]["discovery_filename"],
            name="merged discovery output"),
        _canonical_non_audit(
            paths["discovery_privileged"], protocol=protocol,
            filename=protocol["collection"]["discovery_privileged_filename"],
            name="merged discovery privileged output"),
        _canonical_non_audit(
            paths["discovery_report"], protocol=protocol,
            filename=protocol["collection"]["discovery_merge_report_filename"],
            name="discovery merge report"),
    ]
    _require_free(outputs, "discovery merge")
    commitments = readiness["role_commitments"]
    datasets: list[GroupedBranchDataset] = []
    views: list[PrivilegedBranchView] = []
    for ordinal, (seed, path, privileged_path) in enumerate(zip(
            SOURCE_SEEDS,
            paths["discovery_leaves"],
            paths["discovery_privileged_leaves"],
            strict=True,
    )):
        commitment = commitments["discovery"][ordinal]
        privileged_commitment = commitments["discovery_privileged"][ordinal]
        if Path(commitment["path"]) != path or Path(
                privileged_commitment["path"]) != privileged_path:
            raise ValueError("discovery readiness path order drifted")
        with workflow_evidence_read_scope(
                workflow=PROTOCOL_NAME,
                role="discovery",
                path=path):
            dataset = GroupedBranchDataset.load(path)
        with workflow_evidence_read_scope(
                workflow=PROTOCOL_NAME,
                role="discovery_privileged",
                path=privileged_path):
            view = PrivilegedBranchView.load(
                privileged_path, deployable=dataset)
        _require_exact_v5_discovery_rng_split(dataset.manifest)
        if dataset.group_count != 64 or dataset.candidate_count != 9 or (
                dataset.replica_count != 64) or dataset.horizon_steps != 96 or (
                    not np.all(np.asarray(dataset["source_seed"]) == seed)):
            raise ValueError(f"discovery leaf {seed} violates exact G/K/R/H")
        if _sha256(path) != commitment["file_sha256"] or dataset.manifest[
                "content_sha256"] != commitment["content_sha256"] or (
                    dataset.manifest.get("generator_commit") != commit):
            raise ValueError(f"discovery leaf {seed} differs from readiness")
        if _sha256(privileged_path) != privileged_commitment[
                "file_sha256"] or view.manifest["content_sha256"] != (
                    privileged_commitment["content_sha256"]):
            raise ValueError(
                f"discovery privileged leaf {seed} differs from readiness")
        datasets.append(dataset)
        views.append(view)

    combined = merge_grouped_shards(datasets)
    combined_view = merge_privileged_shards(views, datasets, combined)
    data_gate = _v3_exact_discovery_gate(combined, protocol)
    if not data_gate["pass"]:
        failed = sorted(
            name for name, value in data_gate["checks"].items() if not value)
        raise ValueError("V5 discovery data gate failed: " + ", ".join(failed))

    staged = [_staging_path(path) for path in outputs]
    try:
        combined.save(staged[0])
        combined_view.save(staged[1])
        combined_view.validate(combined)
        report = {
            "schema_version": "qsafe.grouped_merge_report.v3",
            "development_only": True,
            "publication_contract": "atomic_no_clobber_report_last_v1",
            "merge_tool_commit": commit,
            "merge_tool_worktree_clean": True,
            "merge_tool_commit_stable": True,
            "output": str(outputs[0]),
            "output_sha256": _sha256(staged[0]),
            "output_content_sha256": combined.manifest["content_sha256"],
            "privileged_output": str(outputs[1]),
            "privileged_sha256": _sha256(staged[1]),
            "privileged_content_sha256": combined_view.manifest[
                "content_sha256"],
            "input_shards": [
                {
                    "path": str(dataset.path),
                    "file_sha256": _sha256(dataset.path),
                    "content_sha256": dataset.manifest["content_sha256"],
                    "generator_commit": dataset.manifest["generator_commit"],
                    "groups": dataset.group_count,
                    "source_seeds": sorted(set(map(
                        int, dataset["source_seed"]))),
                }
                for dataset in datasets
            ],
            "input_privileged_shards": [
                {
                    "path": str(view.path),
                    "file_sha256": _sha256(view.path),
                    "content_sha256": view.manifest["content_sha256"],
                    "generator_commit": view.manifest["generator_commit"],
                    "deployable_content_sha256": view.manifest[
                        "deployable_content_sha256"],
                }
                for view in views
            ],
            "validation": combined.validate(),
            "data_gate_role": "closed_loop_recovery_triage",
            "collection_data_gate": data_gate,
            "phase1_data_gate": data_gate,
            "phase2_authorized": False,
            "collection_readiness_sha256": readiness["readiness_sha256"],
        }
        staged[2].write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if _clean_git_commit() != commit:
            raise RuntimeError("worktree changed during V5 discovery merge")
        _publish_no_clobber(list(zip(staged, outputs, strict=True)))
    finally:
        for path in staged:
            path.unlink(missing_ok=True)
    return report


def _create_lock(
    *,
    protocol: dict[str, Any],
    paths: dict[str, Any],
    readiness: dict[str, Any],
    commit: str,
) -> dict[str, Any]:
    if readiness["generator_commit"] != commit:
        raise RuntimeError("V5 source shards use a different generator commit")
    return create_state_dependent_selection_lock(
        protocol=protocol,
        admission_path=paths["admission"],
        discovery_path=paths["discovery"],
        collection_report_paths=paths["reports"],
        selection_lock_path=paths["selection_lock"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=("admission", "discovery", "lock", "resume-denied-report"),
    )
    parser.add_argument(
        "--selection-lock-sha256",
        help=("required only for resume-denied-report; exact lowercase SHA-256 "
              "of the existing selection lock"),
    )
    args = parser.parse_args()
    protocol = load_state_dependent_recovery_v5_protocol()
    paths = _paths(protocol)
    if args.operation == "resume-denied-report":
        if args.selection_lock_sha256 is None:
            parser.error(
                "resume-denied-report requires --selection-lock-sha256")
        report = resume_state_dependent_discovery_failure_report(
            protocol=protocol,
            selection_lock_path=paths["selection_lock"],
            expected_selection_lock_sha256=args.selection_lock_sha256,
        )
        print(json.dumps(
            report, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.selection_lock_sha256 is not None:
        parser.error(
            "--selection-lock-sha256 is valid only for resume-denied-report")
    readiness = validate_state_dependent_collection_readiness(
        protocol=protocol,
        collection_report_paths=paths["reports"],
    )
    commit = _clean_git_commit()
    if readiness["generator_commit"] != commit:
        raise RuntimeError(
            "V5 merge/lock must use the exact clean collection commit")
    if readiness["protocol_file_sha256"] != _file_sha256(PROTOCOL_PATH):
        raise RuntimeError("V5 collection reports bind another protocol file")
    if args.operation == "admission":
        report = _merge_admission(
            protocol=protocol, paths=paths, readiness=readiness, commit=commit)
    elif args.operation == "discovery":
        report = _merge_discovery(
            protocol=protocol, paths=paths, readiness=readiness, commit=commit)
    else:
        report = _create_lock(
            protocol=protocol, paths=paths, readiness=readiness, commit=commit)
    if args.operation == "lock":
        rendered = {
            "selection_lock": str(paths["selection_lock"]),
            "selection_lock_sha256": report["selection_lock_sha256"],
            "primary_selection": report["primary_selection"],
            "discovery_informativeness": report["data_gate"][
                "discovery_informativeness"],
            "audit_authorized": report["audit_authorized"],
            "model_training_authorized": False,
            "phase2_authorized": False,
        }
        if report["audit_authorized"] is False:
            rendered.update({
                "stage_A_failure_report": report["stage_A_failure_report"],
                "stage_A_failure_report_sha256": report[
                    "stage_A_failure_report_sha256"],
                "decision": "no_model_training",
            })
    else:
        rendered = report
    print(json.dumps(rendered, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
