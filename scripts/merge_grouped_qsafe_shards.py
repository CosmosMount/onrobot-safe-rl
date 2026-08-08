#!/usr/bin/env python3
"""Merge identity-disjoint native Q_safe shards and evaluate the data gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

import numpy as np
import yaml

from safety_data.merge import merge_grouped_shards, merge_privileged_shards
from safety_data.paths import assert_development_path
from safety_data.schema import GroupedBranchDataset, PrivilegedBranchView


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _staging_path(output: Path) -> Path:
    """Create a process-owned sibling used for no-clobber publication."""
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=output.suffix, dir=output.parent)
    os.close(descriptor)
    return Path(raw_path)


def _publish_no_clobber(
    staged_outputs: Sequence[tuple[Path, Path]],
) -> None:
    """Publish a set of files without replacement and roll back on failure.

    Each staged file is a sibling of its destination, so ``os.link`` is an
    atomic create-if-absent operation.  The report is supplied last by the
    caller and acts as the completion marker.  Recoverable failures remove
    only hard links that still point to files owned by this invocation.
    """
    pairs = list(staged_outputs)
    if not pairs or len({destination for _, destination in pairs}) != len(pairs):
        raise ValueError("staged publication destinations must be nonempty/distinct")
    published: list[tuple[Path, tuple[int, int]]] = []
    try:
        for staged, destination in pairs:
            staged_stat = staged.stat()
            os.link(staged, destination)
            published.append(
                (destination, (staged_stat.st_dev, staged_stat.st_ino)))
    except BaseException:
        for destination, identity in reversed(published):
            try:
                current = destination.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if (current.st_dev, current.st_ino) == identity:
                destination.unlink()
        raise
    finally:
        for staged, _ in pairs:
            try:
                staged.unlink()
            except FileNotFoundError:
                pass


def _data_gate(dataset, thresholds):
    report = dataset.validate()
    mask = np.asarray(dataset["candidate_mask"], dtype=bool)
    checks = {
        "independent_groups": dataset.group_count
        >= int(thresholds["min_independent_groups"]),
        "trajectory_clusters": len(np.unique(dataset["trajectory_id"]))
        >= int(thresholds["min_independent_trajectory_clusters"]),
        "source_seeds": len(np.unique(dataset["source_seed"]))
        >= int(thresholds["min_source_seeds"]),
        "candidates_per_group": int(mask.sum(axis=1).min())
        >= int(thresholds["min_candidates_per_group"]),
        "replicas_per_candidate": dataset.replica_count
        >= int(thresholds["min_replicas_per_candidate"]),
        "mixed_outcomes": float(report["mixed_outcome_fraction"])
        >= float(thresholds["min_mixed_outcome_fraction"]),
        "duplicate_groups": float(report["duplicate_state_fraction"])
        <= float(thresholds["max_duplicate_group_fraction"]),
    }
    return {"pass": all(checks.values()), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", help="At least two deployable .npz shards")
    parser.add_argument(
        "--privileged-shards", nargs="*",
        help="Optional aligned privileged .npz shard list")
    parser.add_argument("--output", required=True)
    parser.add_argument("--privileged-output")
    parser.add_argument("--report")
    parser.add_argument(
        "--protocol", default="config/qsafe_evidence_protocol.yaml")
    args = parser.parse_args()
    if len(args.shards) < 2:
        parser.error("at least two deployable shards are required")
    if args.privileged_shards is not None and not args.privileged_shards:
        parser.error("--privileged-shards requires one path per deployable shard")
    if args.privileged_shards is not None and len(args.privileged_shards) != (
            len(args.shards)):
        parser.error("privileged shard count must equal deployable shard count")
    if args.privileged_output and args.privileged_shards is None:
        parser.error("--privileged-output requires --privileged-shards")

    output = assert_development_path(args.output)
    privileged_output = (
        None if not args.privileged_shards
        else assert_development_path(
            args.privileged_output or output.with_name(
                f"{output.stem}.privileged.npz")))
    report_output = assert_development_path(
        args.report or output.with_name(f"{output.stem}.report.json"))
    outputs = [output, report_output]
    if privileged_output is not None:
        outputs.append(privileged_output)
    if len(set(outputs)) != len(outputs):
        parser.error("merged output paths must be distinct")
    if output.suffix != ".npz" or (
            privileged_output is not None and privileged_output.suffix != ".npz"):
        parser.error("merged dataset outputs must use .npz")
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite merged outputs: {existing}")

    protocol_path = assert_development_path(args.protocol)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))

    datasets = [GroupedBranchDataset.load(path) for path in args.shards]
    combined = merge_grouped_shards(datasets)
    combined_privileged = None
    if args.privileged_shards:
        views = [
            PrivilegedBranchView.load(path, deployable=dataset)
            for path, dataset in zip(
                args.privileged_shards, datasets, strict=True)
        ]
        combined_privileged = merge_privileged_shards(
            views, datasets, combined)
    data_gate = _data_gate(combined, protocol["phase1"]["data_gate"])
    staged_paths: list[Path] = []
    rendered = ""
    try:
        staged_output = _staging_path(output)
        staged_paths.append(staged_output)
        combined.save(staged_output)
        staged_privileged = None
        if combined_privileged is not None:
            assert privileged_output is not None
            staged_privileged = _staging_path(privileged_output)
            staged_paths.append(staged_privileged)
            combined_privileged.save(staged_privileged)
            combined_privileged.validate(combined)

        input_shards = []
        for dataset in datasets:
            assert dataset.path is not None
            input_shards.append({
                "path": str(dataset.path),
                "file_sha256": _sha256(dataset.path),
                "content_sha256": dataset.manifest["content_sha256"],
                "groups": dataset.group_count,
                "source_seeds": sorted(set(map(int, dataset["source_seed"]))),
            })
        input_privileged_shards = None
        if args.privileged_shards:
            input_privileged_shards = [
                {
                    "path": str(view.path),
                    "file_sha256": _sha256(view.path),
                    "content_sha256": view.manifest["content_sha256"],
                    "deployable_content_sha256": view.manifest[
                        "deployable_content_sha256"],
                }
                for view in views
            ]
        report = {
            "schema_version": "qsafe.grouped_merge_report.v2",
            "development_only": True,
            "publication_contract": "atomic_no_clobber_report_last_v1",
            "output": str(output),
            "output_sha256": _sha256(staged_output),
            "output_content_sha256": combined.manifest["content_sha256"],
            "privileged_output": (
                None if privileged_output is None else str(privileged_output)),
            "privileged_sha256": (
                None if staged_privileged is None else _sha256(staged_privileged)),
            "privileged_content_sha256": (
                None if combined_privileged is None
                else combined_privileged.manifest["content_sha256"]),
            "input_shards": input_shards,
            "input_privileged_shards": input_privileged_shards,
            "validation": combined.validate(),
            "phase1_data_gate": data_gate,
            "phase2_authorized": False,
        }
        staged_report = _staging_path(report_output)
        staged_paths.append(staged_report)
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        staged_report.write_text(rendered, encoding="utf-8")

        publication = [(staged_output, output)]
        if staged_privileged is not None:
            assert privileged_output is not None
            publication.append((staged_privileged, privileged_output))
        publication.append((staged_report, report_output))
        _publish_no_clobber(publication)
        combined.path = output
        if combined_privileged is not None:
            combined_privileged.path = privileged_output
    finally:
        for path in staged_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
