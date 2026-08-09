#!/usr/bin/env python3
"""Merge identity-disjoint native Q_safe shards and evaluate the data gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Sequence

import numpy as np
import yaml

from safety_data.merge import merge_grouped_shards, merge_privileged_shards
from safety_data.paths import (
    assert_development_path,
    assert_safe_evidence_output,
    require_v3_audit_consumed_or_safe_input,
)
from safety_data.closed_loop_recovery_triage import (
    validate_closed_loop_recovery_protocol,
    validate_collection_readiness,
)
from safety_data.schema import GroupedBranchDataset, PrivilegedBranchView


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_LOCKED_V3_AUDIT_BASENAMES = frozenset({
    *(f"source-{seed}.audit.npz"
      for seed in (7801, 7802, 7811, 7812, 7821, 7822)),
    *(f"source-{seed}.audit.privileged.npz"
      for seed in (7801, 7802, 7811, 7812, 7821, 7822)),
    "audit-g384.npz",
    "audit-g384-privileged.npz",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lexical_absolute(path: str | os.PathLike[str]) -> Path:
    """Normalize without resolving or touching the final path component."""
    return Path(os.path.abspath(os.fspath(path)))


def _reject_locked_v3_audit_basenames(
    paths: Sequence[str | os.PathLike[str]],
) -> None:
    """Keep the generic merger from ever opening a locked v3 audit name.

    This check is deliberately protocol-independent and purely lexical.  In
    particular, it runs before resolving a parent-directory symlink, reading a
    caller-supplied protocol, or probing any artifact.  The one-shot audit
    command is the only supported consumer of these physical shard names.
    """
    offenders = [
        _lexical_absolute(path)
        for path in paths
        if _lexical_absolute(path).name in _LOCKED_V3_AUDIT_BASENAMES
    ]
    if offenders:
        raise ValueError(
            "locked v3 audit paths are forbidden in the generic merger: "
            f"{offenders}")


def _v3_lexical_discovery_paths(
    *,
    protocol: dict,
    shards: Sequence[str],
    privileged_shards: Sequence[str] | None,
    collection_reports: Sequence[str] | None,
    output: Path,
    privileged_output: Path | None,
    report_output: Path,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Apply the v3 artifact allowlist without any filesystem inspection."""
    collection = protocol["collection"]
    seeds = list(map(
        int, protocol["triage_gates"]["data"]["required_source_seeds"]))
    root = _lexical_absolute(
        _REPOSITORY_ROOT / str(collection["artifact_root"]))

    def expected(template_key: str) -> list[Path]:
        template = str(collection[template_key])
        return [root / template.format(source_seed=seed) for seed in seeds]

    expected_shards = expected("discovery_shard_filename_template")
    expected_privileged = expected(
        "discovery_privileged_shard_filename_template")
    expected_reports = expected("collection_report_shard_filename_template")
    audit_names = {
        *(path.name for path in expected("audit_shard_filename_template")),
        *(path.name for path in expected(
            "audit_privileged_shard_filename_template")),
        Path(str(collection["audit_filename"])).name,
        Path(str(collection["audit_privileged_filename"])).name,
    }
    supplied_paths = [
        *map(_lexical_absolute, shards),
        *([] if privileged_shards is None else map(
            _lexical_absolute, privileged_shards)),
        *([] if collection_reports is None else map(
            _lexical_absolute, collection_reports)),
        _lexical_absolute(output),
        *([] if privileged_output is None else [
            _lexical_absolute(privileged_output)]),
        _lexical_absolute(report_output),
    ]
    if any(path.name in audit_names for path in supplied_paths):
        raise ValueError(
            "canonical v3 audit paths are forbidden before any filesystem probe")
    if list(map(_lexical_absolute, shards)) != expected_shards:
        raise ValueError(
            "v3 merger accepts only the six canonical discovery shards in "
            "preregistered source-seed order; audit paths are forbidden")
    if privileged_shards is None or list(map(
            _lexical_absolute, privileged_shards)) != expected_privileged:
        raise ValueError(
            "v3 merger requires the six canonical discovery privileged shards")
    if collection_reports is None or list(map(
            _lexical_absolute, collection_reports)) != expected_reports:
        raise ValueError(
            "v3 merger requires six report-last completion markers in source order")
    if _lexical_absolute(output) != root / str(
            collection["discovery_filename"]):
        raise ValueError("v3 discovery output must use the canonical locked path")
    if privileged_output is None or _lexical_absolute(
            privileged_output) != root / str(
                collection["discovery_privileged_filename"]):
        raise ValueError(
            "v3 privileged discovery output must use the canonical locked path")
    if _lexical_absolute(report_output) != root / str(
            collection["discovery_merge_report_filename"]):
        raise ValueError(
            "v3 discovery merge report must use the canonical locked path")
    return expected_shards, expected_privileged, expected_reports


def _v3_preopen_discovery_paths(
    *,
    protocol: dict,
    shards: Sequence[str],
    privileged_shards: Sequence[str] | None,
    collection_reports: Sequence[str] | None,
    output: Path,
    privileged_output: Path | None,
    report_output: Path,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Apply the lexical v3 allowlist, then reject symlink escapes."""
    expected_shards, expected_privileged, expected_reports = (
        _v3_lexical_discovery_paths(
            protocol=protocol,
            shards=shards,
            privileged_shards=privileged_shards,
            collection_reports=collection_reports,
            output=output,
            privileged_output=privileged_output,
            report_output=report_output,
        )
    )
    root = _lexical_absolute(
        _REPOSITORY_ROOT / str(protocol["collection"]["artifact_root"]))

    # Only the exact lexical allowlist reaches filesystem inspection.  The
    # shared guard walks ancestors with lstat and refuses the final symlink,
    # without following an alias into a role artifact.
    checked_root = require_v3_audit_consumed_or_safe_input(root)
    if assert_development_path(checked_root) != root:
        raise ValueError("v3 artifact root may not resolve through a symlink")
    for label, paths in (
        ("discovery shard", expected_shards),
        ("discovery privileged shard", expected_privileged),
        ("collection report", expected_reports),
    ):
        for path in paths:
            checked = require_v3_audit_consumed_or_safe_input(path)
            if checked.parent != root:
                raise ValueError(f"v3 {label} is outside artifact root")
    return expected_shards, expected_privileged, expected_reports


def _clean_git_commit() -> str:
    commit = subprocess.run(
        ["git", "-C", str(_REPOSITORY_ROOT), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(_REPOSITORY_ROOT),
         "status", "--porcelain=v1", "-z"], check=True,
        capture_output=True)
    if status.stdout:
        raise RuntimeError(
            "grouped merge requires a clean git worktree so the merge report "
            "identifies the code that actually ran")
    return commit


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


def _v3_exact_discovery_gate(dataset, protocol: dict) -> dict:
    data = protocol["triage_gates"]["data"]
    collection = protocol["collection"]
    required_seeds = list(map(int, data["required_source_seeds"]))
    groups = int(dataset.group_count)
    candidates = int(data["candidates_per_group_exact"])
    discovery_replicas = int(data["discovery_replicas_exact"])
    audit_replicas = int(data["audit_replicas_exact"])
    source_seed = np.asarray(dataset["source_seed"])
    mask = np.asarray(dataset["candidate_mask"], dtype=bool)
    role = dataset.manifest.get("collection_protocol", {}).get("role")
    candidate_protocol = collection.get("candidates")
    manifest_candidate_protocol = dataset.manifest.get("candidate_protocol")

    ordered_names = (
        candidate_protocol.get("ordered_names")
        if isinstance(candidate_protocol, dict) else None)
    expected_behavior_steps = (
        candidate_protocol.get("behavior_override_steps")
        if isinstance(candidate_protocol, dict) else None)
    names_well_formed = (
        isinstance(ordered_names, list)
        and len(ordered_names) == candidates
        and all(isinstance(name, str) and name for name in ordered_names)
    )
    steps_well_formed = (
        isinstance(expected_behavior_steps, list)
        and len(expected_behavior_steps) == candidates
        and all(isinstance(step, int) and not isinstance(step, bool)
                for step in expected_behavior_steps)
    )
    candidate_kind = np.asarray(dataset.arrays.get("candidate_kind"))
    candidate_behavior_steps = np.asarray(
        dataset.arrays.get("candidate_behavior_steps"))
    candidate_kind_exact = bool(
        names_well_formed
        and candidate_kind.shape == (groups, candidates)
        and candidate_kind.dtype.kind in "US"
        and np.array_equal(
            candidate_kind.astype(str),
            np.broadcast_to(
                np.asarray(ordered_names, dtype=str), (groups, candidates)),
        )
    )
    candidate_behavior_steps_exact = bool(
        steps_well_formed
        and candidate_behavior_steps.shape == (groups, candidates)
        and candidate_behavior_steps.dtype.kind in "iu"
        and np.array_equal(
            candidate_behavior_steps,
            np.broadcast_to(
                np.asarray(expected_behavior_steps, dtype=np.int64),
                (groups, candidates),
            ),
        )
    )

    def seed_array(name: str, shape: tuple[int, ...]) -> np.ndarray | None:
        if name not in dataset.arrays:
            return None
        value = np.asarray(dataset.arrays[name])
        if value.shape != shape or value.dtype.kind not in "iu" or np.any(
                value < 0):
            return None
        return value

    discovery_seed_arrays = [
        seed_array(name, (groups, discovery_replicas))
        for name in ("crn_id", "rollout_seed", "perturbation_seed")
    ]
    discovery_seed_arrays.append(seed_array("candidate_seed", (groups,)))
    audit_seed_arrays = [
        seed_array(name, (groups, audit_replicas))
        for name in (
            "preassigned_audit_crn_id",
            "preassigned_audit_rollout_seed",
            "preassigned_audit_perturbation_seed",
        )
    ]
    audit_seed_arrays.append(seed_array(
        "preassigned_audit_candidate_seed", (groups,)))
    discovery_seed_shape_exact = all(
        value is not None for value in discovery_seed_arrays)
    audit_seed_shape_exact = all(
        value is not None for value in audit_seed_arrays)

    def combined_seed_values(
        values: list[np.ndarray | None],
    ) -> np.ndarray | None:
        if any(value is None for value in values):
            return None
        return np.concatenate([
            value.reshape(-1).astype(np.uint64, copy=False)
            for value in values if value is not None
        ])

    discovery_seed_values = combined_seed_values(discovery_seed_arrays)
    audit_seed_values = combined_seed_values(audit_seed_arrays)
    audit_seed_unique = bool(
        audit_seed_values is not None
        and np.unique(audit_seed_values).size == audit_seed_values.size)
    discovery_audit_seed_domains_disjoint = bool(
        discovery_seed_values is not None
        and audit_seed_values is not None
        and not np.intersect1d(
            discovery_seed_values, audit_seed_values,
            assume_unique=False,
        ).size
    )

    groups_per_seed = int(data["groups_per_required_source_seed_exact"])
    expected_source_seed = np.repeat(
        np.asarray(required_seeds, dtype=np.int64), groups_per_seed)
    checks = {
        "physical_role_discovery": role == "discovery",
        "independent_groups_exact": groups
        == int(data["independent_groups_exact"]),
        "trajectory_clusters_exact": len(np.unique(dataset["trajectory_id"]))
        == int(data["unique_source_trajectories_exact"]),
        "source_seed_order_and_counts_exact": source_seed.shape == (
            groups,) and np.array_equal(source_seed, expected_source_seed),
        "candidates_exact": bool(
            dataset.candidate_count == candidates
            and mask.shape == (groups, candidates)
            and np.all(mask)),
        "candidate_kind_exact": candidate_kind_exact,
        "candidate_behavior_steps_exact": candidate_behavior_steps_exact,
        "candidate_protocol_exact": isinstance(candidate_protocol, dict)
        and manifest_candidate_protocol == candidate_protocol,
        "discovery_replicas_exact": dataset.replica_count
        == discovery_replicas,
        "discovery_seed_shape_exact": discovery_seed_shape_exact,
        "horizon_exact": dataset.horizon_steps
        == int(data["horizon_policy_steps_exact"]),
        "audit_seed_preassignment_shape_exact": audit_seed_shape_exact,
        "audit_seed_preassignment_unique": audit_seed_unique,
        "discovery_audit_seed_domains_disjoint": (
            discovery_audit_seed_domains_disjoint),
        "audit_merge_forbidden": collection.get(
            "audit_merge_before_selection") == "forbidden",
    }
    return {"pass": bool(all(checks.values())), "checks": checks}


def _require_v3_exact_discovery_gate(dataset, protocol: dict) -> dict:
    """Fail before staging any deployable or report when the v3 gate drifts."""
    result = _v3_exact_discovery_gate(dataset, protocol)
    if not result["pass"]:
        failures = sorted(
            name for name, passed in result["checks"].items() if not passed)
        raise ValueError(
            "v3 exact discovery gate failed before publication: "
            + ", ".join(failures))
    return result


def _data_gate_thresholds(protocol: dict) -> tuple[dict, str]:
    """Route legacy Phase-1 and recovery-triage merge contracts explicitly."""
    if isinstance(protocol.get("phase1"), dict) and isinstance(
            protocol["phase1"].get("data_gate"), dict):
        return dict(protocol["phase1"]["data_gate"]), "phase1"
    if protocol.get(
            "protocol_name") == "objective1_recovery_option_triage_v2":
        data = protocol["triage_gates"]["data"]
        return {
            "min_independent_groups": int(data["min_independent_groups"]),
            "min_independent_trajectory_clusters": int(
                data["min_trajectory_clusters"]),
            "min_source_seeds": len(data["required_source_seeds"]),
            "min_candidates_per_group": int(data["candidates_per_group"]),
            "min_replicas_per_candidate": int(data["discovery_replicas"])
            + int(data["audit_replicas"]),
            "min_mixed_outcome_fraction": 0.0,
            "max_duplicate_group_fraction": 0.0,
        }, "recovery_option_triage"
    if protocol.get(
            "protocol_name") == "objective1_closed_loop_recovery_triage_v3":
        data = protocol["triage_gates"]["data"]
        return {
            "min_independent_groups": int(data["independent_groups_exact"]),
            "min_independent_trajectory_clusters": int(
                data["unique_source_trajectories_exact"]),
            "min_source_seeds": len(data["required_source_seeds"]),
            "min_candidates_per_group": int(
                data["candidates_per_group_exact"]),
            # Discovery and audit are physically separate and this merger is
            # invoked once per role.
            "min_replicas_per_candidate": int(
                data["discovery_replicas_exact"]),
            "min_mixed_outcome_fraction": 0.0,
            "max_duplicate_group_fraction": 0.0,
        }, "closed_loop_recovery_triage"
    raise ValueError("unsupported grouped-shard merge protocol")


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
        "--collection-reports", nargs="*",
        help="Locked report-last completion markers (required by v3)")
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

    # Compute artifact paths lexically first.  For v3, no supplied artifact
    # final component may be statted or resolved until its entire path set has
    # passed the discovery-only allowlist below.
    lexical_output = _lexical_absolute(args.output)
    lexical_privileged_output = (
        None if args.privileged_shards is None
        else _lexical_absolute(
            args.privileged_output or lexical_output.with_name(
                f"{lexical_output.stem}.privileged.npz")))
    lexical_report_output = _lexical_absolute(
        args.report or lexical_output.with_name(
            f"{lexical_output.stem}.report.json"))
    lexical_outputs = [lexical_output, lexical_report_output]
    if lexical_privileged_output is not None:
        lexical_outputs.append(lexical_privileged_output)
    if len(set(lexical_outputs)) != len(lexical_outputs):
        parser.error("merged output paths must be distinct")
    if lexical_output.suffix != ".npz" or (
            lexical_privileged_output is not None
            and lexical_privileged_output.suffix != ".npz"):
        parser.error("merged dataset outputs must use .npz")

    supplied_shards = list(map(_lexical_absolute, args.shards))
    supplied_artifacts = [
        *supplied_shards,
        *([] if args.privileged_shards is None else map(
            _lexical_absolute, args.privileged_shards)),
        *([] if args.collection_reports is None else map(
            _lexical_absolute, args.collection_reports)),
        *lexical_outputs,
    ]
    _reject_locked_v3_audit_basenames([*supplied_artifacts, args.protocol])

    # Reject a final-component symlink before ``assert_development_path`` can
    # resolve it.  Otherwise an innocently named protocol alias could point at
    # a locked audit shard and disclose it before the consumed marker exists.
    protocol_path = assert_development_path(
        require_v3_audit_consumed_or_safe_input(args.protocol))
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    canonical_v3_root = _lexical_absolute(
        _REPOSITORY_ROOT
        / "saved" / "qsafe_development" / "closed_loop_recovery_triage_v3"
    )
    touches_v3_root = any(
        _lexical_absolute(path).parent == canonical_v3_root
        for path in supplied_artifacts
    )
    is_v3 = protocol.get(
        "protocol_name") == "objective1_closed_loop_recovery_triage_v3"
    if is_v3:
        # Pure mapping validation: reject any same-name protocol drift before
        # probing an input/output artifact path.
        validate_closed_loop_recovery_protocol(protocol)
    if touches_v3_root and not is_v3:
        raise ValueError("v3 artifacts require the canonical v3 workflow")
    readiness = None
    if is_v3:
        # This is intentionally the first operation on supplied v3 artifact
        # paths.  It performs no stat, lstat, open, or resolve.
        _v3_lexical_discovery_paths(
            protocol=protocol,
            shards=args.shards,
            privileged_shards=args.privileged_shards,
            collection_reports=args.collection_reports,
            output=lexical_output,
            privileged_output=lexical_privileged_output,
            report_output=lexical_report_output,
        )
        canonical_protocol_lexical = (
            _REPOSITORY_ROOT
            / "config" / "qsafe_closed_loop_recovery_triage_v3.yaml")
        canonical_protocol = assert_development_path(
            require_v3_audit_consumed_or_safe_input(
                canonical_protocol_lexical))
        if protocol_path != canonical_protocol:
            raise ValueError("v3 merge requires the canonical protocol file")
    elif args.collection_reports:
        parser.error("--collection-reports is supported only by v3")

    # ``lexists`` treats dangling symlinks as occupied.  Run it against the
    # original lexical destinations before assert_development_path resolves a
    # final symlink target.
    existing = [
        path for path in lexical_outputs if os.path.lexists(os.fspath(path))]
    if existing:
        raise FileExistsError(f"refusing to overwrite merged outputs: {existing}")

    def checked_output(path: Path) -> Path:
        # The v3 merger is the reviewed publisher for its exact discovery
        # outputs.  Generic invocations must use the reserved-name denylist.
        guarded = (
            require_v3_audit_consumed_or_safe_input(path)
            if is_v3 else assert_safe_evidence_output(path))
        return assert_development_path(guarded)

    output = checked_output(lexical_output)
    privileged_output = (
        None if lexical_privileged_output is None
        else checked_output(lexical_privileged_output))
    report_output = checked_output(lexical_report_output)
    outputs = [output, report_output]
    if privileged_output is not None:
        outputs.append(privileged_output)
    if len(set(outputs)) != len(outputs):
        parser.error("resolved merged output paths must be distinct")

    if is_v3:
        _, _, expected_reports = _v3_preopen_discovery_paths(
            protocol=protocol,
            shards=args.shards,
            privileged_shards=args.privileged_shards,
            collection_reports=args.collection_reports,
            output=lexical_output,
            privileged_output=lexical_privileged_output,
            report_output=lexical_report_output,
        )
        # Report-last readiness is checked before opening even one discovery
        # NPZ.  This function never stats or opens any role artifact.
        readiness = validate_collection_readiness(
            protocol=protocol,
            collection_report_paths=expected_reports,
        )
    else:
        for path in (
                *args.shards,
                *([] if args.privileged_shards is None
                  else args.privileged_shards),
                *([] if args.collection_reports is None
                  else args.collection_reports)):
            require_v3_audit_consumed_or_safe_input(path)
    merge_tool_commit = _clean_git_commit()
    if readiness is not None and readiness["generator_commit"] != (
            merge_tool_commit):
        raise ValueError(
            "v3 completion reports were generated by a different commit")

    datasets = [GroupedBranchDataset.load(path) for path in args.shards]
    if readiness is not None:
        commitments = readiness["role_commitments"]["discovery"]
        required_seeds = list(map(
            int, protocol["triage_gates"]["data"]["required_source_seeds"]))
        groups_per_seed = int(
            protocol["collection"]["groups_per_source_seed"])
        for ordinal, (dataset, commitment, seed) in enumerate(zip(
                datasets, commitments, required_seeds, strict=True)):
            source_values = np.asarray(dataset["source_seed"], dtype=np.int64)
            role = dataset.manifest.get("collection_protocol", {}).get("role")
            if role != "discovery" or dataset.group_count != groups_per_seed or (
                    not np.all(source_values == seed)) or dataset.candidate_count != 9 or (
                        dataset.replica_count != 64) or dataset.horizon_steps != 96:
                raise ValueError(
                    f"v3 discovery leaf {ordinal} violates exact role/G/K/R/H/seed")
            if dataset.manifest.get("generator_commit") != merge_tool_commit or (
                    dataset.manifest["content_sha256"] != commitment[
                        "content_sha256"]) or _sha256(dataset.path) != commitment[
                            "file_sha256"]:
                raise ValueError(
                    f"v3 discovery leaf {ordinal} differs from completion report")
    combined = merge_grouped_shards(datasets)
    combined_privileged = None
    if args.privileged_shards:
        views = [
            PrivilegedBranchView.load(path, deployable=dataset)
            for path, dataset in zip(
                args.privileged_shards, datasets, strict=True)
        ]
        if readiness is not None:
            privileged_commitments = readiness[
                "role_commitments"]["discovery_privileged"]
            for ordinal, (view, commitment) in enumerate(zip(
                    views, privileged_commitments, strict=True)):
                if view.manifest["content_sha256"] != commitment[
                        "content_sha256"] or _sha256(view.path) != commitment[
                            "file_sha256"]:
                    raise ValueError(
                        "v3 privileged discovery leaf differs from completion "
                        f"report at ordinal {ordinal}")
        combined_privileged = merge_privileged_shards(
            views, datasets, combined)
    gate_thresholds, gate_role = _data_gate_thresholds(protocol)
    data_gate = (
        _require_v3_exact_discovery_gate(combined, protocol)
        if gate_role == "closed_loop_recovery_triage"
        else _data_gate(combined, gate_thresholds)
    )
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
                "generator_commit": dataset.manifest["generator_commit"],
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
                    "generator_commit": view.manifest["generator_commit"],
                    "deployable_content_sha256": view.manifest[
                        "deployable_content_sha256"],
                }
                for view in views
            ]
        if _clean_git_commit() != merge_tool_commit:
            raise RuntimeError(
                "git commit changed while grouped merge was running")
        report = {
            "schema_version": "qsafe.grouped_merge_report.v3",
            "development_only": True,
            "publication_contract": "atomic_no_clobber_report_last_v1",
            "merge_tool_commit": merge_tool_commit,
            "merge_tool_worktree_clean": True,
            "merge_tool_commit_stable": True,
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
            "data_gate_role": gate_role,
            "collection_data_gate": data_gate,
            "phase1_data_gate": data_gate,
            "phase2_authorized": False,
        }
        if readiness is not None:
            report["collection_readiness_sha256"] = readiness[
                "readiness_sha256"]
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
