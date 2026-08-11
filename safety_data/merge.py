"""Deterministic, identity-safe merging of native grouped data shards."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from safety_data.schema import (
    GroupedBranchDataset,
    PRIVILEGED_SCHEMA_VERSION,
    PrivilegedBranchView,
    _content_hash,
    _require_content_hash,
)


def _contract_manifest(
    manifest: dict[str, Any],
    *, per_shard_fields: Sequence[str] = (),
) -> str:
    """Return a forward-compatible causal-contract representation.

    Every manifest field is locked by default.  Only the content hash and
    explicitly named links that necessarily differ per shard may vary.  This
    prevents a newly added collection/labeling field from silently escaping a
    hand-maintained allowlist.
    """
    if "shards" in manifest:
        raise ValueError(
            "nested merged artifacts are not leaf shards; merge original "
            "hashed shards instead")
    value = copy.deepcopy(manifest)
    value.pop("content_sha256", None)
    for name in per_shard_fields:
        value.pop(name, None)
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("shard manifest is not canonical JSON") from exc


def _verified_content_hash(
    manifest: dict[str, Any],
    actual_hash: str,
    *,
    label: str,
) -> str:
    recorded = manifest.get("content_sha256")
    if recorded != actual_hash:
        raise ValueError(
            f"{label} must carry its verified on-disk content_sha256")
    return actual_hash


def _merged_generator_commit(
    manifests: Sequence[dict[str, Any]],
) -> str:
    """Summarize leaf collection commits without pretending they are causal.

    Keeping the original value for a single-commit merge preserves the
    historical manifest shape.  A mixed merge gets an order-sensitive digest;
    the exact values remain available in each ``shards`` entry.
    """
    commits = [manifest.get("generator_commit") for manifest in manifests]
    if any(not isinstance(commit, str) or not commit.strip()
           for commit in commits):
        raise ValueError("leaf generator_commit must be a nonempty string")
    if len(set(commits)) == 1:
        return commits[0]
    payload = json.dumps(
        commits, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return "mixed_leaf_generator_commits_sha256:" + hashlib.sha256(
        payload).hexdigest()


def _assert_disjoint(datasets: Sequence[GroupedBranchDataset]) -> None:
    vector_fields = (
        "group_id", "state_hash", "trajectory_id", "episode_id", "source_seed")
    matrix_fields = ("crn_id", "rollout_seed", "perturbation_seed")
    if all("candidate_seed" in dataset.arrays for dataset in datasets):
        vector_fields = (*vector_fields, "candidate_seed")
    seen: dict[str, set[Any]] = {
        name: set() for name in (*vector_fields, *matrix_fields)}
    for shard_index, dataset in enumerate(datasets):
        for name in vector_fields:
            values = set(np.asarray(dataset[name]).astype(str).tolist())
            overlap = seen[name] & values
            if overlap:
                raise ValueError(
                    f"{name} overlaps before shard {shard_index}: "
                    f"{sorted(overlap)[:3]}")
            seen[name].update(values)
        for name in matrix_fields:
            values = set(np.asarray(dataset[name]).reshape(-1).astype(str).tolist())
            overlap = seen[name] & values
            if overlap:
                raise ValueError(
                    f"{name} overlaps before shard {shard_index}: "
                    f"{sorted(overlap)[:3]}")
            seen[name].update(values)


def merge_grouped_shards(
    datasets: Sequence[GroupedBranchDataset],
) -> GroupedBranchDataset:
    """Concatenate shards only when every causal contract is identical."""
    items = list(datasets)
    if len(items) < 2:
        raise ValueError("at least two grouped shards are required")
    reports = [
        dataset.validate(summarize_outcomes=False) for dataset in items
    ]
    content_hashes = [
        _verified_content_hash(
            dataset.manifest, report["content_sha256"],
            label=f"deployable shard {index}",
        )
        for index, (dataset, report) in enumerate(
            zip(items, reports, strict=True))
    ]
    reference = items[0]
    reference_keys = set(reference.arrays)
    reference_contract = _contract_manifest(
        reference.manifest, per_shard_fields=("generator_commit",))
    for shard_index, dataset in enumerate(items[1:], start=1):
        if set(dataset.arrays) != reference_keys:
            raise ValueError(f"shard {shard_index} array fields differ")
        if _contract_manifest(
                dataset.manifest,
                per_shard_fields=("generator_commit",),
        ) != reference_contract:
            raise ValueError(
                f"shard {shard_index} changes the causal manifest contract")
    for shard_index, dataset in enumerate(items):
        for name, value in dataset.arrays.items():
            array = np.asarray(value)
            reference_array = np.asarray(reference[name])
            if array.ndim == 0 or array.shape[0] != dataset.group_count:
                raise ValueError(
                    f"shard {shard_index} field {name!r} is not group-first")
            if array.shape[1:] != reference_array.shape[1:]:
                raise ValueError(
                    f"shard {shard_index} field {name!r} changes trailing shape")
            text_compatible = (
                array.dtype.kind in "US" and reference_array.dtype.kind in "US")
            if not text_compatible and array.dtype != reference_array.dtype:
                raise ValueError(
                    f"shard {shard_index} field {name!r} changes dtype from "
                    f"{reference_array.dtype} to {array.dtype}")
    _assert_disjoint(items)
    arrays = {
        name: np.concatenate([
            np.asarray(dataset[name]) for dataset in items
        ], axis=0)
        for name in sorted(reference_keys)
    }
    manifest = copy.deepcopy(reference.manifest)
    manifest.pop("content_sha256", None)
    manifest["generator_commit"] = _merged_generator_commit([
        dataset.manifest for dataset in items])
    manifest["shards"] = [
        {
            "ordinal": index,
            "content_sha256": content_hashes[index],
            "generator_commit": dataset.manifest["generator_commit"],
            "groups": dataset.group_count,
            "source_seeds": sorted(set(map(int, dataset["source_seed"]))),
        }
        for index, dataset in enumerate(items)
    ]
    combined = GroupedBranchDataset(manifest, arrays)
    combined.validate(verify_hash=False, summarize_outcomes=False)
    return combined


def load_grouped_shard_blind(path: str | Path) -> GroupedBranchDataset:
    """Load a deployable shard mechanically, without schema/outcome validation."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        if "manifest_json" not in payload.files:
            raise ValueError("blind grouped shard has no manifest_json")
        manifest = json.loads(str(payload["manifest_json"].item()))
        arrays = {
            name: payload[name].copy()
            for name in payload.files if name != "manifest_json"
        }
    _require_content_hash(manifest)
    actual = _content_hash(manifest, arrays)
    if actual != manifest["content_sha256"]:
        raise ValueError("blind grouped shard content hash mismatch")
    return GroupedBranchDataset(manifest, arrays, source)


def merge_grouped_shards_blind(
    datasets: Sequence[GroupedBranchDataset],
) -> GroupedBranchDataset:
    """Mechanically concatenate shards without reading outcome semantics."""
    items = list(datasets)
    if len(items) < 2:
        raise ValueError("at least two grouped shards are required")
    reference = items[0]
    keys = set(reference.arrays)
    contract = _contract_manifest(reference.manifest, per_shard_fields=(
        "generator_commit",))
    hashes: list[str] = []
    for index, item in enumerate(items):
        if set(item.arrays) != keys:
            raise ValueError(f"blind shard {index} array fields differ")
        if _contract_manifest(item.manifest, per_shard_fields=(
                "generator_commit",)) != contract:
            raise ValueError(f"blind shard {index} changes manifest contract")
        for name in keys:
            value = np.asarray(item.arrays[name])
            if value.ndim == 0 or value.shape[0] != item.group_count:
                raise ValueError(f"blind shard {index} field {name} is not group-first")
            reference_value = np.asarray(reference.arrays[name])
            if value.shape[1:] != reference_value.shape[1:] or (
                    value.dtype.kind not in "US" and value.dtype != reference_value.dtype):
                raise ValueError(f"blind shard {index} field {name} shape/dtype drifted")
        actual = _content_hash(item.manifest, item.arrays)
        if item.manifest.get("content_sha256") != actual:
            raise ValueError(f"blind shard {index} content hash mismatch")
        hashes.append(actual)
    _assert_disjoint(items)
    arrays = {
        name: np.concatenate([np.asarray(item.arrays[name]) for item in items], axis=0)
        for name in sorted(keys)
    }
    manifest = copy.deepcopy(reference.manifest)
    manifest.pop("content_sha256", None)
    manifest["generator_commit"] = _merged_generator_commit([
        item.manifest for item in items])
    manifest["shards"] = [
        {
            "ordinal": index,
            "content_sha256": hashes[index],
            "generator_commit": item.manifest["generator_commit"],
            "groups": item.group_count,
            "source_seeds": sorted(set(map(int, item.arrays["source_seed"]))),
        }
        for index, item in enumerate(items)
    ]
    manifest["content_sha256"] = _content_hash(manifest, arrays)
    return GroupedBranchDataset(manifest, arrays)


def save_grouped_shard_blind(dataset: GroupedBranchDataset, path: str | Path) -> Path:
    """Write a mechanically merged shard without semantic outcome validation."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = copy.deepcopy(dataset.manifest)
    manifest["content_sha256"] = _content_hash(manifest, dataset.arrays)
    np.savez_compressed(
        output,
        manifest_json=np.asarray(json.dumps(
            manifest, sort_keys=True, separators=(",", ":"))),
        **dataset.arrays,
    )
    dataset.manifest = manifest
    dataset.path = output
    return output


def merge_privileged_shards(
    views: Sequence[PrivilegedBranchView],
    deployable_shards: Sequence[GroupedBranchDataset],
    combined_deployable: GroupedBranchDataset,
) -> PrivilegedBranchView:
    """Merge physically separate privileged views in deployable shard order."""
    privileged = list(views)
    deployable = list(deployable_shards)
    if len(privileged) != len(deployable) or len(privileged) < 2:
        raise ValueError("privileged and deployable shard lists must align")
    deployable_reports = [
        dataset.validate(summarize_outcomes=False) for dataset in deployable
    ]
    deployable_hashes = [
        _verified_content_hash(
            dataset.manifest, report["content_sha256"],
            label=f"deployable shard {index}",
        )
        for index, (dataset, report) in enumerate(
            zip(deployable, deployable_reports, strict=True))
    ]
    privileged_reports = []
    for index, (view, dataset) in enumerate(
            zip(privileged, deployable, strict=True)):
        report = view.validate(dataset)
        _verified_content_hash(
            view.manifest, report["content_sha256"],
            label=f"privileged shard {index}",
        )
        privileged_reports.append(report)

    combined_report = combined_deployable.validate(
        verify_hash=False, summarize_outcomes=False)
    provenance = combined_deployable.manifest.get("shards")
    expected_provenance = [
        (content_hash, dataset.manifest["generator_commit"])
        for content_hash, dataset in zip(
            deployable_hashes, deployable, strict=True)
    ]
    if not isinstance(provenance, list) or [
            (
                item.get("content_sha256"), item.get("generator_commit"),
            ) if isinstance(item, dict) else (None, None)
            for item in provenance
    ] != expected_provenance or combined_deployable.manifest.get(
            "generator_commit") != _merged_generator_commit([
                dataset.manifest for dataset in deployable]):
        raise ValueError(
            "combined deployable provenance does not match deployable shard order")
    reference_names = np.asarray(privileged[0].feature_names).astype(str)
    reference_contract = _contract_manifest(
        privileged[0].manifest,
        per_shard_fields=("deployable_content_sha256", "generator_commit"),
    )
    for index, view in enumerate(privileged[1:], start=1):
        if not np.array_equal(
                np.asarray(view.feature_names).astype(str), reference_names):
            raise ValueError(f"privileged shard {index} changes feature names/order")
        if _contract_manifest(
                view.manifest,
                per_shard_fields=(
                    "deployable_content_sha256", "generator_commit"),
        ) != reference_contract:
            raise ValueError(
                f"privileged shard {index} changes the causal feature contract")
        if np.asarray(view.features).dtype != np.asarray(
                privileged[0].features).dtype:
            raise ValueError(f"privileged shard {index} changes feature dtype")

    manifest = copy.deepcopy(privileged[0].manifest)
    manifest.pop("content_sha256", None)
    manifest["schema_version"] = PRIVILEGED_SCHEMA_VERSION
    manifest["split"] = combined_deployable.manifest["split"]
    manifest["generator_commit"] = combined_deployable.manifest["generator_commit"]
    manifest["deployable_content_sha256"] = combined_report["content_sha256"]
    manifest["shards"] = [
        {
            "ordinal": index,
            "deployable_content_sha256": deployable_hashes[index],
            "privileged_content_sha256": privileged_reports[index][
                "content_sha256"],
            "generator_commit": dataset.manifest["generator_commit"],
            "groups": dataset.group_count,
        }
        for index, dataset in enumerate(deployable)
    ]
    merged = PrivilegedBranchView(
        manifest=manifest,
        group_id=np.concatenate([view.group_id for view in privileged]),
        state_hash=np.concatenate([view.state_hash for view in privileged]),
        features=np.concatenate([view.features for view in privileged], axis=0),
        feature_names=reference_names.copy(),
    )
    merged.validate(combined_deployable, verify_hash=False)
    return merged


def load_privileged_shard_blind(path: str | Path) -> PrivilegedBranchView:
    """Load privileged sidecar bytes without validating deployable outcomes."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        required = {"manifest_json", "group_id", "state_hash", "features",
                    "feature_names"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"blind privileged shard missing fields: {sorted(missing)}")
        manifest = json.loads(str(payload["manifest_json"].item()))
        arrays = {
            name: payload[name].copy() for name in payload.files
            if name != "manifest_json"
        }
    recorded = manifest.get("content_sha256")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ValueError("blind privileged shard lacks content_sha256")
    actual = _content_hash(manifest, arrays)
    if actual != recorded:
        raise ValueError("blind privileged shard content hash mismatch")
    return PrivilegedBranchView(
        manifest=manifest,
        group_id=arrays["group_id"],
        state_hash=arrays["state_hash"],
        features=arrays["features"],
        feature_names=arrays["feature_names"],
        path=source,
    )


def merge_privileged_shards_blind(
    views: Sequence[PrivilegedBranchView],
    deployable_shards: Sequence[GroupedBranchDataset],
    combined_deployable: GroupedBranchDataset,
) -> PrivilegedBranchView:
    """Merge privileged sidecars using identity/feature bytes only."""
    privileged = list(views)
    deployable = list(deployable_shards)
    if len(privileged) != len(deployable) or len(privileged) < 2:
        raise ValueError("privileged and deployable shard lists must align")
    reference = privileged[0]
    names = np.asarray(reference.feature_names).astype(str)
    contract = _contract_manifest(
        reference.manifest,
        per_shard_fields=("deployable_content_sha256", "generator_commit"),
    )
    hashes: list[str] = []
    deployable_hashes: list[str] = []
    for index, (view, dataset) in enumerate(zip(privileged, deployable, strict=True)):
        actual = _content_hash(view.manifest, {
            "group_id": view.group_id,
            "state_hash": view.state_hash,
            "features": view.features,
            "feature_names": view.feature_names,
        })
        if view.manifest.get("content_sha256") != actual:
            raise ValueError(f"blind privileged shard {index} content hash mismatch")
        if not np.array_equal(np.asarray(view.feature_names).astype(str), names):
            raise ValueError(f"blind privileged shard {index} changes feature names")
        if _contract_manifest(
                view.manifest,
                per_shard_fields=("deployable_content_sha256", "generator_commit"),
        ) != contract:
            raise ValueError(f"blind privileged shard {index} changes feature contract")
        group_id = np.asarray(view.group_id)
        state_hash = np.asarray(view.state_hash)
        features = np.asarray(view.features)
        if group_id.ndim != 1 or state_hash.shape != group_id.shape:
            raise ValueError(f"blind privileged shard {index} identity shape drifted")
        if features.ndim != 2 or features.shape[0] != len(group_id):
            raise ValueError(f"blind privileged shard {index} feature shape drifted")
        if features.shape[1] != len(names):
            raise ValueError(f"blind privileged shard {index} feature width drifted")
        deployable_hash = dataset.manifest.get("content_sha256")
        if not isinstance(deployable_hash, str) or len(deployable_hash) != 64:
            raise ValueError(f"blind deployable shard {index} lacks content hash")
        hashes.append(actual)
        deployable_hashes.append(deployable_hash)
    for name in ("group_id", "state_hash"):
        combined = np.concatenate([
            np.asarray(view.__dict__[name]).astype(str) for view in privileged
        ])
        if len(np.unique(combined)) != combined.size:
            raise ValueError(f"blind privileged shards overlap on {name}")
    merged_manifest = copy.deepcopy(reference.manifest)
    merged_manifest.pop("content_sha256", None)
    merged_manifest["schema_version"] = PRIVILEGED_SCHEMA_VERSION
    merged_manifest["split"] = combined_deployable.manifest.get("split")
    merged_manifest["generator_commit"] = _merged_generator_commit([
        dataset.manifest for dataset in deployable])
    merged_manifest["deployable_content_sha256"] = str(
        combined_deployable.manifest.get("content_sha256"))
    merged_manifest["shards"] = [
        {
            "ordinal": index,
            "deployable_content_sha256": deployable_hashes[index],
            "privileged_content_sha256": hashes[index],
            "generator_commit": dataset.manifest.get("generator_commit"),
            "groups": int(len(np.asarray(view.group_id))),
        }
        for index, (view, dataset) in enumerate(zip(privileged, deployable, strict=True))
    ]
    arrays = {
        "group_id": np.concatenate([np.asarray(view.group_id) for view in privileged]),
        "state_hash": np.concatenate([np.asarray(view.state_hash) for view in privileged]),
        "features": np.concatenate([np.asarray(view.features) for view in privileged], axis=0),
        "feature_names": names.copy(),
    }
    merged_manifest["content_sha256"] = _content_hash(merged_manifest, arrays)
    return PrivilegedBranchView(manifest=merged_manifest, **arrays)


def save_privileged_shard_blind(
    view: PrivilegedBranchView, path: str | Path,
) -> Path:
    """Write privileged sidecar bytes without deployable outcome validation."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "group_id": np.asarray(view.group_id),
        "state_hash": np.asarray(view.state_hash),
        "features": np.asarray(view.features),
        "feature_names": np.asarray(view.feature_names),
    }
    manifest = copy.deepcopy(view.manifest)
    manifest["content_sha256"] = _content_hash(manifest, arrays)
    np.savez_compressed(
        output,
        manifest_json=np.asarray(json.dumps(
            manifest, sort_keys=True, separators=(",", ":"))),
        **arrays,
    )
    view.manifest = manifest
    view.path = output
    return output


__all__ = [
    "load_grouped_shard_blind", "merge_grouped_shards_blind",
    "save_grouped_shard_blind", "merge_grouped_shards", "merge_privileged_shards",
    "load_privileged_shard_blind", "merge_privileged_shards_blind",
    "save_privileged_shard_blind",
]
