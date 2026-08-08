"""Deterministic, identity-safe merging of native grouped data shards."""

from __future__ import annotations

import copy
import json
from typing import Any, Sequence

import numpy as np

from safety_data.schema import (
    GroupedBranchDataset,
    PRIVILEGED_SCHEMA_VERSION,
    PrivilegedBranchView,
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
    reports = [dataset.validate() for dataset in items]
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
    reference_contract = _contract_manifest(reference.manifest)
    for shard_index, dataset in enumerate(items[1:], start=1):
        if set(dataset.arrays) != reference_keys:
            raise ValueError(f"shard {shard_index} array fields differ")
        if _contract_manifest(dataset.manifest) != reference_contract:
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
    manifest["shards"] = [
        {
            "ordinal": index,
            "content_sha256": content_hashes[index],
            "groups": dataset.group_count,
            "source_seeds": sorted(set(map(int, dataset["source_seed"]))),
        }
        for index, dataset in enumerate(items)
    ]
    combined = GroupedBranchDataset(manifest, arrays)
    combined.validate(verify_hash=False)
    return combined


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
    deployable_reports = [dataset.validate() for dataset in deployable]
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

    combined_report = combined_deployable.validate(verify_hash=False)
    provenance = combined_deployable.manifest.get("shards")
    if not isinstance(provenance, list) or [
            item.get("content_sha256") if isinstance(item, dict) else None
            for item in provenance
    ] != deployable_hashes:
        raise ValueError(
            "combined deployable provenance does not match deployable shard order")
    reference_names = np.asarray(privileged[0].feature_names).astype(str)
    reference_contract = _contract_manifest(
        privileged[0].manifest,
        per_shard_fields=("deployable_content_sha256",),
    )
    for index, view in enumerate(privileged[1:], start=1):
        if not np.array_equal(
                np.asarray(view.feature_names).astype(str), reference_names):
            raise ValueError(f"privileged shard {index} changes feature names/order")
        if _contract_manifest(
                view.manifest,
                per_shard_fields=("deployable_content_sha256",),
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


__all__ = ["merge_grouped_shards", "merge_privileged_shards"]
