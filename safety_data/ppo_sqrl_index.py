"""Episode-group index and nested selections for PPO SQRL transition shards."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from safety_data.ppo_sqrl_master import split_role, validate_master_manifest


INDEX_SCHEMA = "qsafe.ppo_sqrl_episode_index.v1"
SELECTION_NAMESPACE = b"qsafe.ppo_sqrl.nested.v1\0"


def episode_key(seed: int, environment_id: int, episode_id: int,
                stage: str) -> str:
    return f"{seed}:{stage}:{environment_id}:{episode_id}"


def _rank(key: str) -> bytes:
    return hashlib.sha256(SELECTION_NAMESPACE + key.encode("ascii")).digest()


def _write_no_clobber(path: Path, value: dict[str, Any]) -> None:
    content = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite {path}") from exc
    temporary.unlink()


def build_episode_index(manifests: Iterable[str | Path]) -> dict[str, Any]:
    episodes: dict[str, dict[str, Any]] = {}
    sources = []
    for raw_manifest in manifests:
        manifest_path = Path(raw_manifest).resolve()
        manifest = validate_master_manifest(manifest_path)
        provenance = manifest["provenance"]
        seed = int(provenance["ppo_seed"])
        stage = str(provenance["collector_stage"])
        sources.append({
            "manifest": str(manifest_path),
            "seed": seed,
            "stage": stage,
            "transitions": int(manifest["transition_count"]),
        })
        for shard in manifest["shards"]:
            path = manifest_path.parent / shard["path"]
            with np.load(path, allow_pickle=False) as loaded:
                env = loaded["env_id"].astype(np.int64)
                episode = loaded["episode_id"].astype(np.int64)
                cost = loaded["c_t_plus_1"].astype(bool)
                randomization = loaded["randomization_identity"].astype(np.uint64)
                for environment_id, episode_id, failed, randomization_id in zip(
                        env.tolist(), episode.tolist(), cost.tolist(),
                        randomization.tolist(), strict=True):
                    key = episode_key(seed, environment_id, episode_id, stage)
                    row = episodes.setdefault(key, {
                        "key": key,
                        "seed": seed,
                        "stage": stage,
                        "environment_id": environment_id,
                        "episode_id": episode_id,
                        "transitions": 0,
                        "fall": False,
                        "randomization_identity": randomization_id,
                        "role": split_role(seed, environment_id, episode_id),
                    })
                    row["transitions"] += 1
                    row["fall"] = bool(row["fall"] or failed)
                    if row["randomization_identity"] != randomization_id:
                        raise ValueError("randomization identity changed within an episode")
    if not episodes:
        raise ValueError("PPO SQRL episode index has no transitions")
    return {
        "schema_version": INDEX_SCHEMA,
        "sources": sources,
        "episodes": sorted(episodes.values(), key=lambda value: value["key"]),
        "transition_count": sum(row["transitions"] for row in episodes.values()),
    }


def nested_episode_selections(
    index: dict[str, Any], budgets: tuple[int, ...] = (1_000_000, 3_000_000, 5_000_000),
) -> dict[str, Any]:
    if index.get("schema_version") != INDEX_SCHEMA or tuple(sorted(budgets)) != budgets:
        raise ValueError("invalid episode index or non-increasing budgets")
    episodes = index["episodes"]
    total = int(index["transition_count"])
    if budgets[-1] > total:
        raise ValueError("largest nested budget exceeds master dataset")
    by_stratum: dict[tuple[int, str, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        by_stratum[(row["seed"], row["stage"], row["fall"])].append(row)
    for rows in by_stratum.values():
        rows.sort(key=lambda row: _rank(row["key"]))

    selections = []
    previous: set[str] = set()
    for budget in budgets:
        selected: set[str] = set()
        stratum_counts = {}
        for stratum, rows in sorted(by_stratum.items()):
            available = sum(row["transitions"] for row in rows)
            target = round(budget * available / total)
            count = sum(
                row["transitions"] for row in rows if row["key"] in previous)
            selected.update(row["key"] for row in rows if row["key"] in previous)
            for row in rows:
                if row["key"] in selected:
                    continue
                if count + row["transitions"] > target:
                    continue
                selected.add(row["key"])
                count += row["transitions"]
            stratum_counts[":".join(map(str, stratum))] = {
                "target": target, "realized": count, "available": available,
            }
        if not previous.issubset(selected):
            raise RuntimeError("nested episode selection lost a smaller-budget episode")
        previous = selected
        realized = sum(
            row["transitions"] for row in episodes if row["key"] in selected)
        selections.append({
            "nominal_budget": budget,
            "realized_transitions": realized,
            "whole_episode_error": realized - budget,
            "episode_keys": sorted(selected),
            "strata": stratum_counts,
        })
    return {
        "schema_version": "qsafe.ppo_sqrl_nested_selection.v1",
        "selection_unit": "whole_episode",
        "selection_hash": "sha256",
        "master_transition_count": total,
        "selections": selections,
    }


def write_index_and_selection(
    manifests: Iterable[str | Path], output: str | Path,
) -> tuple[Path, Path]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    index = build_episode_index(manifests)
    selection = nested_episode_selections(index)
    index_path = output / "episode-index.json"
    selection_path = output / "nested-selection.json"
    _write_no_clobber(index_path, index)
    _write_no_clobber(selection_path, selection)
    return index_path, selection_path
