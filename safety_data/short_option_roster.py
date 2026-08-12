"""Fresh, outcome-blind Boundary roster for the short-option oracle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


REQUIRED_PER_COLLECTOR = 300
LOW_OFFSET = 32
HIGH_OFFSET = 64


def _u64(namespace: bytes, payload: bytes) -> int:
    return int.from_bytes(hashlib.sha256(namespace + b"\0" + payload).digest()[:8],
                          "little")


def _episode_key(seed: int, rollout_seed: int, environment: int, episode: int) -> str:
    payload = np.asarray([seed, rollout_seed, environment, episode], np.uint64).tobytes()
    return hashlib.sha256(b"qsafe.short-option.episode.v1\0" + payload).hexdigest()


def _state_id(episode: str, offset: int) -> str:
    return hashlib.sha256(
        b"qsafe.short-option.state.v1\0" + episode.encode() + b"\0"
        + str(offset).encode()).hexdigest()


def records_from_fresh_archive(
    root: Path, *, collector_seed: int, rollout_seed: int,
) -> list[dict[str, object]]:
    root = root.resolve()
    result_path = root / "collection-result.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = json.loads(result_path.read_text())
    manifest = json.loads(Path(result["manifest"]).read_text())
    provenance = manifest.get("provenance", {})
    if int(provenance.get("collector_seed", provenance.get("seed", -1))) != collector_seed:
        raise RuntimeError("fresh archive collector seed mismatch")
    if int(provenance.get("rollout_seed", -1)) != rollout_seed:
        raise RuntimeError("fresh archive rollout seed mismatch")
    if int(result.get("aggregate_transitions", -1)) != 32_000_000:
        raise RuntimeError("fresh archive exposure differs from frozen protocol")
    rows: list[dict[str, object]] = []
    for path in sorted((root / "natural-falls").glob("falls-*.npz")):
        with np.load(path, allow_pickle=False) as data:
            for row in range(len(data["identity"])):
                identity = bytes(data["identity"][row])
                offset = LOW_OFFSET + _u64(
                    b"qsafe.short-option.boundary-offset.v1", identity,
                ) % (HIGH_OFFSET - LOW_OFFSET + 1)
                length = int(data["trajectory_length"][row])
                if length < offset:
                    continue
                episode = _episode_key(
                    collector_seed, rollout_seed,
                    int(data["environment_id"][row]), int(data["episode_id"][row]))
                rows.append({
                    "state_id": _state_id(episode, offset),
                    "episode_key": episode,
                    "collector_seed": collector_seed,
                    "rollout_seed": rollout_seed,
                    "risk_stratum": "boundary",
                    "offset": offset,
                    "archive_path": str(path.resolve()),
                    "archive_row": row,
                    "trajectory_index": length - offset,
                })
    return rows


def select_fresh_boundary_roster(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seeds = sorted({int(row["collector_seed"]) for row in rows})
    if seeds != [137, 138]:
        raise ValueError("roster requires collectors 137 and 138")
    selected: list[dict[str, object]] = []
    for seed in seeds:
        eligible = [row for row in rows if int(row["collector_seed"]) == seed]
        eligible.sort(key=lambda row: _u64(
            f"qsafe.short-option.roster.v1:{seed}".encode(),
            str(row["state_id"]).encode()))
        if len(eligible) < REQUIRED_PER_COLLECTOR:
            raise RuntimeError(f"insufficient fresh Boundary states for seed{seed}")
        selected.extend(eligible[:REQUIRED_PER_COLLECTOR])
    episodes = [str(row["episode_key"]) for row in selected]
    states = [str(row["state_id"]) for row in selected]
    if len(selected) != 600 or len(set(episodes)) != 600 or len(set(states)) != 600:
        raise RuntimeError("fresh roster violates exact episode/state isolation")
    return selected


def save_roster(path: Path, rows: list[dict[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray([row[name] for row in rows]) for name in sorted(rows[0])}
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
