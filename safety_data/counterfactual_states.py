"""Deterministic episode-disjoint state rosters for counterfactual Q_safe."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np


ROLE_COUNTS = {
    "train": {"boundary": 800, "medium": 400, "normal": 400},
    "calibration": {"boundary": 200, "medium": 100, "normal": 100},
    "protected": {"boundary": 200, "medium": 100, "normal": 100},
}


def stable_u64(namespace: bytes, identity: bytes) -> int:
    return int.from_bytes(hashlib.sha256(namespace + b"\0" + identity).digest()[:8],
                          "little")


def episode_key(seed: int, environment_id: int, episode_id: int) -> str:
    payload = np.asarray([seed, environment_id, episode_id], np.uint64).tobytes()
    return hashlib.sha256(b"qsafe.counterfactual.episode.v2\0" + payload).hexdigest()


def state_identity(episode: str, risk_stratum: str, offset: int) -> str:
    return hashlib.sha256(
        b"qsafe.counterfactual.state.v2\0" + episode.encode("ascii") + b"\0"
        + risk_stratum.encode("ascii") + b"\0" + str(offset).encode("ascii")
    ).hexdigest()


def offset_for(identity: bytes, low: int, high: int, namespace: bytes) -> int:
    if low > high:
        raise ValueError("invalid inclusive offset interval")
    return low + stable_u64(namespace, identity) % (high - low + 1)


def assign_episode_disjoint_roster(
    records: Iterable[dict[str, object]], *, role_counts=ROLE_COUNTS,
) -> list[dict[str, object]]:
    """Assign exact role/stratum/seed quotas without outcome-based replacement."""
    records = list(records)
    seeds = sorted({int(record["collector_seed"]) for record in records})
    if seeds != [137, 138]:
        raise ValueError("counterfactual roster requires collectors 137 and 138")
    used_episodes: set[str] = set()
    selected: list[dict[str, object]] = []
    for role, strata in role_counts.items():
        for stratum, total in strata.items():
            if total % 2:
                raise ValueError("every role/stratum quota must split 1:1 by seed")
            for seed in seeds:
                candidates = [record for record in records
                              if int(record["collector_seed"]) == seed
                              and record["risk_stratum"] == stratum
                              and record["episode_key"] not in used_episodes]
                candidates.sort(key=lambda record: stable_u64(
                    f"qsafe.counterfactual.roster.{role}.{stratum}.{seed}.v2".encode(),
                    str(record["state_id"]).encode("ascii")))
                needed = total // 2
                if len(candidates) < needed:
                    raise RuntimeError(
                        f"insufficient {seed}/{role}/{stratum} states: "
                        f"{len(candidates)} < {needed}")
                for record in candidates[:needed]:
                    copy = dict(record)
                    copy["split"] = role
                    selected.append(copy)
                    used_episodes.add(str(copy["episode_key"]))
    identities = [str(row["state_id"]) for row in selected]
    if len(identities) != len(set(identities)):
        raise RuntimeError("state identity collision")
    return selected


def save_roster(path: Path, rows: list[dict[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(rows[0])
    arrays = {name: np.asarray([row[name] for row in rows]) for name in names}
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)

