"""Validation and deterministic normal matching for MjLab PPO archives."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from safety_data.natural_ppo_falls import (
    NORMAL_TERMINAL_DISTANCE,
    PREFALL_OFFSETS,
    RING_POLICY_STEPS,
)


AGE_BOUNDARIES = (1_000_000, 2_000_000, 5_000_000, 10_000_000,
                  20_000_000, 30_000_001)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    content = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def checkpoint_age_bucket(policy_step: int) -> int:
    if policy_step < 0:
        raise ValueError("policy_step must be non-negative")
    for index, boundary in enumerate(AGE_BOUNDARIES):
        if policy_step < boundary:
            return index
    raise ValueError("policy_step exceeds registered 30M exposure")


def randomization_stratum(friction: np.ndarray, body_ipos: np.ndarray,
                          encoder_bias: np.ndarray) -> str:
    values = (
        round(float(np.mean(np.asarray(friction)[..., 0])), 1),
        round(float(np.linalg.norm(body_ipos)), 1),
        round(float(np.linalg.norm(encoder_bias)), 2),
    )
    return ":".join(map(str, values))


def deterministic_pairs(
    prefall: Iterable[tuple[str, int, str]],
    normals: Iterable[tuple[str, str]],
) -> list[tuple[str, int, str, str]]:
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for identity, stratum in normals:
        by_stratum[stratum].append(identity)
    for values in by_stratum.values():
        values.sort()
    consumed: dict[str, int] = defaultdict(int)
    result = []
    for fall_identity, offset, stratum in sorted(prefall):
        index = consumed[stratum]
        candidates = by_stratum.get(stratum, [])
        if index >= len(candidates):
            raise RuntimeError(f"insufficient normal states in stratum {stratum}")
        result.append((fall_identity, offset, candidates[index], stratum))
        consumed[stratum] += 1
    return result


def _load_manifest(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != schema:
        raise ValueError(f"unexpected archive schema at {path}")
    return value


def validate_and_match_archive(root: str | Path, output: str | Path) -> dict[str, Any]:
    root = Path(root)
    output = Path(output)
    falls = _load_manifest(root / "manifest.json", "qsafe.mjlab_natural_falls.v1")
    normals = _load_manifest(
        root / falls["provenance"]["normal_manifest"],
        "qsafe.mjlab_natural_normals.v1")
    if falls.get("ppo_outcomes_are_qsafe_labels") is not False:
        raise ValueError("PPO archive attempted to provide Q_safe labels")
    if falls.get("external_force") != "verified_zero":
        raise ValueError("natural archive lacks zero-force proof")
    if falls.get("prefall_offsets") != list(PREFALL_OFFSETS):
        raise ValueError("natural archive prefall offsets drifted")
    if normals.get("minimum_future_nonterminal_steps") != NORMAL_TERMINAL_DISTANCE:
        raise ValueError("normal archive terminal-distance contract drifted")

    seen: set[str] = set()
    prefall_rows: list[tuple[str, int, str]] = []
    fall_count = 0
    for shard in falls["shards"]:
        path = root / shard["path"]
        if _sha256(path) != shard["sha256"]:
            raise ValueError(f"fall shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as arrays:
            forbidden = {"fall_label", "qsafe_label", "candidate_outcome"}
            if forbidden.intersection(arrays.files):
                raise ValueError("PPO shard contains forbidden Q_safe labels")
            count = len(arrays["identity"])
            if count != shard["event_count"]:
                raise ValueError("fall shard count mismatch")
            for row in range(count):
                identity = bytes(arrays["identity"][row]).decode("ascii")
                if identity in seen:
                    raise ValueError("duplicate fall/normal identity")
                seen.add(identity)
                length = int(arrays["trajectory_length"][row])
                mask = arrays["trajectory_mask"][row]
                if length < 1 or length > RING_POLICY_STEPS:
                    raise ValueError("invalid fall trajectory length")
                expected_mask = np.arange(RING_POLICY_STEPS) < length
                if not np.array_equal(mask, expected_mask):
                    raise ValueError("fall trajectory mask is not contiguous")
                command = arrays["trajectory_command"][row][mask]
                if not np.allclose(command, [0.4, 0.0, 0.0], atol=1e-6):
                    raise ValueError("PPO command drifted from constant +0.4 m/s")
                availability = arrays["prefall_availability"][row]
                expected = np.asarray(
                    [length >= offset for offset in PREFALL_OFFSETS], dtype=bool)
                if not np.array_equal(availability, expected):
                    raise ValueError("prefall availability mask drifted")
                stratum_base = randomization_stratum(
                    arrays["randomized_geom_friction"][row],
                    arrays["randomized_body_ipos"][row],
                    arrays["randomized_encoder_bias"][row])
                for offset_index, offset in enumerate(PREFALL_OFFSETS):
                    if not availability[offset_index]:
                        continue
                    index = int(arrays["prefall_trajectory_index"][row, offset_index])
                    if index != length - offset:
                        raise ValueError("prefall index does not match registered offset")
                    policy_step = int(arrays["trajectory_policy_step"][row, index])
                    stratum = f"{checkpoint_age_bucket(policy_step)}:{stratum_base}"
                    prefall_rows.append((identity, offset, stratum))
            fall_count += count
    if fall_count != falls["event_count"]:
        raise ValueError("aggregate fall count mismatch")
    if fall_count != falls["provenance"]["independent_fall_episodes"]:
        raise ValueError("fall event count is not independent episode count")

    normal_rows: list[tuple[str, str]] = []
    normal_count = 0
    normal_root = (root / falls["provenance"]["normal_manifest"]).parent
    for shard in normals["shards"]:
        path = normal_root / shard["path"]
        if _sha256(path) != shard["sha256"]:
            raise ValueError(f"normal shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as arrays:
            count = len(arrays["identity"])
            if count != shard["event_count"]:
                raise ValueError("normal shard count mismatch")
            if not np.all(
                    arrays["qualification_future_nonterminal_steps"]
                    == NORMAL_TERMINAL_DISTANCE):
                raise ValueError("normal state lacks 96-step future survival")
            if not np.allclose(arrays["command"], [0.4, 0.0, 0.0], atol=1e-6):
                raise ValueError("normal command drifted from constant +0.4 m/s")
            for row in range(count):
                identity = bytes(arrays["identity"][row]).decode("ascii")
                if identity in seen:
                    raise ValueError("duplicate fall/normal identity")
                seen.add(identity)
                base = randomization_stratum(
                    arrays["randomized_geom_friction"][row],
                    arrays["randomized_body_ipos"][row],
                    arrays["randomized_encoder_bias"][row])
                age = checkpoint_age_bucket(int(arrays["policy_step"][row]))
                normal_rows.append((identity, f"{age}:{base}"))
            normal_count += count
    if normal_count != normals["event_count"]:
        raise ValueError("aggregate normal count mismatch")

    pairs = deterministic_pairs(prefall_rows, normal_rows)
    _atomic_npz(
        output,
        fall_identity=np.asarray([row[0] for row in pairs], dtype="S64"),
        prefall_offset=np.asarray([row[1] for row in pairs], dtype=np.int16),
        normal_identity=np.asarray([row[2] for row in pairs], dtype="S64"),
        stratum=np.asarray([row[3] for row in pairs], dtype="S64"),
    )
    report = {
        "schema_version": "qsafe.natural_ppo_archive_validation.v1",
        "fall_events": fall_count,
        "available_prefall_states": len(prefall_rows),
        "normal_candidates": normal_count,
        "matched_normal_states": len(pairs),
        "one_to_one_matching_complete": len(pairs) == len(prefall_rows),
        "pair_file": output.name,
        "pair_file_sha256": _sha256(output),
        "command_vx_mps": 0.4,
        "external_force_verified_zero": True,
        "ppo_outcomes_are_qsafe_labels": False,
    }
    report_path = output.with_suffix(".report.json")
    _atomic_json(report_path, report)
    return report
