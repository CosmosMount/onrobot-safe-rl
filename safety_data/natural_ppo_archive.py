"""Validation and deterministic normal matching for MjLab PPO archives."""

from __future__ import annotations

from collections import Counter, defaultdict
import heapq
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
ROLE_BY_BUCKET = (
    "fit", "fit", "fit", "fit", "fit", "fit", "fit",
    "fit", "fit", "fit", "fit", "fit", "fit", "fit",
    "calibration", "calibration", "calibration",
    "test", "test", "test",
)


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
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite archive validation output: {path}") from exc
    temporary.unlink()
    _fsync_directory(path.parent)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    content = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite archive validation report: {path}") from exc
    temporary.unlink()
    _fsync_directory(path.parent)


def checkpoint_age_bucket(policy_step: int) -> int:
    if policy_step < 0:
        raise ValueError("policy_step must be non-negative")
    for index, boundary in enumerate(AGE_BOUNDARIES):
        if policy_step < boundary:
            return index
    raise ValueError("policy_step exceeds registered 30M exposure")


def episode_split_role(seed: int, environment_id: int, episode_id: int) -> str:
    values = f"{int(seed)}:{int(environment_id)}:{int(episode_id)}".encode("ascii")
    digest = hashlib.sha256(
        b"qsafe.ppo.direct.episode.role.v1\0" + values).hexdigest()
    return ROLE_BY_BUCKET[int(digest[:16], 16) % len(ROLE_BY_BUCKET)]


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


def deterministic_role_pairs(
    prefall: Iterable[tuple[str, int, str, str]],
    normals: Iterable[tuple[str, str, str]],
) -> list[tuple[str, int, str, str, str]]:
    """Match within fixed whole-episode roles and randomization strata."""
    by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    for identity, stratum, role in normals:
        by_key[(stratum, role)].append(identity)
    for values in by_key.values():
        values.sort()
    consumed: dict[tuple[str, str], int] = defaultdict(int)
    result = []
    for fall_identity, offset, stratum, role in sorted(prefall):
        key = (stratum, role)
        index = consumed[key]
        candidates = by_key.get(key, [])
        if index >= len(candidates):
            raise RuntimeError(
                f"insufficient {role} normal states in stratum {stratum}")
        result.append((fall_identity, offset, candidates[index], stratum, role))
        consumed[key] += 1
    return result


def _retain_smallest_identity(
    pools: dict[tuple[str, str], list[tuple[int, str]]],
    key: tuple[str, str],
    identity: str,
    limit: int,
) -> None:
    """Keep the lexicographically smallest ``limit`` SHA-256 identities."""
    if limit <= 0:
        return
    heap = pools.setdefault(key, [])
    numeric = int(identity, 16)
    item = (-numeric, identity)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif numeric < -heap[0][0]:
        heapq.heapreplace(heap, item)


def _load_manifest(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != schema:
        raise ValueError(f"unexpected archive schema at {path}")
    return value


def validate_and_match_archive(root: str | Path, output: str | Path) -> dict[str, Any]:
    root = Path(root)
    output = Path(output)
    if output.exists() or output.with_suffix(".report.json").exists():
        raise FileExistsError("archive validation output path was already consumed")
    fall_manifest_path = root / "manifest.json"
    falls = _load_manifest(fall_manifest_path, "qsafe.mjlab_natural_falls.v2")
    normal_manifest_path = root / falls["provenance"]["normal_manifest"]
    normals = _load_manifest(
        normal_manifest_path,
        "qsafe.mjlab_natural_normals.v2")
    supervision = falls.get("direct_qsafe_supervision", {})
    if supervision != {
            "state_risk": True,
            "executed_action_risk_under_ppo_continuation": "diagnostic_only",
            "counterfactual_recovery_action_risk": False,
            "horizon_policy_steps": 96}:
        raise ValueError("PPO direct-supervision contract drifted")
    if falls.get("external_force") != "verified_zero":
        raise ValueError("natural archive lacks zero-force proof")
    if falls.get("prefall_offsets") != list(PREFALL_OFFSETS):
        raise ValueError("natural archive prefall offsets drifted")
    if normals.get("minimum_future_nonterminal_steps") != NORMAL_TERMINAL_DISTANCE:
        raise ValueError("normal archive terminal-distance contract drifted")

    seen: set[str] = set()
    seed = int(falls["provenance"]["seed"])
    prefall_rows: list[tuple[str, int, str, str]] = []
    fall_count = 0
    for shard in falls["shards"]:
        path = root / shard["path"]
        if _sha256(path) != shard["sha256"]:
            raise ValueError(f"fall shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
            forbidden = {
                "counterfactual_candidate_outcome", "recovery_action",
                "recovery_sequence", "recovery_policy_action",
            }
            if forbidden.intersection(arrays):
                raise ValueError("PPO shard contains counterfactual action labels")
            required = {
                "trajectory_time", "trajectory_qpos", "trajectory_qvel",
                "trajectory_act", "trajectory_qacc_warmstart", "trajectory_ctrl",
                "terminal_time", "terminal_qpos", "terminal_qvel", "terminal_act",
                "terminal_qacc_warmstart", "terminal_ctrl", "rng_identity",
                "ppo_iteration",
            }
            if not required.issubset(arrays):
                raise ValueError("fall shard lacks complete integration/RNG state")
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
                if not np.array_equal(
                        arrays["trajectory_fall_within_96_steps"][row], mask):
                    raise ValueError("positive PPO state-risk labels drifted")
                expected_steps = np.zeros(RING_POLICY_STEPS, dtype=np.int16)
                expected_steps[:length] = np.arange(length, 0, -1)
                if not np.array_equal(
                        arrays["trajectory_steps_to_fall"][row], expected_steps):
                    raise ValueError("PPO steps-to-fall labels drifted")
                command = arrays["trajectory_command"][row][mask]
                if not np.allclose(command, [0.3, 0.0, 0.0], atol=1e-6):
                    raise ValueError("PPO command drifted from constant +0.3 m/s")
                for field in (
                    "trajectory_time", "trajectory_qpos", "trajectory_qvel",
                    "trajectory_act", "trajectory_qacc_warmstart", "trajectory_ctrl",
                    "terminal_time", "terminal_qpos", "terminal_qvel", "terminal_act",
                    "terminal_qacc_warmstart", "terminal_ctrl",
                ):
                    if not np.all(np.isfinite(arrays[field][row])):
                        raise ValueError(f"PPO integration field {field} is non-finite")
                terminal_qpos = np.asarray(arrays["terminal_qpos"][row])
                quat = terminal_qpos[3:7]
                quat = quat / max(float(np.linalg.norm(quat)), 1e-12)
                w, x, y, z = quat
                roll = np.arctan2(
                    2.0 * (w * x + y * z),
                    1.0 - 2.0 * (x * x + y * y))
                pitch = np.arcsin(np.clip(
                    2.0 * (w * y - z * x), -1.0, 1.0))
                if not (terminal_qpos[2] < 0.18 or max(
                        abs(float(roll)), abs(float(pitch))) >= 1.047198):
                    raise ValueError("recorded terminal event is not a target fall")
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
                    role = episode_split_role(
                        seed, int(arrays["environment_id"][row]),
                        int(arrays["episode_id"][row]))
                    prefall_rows.append((identity, offset, stratum, role))
            fall_count += count
    if fall_count != falls["event_count"]:
        raise ValueError("aggregate fall count mismatch")
    if fall_count != falls["provenance"]["independent_fall_episodes"]:
        raise ValueError("fall event count is not independent episode count")

    demand_by_key = Counter((stratum, role) for _, _, stratum, role in prefall_rows)
    normal_pools: dict[tuple[str, str], list[tuple[int, str]]] = {}
    episode_roles: dict[tuple[int, int], str] = {}
    normal_count = 0
    normal_root = (root / falls["provenance"]["normal_manifest"]).parent
    for shard in normals["shards"]:
        path = normal_root / shard["path"]
        if _sha256(path) != shard["sha256"]:
            raise ValueError(f"normal shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
            count = len(arrays["identity"])
            if count != shard["event_count"]:
                raise ValueError("normal shard count mismatch")
            if not np.all(
                    arrays["qualification_future_nonterminal_steps"]
                    == NORMAL_TERMINAL_DISTANCE):
                raise ValueError("normal state lacks 96-step future survival")
            if np.any(arrays["fall_within_96_steps"]):
                raise ValueError("normal state has a positive fall label")
            if not np.all(
                    arrays["outcome_horizon_policy_steps"]
                    == NORMAL_TERMINAL_DISTANCE):
                raise ValueError("normal outcome horizon drifted")
            if not np.allclose(arrays["command"], [0.3, 0.0, 0.0], atol=1e-6):
                raise ValueError("normal command drifted from constant +0.3 m/s")
            if "rng_identity" not in arrays or "ppo_iteration" not in arrays:
                raise ValueError("normal shard lacks RNG/training-age identity")
            identities = [bytes(value).decode("ascii") for value in arrays["identity"]]
            chunk_identities = set(identities)
            if len(chunk_identities) != count or seen.intersection(chunk_identities):
                raise ValueError("duplicate fall/normal identity")
            seen.update(chunk_identities)

            friction_values = arrays["randomized_geom_friction"][..., 0]
            friction = np.round(np.mean(
                friction_values,
                axis=tuple(range(1, friction_values.ndim)),
            ).astype(np.float64), 1)
            body_ipos = np.round(np.linalg.norm(
                arrays["randomized_body_ipos"], axis=(1, 2)
            ).astype(np.float64), 1)
            encoder_bias = np.round(np.linalg.norm(
                arrays["randomized_encoder_bias"], axis=1
            ).astype(np.float64), 2)
            policy_steps = arrays["policy_step"].astype(np.int64, copy=False)
            ages = np.searchsorted(
                np.asarray(AGE_BOUNDARIES, dtype=np.int64), policy_steps,
                side="right")
            if np.any(ages >= len(AGE_BOUNDARIES)):
                raise ValueError("normal policy step exceeds registered 30M exposure")
            environment_ids = arrays["environment_id"].astype(np.int64, copy=False)
            episode_ids = arrays["episode_id"].astype(np.int64, copy=False)
            for row, identity in enumerate(identities):
                base = ":".join(map(str, (
                    float(friction[row]), float(body_ipos[row]),
                    float(encoder_bias[row]))))
                stratum = f"{int(ages[row])}:{base}"
                episode_key = (int(environment_ids[row]), int(episode_ids[row]))
                role = episode_roles.get(episode_key)
                if role is None:
                    role = episode_split_role(seed, *episode_key)
                    episode_roles[episode_key] = role
                key = (stratum, role)
                _retain_smallest_identity(
                    normal_pools, key, identity, demand_by_key.get(key, 0))
            normal_count += count
    if normal_count != normals["event_count"]:
        raise ValueError("aggregate normal count mismatch")

    normal_rows = [
        (identity, stratum, role)
        for (stratum, role), heap in normal_pools.items()
        for _, identity in heap
    ]
    pairs = deterministic_role_pairs(prefall_rows, normal_rows)
    _atomic_npz(
        output,
        fall_identity=np.asarray([row[0] for row in pairs], dtype="S64"),
        prefall_offset=np.asarray([row[1] for row in pairs], dtype=np.int16),
        normal_identity=np.asarray([row[2] for row in pairs], dtype="S64"),
        stratum=np.asarray([row[3] for row in pairs], dtype="S64"),
        split_role=np.asarray([row[4] for row in pairs], dtype="S16"),
    )
    report = {
        "schema_version": "qsafe.natural_ppo_archive_validation.v1",
        "fall_manifest_sha256": _sha256(fall_manifest_path),
        "normal_manifest_sha256": _sha256(normal_manifest_path),
        "fall_events": fall_count,
        "available_prefall_states": len(prefall_rows),
        "normal_candidates": normal_count,
        "normal_candidates_retained_for_matching": len(normal_rows),
        "normal_matching_selection": (
            "lexicographically_smallest_identity_within_frozen_"
            "age_randomization_episode_role_stratum"),
        "matched_normal_states": len(pairs),
        "matched_pairs_by_role": {
            role: sum(row[4] == role for row in pairs)
            for role in ("fit", "calibration", "test")
        },
        "one_to_one_matching_complete": len(pairs) == len(prefall_rows),
        "pair_file": output.name,
        "pair_file_sha256": _sha256(output),
        "command_vx_mps": 0.3,
        "external_force_verified_zero": True,
        "ppo_direct_state_supervision": True,
        "ppo_executed_action_use": "diagnostic_only",
        "ppo_counterfactual_recovery_labels": False,
    }
    report_path = output.with_suffix(".report.json")
    _atomic_json(report_path, report)
    return report
