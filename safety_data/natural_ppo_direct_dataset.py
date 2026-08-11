"""Compile matched natural-PPO states into direct Q_safe supervision."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from safety_data.natural_ppo_falls import NORMAL_TERMINAL_DISTANCE
from safety_data.natural_ppo_archive import episode_split_role


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(namespace: bytes, *values: str) -> str:
    digest = hashlib.sha256(namespace + b"\0")
    for value in values:
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _decode(value: np.generic) -> str:
    return bytes(value).decode("ascii")


def _git_head() -> str:
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True).stdout.strip()


def _publish_no_clobber(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite direct Q_safe artifact: {destination}") from exc
    temporary.unlink()
    descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def compile_direct_qsafe_dataset(
    archive: str | Path,
    pairs_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Create balanced positive/negative samples without frame-level leakage."""
    archive = Path(archive)
    pairs_path = Path(pairs_path)
    output = Path(output)
    report_output = output.with_suffix(".manifest.json")
    if output.exists() or report_output.exists():
        raise FileExistsError("direct Q_safe output path was already consumed")
    manifest_path = archive / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "qsafe.mjlab_natural_falls.v2":
        raise ValueError("direct compiler requires validated natural-fall schema v2")
    supervision = manifest.get("direct_qsafe_supervision", {})
    if supervision.get("state_risk") is not True or supervision.get(
            "executed_action_risk_under_ppo_continuation") != "diagnostic_only" or (
            supervision.get("counterfactual_recovery_action_risk") is not False):
        raise ValueError("archive does not authorize direct state-risk supervision")

    report_path = pairs_path.with_suffix(".report.json")
    pair_report = json.loads(report_path.read_text(encoding="utf-8"))
    if pair_report.get("pair_file_sha256") != _sha256(pairs_path) or not (
            pair_report.get("one_to_one_matching_complete")):
        raise ValueError("normal-match report does not bind a complete pair file")
    with np.load(pairs_path, allow_pickle=False) as pairs:
        fall_ids = [_decode(value) for value in pairs["fall_identity"]]
        offsets = pairs["prefall_offset"].astype(np.int16).tolist()
        normal_ids = [_decode(value) for value in pairs["normal_identity"]]
        strata = [_decode(value) for value in pairs["stratum"]]
        split_roles = [_decode(value) for value in pairs["split_role"]]
    pair_count = len(fall_ids)
    if not (pair_count == len(offsets) == len(normal_ids) == len(strata)
            == len(split_roles)):
        raise ValueError("matched arrays differ in length")
    if len(set(normal_ids)) != pair_count:
        raise ValueError("normal matches are not one-to-one")

    requested_positive = set(zip(fall_ids, offsets, strict=True))
    offsets_by_fall: dict[str, list[int]] = {}
    for fall_identity, offset in requested_positive:
        offsets_by_fall.setdefault(fall_identity, []).append(offset)
    positives: dict[tuple[str, int], dict[str, Any]] = {}
    for shard in manifest["shards"]:
        path = archive / shard["path"]
        if _sha256(path) != shard["sha256"]:
            raise ValueError("fall shard changed after archive validation")
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
            for row, raw_identity in enumerate(arrays["identity"]):
                fall_identity = _decode(raw_identity)
                candidate_offsets = offsets_by_fall.get(fall_identity, ())
                for offset in candidate_offsets:
                    offset_index = manifest["prefall_offsets"].index(offset)
                    index = int(arrays["prefall_trajectory_index"][row, offset_index])
                    if index < 0 or not arrays[
                            "trajectory_fall_within_96_steps"][row, index]:
                        raise ValueError("requested positive lacks a direct fall label")
                    positives[(fall_identity, offset)] = {
                        "observation_history": arrays[
                            "trajectory_observation_history"][row, index].copy(),
                        "action_requested": arrays[
                            "trajectory_action_requested"][row, index].copy(),
                        "action_executed": arrays[
                            "trajectory_action_executed"][row, index].copy(),
                        "q_target": arrays["trajectory_q_target"][row, index].copy(),
                        "command": arrays["trajectory_command"][row, index].copy(),
                        "policy_step": int(arrays[
                            "trajectory_policy_step"][row, index]),
                        "environment_id": int(arrays["environment_id"][row]),
                        "episode_id": int(arrays["episode_id"][row]),
                    }
    if set(positives) != requested_positive:
        raise ValueError("pair file references missing positive states")

    normal_manifest_path = archive / manifest["provenance"]["normal_manifest"]
    normal_manifest = json.loads(normal_manifest_path.read_text(encoding="utf-8"))
    normal_root = normal_manifest_path.parent
    requested_normals = set(normal_ids)
    negatives: dict[str, dict[str, Any]] = {}
    for shard in normal_manifest["shards"]:
        path = normal_root / shard["path"]
        if _sha256(path) != shard["sha256"]:
            raise ValueError("normal shard changed after archive validation")
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
            for row, raw_identity in enumerate(arrays["identity"]):
                identity = _decode(raw_identity)
                if identity not in requested_normals:
                    continue
                if bool(arrays["fall_within_96_steps"][row]) or int(
                        arrays["qualification_future_nonterminal_steps"][row]
                ) < NORMAL_TERMINAL_DISTANCE:
                    raise ValueError("matched normal lacks a negative direct label")
                negatives[identity] = {
                    "observation_history": arrays["observation_history"][row].copy(),
                    "action_requested": arrays["action_requested"][row].copy(),
                    "action_executed": arrays["action_executed"][row].copy(),
                    "q_target": arrays["q_target"][row].copy(),
                    "command": arrays["command"][row].copy(),
                    "policy_step": int(arrays["policy_step"][row]),
                    "environment_id": int(arrays["environment_id"][row]),
                    "episode_id": int(arrays["episode_id"][row]),
                }
    if set(negatives) != requested_normals:
        raise ValueError("pair file references missing normal states")

    seed = int(manifest["provenance"]["seed"])
    pair_episode_keys = []
    for fall_identity, offset, normal_identity, split_role in zip(
            fall_ids, offsets, normal_ids, split_roles, strict=True):
        positive = positives[(fall_identity, offset)]
        negative = negatives[normal_identity]
        fall_episode = f"{seed}:{positive['environment_id']}:{positive['episode_id']}"
        normal_episode = f"{seed}:{negative['environment_id']}:{negative['episode_id']}"
        if episode_split_role(
                seed, positive["environment_id"], positive["episode_id"]
        ) != split_role or episode_split_role(
                seed, negative["environment_id"], negative["episode_id"]
        ) != split_role:
            raise ValueError("matched pair crosses fixed PPO episode roles")
        pair_episode_keys.append((fall_episode, normal_episode))

    rows: dict[str, list[Any]] = {
        name: [] for name in (
            "identity", "pair_identity", "episode_identity",
            "role", "label", "source_kind",
            "steps_to_outcome", "observation_history", "action_requested",
            "action_executed", "q_target", "command", "policy_step",
            "ppo_seed", "environment_id", "episode_id", "fall_identity",
            "prefall_offset", "stratum",
        )
    }
    for index, (fall_identity, offset, normal_identity, stratum, role) in enumerate(zip(
            fall_ids, offsets, normal_ids, strata, split_roles, strict=True)):
        pair_identity = _identity(
            b"qsafe.ppo.direct.pair.v1", fall_identity, str(offset), normal_identity)
        for positive_label, source_identity, episode_key, value in (
            (True, _identity(b"qsafe.ppo.direct.positive.v1",
                             fall_identity, str(offset)),
             pair_episode_keys[index][0], positives[(fall_identity, offset)]),
            (False, normal_identity, pair_episode_keys[index][1],
             negatives[normal_identity]),
        ):
            rows["identity"].append(source_identity)
            rows["pair_identity"].append(pair_identity)
            rows["episode_identity"].append(_identity(
                b"qsafe.ppo.direct.episode.v1", episode_key))
            rows["role"].append(role)
            rows["label"].append(positive_label)
            rows["source_kind"].append("prefall" if positive_label else "normal")
            rows["steps_to_outcome"].append(
                offset if positive_label else NORMAL_TERMINAL_DISTANCE)
            for name in (
                    "observation_history", "action_requested", "action_executed",
                    "q_target", "command", "policy_step", "environment_id",
                    "episode_id"):
                rows[name].append(value[name])
            rows["ppo_seed"].append(seed)
            rows["fall_identity"].append(fall_identity)
            rows["prefall_offset"].append(offset if positive_label else 0)
            rows["stratum"].append(stratum)

    arrays = {
        "identity": np.asarray(rows["identity"], dtype="S64"),
        "pair_identity": np.asarray(rows["pair_identity"], dtype="S64"),
        "episode_identity": np.asarray(rows["episode_identity"], dtype="S64"),
        "role": np.asarray(rows["role"], dtype="S16"),
        "label": np.asarray(rows["label"], dtype=bool),
        "source_kind": np.asarray(rows["source_kind"], dtype="S8"),
        "steps_to_outcome": np.asarray(rows["steps_to_outcome"], dtype=np.int16),
        "observation_history": np.asarray(rows["observation_history"], dtype=np.float32),
        "action_requested": np.asarray(rows["action_requested"], dtype=np.float32),
        "action_executed": np.asarray(rows["action_executed"], dtype=np.float32),
        "q_target": np.asarray(rows["q_target"], dtype=np.float32),
        "command": np.asarray(rows["command"], dtype=np.float32),
        "policy_step": np.asarray(rows["policy_step"], dtype=np.int64),
        "ppo_seed": np.asarray(rows["ppo_seed"], dtype=np.int32),
        "environment_id": np.asarray(rows["environment_id"], dtype=np.int32),
        "episode_id": np.asarray(rows["episode_id"], dtype=np.int64),
        "fall_identity": np.asarray(rows["fall_identity"], dtype="S64"),
        "prefall_offset": np.asarray(rows["prefall_offset"], dtype=np.int16),
        "stratum": np.asarray(rows["stratum"], dtype="S64"),
    }
    for name, value in arrays.items():
        if len(value) != 2 * pair_count:
            raise RuntimeError(f"compiled array {name} has the wrong length")
    if not np.allclose(arrays["command"], [0.3, 0.0, 0.0], atol=1e-6):
        raise ValueError("compiled direct dataset command drifted")
    role_by_episode: dict[bytes, bytes] = {}
    for episode, role in zip(arrays["episode_identity"], arrays["role"], strict=True):
        previous = role_by_episode.setdefault(bytes(episode), bytes(role))
        if previous != bytes(role):
            raise RuntimeError("one PPO episode crossed dataset roles")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    _publish_no_clobber(temporary, output)

    counts = {
        role: int(np.sum(arrays["role"] == role.encode("ascii")))
        for role in ("fit", "calibration", "test")
    }
    report = {
        "schema_version": "qsafe.natural_ppo_direct_dataset.v1",
        "archive_manifest_sha256": _sha256(manifest_path),
        "match_file_sha256": _sha256(pairs_path),
        "dataset_file": output.name,
        "dataset_file_sha256": _sha256(output),
        "compiler_commit": _git_head(),
        "pair_count": pair_count,
        "sample_count": 2 * pair_count,
        "positive_count": pair_count,
        "negative_count": pair_count,
        "role_sample_counts": counts,
        "split_unit": "deterministic_whole_ppo_episode_hash_before_matching",
        "command_vx_mps": 0.3,
        "risk_horizon_policy_steps": NORMAL_TERMINAL_DISTANCE,
        "direct_state_risk_supervision": True,
        "direct_executed_action_risk_supervision": False,
        "executed_action_risk_role": "diagnostic_only",
        "counterfactual_recovery_action_labels": False,
    }
    content = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode("utf-8")
    temporary_json = report_output.with_name(
        f".{report_output.name}.tmp-{os.getpid()}")
    with temporary_json.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    _publish_no_clobber(temporary_json, report_output)
    return report
