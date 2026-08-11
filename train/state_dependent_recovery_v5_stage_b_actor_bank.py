"""Frozen Stage-B SAC actor-bank checkpoint and inventory contracts.

The Stage-B evidence protocol needs the policy produced at three exact
interaction boundaries for every preregistered SAC-from-zero seed.  Ordinary
asynchronous checkpoints are episode-boundary snapshots and therefore cannot
provide that identity.  This module implements a separate, policy-only export
path and a compiler that accepts the complete frozen 14 x 3 roster or nothing.

No return, fall, or other outcome is inspected when an actor is exported or
selected.  Inclusion is the fixed Cartesian product of the preregistered seed
roster and exact checkpoint steps.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import yaml

from safety_data.paths import (
    assert_development_path,
    assert_safe_evidence_output,
    require_v3_audit_consumed_or_safe_input,
)
from safety_data.policies import load_frozen_droq_policy
from safety_data.state_dependent_recovery_v5 import (
    PROTOCOL_PATH as V5_PROTOCOL_PATH,
    load_state_dependent_recovery_v5_protocol,
)
from safety_data.state_dependent_recovery_v5_stage_b import (
    EXECUTION_PROTOCOL_PATH,
    load_stage_b_execution_protocol,
    stage_b_artifact_root,
    validate_stage_a_authorization,
)


RUN_CONTRACT_SCHEMA_VERSION = (
    "qsafe.state_dependent_recovery_v5_stage_b.actor_run.v1")
ATTEMPT_SCHEMA_VERSION = (
    "qsafe.state_dependent_recovery_v5_stage_b.actor_bank_attempt.v1")
SNAPSHOT_SCHEMA_VERSION = (
    "qsafe.state_dependent_recovery_v5_stage_b.policy_checkpoint.v1")
RUN_COMPLETION_SCHEMA_VERSION = (
    "qsafe.state_dependent_recovery_v5_stage_b.actor_run_complete.v1")
ACTOR_BANK_SCHEMA_VERSION = (
    "qsafe.state_dependent_recovery_v5.stage_b_actor_bank.v1")
CHECKPOINT_SEMANTICS = (
    "after_transition_and_scheduled_update_before_next_transition")
EXACT_CHECKPOINT_STEPS = (25_000, 50_000, 100_000)
ROLE_SEEDS: dict[str, tuple[int, ...]] = {
    "fit": (43, 44, 45, 46),
    "probability_calibration": (47, 48),
    "uncertainty_calibration": (49, 50),
    "selector_calibration": (51, 52),
    "model_test": (53, 54, 55, 56),
}
ALL_ACTOR_SEEDS = tuple(
    seed for seeds in ROLE_SEEDS.values() for seed in seeds)
EXPECTED_IDENTITY_COUNT = len(ALL_ACTOR_SEEDS) * len(EXACT_CHECKPOINT_STEPS)
PRODUCTION_OBSERVATION_DIM = 46
PRODUCTION_ACTION_DIM = 12
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_STEP_DIR = re.compile(r"^step[-_](\d+)$")
_EXECUTION_CHECKPOINT_PATH_TEMPLATE = (
    "stage-b/actor-bank/seed-{training_seed}/"
    "step-{checkpoint_step}/agent")
_EXECUTION_ATTEMPT_MARKER_PATH = "stage-b/actor-bank-attempt-started.json"
_CANONICAL_ACTOR_BANK_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "saved" / "qsafe_development" / "state_dependent_recovery_v5"
    / "stage-b" / "actor-bank-manifest.json"
).resolve(strict=False)


class StageBActorBankError(RuntimeError):
    """Raised when an exact-checkpoint or actor-bank contract fails closed."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _checked_input(path: str | Path, name: str) -> Path:
    checked = assert_development_path(
        require_v3_audit_consumed_or_safe_input(path))
    if not checked.is_file():
        raise StageBActorBankError(f"{name} is not a regular file: {checked}")
    return checked


def _sha256_file(path: str | Path, name: str = "file") -> str:
    checked = _checked_input(path, name)
    digest = hashlib.sha256()
    try:
        with checked.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:  # pragma: no cover - race/IO defensive path.
        raise StageBActorBankError(f"cannot hash {name}: {checked}") from exc
    return digest.hexdigest()


def _read_json(path: str | Path, name: str) -> dict[str, Any]:
    checked = _checked_input(path, name)
    try:
        value = json.loads(checked.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StageBActorBankError(f"{name} is not valid JSON: {checked}") from exc
    if not isinstance(value, dict):
        raise StageBActorBankError(f"{name} must be a JSON object")
    return value


def _read_yaml(path: str | Path, name: str) -> dict[str, Any]:
    checked = _checked_input(path, name)
    try:
        value = yaml.safe_load(checked.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise StageBActorBankError(f"{name} is not valid YAML: {checked}") from exc
    if not isinstance(value, dict):
        raise StageBActorBankError(f"{name} must be a YAML mapping")
    return value


def _atomic_no_clobber_json(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    allow_canonical_actor_bank_manifest: bool = False,
) -> str:
    resolved = assert_development_path(path)
    if (allow_canonical_actor_bank_manifest
            and resolved == _CANONICAL_ACTOR_BANK_MANIFEST):
        # actor-bank-manifest.json is intentionally reserved from generic
        # writers.  This dedicated compiler owns exactly its canonical V5
        # Stage-B path and no same-basename fixture or alias.
        target = resolved
    else:
        target = assert_development_path(assert_safe_evidence_output(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(
        _jsonable(value), sort_keys=True, indent=2, ensure_ascii=True,
        allow_nan=False,
    ) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.pending-")
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise StageBActorBankError(
                f"refusing to clobber existing artifact: {target}") from exc
        published = True
        _fsync_directory(target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if not published:  # pragma: no cover - exceptions leave first.
        raise StageBActorBankError(f"failed to publish {target}")
    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(path: Path) -> None:
    """Durably publish prior directory-entry mutations at ``path``."""
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _self_hashed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    if field in result:
        raise StageBActorBankError(f"self-hash field already present: {field}")
    result[field] = canonical_sha256(result)
    return result


def _validate_self_hash(value: Mapping[str, Any], field: str, name: str) -> str:
    observed = value.get(field)
    basis = dict(value)
    basis.pop(field, None)
    expected = canonical_sha256(basis)
    if not isinstance(observed, str) or observed != expected:
        raise StageBActorBankError(f"{name} {field} is invalid")
    return observed


def _git_identity() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError) as exc:
        raise StageBActorBankError("cannot establish git identity") from exc
    if _HEX40.fullmatch(commit) is None:
        raise StageBActorBankError("git HEAD is not a full lowercase commit")
    return commit, dirty


def _stage_b_node(supplement: Mapping[str, Any]) -> Mapping[str, Any]:
    stage = supplement.get("stage_B", supplement.get("stage_b"))
    if not isinstance(stage, Mapping):
        raise StageBActorBankError(
            "execution supplement requires a stage_B mapping")
    return stage


def validate_actor_roster(
    actor_training_seeds: Mapping[str, Sequence[int]],
    checkpoint_steps: Sequence[int],
) -> dict[str, tuple[int, ...]]:
    """Validate the immutable Stage-B roles and 42 exact identities."""
    if set(actor_training_seeds) != set(ROLE_SEEDS):
        raise StageBActorBankError(
            "Stage-B actor roles differ from the frozen five-role roster")
    normalized: dict[str, tuple[int, ...]] = {}
    for role, expected in ROLE_SEEDS.items():
        raw = actor_training_seeds.get(role)
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise StageBActorBankError(f"actor roster {role} must be a sequence")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in raw):
            raise StageBActorBankError(f"actor roster {role} contains a non-integer")
        observed = tuple(int(item) for item in raw)
        if observed != expected:
            raise StageBActorBankError(
                f"actor roster {role} must equal {list(expected)}")
        normalized[role] = observed
    flattened = tuple(seed for role in ROLE_SEEDS for seed in normalized[role])
    if flattened != ALL_ACTOR_SEEDS or len(set(flattened)) != len(flattened):
        raise StageBActorBankError("actor seeds must be ordered and pairwise distinct")
    if (any(isinstance(step, bool) or not isinstance(step, int)
            for step in checkpoint_steps)
            or tuple(int(step) for step in checkpoint_steps)
            != EXACT_CHECKPOINT_STEPS):
        raise StageBActorBankError(
            "actor checkpoint steps must equal [25000, 50000, 100000]")
    return normalized


def actor_roster_from_supplement(
    supplement: Mapping[str, Any],
) -> dict[str, tuple[int, ...]]:
    # The immutable parent protocol records the role-partitioned roster under
    # stage_B.  The execution supplement intentionally repeats the same
    # identity as one ordered flat list under actor_bank.  Accept either
    # representation, but normalize both to the one frozen role mapping.
    if isinstance(supplement.get("stage_B"), Mapping) or isinstance(
            supplement.get("stage_b"), Mapping):
        stage = _stage_b_node(supplement)
        roster = stage.get("actor_training_seeds")
        assignment = stage.get("actor_source_assignment")
        if not isinstance(roster, Mapping) or not isinstance(
                assignment, Mapping):
            raise StageBActorBankError(
                "stage_B requires actor_training_seeds and "
                "actor_source_assignment")
        steps = assignment.get("actor_checkpoint_steps")
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            raise StageBActorBankError(
                "stage_B.actor_source_assignment.actor_checkpoint_steps "
                "is missing")
        return validate_actor_roster(roster, steps)

    actor_bank = supplement.get("actor_bank")
    if not isinstance(actor_bank, Mapping):
        raise StageBActorBankError(
            "execution supplement requires an actor_bank mapping")
    seeds = actor_bank.get("training_seeds_exact")
    steps = actor_bank.get("checkpoint_steps_exact")
    if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)) or (
            any(isinstance(seed, bool) or not isinstance(seed, int)
                for seed in seeds)) or tuple(int(seed) for seed in seeds) != (
                    ALL_ACTOR_SEEDS):
        raise StageBActorBankError(
            "actor_bank.training_seeds_exact must equal seeds 43..56")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        raise StageBActorBankError(
            "actor_bank.checkpoint_steps_exact is missing")
    roster = validate_actor_roster(ROLE_SEEDS, steps)
    expected = {
        "checkpoint_count_exact": EXPECTED_IDENTITY_COUNT,
        "snapshot_kind": "policy_only",
        "snapshot_timing": CHECKPOINT_SEMANTICS,
        "nearby_or_episode_boundary_checkpoint_substitution": "forbidden",
        "retain_every_seed_and_checkpoint_without_return_or_fall_filter": True,
        "checkpoint_path_template": _EXECUTION_CHECKPOINT_PATH_TEMPLATE,
        "attempt_marker_path": _EXECUTION_ATTEMPT_MARKER_PATH,
    }
    if any(actor_bank.get(key) != value for key, value in expected.items()):
        raise StageBActorBankError(
            "execution supplement actor_bank frozen semantics differ")
    return roster


def _execution_checkpoint_template(supplement: Mapping[str, Any]) -> str:
    actor_bank = supplement.get("actor_bank")
    if not isinstance(actor_bank, Mapping) or actor_bank.get(
            "checkpoint_path_template") != _EXECUTION_CHECKPOINT_PATH_TEMPLATE:
        raise StageBActorBankError(
            "execution supplement checkpoint path template differs")
    return _EXECUTION_CHECKPOINT_PATH_TEMPLATE


def _execution_attempt_marker_path(
    supplement: Mapping[str, Any], actor_root: Path,
) -> Path:
    actor_bank = supplement.get("actor_bank")
    if not isinstance(actor_bank, Mapping) or actor_bank.get(
            "attempt_marker_path") != _EXECUTION_ATTEMPT_MARKER_PATH:
        raise StageBActorBankError(
            "execution supplement actor-bank attempt path differs")
    expected = actor_root.parent / "actor-bank-attempt-started.json"
    if expected.parts[-2:] != ("stage-b", "actor-bank-attempt-started.json"):
        raise StageBActorBankError("actor-bank attempt path is not canonical")
    return expected


def _checkpoint_directory_name(step: int) -> str:
    if step not in EXACT_CHECKPOINT_STEPS:
        raise StageBActorBankError(f"non-frozen checkpoint step {step}")
    return f"step-{step}"


def _validate_actor_root_shape(path: str | Path) -> Path:
    root = assert_development_path(path)
    if len(root.parts) < 2 or root.parts[-2:] != ("stage-b", "actor-bank"):
        raise StageBActorBankError(
            "actor_root must end in stage-b/actor-bank as frozen by the "
            "execution supplement")
    return root


def _role_for_seed(seed: int) -> str:
    matches = [role for role, seeds in ROLE_SEEDS.items() if seed in seeds]
    if len(matches) != 1:
        raise StageBActorBankError(f"seed {seed} is outside the Stage-B roster")
    return matches[0]


def _binding(path: str | Path, name: str) -> dict[str, str]:
    checked = _checked_input(path, name)
    return {"path": str(checked), "file_sha256": _sha256_file(checked, name)}


def _validate_canonical_science_bindings(
    *,
    supplement_path: str | Path,
    protocol_path: str | Path,
    stage_a_report_path: str | Path,
    training_config_path: str | Path,
    actor_root: str | Path,
    output_path: str | Path | None = None,
) -> None:
    supplement_checked = _checked_input(
        supplement_path, "execution supplement")
    protocol_checked = _checked_input(protocol_path, "V5 protocol")
    if supplement_checked != EXECUTION_PROTOCOL_PATH.resolve(strict=True) or (
            protocol_checked != V5_PROTOCOL_PATH.resolve(strict=True)):
        raise StageBActorBankError(
            "production actor bank requires the canonical V5 protocol and "
            "Stage-B execution supplement")
    execution = load_stage_b_execution_protocol(supplement_checked)
    parent = load_state_dependent_recovery_v5_protocol(protocol_checked)
    validate_stage_a_authorization(execution)
    stage_b_root = stage_b_artifact_root(parent).resolve(strict=False)
    expected_report = stage_b_root.parent / str(
        execution["authorization"]["stage_a_report_relative_path"])
    expected_config = (Path(__file__).resolve().parents[1] / str(
        execution["actor_bank"]["policy_config_path"])).resolve(strict=False)
    if _checked_input(stage_a_report_path, "Stage-A report") != expected_report:
        raise StageBActorBankError("Stage-A report path is not canonical")
    if _checked_input(training_config_path, "training config") != expected_config:
        raise StageBActorBankError("actor training config path is not canonical")
    if _validate_actor_root_shape(actor_root) != stage_b_root / "actor-bank":
        raise StageBActorBankError("actor_root is not the canonical Stage-B root")
    if output_path is not None and assert_development_path(
            output_path) != stage_b_root / "actor-bank-manifest.json":
        raise StageBActorBankError(
            "actor-bank manifest output is not the canonical Stage-B path")


def prepare_actor_run_contracts(
    *,
    supplement_path: str | Path,
    protocol_path: str | Path,
    stage_a_report_path: str | Path,
    training_config_path: str | Path,
    actor_root: str | Path,
    contracts_root: str | Path,
    generator_commit: str,
    require_clean_git: bool = True,
    enforce_canonical_bindings: bool = True,
) -> list[Path]:
    """Publish the 14 no-clobber run contracts before actor training."""
    if _HEX40.fullmatch(generator_commit) is None:
        raise StageBActorBankError("generator_commit must be a full commit hash")
    if require_clean_git:
        live_commit, dirty = _git_identity()
        if dirty or live_commit != generator_commit:
            raise StageBActorBankError(
                "actor run contracts require the named clean generator commit")
    if enforce_canonical_bindings:
        _validate_canonical_science_bindings(
            supplement_path=supplement_path,
            protocol_path=protocol_path,
            stage_a_report_path=stage_a_report_path,
            training_config_path=training_config_path,
            actor_root=actor_root,
        )

    supplement = _read_yaml(supplement_path, "execution supplement")
    roster = actor_roster_from_supplement(supplement)
    checkpoint_path_template = _execution_checkpoint_template(supplement)
    protocol = _read_yaml(protocol_path, "V5 protocol")
    protocol_stage = _stage_b_node(protocol)
    protocol_roster = protocol_stage.get("actor_training_seeds")
    protocol_assignment = protocol_stage.get("actor_source_assignment")
    if not isinstance(protocol_roster, Mapping) or not isinstance(
            protocol_assignment, Mapping):
        raise StageBActorBankError("V5 protocol Stage-B actor contract is missing")
    validate_actor_roster(
        protocol_roster,
        protocol_assignment.get("actor_checkpoint_steps", []),
    )

    actor_root_path = _validate_actor_root_shape(actor_root)
    contracts_root_path = assert_development_path(contracts_root)
    attempt_marker_path = _execution_attempt_marker_path(
        supplement, actor_root_path)
    bindings = {
        "protocol": _binding(protocol_path, "V5 protocol"),
        "execution_supplement": _binding(
            supplement_path, "execution supplement"),
        "stage_a_report": _binding(stage_a_report_path, "Stage-A report"),
        "training_config": _binding(training_config_path, "training config"),
    }
    bindings["protocol"]["contract_sha256"] = canonical_sha256(protocol)
    bindings["execution_supplement"]["contract_sha256"] = canonical_sha256(
        supplement)
    roster_manifest = {role: list(seeds) for role, seeds in roster.items()}
    roster_sha256 = canonical_sha256({
        "actor_training_seeds": roster_manifest,
        "actor_checkpoint_steps": list(EXACT_CHECKPOINT_STEPS),
    })

    outputs = [contracts_root_path / f"seed-{seed}.json"
               for seed in ALL_ACTOR_SEEDS]
    occupied = [path for path in [attempt_marker_path, *outputs]
                if os.path.lexists(path)]
    if occupied:
        raise StageBActorBankError(
            f"refusing to clobber actor run contracts: {occupied}")
    for seed in ALL_ACTOR_SEEDS:
        run_dir = actor_root_path / f"seed-{seed}"
        if os.path.lexists(run_dir):
            raise StageBActorBankError(
                f"actor run directory already exists: {run_dir}")

    attempt_marker = _self_hashed({
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "generator_commit": generator_commit,
        "actor_root": str(actor_root_path),
        "contracts_root": str(contracts_root_path),
        "fixed_actor_roster": roster_manifest,
        "actor_roster_sha256": roster_sha256,
        "checkpoint_steps": list(EXACT_CHECKPOINT_STEPS),
        "checkpoint_semantics": CHECKPOINT_SEMANTICS,
        "checkpoint_path_template": checkpoint_path_template,
        "expected_actor_identity_count": EXPECTED_IDENTITY_COUNT,
        "actor_inclusion_rule": (
            "all_preregistered_seed_step_identities_no_filtering"),
        "return_or_fall_filtering": "forbidden",
        "created_before_first_training_transition": True,
        "bindings": bindings,
    }, "attempt_contract_sha256")
    attempt_file_sha256 = _atomic_no_clobber_json(
        attempt_marker_path, attempt_marker)
    attempt_binding = {
        "path": str(attempt_marker_path),
        "file_sha256": attempt_file_sha256,
        "contract_sha256": attempt_marker["attempt_contract_sha256"],
    }

    published: list[Path] = []
    for seed, output in zip(ALL_ACTOR_SEEDS, outputs, strict=True):
        contract = _self_hashed({
            "schema_version": RUN_CONTRACT_SCHEMA_VERSION,
            "generator_commit": generator_commit,
            "actor_training_seed": seed,
            "role": _role_for_seed(seed),
            "checkpoint_steps": list(EXACT_CHECKPOINT_STEPS),
            "checkpoint_semantics": CHECKPOINT_SEMANTICS,
            "checkpoint_path_template": checkpoint_path_template,
            "checkpoint_directory_names": [
                _checkpoint_directory_name(step)
                for step in EXACT_CHECKPOINT_STEPS
            ],
            "training_max_steps_exact": EXACT_CHECKPOINT_STEPS[-1],
            "training_save_dir": str(actor_root_path / f"seed-{seed}"),
            "fixed_actor_roster": roster_manifest,
            "actor_roster_sha256": roster_sha256,
            "expected_actor_identity_count": EXPECTED_IDENTITY_COUNT,
            "actor_inclusion_rule": (
                "all_preregistered_seed_step_identities_no_filtering"),
            "return_or_fall_filtering": "forbidden",
            "checkpoint_substitution": "forbidden",
            "nearby_checkpoint_substitution": "forbidden",
            "policy_only_actor_pt": True,
            "actor_bank_attempt_binding": attempt_binding,
            "bindings": bindings,
        }, "run_contract_sha256")
        _atomic_no_clobber_json(output, contract)
        published.append(output)
    return published


def _validate_actor_bank_attempt_binding(
    raw: object,
    *,
    contract: Mapping[str, Any],
    run_contract_path: Path,
) -> dict[str, str]:
    if not isinstance(raw, Mapping) or set(raw) != {
            "path", "file_sha256", "contract_sha256"}:
        raise StageBActorBankError("actor-bank attempt binding is malformed")
    binding = {str(key): str(value) for key, value in raw.items()}
    if any(_HEX64.fullmatch(binding[field]) is None
           for field in ("file_sha256", "contract_sha256")):
        raise StageBActorBankError("actor-bank attempt hashes are invalid")
    marker_path = _checked_input(binding["path"], "actor-bank attempt marker")
    run_dir = assert_development_path(contract["training_save_dir"])
    expected_path = run_dir.parent.parent / "actor-bank-attempt-started.json"
    if marker_path != expected_path or _sha256_file(
            marker_path, "actor-bank attempt marker") != binding[
                "file_sha256"]:
        raise StageBActorBankError("actor-bank attempt marker binding differs")
    marker = _read_json(marker_path, "actor-bank attempt marker")
    marker_contract_sha256 = _validate_self_hash(
        marker, "attempt_contract_sha256", "actor-bank attempt marker")
    if marker_contract_sha256 != binding["contract_sha256"]:
        raise StageBActorBankError("actor-bank attempt contract hash differs")
    expected = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "generator_commit": contract["generator_commit"],
        "actor_root": str(run_dir.parent),
        "fixed_actor_roster": contract["fixed_actor_roster"],
        "actor_roster_sha256": contract["actor_roster_sha256"],
        "checkpoint_steps": list(EXACT_CHECKPOINT_STEPS),
        "checkpoint_semantics": CHECKPOINT_SEMANTICS,
        "checkpoint_path_template": _EXECUTION_CHECKPOINT_PATH_TEMPLATE,
        "expected_actor_identity_count": EXPECTED_IDENTITY_COUNT,
        "actor_inclusion_rule": (
            "all_preregistered_seed_step_identities_no_filtering"),
        "return_or_fall_filtering": "forbidden",
        "created_before_first_training_transition": True,
        "bindings": contract["bindings"],
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise StageBActorBankError("actor-bank attempt marker identity differs")
    contracts_root = marker.get("contracts_root")
    if not isinstance(contracts_root, str) or Path(
            contracts_root).resolve(strict=False) != run_contract_path.parent:
        raise StageBActorBankError(
            "actor-bank attempt marker contracts_root is invalid")
    return binding


def load_actor_run_contract(
    path: str | Path,
    *,
    verify_live_bindings: bool = True,
    require_clean_git: bool = True,
) -> dict[str, Any]:
    contract = _read_json(path, "actor run contract")
    _validate_self_hash(contract, "run_contract_sha256", "actor run contract")
    required = {
        "schema_version", "generator_commit", "actor_training_seed", "role",
        "checkpoint_steps", "checkpoint_semantics", "training_max_steps_exact",
        "checkpoint_path_template", "checkpoint_directory_names",
        "training_save_dir", "fixed_actor_roster", "actor_roster_sha256",
        "expected_actor_identity_count", "actor_inclusion_rule",
        "return_or_fall_filtering", "checkpoint_substitution",
        "nearby_checkpoint_substitution", "policy_only_actor_pt",
        "actor_bank_attempt_binding", "bindings", "run_contract_sha256",
    }
    if set(contract) != required:
        raise StageBActorBankError(
            "actor run contract fields differ from the frozen schema")
    seed = contract.get("actor_training_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise StageBActorBankError("actor_training_seed must be an integer")
    if contract.get("schema_version") != RUN_CONTRACT_SCHEMA_VERSION or (
            contract.get("generator_commit") is None) or _HEX40.fullmatch(
                str(contract["generator_commit"])) is None or (
                    contract.get("role") != _role_for_seed(seed)) or (
                        contract.get("checkpoint_steps") !=
                        list(EXACT_CHECKPOINT_STEPS)) or (
                            contract.get("checkpoint_semantics") !=
                            CHECKPOINT_SEMANTICS) or (
                                contract.get("checkpoint_path_template") !=
                                _EXECUTION_CHECKPOINT_PATH_TEMPLATE) or (
                                    contract.get("checkpoint_directory_names") != [
                                        _checkpoint_directory_name(step)
                                        for step in EXACT_CHECKPOINT_STEPS]) or (
                                contract.get("training_max_steps_exact") !=
                                EXACT_CHECKPOINT_STEPS[-1]) or (
                                    contract.get("expected_actor_identity_count") !=
                                    EXPECTED_IDENTITY_COUNT) or (
                                        contract.get("actor_inclusion_rule") !=
                                        "all_preregistered_seed_step_identities_no_filtering") or (
                                            contract.get("return_or_fall_filtering") !=
                                            "forbidden") or (
                                                contract.get(
                                                    "checkpoint_substitution") !=
                                                "forbidden") or (
                                                    contract.get(
                                                        "nearby_checkpoint_substitution") !=
                                                    "forbidden") or (
                                                        contract.get(
                                                            "policy_only_actor_pt")
                                                        is not True):
        raise StageBActorBankError("actor run contract frozen values differ")
    roster = validate_actor_roster(
        contract.get("fixed_actor_roster", {}),
        contract.get("checkpoint_steps", []),
    )
    roster_sha256 = canonical_sha256({
        "actor_training_seeds": {
            role: list(seeds) for role, seeds in roster.items()},
        "actor_checkpoint_steps": list(EXACT_CHECKPOINT_STEPS),
    })
    if contract.get("actor_roster_sha256") != roster_sha256:
        raise StageBActorBankError("actor roster hash differs")
    _validate_actor_bank_attempt_binding(
        contract.get("actor_bank_attempt_binding"),
        contract=contract,
        run_contract_path=_checked_input(path, "actor run contract"),
    )

    run_dir = assert_development_path(str(contract["training_save_dir"]))
    if run_dir.name != f"seed-{seed}":
        raise StageBActorBankError("training_save_dir does not identify its seed")
    bindings = contract.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
            "protocol", "execution_supplement", "stage_a_report",
            "training_config"}:
        raise StageBActorBankError("actor run contract bindings are invalid")
    if verify_live_bindings:
        for name, raw in bindings.items():
            if not isinstance(raw, Mapping) or set(raw) not in (
                    {"path", "file_sha256"},
                    {"path", "file_sha256", "contract_sha256"}):
                raise StageBActorBankError(f"binding {name} is malformed")
            expected = raw.get("file_sha256")
            if _HEX64.fullmatch(str(expected)) is None or _sha256_file(
                    str(raw.get("path")), f"{name} binding") != expected:
                raise StageBActorBankError(f"binding {name} changed")
            if "contract_sha256" in raw:
                parsed = _read_yaml(str(raw["path"]), f"{name} binding")
                if canonical_sha256(parsed) != raw["contract_sha256"]:
                    raise StageBActorBankError(
                        f"binding {name} canonical contract changed")
        supplement = _read_yaml(
            str(bindings["execution_supplement"]["path"]),
            "execution supplement binding",
        )
        actor_roster_from_supplement(supplement)
        _execution_checkpoint_template(supplement)
        live_commit, dirty = _git_identity()
        if live_commit != contract["generator_commit"] or (
                require_clean_git and dirty):
            raise StageBActorBankError(
                "live git identity differs from actor run contract")
    return contract


def configure_training_for_actor_run_contract(
    cfg: Any,
    agent_cfg: Any,
    robot_cfg: Any,
    *,
    contract_path: str | Path,
    source_config_path: str | Path,
) -> dict[str, Any]:
    """Validate a run contract and apply only its frozen run identity."""
    contract = load_actor_run_contract(contract_path)
    binding = contract["bindings"]["training_config"]
    supplied_config = _checked_input(source_config_path, "training config")
    if str(supplied_config) != str(binding["path"]) or _sha256_file(
            supplied_config, "training config") != binding["file_sha256"]:
        raise StageBActorBankError(
            "training CLI config differs from the run-contract binding")
    if str(getattr(agent_cfg, "agent_type", "")) != "droq" or (
            str(getattr(cfg, "agent", "")) != "droq"):
        raise StageBActorBankError("Stage-B actor runs require plain DroQ SAC")
    if not bool(getattr(cfg, "async_collection", False)):
        raise StageBActorBankError("Stage-B actor runs require async collection")
    if float(getattr(robot_cfg, "move_speed", float("nan"))) != 0.30:
        raise StageBActorBankError("Stage-B actor runs require 0.30 m/s")
    if int(getattr(cfg, "start_training", -1)) >= EXACT_CHECKPOINT_STEPS[0]:
        raise StageBActorBankError("training must start before the first checkpoint")

    seed = int(contract["actor_training_seed"])
    cfg.seed = seed
    agent_cfg.seed = seed
    cfg.save_dir = str(contract["training_save_dir"])
    cfg.max_steps = EXACT_CHECKPOINT_STEPS[-1]
    cfg.save_checkpoints = False
    cfg.resume_checkpoint = False
    cfg.benchmark_only = False
    cfg.stage_b_actor_run_contract = str(
        _checked_input(contract_path, "actor run contract"))
    cfg.experiment_name = f"qsafe-v5-stage-b-sac-from-zero-seed-{seed}"
    cfg.wandb_run_name = f"qsafe-v5-stage-b-sac-030-seed-{seed}"
    if os.path.lexists(cfg.save_dir):
        raise StageBActorBankError(
            f"actor run output already exists: {cfg.save_dir}")
    return contract


def _state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
    if not state_dict:
        raise StageBActorBankError("actor state_dict must be nonempty")
    digest = hashlib.sha256(b"qsafe_actor_state_dict_v1\0")
    for name in sorted(state_dict):
        if not isinstance(name, str):
            raise StageBActorBankError("actor state_dict names must be strings")
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise StageBActorBankError(
                f"actor state_dict entry {name!r} must be a dense tensor")
        value = tensor.detach()
        if value.is_meta:
            raise StageBActorBankError(
                f"actor state_dict entry {name!r} is not materialized")
        value = value.cpu().contiguous()
        if (value.is_floating_point() or value.is_complex()) and not bool(
                torch.isfinite(value).all()):
            raise StageBActorBankError(
                f"actor state_dict entry {name!r} is not finite")
        metadata = json.dumps({
            "name": name,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8")
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


@dataclass(frozen=True)
class ExportedPolicyCheckpoint:
    actor_training_seed: int
    checkpoint_step: int
    actor_path: str
    actor_sha256: str
    actor_state_dict_sha256: str
    snapshot_manifest_path: str
    snapshot_manifest_sha256: str


class ExactPolicyCheckpointExporter:
    """One-shot exact-step exporter used inside the async learner loop."""

    def __init__(
        self,
        contract_path: str | Path,
        *,
        cfg: Any,
        verify_live_bindings: bool = True,
        require_clean_git: bool = True,
    ) -> None:
        self._contract_path = _checked_input(
            contract_path, "actor run contract")
        self._contract = load_actor_run_contract(
            self._contract_path,
            verify_live_bindings=verify_live_bindings,
            require_clean_git=require_clean_git,
        )
        self._verify_live_bindings = bool(verify_live_bindings)
        self._require_clean_git = bool(require_clean_git)
        self._seed = int(self._contract["actor_training_seed"])
        self._run_dir = assert_development_path(
            self._contract["training_save_dir"])
        if int(getattr(cfg, "seed", -1)) != self._seed or Path(
                str(getattr(cfg, "save_dir", ""))).resolve(strict=False) != (
                    self._run_dir):
            raise StageBActorBankError(
                "live training seed/save_dir differ from actor run contract")
        if int(getattr(cfg, "max_steps", -1)) != EXACT_CHECKPOINT_STEPS[-1]:
            raise StageBActorBankError(
                "live actor training max_steps must equal 100000")
        if bool(getattr(cfg, "resume_checkpoint", False)):
            raise StageBActorBankError("Stage-B actor training cannot resume")
        self._run_dir.mkdir(parents=True, exist_ok=True)
        # ``run_async_training`` normally creates save_dir immediately before
        # constructing this exporter.  Persist that entry now so a crash after
        # the first transition cannot make the attempted run disappear and be
        # silently retried.
        _fsync_directory(self._run_dir.parent)
        _fsync_directory(self._run_dir)
        nearby = [path for path in self._run_dir.iterdir()
                  if _STEP_DIR.fullmatch(path.name)]
        if nearby:
            raise StageBActorBankError(
                f"actor run already contains checkpoint directories: {nearby}")
        self._exported: list[ExportedPolicyCheckpoint] = []

    @property
    def contract_sha256(self) -> str:
        return str(self._contract["run_contract_sha256"])

    @property
    def exported_steps(self) -> tuple[int, ...]:
        return tuple(item.checkpoint_step for item in self._exported)

    def _require_live_identity(self) -> None:
        # Revalidate every bound file and clean generator identity at each of
        # the three irreversible checkpoint publications.
        live = load_actor_run_contract(
            self._contract_path,
            verify_live_bindings=self._verify_live_bindings,
            require_clean_git=self._require_clean_git,
        )
        if live != self._contract:
            raise StageBActorBankError("actor run contract changed after startup")

    def maybe_export(self, agent: Any, policy_training_step: int) -> bool:
        if isinstance(policy_training_step, bool) or not isinstance(
                policy_training_step, int) or policy_training_step < 0:
            raise StageBActorBankError(
                "policy_training_step must be a nonnegative integer")
        if policy_training_step in self.exported_steps:
            raise StageBActorBankError(
                f"duplicate exact checkpoint hook at step {policy_training_step}")
        next_index = len(self._exported)
        if next_index >= len(EXACT_CHECKPOINT_STEPS):
            return False
        expected = EXACT_CHECKPOINT_STEPS[next_index]
        if policy_training_step > expected:
            raise StageBActorBankError(
                f"missed exact checkpoint {expected}; observed {policy_training_step}")
        if policy_training_step != expected:
            return False
        self._export(agent, policy_training_step)
        return True

    def _export(self, agent: Any, step: int) -> None:
        self._require_live_identity()
        snapshot = agent.export_inference_snapshot(snapshot_version=step)
        if not isinstance(snapshot, Mapping) or snapshot.get(
                "agent_type") != "droq":
            raise StageBActorBankError(
                "exact actor checkpoint requires a DroQ inference snapshot")
        state_dict = snapshot.get("actor_state_dict")
        if not isinstance(state_dict, Mapping):
            raise StageBActorBankError("inference snapshot lacks actor_state_dict")
        state_sha256 = _state_dict_sha256(state_dict)
        counters = agent.get_update_counters()
        if not isinstance(counters, Mapping):
            raise StageBActorBankError("agent update counters must be a mapping")

        step_dir = self._run_dir / _checkpoint_directory_name(step)
        if os.path.lexists(step_dir):
            raise StageBActorBankError(
                f"refusing to clobber exact checkpoint: {step_dir}")
        try:
            step_dir.mkdir()
            # Reserve the exact-step attempt before creating any children.
            # Any later failure must leave a durable partial that fails closed.
            _fsync_directory(self._run_dir)
            agent_dir = step_dir / "agent"
            agent_dir.mkdir()
            _fsync_directory(step_dir)
            actor_path = assert_development_path(
                assert_safe_evidence_output(agent_dir / "actor.pt"))
            # Keep the established actor.pt loader contract while omitting
            # critic, replay, optimizer, scheduler, and temperature state.
            torch.save({
                "network_state_dict": {
                    str(name): tensor.detach().cpu().clone()
                    for name, tensor in state_dict.items()
                },
                "optimizer_state_dict": None,
                "scheduler_state_dict": None,
                "update_step": int(snapshot.get("actor_steps", 0)),
            }, actor_path)
            with actor_path.open("rb") as stream:
                os.fsync(stream.fileno())
            # A durable inode is insufficient unless its name in agent/ is
            # durable as well.
            _fsync_directory(agent_dir)
            actor_sha256 = _sha256_file(actor_path, "policy-only actor")
            snapshot_manifest = _self_hashed({
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "generator_commit": self._contract["generator_commit"],
                "actor_training_seed": self._seed,
                "role": self._contract["role"],
                "policy_training_step": step,
                "checkpoint_semantics": CHECKPOINT_SEMANTICS,
                "run_contract_path": str(self._contract_path),
                "run_contract_sha256": self.contract_sha256,
                "actor_relative_path": "agent/actor.pt",
                "actor_sha256": actor_sha256,
                "actor_state_dict_sha256": state_sha256,
                "policy_only": True,
                "excluded_state": [
                    "critic", "target_critic", "temperature", "replay_buffer",
                    "actor_optimizer", "actor_scheduler",
                ],
                "update_counters_after_scheduled_update": {
                    str(key): int(value) for key, value in counters.items()
                },
            }, "snapshot_manifest_contract_sha256")
            manifest_path = step_dir / "snapshot-manifest.json"
            manifest_file_sha256 = _atomic_no_clobber_json(
                manifest_path, snapshot_manifest)
            # The manifest publisher fsyncs step_dir; persist the already
            # complete step directory in the run directory before returning.
            _fsync_directory(self._run_dir)
        except Exception:
            # A partial directory is intentionally retained.  Its presence
            # makes retries fail closed instead of silently repairing a run.
            raise
        self._exported.append(ExportedPolicyCheckpoint(
            actor_training_seed=self._seed,
            checkpoint_step=step,
            actor_path=str(actor_path),
            actor_sha256=actor_sha256,
            actor_state_dict_sha256=state_sha256,
            snapshot_manifest_path=str(manifest_path),
            snapshot_manifest_sha256=manifest_file_sha256,
        ))
        if step == EXACT_CHECKPOINT_STEPS[-1]:
            completion = _self_hashed({
                "schema_version": RUN_COMPLETION_SCHEMA_VERSION,
                "actor_training_seed": self._seed,
                "role": self._contract["role"],
                "run_contract_sha256": self.contract_sha256,
                "checkpoint_steps": list(self.exported_steps),
                "checkpoint_semantics": CHECKPOINT_SEMANTICS,
                "snapshot_manifest_file_sha256": {
                    str(item.checkpoint_step): item.snapshot_manifest_sha256
                    for item in self._exported
                },
                "actor_inclusion_rule": (
                    "all_preregistered_seed_step_identities_no_filtering"),
                "return_or_fall_filtering": "forbidden",
            }, "completion_contract_sha256")
            _atomic_no_clobber_json(
                self._run_dir / "actor-run-completed.json", completion)

    def require_complete(self) -> None:
        if self.exported_steps != EXACT_CHECKPOINT_STEPS:
            raise StageBActorBankError(
                "actor run finished without all three exact checkpoints")


def maybe_create_exact_policy_exporter(
    cfg: Any,
) -> ExactPolicyCheckpointExporter | None:
    contract_path = getattr(cfg, "stage_b_actor_run_contract", None)
    if contract_path is None:
        return None
    return ExactPolicyCheckpointExporter(contract_path, cfg=cfg)


def _validate_snapshot_manifest(
    path: Path,
    *,
    contract: Mapping[str, Any],
    seed: int,
    step: int,
) -> tuple[dict[str, Any], str]:
    value = _read_json(path, "exact checkpoint manifest")
    _validate_self_hash(
        value, "snapshot_manifest_contract_sha256",
        "exact checkpoint manifest",
    )
    if value.get("schema_version") != SNAPSHOT_SCHEMA_VERSION or (
            value.get("generator_commit") != contract["generator_commit"]) or (
                value.get("actor_training_seed") != seed) or (
                    value.get("role") != _role_for_seed(seed)) or (
                        value.get("policy_training_step") != step) or (
                            value.get("checkpoint_semantics") !=
                            CHECKPOINT_SEMANTICS) or (
                                value.get("run_contract_sha256") !=
                                contract["run_contract_sha256"]) or (
                                    value.get("actor_relative_path") !=
                                    "agent/actor.pt") or (
                                        value.get("policy_only") is not True):
        raise StageBActorBankError(
            f"exact checkpoint manifest identity differs for seed={seed} step={step}")
    for field in ("actor_sha256", "actor_state_dict_sha256"):
        if _HEX64.fullmatch(str(value.get(field))) is None:
            raise StageBActorBankError(
                f"exact checkpoint manifest has invalid {field}")
    return value, _sha256_file(path, "exact checkpoint manifest")


def _inspect_policy_default(
    checkpoint: Path,
    config: Path,
    observation_dim: int,
    action_dim: int,
    training_step: int,
) -> Mapping[str, Any]:
    return load_frozen_droq_policy(
        checkpoint,
        config,
        observation_dim=observation_dim,
        action_dim=action_dim,
        training_step=training_step,
        device="cpu",
    ).manifest()


def compile_actor_bank_manifest(
    *,
    supplement_path: str | Path,
    protocol_path: str | Path,
    stage_a_report_path: str | Path,
    training_config_path: str | Path,
    actor_root: str | Path,
    contracts_root: str | Path,
    output_path: str | Path,
    observation_dim: int = PRODUCTION_OBSERVATION_DIM,
    action_dim: int = PRODUCTION_ACTION_DIM,
    policy_inspector: Callable[
        [Path, Path, int, int, int], Mapping[str, Any]
    ] = _inspect_policy_default,
    verify_live_bindings: bool = True,
    require_clean_git: bool = True,
    enforce_canonical_bindings: bool = True,
) -> tuple[dict[str, Any], str]:
    """Compile all 42 exact identities without consulting run outcomes."""
    if enforce_canonical_bindings:
        if observation_dim != PRODUCTION_OBSERVATION_DIM or action_dim != (
                PRODUCTION_ACTION_DIM):
            raise StageBActorBankError(
                "canonical actor-bank compilation requires observation_dim=46 "
                "and action_dim=12")
        _validate_canonical_science_bindings(
            supplement_path=supplement_path,
            protocol_path=protocol_path,
            stage_a_report_path=stage_a_report_path,
            training_config_path=training_config_path,
            actor_root=actor_root,
            output_path=output_path,
        )
    supplement = _read_yaml(supplement_path, "execution supplement")
    roster = actor_roster_from_supplement(supplement)
    protocol = _read_yaml(protocol_path, "V5 protocol")
    protocol_roster = actor_roster_from_supplement(protocol)
    if roster != protocol_roster:
        raise StageBActorBankError(
            "execution supplement actor roster differs from V5")
    expected_bindings = {
        "protocol": _binding(protocol_path, "V5 protocol"),
        "execution_supplement": _binding(
            supplement_path, "execution supplement"),
        "stage_a_report": _binding(stage_a_report_path, "Stage-A report"),
        "training_config": _binding(training_config_path, "training config"),
    }
    expected_bindings["protocol"]["contract_sha256"] = canonical_sha256(
        protocol)
    expected_bindings["execution_supplement"][
        "contract_sha256"] = canonical_sha256(supplement)

    actor_root_path = _validate_actor_root_shape(actor_root)
    contracts_root_path = assert_development_path(contracts_root)
    expected_attempt_path = _execution_attempt_marker_path(
        supplement, actor_root_path)
    config_path = _checked_input(training_config_path, "training config")
    if observation_dim <= 0 or action_dim <= 0:
        raise StageBActorBankError("observation_dim/action_dim must be positive")
    contract_files = sorted(contracts_root_path.glob("seed-*.json"))
    expected_contract_files = [
        contracts_root_path / f"seed-{seed}.json" for seed in ALL_ACTOR_SEEDS]
    if set(contract_files) != set(expected_contract_files):
        raise StageBActorBankError(
            "actor run contract files differ from the exact 14-seed roster")

    identities: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_fingerprints: set[str] = set()
    attempt_bindings: dict[str, dict[str, str]] = {}
    for role, seeds in roster.items():
        for seed in seeds:
            contract_path = contracts_root_path / f"seed-{seed}.json"
            contract = load_actor_run_contract(
                contract_path,
                verify_live_bindings=verify_live_bindings,
                require_clean_git=require_clean_git,
            )
            if contract["bindings"] != expected_bindings or (
                    contract["role"] != role):
                raise StageBActorBankError(
                    f"seed {seed} run contract binding/role differs")
            attempt_binding = dict(contract["actor_bank_attempt_binding"])
            if Path(attempt_binding["path"]) != expected_attempt_path:
                raise StageBActorBankError(
                    "actor run contract binds a noncanonical attempt marker")
            attempt_bindings[canonical_sha256(attempt_binding)] = (
                attempt_binding)
            seed_root = actor_root_path / f"seed-{seed}"
            if Path(contract["training_save_dir"]) != seed_root:
                raise StageBActorBankError(
                    f"seed {seed} run contract points outside actor_root")
            if not seed_root.is_dir():
                raise StageBActorBankError(f"missing actor run: seed {seed}")
            observed_step_dirs: dict[int, Path] = {}
            for child in seed_root.iterdir():
                match = _STEP_DIR.fullmatch(child.name)
                if match is None:
                    if child.name.startswith("step"):
                        raise StageBActorBankError(
                            f"seed {seed} has a noncanonical checkpoint "
                            f"directory: {child.name}")
                    continue
                checkpoint_step = int(match.group(1))
                if (checkpoint_step in EXACT_CHECKPOINT_STEPS
                        and child.name != _checkpoint_directory_name(
                            checkpoint_step)):
                    raise StageBActorBankError(
                        f"seed {seed} checkpoint spelling is noncanonical: "
                        f"{child.name}")
                if checkpoint_step in observed_step_dirs:
                    raise StageBActorBankError(
                        f"duplicate checkpoint identity for seed {seed} step {checkpoint_step}")
                observed_step_dirs[checkpoint_step] = child
            if set(observed_step_dirs) != set(EXACT_CHECKPOINT_STEPS):
                raise StageBActorBankError(
                    f"seed {seed} has missing or nearby checkpoint directories: "
                    f"{sorted(observed_step_dirs)}")

            completion_path = seed_root / "actor-run-completed.json"
            completion = _read_json(completion_path, "actor run completion")
            _validate_self_hash(
                completion, "completion_contract_sha256",
                "actor run completion",
            )
            completion_identity = {
                "schema_version": RUN_COMPLETION_SCHEMA_VERSION,
                "actor_training_seed": seed,
                "role": role,
                "run_contract_sha256": contract["run_contract_sha256"],
                "checkpoint_steps": list(EXACT_CHECKPOINT_STEPS),
                "checkpoint_semantics": CHECKPOINT_SEMANTICS,
                "return_or_fall_filtering": "forbidden",
            }
            if any(completion.get(key) != expected
                   for key, expected in completion_identity.items()):
                raise StageBActorBankError(
                    f"seed {seed} completion contract differs")
            completion_hashes = completion.get(
                "snapshot_manifest_file_sha256")
            if not isinstance(completion_hashes, Mapping):
                raise StageBActorBankError(
                    f"seed {seed} completion hashes are missing")

            for step in EXACT_CHECKPOINT_STEPS:
                step_dir = observed_step_dirs[step]
                if not step_dir.is_dir() or step_dir.is_symlink():
                    raise StageBActorBankError(
                        f"checkpoint directory is not physical: {step_dir}")
                actor_dir = step_dir / "agent"
                if not actor_dir.is_dir() or actor_dir.is_symlink():
                    raise StageBActorBankError(
                        f"policy-only actor directory is missing: {actor_dir}")
                actor_leaves = sorted(path.name for path in actor_dir.iterdir())
                if actor_leaves != ["actor.pt"]:
                    raise StageBActorBankError(
                        f"checkpoint is not policy-only: {actor_dir}")
                actor_path = actor_dir / "actor.pt"
                snapshot_path = step_dir / "snapshot-manifest.json"
                snapshot, snapshot_file_sha256 = _validate_snapshot_manifest(
                    snapshot_path, contract=contract, seed=seed, step=step)
                if completion_hashes.get(str(step)) != snapshot_file_sha256:
                    raise StageBActorBankError(
                        f"seed {seed} completion does not bind step {step}")
                if _sha256_file(actor_path, "policy-only actor") != snapshot[
                        "actor_sha256"]:
                    raise StageBActorBankError(
                        f"actor bytes changed for seed {seed} step {step}")
                policy = dict(policy_inspector(
                    actor_dir, config_path, observation_dim, action_dim, step))
                required_policy = {
                    "actor_sha256", "actor_state_dict_sha256",
                    "policy_fingerprint_sha256", "checkpoint_fingerprint_sha256",
                }
                if not required_policy.issubset(policy):
                    raise StageBActorBankError(
                        "policy inspector omitted frozen identity fields")
                if policy["actor_sha256"] != snapshot["actor_sha256"] or (
                        policy["actor_state_dict_sha256"] != snapshot[
                            "actor_state_dict_sha256"]):
                    raise StageBActorBankError(
                        f"loaded actor identity differs for seed {seed} step {step}")
                actor_path_text = str(actor_dir)
                checkpoint_fingerprint = str(
                    policy["checkpoint_fingerprint_sha256"])
                if actor_path_text in seen_paths or checkpoint_fingerprint in (
                        seen_fingerprints):
                    raise StageBActorBankError(
                        f"duplicate actor checkpoint for seed {seed} step {step}")
                seen_paths.add(actor_path_text)
                seen_fingerprints.add(checkpoint_fingerprint)
                identities.append({
                    "role": role,
                    "actor_training_seed": seed,
                    "checkpoint_step": step,
                    "checkpoint_path": actor_path_text,
                    "actor_checkpoint_sha256": policy["actor_sha256"],
                    "actor_sha256": policy["actor_sha256"],
                    "actor_state_dict_sha256": policy[
                        "actor_state_dict_sha256"],
                    "policy_fingerprint_sha256": policy[
                        "policy_fingerprint_sha256"],
                    "checkpoint_fingerprint_sha256": checkpoint_fingerprint,
                    "policy_config_sha256": expected_bindings[
                        "training_config"]["file_sha256"],
                    "generator_commit": contract["generator_commit"],
                    "run_contract_sha256": contract["run_contract_sha256"],
                    "snapshot_manifest_file_sha256": snapshot_file_sha256,
                })

    expected_order = [
        (role, seed, step)
        for role, seeds in ROLE_SEEDS.items()
        for seed in seeds
        for step in EXACT_CHECKPOINT_STEPS
    ]
    observed_order = [
        (item["role"], item["actor_training_seed"], item["checkpoint_step"])
        for item in identities
    ]
    if observed_order != expected_order or len(identities) != (
            EXPECTED_IDENTITY_COUNT):
        raise StageBActorBankError("compiled actor identities are incomplete")

    generator_commits = {item["generator_commit"] for item in identities}
    if len(generator_commits) != 1:
        raise StageBActorBankError("actor runs span multiple generator commits")
    if len(attempt_bindings) != 1:
        raise StageBActorBankError(
            "actor runs do not share one actor-bank attempt marker")
    attempt_binding = next(iter(attempt_bindings.values()))
    manifest = _self_hashed({
        "schema_version": ACTOR_BANK_SCHEMA_VERSION,
        "protocol_binding": expected_bindings["protocol"],
        "execution_supplement_binding": expected_bindings[
            "execution_supplement"],
        "stage_a_report_binding": expected_bindings["stage_a_report"],
        "training_config_binding": expected_bindings["training_config"],
        "actor_bank_attempt_binding": attempt_binding,
        "generator_commit": next(iter(generator_commits)),
        "actor_training_seeds": {
            role: list(seeds) for role, seeds in roster.items()},
        "checkpoint_steps": list(EXACT_CHECKPOINT_STEPS),
        "checkpoint_semantics": CHECKPOINT_SEMANTICS,
        "identity_count": len(identities),
        "expected_identity_count": EXPECTED_IDENTITY_COUNT,
        "actor_inclusion_rule": (
            "all_preregistered_seed_step_identities_no_filtering"),
        "return_or_fall_filtering": "forbidden",
        "checkpoint_substitution": "forbidden",
        "nearby_checkpoint_substitution": "forbidden",
        "policy_only": True,
        "identities": identities,
    }, "actor_bank_contract_sha256")
    output_file_sha256 = _atomic_no_clobber_json(
        output_path,
        manifest,
        allow_canonical_actor_bank_manifest=True,
    )
    return manifest, output_file_sha256


def load_actor_bank_manifest(
    path: str | Path = _CANONICAL_ACTOR_BANK_MANIFEST,
    *,
    expected_bindings: Mapping[str, str] | None = None,
    enforce_canonical_path: bool = True,
    verify_bound_files: bool = True,
    verify_checkpoint_files: bool = True,
) -> dict[str, Any]:
    """Load and validate the frozen 42-identity actor bank.

    ``expected_bindings`` may contain any of the following keys:
    ``manifest_file_sha256``, ``actor_bank_contract_sha256``,
    ``protocol_file_sha256``,
    ``protocol_contract_sha256``, ``execution_supplement_file_sha256``,
    ``execution_supplement_contract_sha256``, ``stage_a_report_sha256``,
    ``training_config_sha256``, and ``generator_commit``.  Supplying an
    unknown key fails closed.

    Checkpoint verification opens policy-only actor files, never training
    manifests, episode records, returns, falls, or branch outcomes.
    """
    manifest_path = _checked_input(path, "actor-bank manifest")
    if enforce_canonical_path and manifest_path != _CANONICAL_ACTOR_BANK_MANIFEST:
        raise StageBActorBankError(
            "actor-bank loader requires the canonical Stage-B manifest path")
    manifest = _read_json(manifest_path, "actor-bank manifest")
    _validate_self_hash(
        manifest, "actor_bank_contract_sha256", "actor-bank manifest")
    required_manifest_fields = {
        "schema_version", "protocol_binding", "execution_supplement_binding",
        "stage_a_report_binding", "training_config_binding",
        "actor_bank_attempt_binding", "generator_commit",
        "actor_training_seeds", "checkpoint_steps", "checkpoint_semantics",
        "identity_count", "expected_identity_count", "actor_inclusion_rule",
        "return_or_fall_filtering", "checkpoint_substitution",
        "nearby_checkpoint_substitution", "policy_only", "identities",
        "actor_bank_contract_sha256",
    }
    if set(manifest) != required_manifest_fields:
        raise StageBActorBankError(
            "actor-bank manifest fields differ from the frozen schema")
    if manifest.get("schema_version") != ACTOR_BANK_SCHEMA_VERSION or (
            manifest.get("actor_training_seeds") != {
                role: list(seeds) for role, seeds in ROLE_SEEDS.items()}) or (
                    manifest.get("checkpoint_steps") !=
                    list(EXACT_CHECKPOINT_STEPS)) or (
                        manifest.get("checkpoint_semantics") !=
                        CHECKPOINT_SEMANTICS) or (
                            manifest.get("identity_count") !=
                            EXPECTED_IDENTITY_COUNT) or (
                                manifest.get("expected_identity_count") !=
                                EXPECTED_IDENTITY_COUNT) or (
                                    manifest.get("actor_inclusion_rule") !=
                                    "all_preregistered_seed_step_identities_no_filtering") or (
                                        manifest.get("return_or_fall_filtering") !=
                                        "forbidden") or (
                                            manifest.get("checkpoint_substitution") !=
                                            "forbidden") or (
                                                manifest.get("nearby_checkpoint_substitution") !=
                                                "forbidden") or (
                                                    manifest.get("policy_only") is not True):
        raise StageBActorBankError("actor-bank frozen contract values differ")
    generator_commit = manifest.get("generator_commit")
    if _HEX40.fullmatch(str(generator_commit)) is None:
        raise StageBActorBankError("actor-bank generator commit is malformed")

    binding_specs = {
        "protocol_binding": ("protocol_file_sha256", "protocol_contract_sha256"),
        "execution_supplement_binding": (
            "execution_supplement_file_sha256",
            "execution_supplement_contract_sha256",
        ),
        "stage_a_report_binding": ("stage_a_report_sha256", None),
        "training_config_binding": ("training_config_sha256", None),
    }
    observed_expected_values: dict[str, str] = {
        "manifest_file_sha256": _sha256_file(
            manifest_path, "actor-bank manifest"),
        "actor_bank_contract_sha256": str(
            manifest["actor_bank_contract_sha256"]),
        "generator_commit": str(generator_commit),
    }
    for manifest_field, (file_key, contract_key) in binding_specs.items():
        binding = manifest.get(manifest_field)
        required_binding_fields = {"path", "file_sha256"}
        if contract_key is not None:
            required_binding_fields.add("contract_sha256")
        if not isinstance(binding, Mapping) or set(binding) != (
                required_binding_fields):
            raise StageBActorBankError(
                f"actor-bank {manifest_field} is malformed")
        if _HEX64.fullmatch(str(binding.get("file_sha256"))) is None or (
                contract_key is not None and _HEX64.fullmatch(
                    str(binding.get("contract_sha256"))) is None):
            raise StageBActorBankError(
                f"actor-bank {manifest_field} hash is malformed")
        observed_expected_values[file_key] = str(binding["file_sha256"])
        if contract_key is not None:
            observed_expected_values[contract_key] = str(
                binding["contract_sha256"])
        if verify_bound_files and _sha256_file(
                str(binding.get("path")), manifest_field) != binding[
                    "file_sha256"]:
            raise StageBActorBankError(
                f"actor-bank {manifest_field} live bytes changed")
        if verify_bound_files and contract_key is not None:
            parsed_binding = _read_yaml(
                str(binding["path"]), manifest_field)
            if canonical_sha256(parsed_binding) != binding[
                    "contract_sha256"]:
                raise StageBActorBankError(
                    f"actor-bank {manifest_field} canonical hash differs")

    allowed_expected = set(observed_expected_values)
    supplied_expected = dict(expected_bindings or {})
    if not set(supplied_expected).issubset(allowed_expected):
        raise StageBActorBankError(
            "actor-bank loader received an unknown expected binding")
    for name, expected in supplied_expected.items():
        if not isinstance(expected, str) or observed_expected_values[name] != expected:
            raise StageBActorBankError(
                f"actor-bank expected binding differs: {name}")

    attempt_binding = manifest.get("actor_bank_attempt_binding")
    if not isinstance(attempt_binding, Mapping) or set(attempt_binding) != {
            "path", "file_sha256", "contract_sha256"}:
        raise StageBActorBankError("actor-bank attempt binding is malformed")
    if any(_HEX64.fullmatch(str(attempt_binding.get(field))) is None
           for field in ("file_sha256", "contract_sha256")):
        raise StageBActorBankError("actor-bank attempt hash is malformed")
    attempt_path = _checked_input(
        str(attempt_binding["path"]), "actor-bank attempt marker")
    if verify_bound_files and _sha256_file(
            attempt_path, "actor-bank attempt marker") != attempt_binding[
                "file_sha256"]:
        raise StageBActorBankError("actor-bank attempt marker bytes changed")
    attempt = _read_json(attempt_path, "actor-bank attempt marker")
    attempt_contract_sha256 = _validate_self_hash(
        attempt, "attempt_contract_sha256", "actor-bank attempt marker")
    attempt_expected_bindings = {
        "protocol": manifest["protocol_binding"],
        "execution_supplement": manifest["execution_supplement_binding"],
        "stage_a_report": manifest["stage_a_report_binding"],
        "training_config": manifest["training_config_binding"],
    }
    expected_roster_sha256 = canonical_sha256({
        "actor_training_seeds": manifest["actor_training_seeds"],
        "actor_checkpoint_steps": list(EXACT_CHECKPOINT_STEPS),
    })
    if attempt_contract_sha256 != attempt_binding["contract_sha256"] or (
            attempt.get("schema_version") != ATTEMPT_SCHEMA_VERSION) or (
                attempt.get("generator_commit") != generator_commit) or (
                    attempt.get("fixed_actor_roster") !=
                    manifest["actor_training_seeds"]) or (
                        attempt.get("actor_roster_sha256") !=
                        expected_roster_sha256) or (
                        attempt.get("checkpoint_steps") !=
                        list(EXACT_CHECKPOINT_STEPS)) or (
                            attempt.get("checkpoint_semantics") !=
                            CHECKPOINT_SEMANTICS) or (
                                attempt.get("checkpoint_path_template") !=
                                _EXECUTION_CHECKPOINT_PATH_TEMPLATE) or (
                                attempt.get("expected_actor_identity_count") !=
                                EXPECTED_IDENTITY_COUNT) or (
                                    attempt.get("actor_inclusion_rule") !=
                                    "all_preregistered_seed_step_identities_no_filtering") or (
                                    attempt.get("return_or_fall_filtering") !=
                                    "forbidden") or (
                                        attempt.get("created_before_first_training_transition")
                                        is not True) or (
                                            attempt.get("bindings") !=
                                            attempt_expected_bindings):
        raise StageBActorBankError("actor-bank attempt marker differs")
    actor_root = _validate_actor_root_shape(str(attempt.get("actor_root")))
    if attempt_path != actor_root.parent / "actor-bank-attempt-started.json":
        raise StageBActorBankError("actor-bank attempt marker path differs")
    if enforce_canonical_path and actor_root != (
            _CANONICAL_ACTOR_BANK_MANIFEST.parent / "actor-bank"):
        raise StageBActorBankError("actor-bank checkpoint root is not canonical")

    identities = manifest.get("identities")
    if not isinstance(identities, list) or len(identities) != (
            EXPECTED_IDENTITY_COUNT):
        raise StageBActorBankError("actor-bank identities are incomplete")
    required_identity_fields = {
        "role", "actor_training_seed", "checkpoint_step", "checkpoint_path",
        "actor_checkpoint_sha256", "actor_sha256",
        "actor_state_dict_sha256", "policy_fingerprint_sha256",
        "checkpoint_fingerprint_sha256", "policy_config_sha256",
        "generator_commit", "run_contract_sha256",
        "snapshot_manifest_file_sha256",
    }
    expected_order = [
        (role, seed, step)
        for role, seeds in ROLE_SEEDS.items()
        for seed in seeds
        for step in EXACT_CHECKPOINT_STEPS
    ]
    observed_order: list[tuple[str, int, int]] = []
    unique_paths: set[str] = set()
    unique_fingerprints: set[str] = set()
    for raw_identity in identities:
        if not isinstance(raw_identity, Mapping) or set(raw_identity) != (
                required_identity_fields):
            raise StageBActorBankError(
                "actor-bank identity fields differ from the frozen schema")
        role = raw_identity.get("role")
        seed = raw_identity.get("actor_training_seed")
        step = raw_identity.get("checkpoint_step")
        if not isinstance(role, str) or isinstance(seed, bool) or not isinstance(
                seed, int) or isinstance(step, bool) or not isinstance(step, int):
            raise StageBActorBankError("actor-bank identity tuple is malformed")
        observed_order.append((role, seed, step))
        if raw_identity.get("generator_commit") != generator_commit or (
                raw_identity.get("actor_checkpoint_sha256") !=
                raw_identity.get("actor_sha256")) or (
                    raw_identity.get("policy_config_sha256") != manifest[
                        "training_config_binding"]["file_sha256"]):
            raise StageBActorBankError(
                f"actor-bank identity binding differs for {role}/{seed}/{step}")
        hash_fields = (
            "actor_checkpoint_sha256", "actor_sha256",
            "actor_state_dict_sha256", "policy_fingerprint_sha256",
            "checkpoint_fingerprint_sha256", "policy_config_sha256",
            "run_contract_sha256", "snapshot_manifest_file_sha256",
        )
        if any(_HEX64.fullmatch(str(raw_identity.get(field))) is None
               for field in hash_fields):
            raise StageBActorBankError(
                f"actor-bank identity hash malformed for {role}/{seed}/{step}")
        checkpoint_path = assert_development_path(
            str(raw_identity.get("checkpoint_path")))
        expected_checkpoint_path = (
            actor_root / f"seed-{seed}" / _checkpoint_directory_name(step)
            / "agent")
        checkpoint_text = str(checkpoint_path)
        fingerprint = str(raw_identity["checkpoint_fingerprint_sha256"])
        if checkpoint_path != expected_checkpoint_path or checkpoint_text in (
                unique_paths) or fingerprint in unique_fingerprints:
            raise StageBActorBankError(
                f"actor-bank path/fingerprint duplicate for {role}/{seed}/{step}")
        unique_paths.add(checkpoint_text)
        unique_fingerprints.add(fingerprint)
        if verify_checkpoint_files:
            if not checkpoint_path.is_dir() or checkpoint_path.is_symlink() or (
                    sorted(item.name for item in checkpoint_path.iterdir()) !=
                    ["actor.pt"]):
                raise StageBActorBankError(
                    f"actor-bank checkpoint is not policy-only: {checkpoint_path}")
            actor_path = checkpoint_path / "actor.pt"
            if _sha256_file(actor_path, "actor-bank actor") != raw_identity[
                    "actor_sha256"]:
                raise StageBActorBankError(
                    f"actor-bank actor bytes changed for {role}/{seed}/{step}")
            try:
                actor_payload = torch.load(
                    actor_path, map_location="cpu", weights_only=True)
            except Exception as exc:
                raise StageBActorBankError(
                    f"actor-bank actor cannot be loaded for {role}/{seed}/{step}") from exc
            if not isinstance(actor_payload, Mapping) or set(actor_payload) != {
                    "network_state_dict", "optimizer_state_dict",
                    "scheduler_state_dict", "update_step"} or actor_payload.get(
                        "optimizer_state_dict") is not None or actor_payload.get(
                            "scheduler_state_dict") is not None or not isinstance(
                                actor_payload.get("network_state_dict"), Mapping):
                raise StageBActorBankError(
                    f"actor-bank actor payload is not policy-only for "
                    f"{role}/{seed}/{step}")
            if _state_dict_sha256(actor_payload[
                    "network_state_dict"]) != raw_identity[
                        "actor_state_dict_sha256"]:
                raise StageBActorBankError(
                    f"actor-bank state_dict changed for {role}/{seed}/{step}")
            snapshot_path = checkpoint_path.parent / "snapshot-manifest.json"
            if _sha256_file(snapshot_path, "checkpoint manifest") != raw_identity[
                    "snapshot_manifest_file_sha256"]:
                raise StageBActorBankError(
                    f"checkpoint manifest changed for {role}/{seed}/{step}")
    if observed_order != expected_order:
        raise StageBActorBankError(
            "actor-bank identities are missing, duplicated, or reordered")
    return json.loads(json.dumps(manifest))


def actor_identity_for(
    manifest: Mapping[str, Any],
    *,
    role: str,
    actor_seed: int,
    checkpoint_step: int,
) -> dict[str, Any]:
    """Return one exact identity from a validated actor-bank manifest."""
    if role not in ROLE_SEEDS or actor_seed not in ROLE_SEEDS[role] or (
            checkpoint_step not in EXACT_CHECKPOINT_STEPS):
        raise StageBActorBankError(
            "requested actor identity is outside the frozen Stage-B roster")
    if manifest.get("schema_version") != ACTOR_BANK_SCHEMA_VERSION:
        raise StageBActorBankError("actor-bank manifest schema is invalid")
    identities = manifest.get("identities")
    if not isinstance(identities, list):
        raise StageBActorBankError("actor-bank manifest identities are missing")
    matches = [
        item for item in identities
        if isinstance(item, Mapping)
        and item.get("role") == role
        and item.get("actor_training_seed") == actor_seed
        and item.get("checkpoint_step") == checkpoint_step
    ]
    if len(matches) != 1:
        raise StageBActorBankError(
            "actor-bank identity is missing or duplicated")
    return json.loads(json.dumps(matches[0]))


__all__ = [
    "ACTOR_BANK_SCHEMA_VERSION",
    "ATTEMPT_SCHEMA_VERSION",
    "ALL_ACTOR_SEEDS",
    "CHECKPOINT_SEMANTICS",
    "EXPECTED_IDENTITY_COUNT",
    "EXACT_CHECKPOINT_STEPS",
    "ExactPolicyCheckpointExporter",
    "ExportedPolicyCheckpoint",
    "PRODUCTION_ACTION_DIM",
    "PRODUCTION_OBSERVATION_DIM",
    "ROLE_SEEDS",
    "RUN_CONTRACT_SCHEMA_VERSION",
    "SNAPSHOT_SCHEMA_VERSION",
    "StageBActorBankError",
    "actor_roster_from_supplement",
    "actor_identity_for",
    "canonical_sha256",
    "compile_actor_bank_manifest",
    "configure_training_for_actor_run_contract",
    "load_actor_run_contract",
    "load_actor_bank_manifest",
    "maybe_create_exact_policy_exporter",
    "prepare_actor_run_contracts",
    "validate_actor_roster",
]
