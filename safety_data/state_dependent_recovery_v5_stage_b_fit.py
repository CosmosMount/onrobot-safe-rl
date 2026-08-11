"""One-shot development fitting for V5 Stage B.

The workflow in this module is intentionally narrower than a generic Q_safe
trainer.  It accepts only the four preregistered development roles, requires
the outcome-free Model-Test byte commitment to exist first, and has no API for
opening Model-Test evidence.  A successful run freezes, in order:

1. fit-only observation normalization;
2. the exact five-member recovery Q_safe ensemble and member temperatures;
3. signed split-conformal offsets;
4. the exact 100-point selector search;
5. a canonical selector bundle and outcome-free matched-random placebo; and
6. the self-describing Q_safe artifact.

Production paths and statistical constants are derived from the immutable V5
protocol.  Smaller bootstrap counts are available only through the in-memory
pure-function seam used by unit tests; the production CLI exposes no such
override and no Model-Test path argument.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping

import numpy as np

from rl.qsafe.artifact import save_qsafe_artifact
from rl.qsafe.data import NormalizationStats, TorchGroupedView
from rl.qsafe.recovery_calibration import (
    STAGE_B_SELECTOR_BOOTSTRAP_REPLICATES,
    STAGE_B_SELECTOR_BOOTSTRAP_SEED,
    RecoverySelectorSearchResult,
    SignedConformalCalibration,
    fit_signed_recovery_conformal,
    predict_recovery_member_risk,
    search_recovery_selector_grid,
)
from rl.qsafe.recovery_placebo import (
    MatchedRandomPlaceboBundle,
    fit_matched_random_placebo,
)
from rl.qsafe.recovery_program import RECOVERY_PROGRAM_VIEW
from rl.qsafe.recovery_selector import (
    RecoveryConformalOffsets,
    RecoverySelectorBundle,
    select_recovery_program,
)
from rl.qsafe.training import (
    RECOVERY_PROGRAM_V4_LOSS_CONFIG,
    RECOVERY_PROGRAM_V4_NETWORK_CONFIG,
    RECOVERY_PROGRAM_V4_TRAINING_CONFIG,
    TrainedQSafeEnsemble,
    train_qsafe_ensemble,
)
from safety_data.paths import (
    STAGE_B_EXECUTION_PROTOCOL_NAME,
    STAGE_B_MODEL_TEST_COMMITMENT_SCHEMA,
    STAGE_B_PROTOCOL_NAME,
    _validate_stage_b_model_test_commitment,
)
from safety_data.schema import GroupedBranchDataset
from safety_data.stage_b_paths import stage_b_evidence_read_scope
from safety_data.state_dependent_recovery_v5 import (
    PROTOCOL_CONTRACT_SHA256 as PARENT_PROTOCOL_CONTRACT_SHA256,
    PROTOCOL_FILE_SHA256 as PARENT_PROTOCOL_FILE_SHA256,
    load_state_dependent_recovery_v5_protocol,
)
from safety_data.state_dependent_recovery_v5_stage_b import (
    EXECUTION_PROTOCOL_CONTRACT_SHA256,
    EXECUTION_PROTOCOL_FILE_SHA256,
    GROUPS_PER_SOURCE,
    HORIZON_POLICY_STEPS,
    LABEL_REPLICAS,
    RECOVERY_LIBRARY_FINGERPRINT_SHA256,
    ROLE_ACTOR_SEEDS,
    ROLE_ORDER,
    ROLE_SOURCE_SEEDS,
    SPLIT_COLLISION_DIMENSIONS,
    STAGE_A_DISPOSITION_COMMIT,
    STAGE_A_REPORT_SHA256,
    StageBExecutionError,
    assignment_for,
    canonical_sha256,
    execution_identity,
    load_stage_b_reduced7_amendment,
    load_stage_b_execution_protocol,
    require_clean_stage_b_generator,
    stage_b_artifact_root,
    validate_stage_a_authorization,
)
from train.state_dependent_recovery_v5_stage_b_actor_bank import (
    load_reduced7_actor_bank_manifest,
)


DEVELOPMENT_ROLES = (
    "fit",
    "probability_calibration",
    "uncertainty_calibration",
    "selector_calibration",
)
NORMALIZATION_REPORT_SCHEMA_VERSION = (
    "qsafe.state_dependent_recovery_v5.stage_b.normalization_fit_only.v1"
)
PROBABILITY_REPORT_SCHEMA_VERSION = (
    "qsafe.state_dependent_recovery_v5.stage_b.probability_calibration.v1"
)
UNCERTAINTY_REPORT_SCHEMA_VERSION = (
    "qsafe.state_dependent_recovery_v5.stage_b.uncertainty_calibration.v1"
)
SELECTOR_REPORT_SCHEMA_VERSION = (
    "qsafe.state_dependent_recovery_v5.stage_b.selector_search.v1"
)
DEVELOPMENT_FAILURE_SCHEMA_VERSION = (
    "qsafe.state_dependent_recovery_v5.stage_b.development_failure.v1"
)
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ROLE_DIRECTORIES = {
    role: role.replace("_", "-") for role in DEVELOPMENT_ROLES
}
_EXPECTED_GROUPS = {
    role: len(ROLE_SOURCE_SEEDS[role]) * GROUPS_PER_SOURCE[role]
    for role in DEVELOPMENT_ROLES
}
_REPORT_FILENAMES = {
    "normalization": "normalization-fit-only-report.json",
    "probability": "probability-calibration-report.json",
    "uncertainty": "uncertainty-calibration-report.json",
    "selector": "selector-search-report.json",
    "selector_bundle": "recovery-selector-bundle.json",
    "placebo_bundle": "matched-random-placebo-bundle.json",
}


class StageBFitError(StageBExecutionError):
    """The development-only Stage-B model compiler failed closed."""


@dataclass(frozen=True)
class StageBRoleInput:
    role: str
    path: Path
    dataset: GroupedBranchDataset
    file_sha256: str
    content_sha256: str
    role_report_path: Path
    role_report_file_sha256: str
    completion_marker_path: Path
    completion_marker_file_sha256: str

    def binding(self, *, stage_b_root: Path) -> dict[str, Any]:
        return {
            "role": self.role,
            "relative_path": self.path.relative_to(stage_b_root).as_posix(),
            "file_sha256": self.file_sha256,
            "content_sha256": self.content_sha256,
            "groups": self.dataset.group_count,
            "candidates": self.dataset.candidate_count,
            "replicas": self.dataset.replica_count,
            "source_seeds": sorted(set(
                map(int, np.asarray(self.dataset["source_seed"]))
            )),
            "actor_training_seeds": sorted(set(
                map(int, np.asarray(self.dataset["policy_training_seed"]))
            )),
            "role_report": {
                "relative_path": self.role_report_path.relative_to(
                    stage_b_root).as_posix(),
                "file_sha256": self.role_report_file_sha256,
            },
            "completion_marker": {
                "relative_path": self.completion_marker_path.relative_to(
                    stage_b_root).as_posix(),
                "file_sha256": self.completion_marker_file_sha256,
            },
        }


@dataclass(frozen=True)
class StageBFrozenDevelopmentInputs:
    stage_b_root: Path
    generator_commit: str
    execution_protocol: Mapping[str, Any]
    actor_bank_manifest: Mapping[str, Any]
    actor_bank_manifest_file_sha256: str
    split_disjointness_report: Mapping[str, Any]
    split_disjointness_report_file_sha256: str
    model_test_commitment: Mapping[str, Any]
    model_test_commitment_file_sha256: str
    roles: Mapping[str, StageBRoleInput]
    frozen_identity: Mapping[str, Any]


@dataclass(frozen=True)
class StageBFitResult:
    status: str
    selector_feasible: bool
    placebo_balanced: bool
    frozen_artifact_sha256: Mapping[str, str]
    failure_report: Path | None = None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise StageBFitError("reports may not contain non-finite floats")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise StageBFitError(
        f"value of type {type(value).__name__} is not canonical JSON")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        rendered = json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise StageBFitError("value is not canonical finite JSON") from exc
    return (rendered + "\n").encode("utf-8")


def _self_hashed_report(value: Mapping[str, Any]) -> dict[str, Any]:
    if "report_sha256" in value:
        raise StageBFitError("report payload already contains report_sha256")
    result = _jsonable(value)
    assert isinstance(result, dict)
    result["report_sha256"] = canonical_sha256(result)
    return result


def _validate_self_hashed_report(
    value: Mapping[str, Any],
    *,
    name: str,
) -> str:
    observed = value.get("report_sha256")
    basis = dict(value)
    basis.pop("report_sha256", None)
    expected = canonical_sha256(basis)
    if observed != expected:
        raise StageBFitError(f"{name} report_sha256 is invalid")
    return expected


def _regular_bytes(path: Path, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise StageBFitError(
                    f"{name} must be a single-link regular file")
            return stream.read()
    except StageBFitError:
        raise
    except OSError as exc:
        raise StageBFitError(f"{name} is missing or unreadable") from exc


def _file_sha256(path: Path, name: str) -> str:
    return hashlib.sha256(_regular_bytes(path, name)).hexdigest()


def _read_json(path: Path, name: str, *, canonical: bool = False) -> tuple[
    dict[str, Any], str
]:
    raw = _regular_bytes(path, name)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise StageBFitError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise StageBFitError(f"{name} must be a JSON object")
    if canonical and raw != _canonical_json_bytes(value):
        raise StageBFitError(f"{name} must use canonical JSON plus newline")
    return value, hashlib.sha256(raw).hexdigest()


def _atomic_no_clobber_json(path: Path, value: Mapping[str, Any]) -> str:
    """Publish exact canonical JSON without replacing an existing leaf."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_json_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(
        os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise StageBFitError(
            f"refusing to overwrite or alias frozen output: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise StageBFitError("new report is not a regular file")
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        # Do not unlink an interrupted one-shot report.  Its reserved path is
        # evidence that the deterministic development compiler was attempted.
        raise
    return hashlib.sha256(raw).hexdigest()


def _array_f4_sha256(value: Any) -> str:
    array = np.ascontiguousarray(value, dtype="<f4")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _ordered_text_sha256(value: Any) -> str:
    array = np.asarray(value).astype(str).reshape(-1)
    digest = hashlib.sha256(b"qsafe.ordered_text_vector.v1\0")
    for item in array:
        raw = item.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _artifact_manifest_canonical_sha256(path: Path) -> str:
    manifest, _ = _read_json(path / "manifest.json", "Q_safe manifest")
    return canonical_sha256(manifest)


def _role_label_path(stage_b_root: Path, role: str) -> Path:
    if role not in DEVELOPMENT_ROLES:
        raise StageBFitError(f"unsupported development role {role!r}")
    replicas = LABEL_REPLICAS[role]
    return (
        stage_b_root / _ROLE_DIRECTORIES[role]
        / f"labels-r{replicas}-deployable.npz"
    )


def _read_role_json(
    *,
    role: str,
    kind: str,
    path: Path,
    name: str,
) -> tuple[dict[str, Any], str]:
    with stage_b_evidence_read_scope(
        scientific_role=role,
        evidence_kind=kind,
        path=path,
    ):
        return _read_json(path, name, canonical=True)


def _load_role_input(
    *,
    stage_b_root: Path,
    role: str,
    generator_commit: str,
    actor_bank_manifest: Mapping[str, Any],
    actor_bank_manifest_file_sha256: str,
) -> StageBRoleInput:
    path = _role_label_path(stage_b_root, role)
    with stage_b_evidence_read_scope(
        scientific_role=role,
        evidence_kind="label",
        path=path,
    ):
        before = _file_sha256(path, f"{role} merged labels")
        dataset = GroupedBranchDataset.load(path)
        after = _file_sha256(path, f"{role} merged labels")
    if before != after:
        raise StageBFitError(f"{role} labels changed while loading")

    report_path = stage_b_root / _ROLE_DIRECTORIES[role] / "report.json"
    role_report, role_report_sha256 = _read_role_json(
        role=role,
        kind="report",
        path=report_path,
        name=f"{role} collection report",
    )
    completion_path = stage_b_root / _ROLE_DIRECTORIES[role] / "completed.json"
    completion, completion_sha256 = _read_role_json(
        role=role,
        kind="completion_marker",
        path=completion_path,
        name=f"{role} completion marker",
    )

    validation = dataset.validate()
    expected_groups = _EXPECTED_GROUPS[role]
    if dataset.group_count != expected_groups or dataset.candidate_count != 9 or (
        dataset.replica_count != LABEL_REPLICAS[role]
    ) or dataset.horizon_steps != HORIZON_POLICY_STEPS:
        raise StageBFitError(f"{role} merged dimensions differ from Stage B")
    if validation["unique_trajectory_clusters"] != expected_groups:
        raise StageBFitError(f"{role} must contain one group per trajectory")
    if dataset.manifest.get("generator_commit") != generator_commit:
        raise StageBFitError(f"{role} generator commit differs")
    if dataset.manifest.get("execution_protocol_file_sha256") != (
        EXECUTION_PROTOCOL_FILE_SHA256
    ) or dataset.manifest.get("execution_protocol_contract_sha256") != (
        EXECUTION_PROTOCOL_CONTRACT_SHA256
    ):
        raise StageBFitError(f"{role} execution protocol binding differs")
    if dataset.manifest.get("actor_bank_manifest_file_sha256") != (
        actor_bank_manifest_file_sha256
    ):
        raise StageBFitError(f"{role} actor-bank file binding differs")
    if dataset.manifest.get("actor_bank_contract_sha256") != (
        actor_bank_manifest.get("actor_bank_contract_sha256")
    ):
        raise StageBFitError(f"{role} actor-bank contract binding differs")
    collection = dataset.manifest.get("collection_protocol")
    if not isinstance(collection, Mapping) or collection.get("role") != role:
        raise StageBFitError(f"{role} collection role binding is invalid")
    recovery = dataset.manifest.get("recovery_program")
    if not isinstance(recovery, Mapping) or recovery.get(
        "fingerprint_sha256"
    ) != RECOVERY_LIBRARY_FINGERPRINT_SHA256:
        raise StageBFitError(f"{role} recovery library binding differs")
    if not np.allclose(
        np.asarray(dataset["command_vx"], dtype=np.float64),
        0.30,
        rtol=0.0,
        atol=1e-6,
    ):
        raise StageBFitError(f"{role} must target command_vx=0.30")
    sources = set(map(int, np.asarray(dataset["source_seed"])))
    actors = set(map(int, np.asarray(dataset["policy_training_seed"])))
    if sources != set(ROLE_SOURCE_SEEDS[role]) or actors != set(
        ROLE_ACTOR_SEEDS[role]
    ):
        raise StageBFitError(f"{role} actor/source roster differs")
    if dataset.manifest.get("source_seed_order") != list(
        ROLE_SOURCE_SEEDS[role]
    ):
        raise StageBFitError(f"{role} source order differs")
    for source_seed, actor_seed in zip(
        np.asarray(dataset["source_seed"], dtype=np.int64),
        np.asarray(dataset["policy_training_seed"], dtype=np.int64),
        strict=True,
    ):
        if assignment_for(role, int(source_seed)).actor_training_seed != int(
            actor_seed
        ):
            raise StageBFitError(f"{role} source-to-actor mapping differs")
    actor_identities = actor_bank_manifest.get("identities")
    if not isinstance(actor_identities, list):
        raise StageBFitError("actor bank has no identities")
    expected_policies = {
        str(item["policy_fingerprint_sha256"])
        for item in actor_identities
        if isinstance(item, Mapping) and item.get("role") == role
    }
    observed_policies = set(
        np.asarray(dataset["policy_source"]).astype(str).tolist()
    )
    if observed_policies != expected_policies:
        raise StageBFitError(f"{role} policies differ from the actor bank")
    for control_name, control in (
        ("collection report", role_report),
        ("completion marker", completion),
    ):
        frozen_control = {
            "parent_protocol_name": STAGE_B_PROTOCOL_NAME,
            "parent_protocol_contract_sha256": (
                PARENT_PROTOCOL_CONTRACT_SHA256),
            "parent_protocol_file_sha256": PARENT_PROTOCOL_FILE_SHA256,
            "execution_protocol_name": STAGE_B_EXECUTION_PROTOCOL_NAME,
            "execution_protocol_contract_sha256": (
                EXECUTION_PROTOCOL_CONTRACT_SHA256),
            "execution_protocol_file_sha256": (
                EXECUTION_PROTOCOL_FILE_SHA256),
            "stage_a_report_sha256": STAGE_A_REPORT_SHA256,
            "stage_a_disposition_commit": STAGE_A_DISPOSITION_COMMIT,
            "generator_commit": generator_commit,
        }
        for name, expected in frozen_control.items():
            if control.get(name) != expected:
                raise StageBFitError(
                    f"{role} {control_name} {name} differs")
        if "role" in control and control["role"] != role:
            raise StageBFitError(f"{role} {control_name} role differs")

    if role_report.get("schema_version") != (
        "qsafe.state_dependent_recovery_v5.stage_b_role_outcome_free_report.v1"
    ) or role_report.get("source_seeds") != list(ROLE_SOURCE_SEEDS[role]) or (
        role_report.get("groups") != expected_groups
    ) or role_report.get("label_replicas") != LABEL_REPLICAS[role] or (
        role_report.get("status") != "complete_evidence_hashes_only"
    ):
        raise StageBFitError(f"{role} collection report contract differs")
    aggregate_relative = (
        f"stage-b/{_ROLE_DIRECTORIES[role]}/"
        f"labels-r{LABEL_REPLICAS[role]}-deployable.npz"
    )
    report_records = role_report.get("evidence_artifacts")
    report_label = [
        record for record in report_records or []
        if isinstance(record, Mapping)
        and record.get("kind") == "label"
        and record.get("path") == aggregate_relative
    ] if isinstance(report_records, list) else []
    completion_hashes = completion.get("artifact_sha256_by_path")
    if len(report_label) != 1 or report_label[0].get("sha256") != before or (
        not isinstance(completion_hashes, Mapping)
    ) or completion_hashes.get(aggregate_relative) != before:
        raise StageBFitError(f"{role} controls do not bind merged labels")
    if completion.get("schema_version") != (
        "qsafe.state_dependent_recovery_v5.stage_b_role_completion.v1"
    ) or completion.get("source_seeds") != list(ROLE_SOURCE_SEEDS[role]) or (
        completion.get("groups") != expected_groups
    ) or completion.get("candidate_outcomes_summarized") is not False or (
        completion.get("status") != "role_complete_report_pending"
    ):
        raise StageBFitError(f"{role} completion marker contract differs")

    content_sha256 = str(dataset.manifest.get("content_sha256"))
    if content_sha256 != validation["content_sha256"] or _HEX64.fullmatch(
        content_sha256
    ) is None:
        raise StageBFitError(f"{role} content hash is invalid")
    return StageBRoleInput(
        role=role,
        path=path,
        dataset=dataset,
        file_sha256=before,
        content_sha256=content_sha256,
        role_report_path=report_path,
        role_report_file_sha256=role_report_sha256,
        completion_marker_path=completion_path,
        completion_marker_file_sha256=completion_sha256,
    )


def _load_model_test_commitment(
    *,
    stage_b_root: Path,
    generator_commit: str,
) -> tuple[dict[str, Any], str]:
    """Read only the fixed outcome-free commitment control."""
    path = stage_b_root / "model-test-committed.json"
    value, file_sha256 = _read_json(
        path, "Stage-B Model-Test commitment", canonical=True)
    try:
        checked = _validate_stage_b_model_test_commitment(value)
    except Exception as exc:
        raise StageBFitError("Model-Test commitment contract is invalid") from exc
    checked.pop("_artifact_sha256_by_path", None)
    expected = {
        "schema_version": STAGE_B_MODEL_TEST_COMMITMENT_SCHEMA,
        "parent_protocol_name": STAGE_B_PROTOCOL_NAME,
        "parent_protocol_contract_sha256": PARENT_PROTOCOL_CONTRACT_SHA256,
        "parent_protocol_file_sha256": PARENT_PROTOCOL_FILE_SHA256,
        "execution_protocol_name": STAGE_B_EXECUTION_PROTOCOL_NAME,
        "execution_protocol_contract_sha256": (
            EXECUTION_PROTOCOL_CONTRACT_SHA256
        ),
        "execution_protocol_file_sha256": EXECUTION_PROTOCOL_FILE_SHA256,
        "stage_a_report_sha256": STAGE_A_REPORT_SHA256,
        "stage_a_disposition_commit": STAGE_A_DISPOSITION_COMMIT,
        "generator_commit": generator_commit,
        "model_test_report_path": "stage-b/model-test/report.json",
    }
    for name, expected_value in expected.items():
        if checked.get(name) != expected_value:
            raise StageBFitError(
                f"Model-Test commitment {name} differs from frozen identity")
    # A fit run is a prerequisite to consumption, never a post-consumption
    # analysis.  This probes only the fixed control marker, not evidence.
    if os.path.lexists(stage_b_root / "model-test-consumed.json"):
        raise StageBFitError(
            "Model-Test is already consumed; development fitting is too late")
    return checked, file_sha256


def _load_split_report(
    path: Path,
    *,
    generator_commit: str,
    actor_bank_manifest: Mapping[str, Any],
    actor_bank_manifest_file_sha256: str,
    model_test_commitment: Mapping[str, Any],
    development_roles: Mapping[str, StageBRoleInput],
    stage_b_root: Path,
) -> tuple[dict[str, Any], str]:
    value, file_sha256 = _read_json(path, "Stage-B split report")
    frozen_fields = {
        "parent_protocol_name": STAGE_B_PROTOCOL_NAME,
        "parent_protocol_contract_sha256": PARENT_PROTOCOL_CONTRACT_SHA256,
        "parent_protocol_file_sha256": PARENT_PROTOCOL_FILE_SHA256,
        "execution_protocol_name": STAGE_B_EXECUTION_PROTOCOL_NAME,
        "execution_protocol_contract_sha256": (
            EXECUTION_PROTOCOL_CONTRACT_SHA256
        ),
        "execution_protocol_file_sha256": EXECUTION_PROTOCOL_FILE_SHA256,
        "stage_a_report_sha256": STAGE_A_REPORT_SHA256,
        "stage_a_disposition_commit": STAGE_A_DISPOSITION_COMMIT,
        "generator_commit": generator_commit,
    }
    expected_fields = {
        "schema_version",
        *frozen_fields,
        "actor_bank_manifest_file_sha256",
        "actor_bank_contract_sha256",
        "role_order",
        "role_aggregate_labels",
        "identity_proof",
        "model_test_source",
        "outcome_columns_read",
        "pass",
        "report_sha256",
    }
    if set(value) != expected_fields or value.get("schema_version") != (
        "qsafe.state_dependent_recovery_v5.stage_b_split_disjointness_bound.v1"
    ) or value.get("pass") is not True or value.get(
        "outcome_columns_read"
    ) is not False or value.get("model_test_source") != (
        "in_memory_merged_dataset_and_staged_label_bytes_before_role_report"
    ):
        raise StageBFitError(
            "Stage-B split-disjointness top-level contract did not pass")
    for name, expected in frozen_fields.items():
        if value.get(name) != expected:
            raise StageBFitError(f"split report {name} differs")
    if value.get("actor_bank_manifest_file_sha256") != (
        actor_bank_manifest_file_sha256
    ) or value.get("actor_bank_contract_sha256") != actor_bank_manifest.get(
        "actor_bank_contract_sha256"
    ):
        raise StageBFitError("split report actor-bank binding differs")
    if value.get("role_order") != list(ROLE_ORDER):
        raise StageBFitError("split report role order differs")
    _validate_self_hashed_report(value, name="split-disjointness")

    aggregates = value.get("role_aggregate_labels")
    if not isinstance(aggregates, list) or len(aggregates) != len(ROLE_ORDER):
        raise StageBFitError("split report aggregate roles differ")
    commitment_records = model_test_commitment.get("evidence_artifacts")
    if not isinstance(commitment_records, list):
        raise StageBFitError("Model-Test commitment evidence roster is missing")
    model_test_relative = "stage-b/model-test/labels-r64-deployable.npz"
    committed_model_test = [
        record
        for record in commitment_records
        if isinstance(record, Mapping)
        and record.get("path") == model_test_relative
        and record.get("kind") == "label"
    ]
    if len(committed_model_test) != 1:
        raise StageBFitError(
            "Model-Test commitment has no unique aggregate-label hash")
    for item, role in zip(aggregates, ROLE_ORDER, strict=True):
        expected_groups = (
            _EXPECTED_GROUPS[role]
            if role in _EXPECTED_GROUPS
            else len(ROLE_SOURCE_SEEDS[role]) * GROUPS_PER_SOURCE[role]
        )
        replicas = LABEL_REPLICAS[role]
        expected_relative = (
            f"stage-b/{role.replace('_', '-')}/"
            f"labels-r{replicas}-deployable.npz"
        )
        if not isinstance(item, Mapping) or set(item) != {
            "role", "path", "file_sha256", "content_sha256", "groups",
            "role_report_file_sha256",
        } or item.get("role") != role or item.get(
            "path"
        ) != expected_relative or item.get(
            "groups"
        ) != expected_groups or _HEX64.fullmatch(
            str(item.get("file_sha256"))
        ) is None or _HEX64.fullmatch(
            str(item.get("content_sha256"))
        ) is None:
            raise StageBFitError(f"split report {role} aggregate is malformed")
        if role in DEVELOPMENT_ROLES:
            loaded = development_roles[role]
            if item.get("file_sha256") != loaded.file_sha256 or item.get(
                "content_sha256"
            ) != loaded.content_sha256 or item.get(
                "role_report_file_sha256"
            ) != loaded.role_report_file_sha256 or Path(expected_relative).name != (
                loaded.path.name
            ) or loaded.path.relative_to(stage_b_root.parent).as_posix() != (
                expected_relative
            ):
                raise StageBFitError(
                    f"split report {role} aggregate differs from loaded input")
        elif item.get("role_report_file_sha256") is not None or item.get(
            "file_sha256"
        ) != committed_model_test[0].get("sha256"):
            # Compare commitment text only.  Never resolve, stat, hash, or open
            # the Model-Test path named by either control object.
            raise StageBFitError(
                "split report Model-Test label hash differs from commitment")

    proof = value.get("identity_proof")
    expected_proof_fields = {
        "schema_version", "dimensions", "roles", "pairs_checked", "pairs",
        "outcome_columns_read", "pass", "report_sha256",
    }
    if not isinstance(proof, Mapping) or set(proof) != expected_proof_fields or (
        proof.get("schema_version") != (
            "qsafe.state_dependent_recovery_v5."
            "stage_b_split_disjointness.v1")
    ) or (
        proof.get("pairs_checked") != 10
    ) or proof.get("outcome_columns_read") is not False or proof.get(
        "pass"
    ) is not True:
        raise StageBFitError("split identity proof did not pass")
    _validate_self_hashed_report(proof, name="nested split identity proof")
    dimensions = proof.get("dimensions")
    if dimensions != list(SPLIT_COLLISION_DIMENSIONS):
        raise StageBFitError("split identity dimensions differ")
    roles = proof.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != set(ROLE_ORDER):
        raise StageBFitError("split identity role commitments differ")
    for role in ROLE_ORDER:
        item = roles.get(role)
        expected_groups = len(ROLE_SOURCE_SEEDS[role]) * GROUPS_PER_SOURCE[role]
        if not isinstance(item, Mapping) or item.get("groups") != (
            expected_groups
        ) or item.get("source_seeds") != sorted(ROLE_SOURCE_SEEDS[role]) or (
            item.get("actor_training_seeds") != sorted(ROLE_ACTOR_SEEDS[role])
        ) or item.get("outcome_columns_read") is not False or _HEX64.fullmatch(
            str(item.get("identity_commitment_sha256"))
        ) is None:
            raise StageBFitError(f"split report {role} commitment differs")
    pairs = proof.get("pairs")
    expected_pairs = [
        (left, right)
        for index, left in enumerate(ROLE_ORDER)
        for right in ROLE_ORDER[index + 1:]
    ]
    if not isinstance(pairs, list) or len(pairs) != 10:
        raise StageBFitError("split proof must contain ten pairs")
    for record, (left, right) in zip(pairs, expected_pairs, strict=True):
        collisions = record.get("collision_counts") if isinstance(
            record, Mapping
        ) else None
        if not isinstance(record, Mapping) or set(record) != {
            "left", "right", "collision_counts", "pass"
        } or record.get("left") != left or record.get("right") != right or (
            record.get("pass") is not True
        ) or not isinstance(collisions, Mapping) or set(collisions) != set(
            dimensions
        ) or any(
            isinstance(count, bool) or not isinstance(count, int) or count != 0
            for count in collisions.values()
        ):
            raise StageBFitError(
                f"split identity collisions exist for {left}/{right}")
    return value, file_sha256


def _frozen_identity(
    *,
    execution_protocol: Mapping[str, Any],
    generator_commit: str,
    actor_bank_manifest: Mapping[str, Any],
    actor_bank_manifest_file_sha256: str,
    split_report: Mapping[str, Any],
    split_report_file_sha256: str,
    model_test_commitment_file_sha256: str,
    roles: Mapping[str, StageBRoleInput],
    stage_b_root: Path,
) -> dict[str, Any]:
    identity = execution_identity(execution_protocol)
    identity.update({
        "generator_commit": generator_commit,
        "model_test_commitment_file_sha256": (
            model_test_commitment_file_sha256
        ),
        "model_test_outcomes_read": False,
        "model_test_consumed": False,
        "actor_bank_manifest": {
            "relative_path": "actor-bank-manifest.json",
            "file_sha256": actor_bank_manifest_file_sha256,
            "contract_sha256": actor_bank_manifest[
                "actor_bank_contract_sha256"
            ],
            "identity_count": actor_bank_manifest["identity_count"],
        },
        "split_disjointness_report": {
            "relative_path": "stage-b-split-disjointness-report.json",
            "file_sha256": split_report_file_sha256,
            "report_sha256": split_report["report_sha256"],
            "pairs_checked": 10,
            "pass": True,
        },
        "development_role_inputs": {
            role: roles[role].binding(stage_b_root=stage_b_root)
            for role in DEVELOPMENT_ROLES
        },
    })
    return identity


def load_frozen_development_inputs() -> StageBFrozenDevelopmentInputs:
    """Load and bind the canonical four-role cohort without Model-Test data."""
    execution_protocol = load_stage_b_execution_protocol()
    load_stage_b_reduced7_amendment()
    validate_stage_a_authorization(execution_protocol)
    generator_commit = require_clean_stage_b_generator()
    parent_protocol = load_state_dependent_recovery_v5_protocol()
    stage_b_root = stage_b_artifact_root(parent_protocol)

    commitment, commitment_file_sha256 = _load_model_test_commitment(
        stage_b_root=stage_b_root,
        generator_commit=generator_commit,
    )
    # Refuse a retry before opening any development outcome array.
    _require_outputs_absent(stage_b_root)
    actor_path = stage_b_root / "actor-bank-manifest.json"
    actor_file_sha256 = _file_sha256(actor_path, "actor-bank manifest")
    actor_manifest = load_reduced7_actor_bank_manifest(
        actor_path,
        expected_bindings={
            "manifest_file_sha256": actor_file_sha256,
            "protocol_file_sha256": PARENT_PROTOCOL_FILE_SHA256,
            "protocol_contract_sha256": PARENT_PROTOCOL_CONTRACT_SHA256,
            "execution_supplement_file_sha256": (
                EXECUTION_PROTOCOL_FILE_SHA256
            ),
            "execution_supplement_contract_sha256": (
                EXECUTION_PROTOCOL_CONTRACT_SHA256
            ),
            "stage_a_report_sha256": STAGE_A_REPORT_SHA256,
            "stage_b_generator_commit": generator_commit,
        },
        verify_checkpoint_files=True,
    )
    if actor_manifest.get("stage_b_generator_commit") != generator_commit:
        raise StageBFitError("actor bank amendment generator commit differs")

    roles = {
        role: _load_role_input(
            stage_b_root=stage_b_root,
            role=role,
            generator_commit=generator_commit,
            actor_bank_manifest=actor_manifest,
            actor_bank_manifest_file_sha256=actor_file_sha256,
        )
        for role in DEVELOPMENT_ROLES
    }
    split_path = stage_b_root / "stage-b-split-disjointness-report.json"
    split_report, split_file_sha256 = _load_split_report(
        split_path,
        generator_commit=generator_commit,
        actor_bank_manifest=actor_manifest,
        actor_bank_manifest_file_sha256=actor_file_sha256,
        model_test_commitment=commitment,
        development_roles=roles,
        stage_b_root=stage_b_root,
    )
    identity = _frozen_identity(
        execution_protocol=execution_protocol,
        generator_commit=generator_commit,
        actor_bank_manifest=actor_manifest,
        actor_bank_manifest_file_sha256=actor_file_sha256,
        split_report=split_report,
        split_report_file_sha256=split_file_sha256,
        model_test_commitment_file_sha256=commitment_file_sha256,
        roles=roles,
        stage_b_root=stage_b_root,
    )
    return StageBFrozenDevelopmentInputs(
        stage_b_root=stage_b_root,
        generator_commit=generator_commit,
        execution_protocol=execution_protocol,
        actor_bank_manifest=actor_manifest,
        actor_bank_manifest_file_sha256=actor_file_sha256,
        split_disjointness_report=split_report,
        split_disjointness_report_file_sha256=split_file_sha256,
        model_test_commitment=commitment,
        model_test_commitment_file_sha256=commitment_file_sha256,
        roles=roles,
        frozen_identity=identity,
    )


def build_normalization_report(
    *,
    normalization: NormalizationStats,
    fit_input: StageBRoleInput,
    frozen_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the result-blind fit-only normalization commitment."""
    if normalization.privileged_mean is not None or (
        normalization.privileged_std is not None
    ):
        raise StageBFitError("Stage-B normalization must be deployable-only")
    if normalization.fit_content_sha256 != fit_input.content_sha256 or (
        normalization.fit_split != fit_input.dataset.manifest.get("split")
    ):
        raise StageBFitError("normalization is not fitted to the exact fit role")
    return _self_hashed_report({
        "schema_version": NORMALIZATION_REPORT_SCHEMA_VERSION,
        "frozen_identity": frozen_identity,
        "source_role": "fit",
        "source_array_sha256": fit_input.file_sha256,
        "source_content_sha256": fit_input.content_sha256,
        "source_group_ids_sha256": _ordered_text_sha256(
            fit_input.dataset["group_id"]
        ),
        "observation_mean_f4_sha256": _array_f4_sha256(
            normalization.observation_mean
        ),
        "observation_std_f4_sha256": _array_f4_sha256(
            normalization.observation_std
        ),
        "privileged_features_absent": True,
        "contract": {
            "population": "complete_fit_groups_only",
            "weighting": "one_row_per_group_candidate_no_outcome_weighting",
            "accumulator_dtype": "float64",
            "published_dtype": "float32",
            "population_std_ddof": 0,
            "std_floor": 1e-6,
            "frozen_before_probability_calibration": True,
        },
    })


def _binary_nll(member_risk: np.ndarray, empirical_risk: np.ndarray) -> list[
    float
]:
    risk = np.clip(np.asarray(member_risk, dtype=np.float64), 1e-12, 1 - 1e-12)
    target = np.asarray(empirical_risk, dtype=np.float64)[:, None, :]
    loss = -(target * np.log(risk) + (1.0 - target) * np.log1p(-risk))
    return [float(value) for value in loss.mean(axis=(0, 2))]


def build_probability_report(
    *,
    trained: TrainedQSafeEnsemble,
    member_risk: np.ndarray,
    empirical_risk: np.ndarray,
    probability_input: StageBRoleInput,
    normalization_report: Mapping[str, Any],
    normalization_report_file_sha256: str,
    frozen_identity: Mapping[str, Any],
) -> dict[str, Any]:
    temperatures = [float(member.temperature) for member in trained.members]
    if len(temperatures) != 5 or any(
        not math.isfinite(value) or not math.exp(-4.0) <= value <= math.exp(4.0)
        for value in temperatures
    ):
        raise StageBFitError("member temperatures violate the frozen contract")
    members = []
    for index, member in enumerate(trained.members):
        epoch_loss = np.asarray(member.epoch_loss, dtype="<f8")
        if member.seed != 20_260_810 + 1009 * index or len(epoch_loss) != 100 or (
            not np.all(np.isfinite(epoch_loss))
        ):
            raise StageBFitError("trained member metadata violates Stage B")
        members.append({
            "member_index": index,
            "seed": member.seed,
            "epochs": len(member.epoch_loss),
            "epoch_loss_f8_sha256": hashlib.sha256(
                epoch_loss.tobytes(order="C")
            ).hexdigest(),
            "temperature": temperatures[index],
            "trajectory_bootstrap_count": len(member.bootstrap_trajectories),
            "trajectory_bootstrap_sha256": _ordered_text_sha256(
                member.bootstrap_trajectories
            ),
        })
    return _self_hashed_report({
        "schema_version": PROBABILITY_REPORT_SCHEMA_VERSION,
        "frozen_identity": frozen_identity,
        "normalization_report_sha256": normalization_report["report_sha256"],
        "normalization_report_file_sha256": (
            normalization_report_file_sha256
        ),
        "source_role": "probability_calibration",
        "source_array_sha256": probability_input.file_sha256,
        "source_content_sha256": probability_input.content_sha256,
        "model": {
            "network_config": asdict(RECOVERY_PROGRAM_V4_NETWORK_CONFIG),
            "training_config": asdict(RECOVERY_PROGRAM_V4_TRAINING_CONFIG),
            "loss_config": asdict(RECOVERY_PROGRAM_V4_LOSS_CONFIG),
            "members": members,
        },
        "temperature_calibration": {
            "member_temperatures": temperatures,
            "steps": 100,
            "optimizer": "Adam",
            "learning_rate": 0.05,
            "log_temperature_clamp": [-4.0, 4.0],
            "calibrated_member_binary_nll": _binary_nll(
                member_risk, empirical_risk
            ),
            "weighting": "equal_group_then_equal_valid_K9_candidate",
        },
    })


def build_uncertainty_report(
    *,
    calibration: SignedConformalCalibration,
    uncertainty_input: StageBRoleInput,
    probability_report: Mapping[str, Any],
    probability_report_file_sha256: str,
    frozen_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return _self_hashed_report({
        "schema_version": UNCERTAINTY_REPORT_SCHEMA_VERSION,
        "frozen_identity": frozen_identity,
        "probability_calibration_report_sha256": probability_report[
            "report_sha256"
        ],
        "probability_calibration_report_file_sha256": (
            probability_report_file_sha256
        ),
        "source_role": "uncertainty_calibration",
        "source_array_sha256": uncertainty_input.file_sha256,
        "source_content_sha256": uncertainty_input.content_sha256,
        "signed_conformal": calibration.report_payload(),
    })


def build_selector_report(
    *,
    search: RecoverySelectorSearchResult,
    selector_input: StageBRoleInput,
    probability_report: Mapping[str, Any],
    uncertainty_report: Mapping[str, Any],
    uncertainty_report_file_sha256: str,
    frozen_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return _self_hashed_report({
        "schema_version": SELECTOR_REPORT_SCHEMA_VERSION,
        "frozen_identity": frozen_identity,
        "probability_calibration_report_sha256": probability_report[
            "report_sha256"
        ],
        "uncertainty_calibration_report_sha256": uncertainty_report[
            "report_sha256"
        ],
        "uncertainty_calibration_report_file_sha256": (
            uncertainty_report_file_sha256
        ),
        "source_role": "selector_calibration",
        "source_array_sha256": selector_input.file_sha256,
        "source_content_sha256": selector_input.content_sha256,
        "selector_search": search.report_payload(),
        "development_decision": (
            "freeze_selected_selector"
            if search.selected_config is not None
            else "terminal_no_feasible_selector_before_model_test"
        ),
        "model_test_outcomes_read": False,
        "model_test_consumed": False,
    })


def fit_development_calibration(
    *,
    trained: TrainedQSafeEnsemble,
    uncertainty_view: TorchGroupedView,
    selector_view: TorchGroupedView,
    uncertainty_input: StageBRoleInput,
    selector_input: StageBRoleInput,
    execution_lock_sha256: str,
    bootstrap_replicates: int = STAGE_B_SELECTOR_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = STAGE_B_SELECTOR_BOOTSTRAP_SEED,
    production_contract: bool = True,
) -> tuple[
    SignedConformalCalibration,
    RecoverySelectorSearchResult,
    np.ndarray,
]:
    """Pure-function seam for conformal/search testing.

    Production rejects every statistical override.  Unit tests may inject a
    smaller count only by calling this function with
    ``production_contract=False``; the production CLI never exposes it.
    """
    if production_contract and (
        bootstrap_replicates != STAGE_B_SELECTOR_BOOTSTRAP_REPLICATES
        or bootstrap_seed != STAGE_B_SELECTOR_BOOTSTRAP_SEED
    ):
        raise StageBFitError("production selector bootstrap is immutable")
    conformal = fit_development_conformal(
        trained=trained,
        uncertainty_view=uncertainty_view,
        uncertainty_input=uncertainty_input,
        execution_lock_sha256=execution_lock_sha256,
        production_contract=production_contract,
    )
    selector_member, search = search_development_selector(
        trained=trained,
        selector_view=selector_view,
        selector_input=selector_input,
        offsets=conformal.offsets,
        execution_lock_sha256=execution_lock_sha256,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        production_contract=production_contract,
    )
    return conformal, search, selector_member


def fit_development_conformal(
    *,
    trained: TrainedQSafeEnsemble,
    uncertainty_view: TorchGroupedView,
    uncertainty_input: StageBRoleInput,
    execution_lock_sha256: str,
    production_contract: bool = True,
) -> SignedConformalCalibration:
    uncertainty_member = predict_recovery_member_risk(
        trained, uncertainty_view, device="cpu", batch_size=256)
    uncertainty_empirical = np.asarray(
        uncertainty_input.dataset["fall"], dtype=np.float64
    ).mean(axis=2)
    return fit_signed_recovery_conformal(
        uncertainty_member,
        uncertainty_empirical,
        candidate_mask=uncertainty_input.dataset["candidate_mask"],
        execution_lock=execution_lock_sha256,
        expected_group_count=(
            _EXPECTED_GROUPS["uncertainty_calibration"]
            if production_contract else None
        ),
    )


def search_development_selector(
    *,
    trained: TrainedQSafeEnsemble,
    selector_view: TorchGroupedView,
    selector_input: StageBRoleInput,
    offsets: RecoveryConformalOffsets,
    execution_lock_sha256: str,
    bootstrap_replicates: int = STAGE_B_SELECTOR_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = STAGE_B_SELECTOR_BOOTSTRAP_SEED,
    production_contract: bool = True,
) -> tuple[np.ndarray, RecoverySelectorSearchResult]:
    if production_contract and (
        bootstrap_replicates != STAGE_B_SELECTOR_BOOTSTRAP_REPLICATES
        or bootstrap_seed != STAGE_B_SELECTOR_BOOTSTRAP_SEED
    ):
        raise StageBFitError("production selector bootstrap is immutable")
    selector_member = predict_recovery_member_risk(
        trained, selector_view, device="cpu", batch_size=256)
    selector_empirical = np.asarray(
        selector_input.dataset["fall"], dtype=np.float64
    ).mean(axis=2)
    search = search_recovery_selector_grid(
        selector_member,
        selector_empirical,
        candidate_requested=selector_input.dataset["candidate_requested"],
        candidate_executed=selector_input.dataset["candidate_executed"],
        candidate_q_target=selector_input.dataset["candidate_q_target"],
        candidate_mask=selector_input.dataset["candidate_mask"],
        offsets=offsets,
        actor_training_seed=selector_input.dataset["policy_training_seed"],
        source_seed=selector_input.dataset["source_seed"],
        inner_cluster_id=selector_input.dataset["trajectory_id"],
        execution_lock=execution_lock_sha256,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        bootstrap_inner_unit="trajectory",
        expected_group_count=(
            _EXPECTED_GROUPS["selector_calibration"]
            if production_contract else None
        ),
    )
    return selector_member, search


def _selector_decisions(
    *,
    member_risk: np.ndarray,
    dataset: GroupedBranchDataset,
    offsets: RecoveryConformalOffsets,
    search: RecoverySelectorSearchResult,
) -> tuple[np.ndarray, np.ndarray]:
    config = search.selected_config
    if config is None:
        raise StageBFitError("no selector configuration was frozen")
    selected = np.empty(dataset.group_count, dtype=np.int64)
    nominal_risk_lcb = np.empty(dataset.group_count, dtype=np.float64)
    for group in range(dataset.group_count):
        decision = select_recovery_program(
            member_risk[group],
            candidate_requested=dataset["candidate_requested"][group],
            candidate_executed=dataset["candidate_executed"][group],
            candidate_q_target=dataset["candidate_q_target"][group],
            candidate_mask=np.asarray(
                dataset["candidate_mask"][group], dtype=np.bool_
            ),
            offsets=offsets,
            config=config,
        )
        selected[group] = decision.selected_index
        nominal_risk_lcb[group] = decision.nominal_risk_lcb
    return selected, nominal_risk_lcb


def fit_outcome_free_placebo(
    *,
    member_risk: np.ndarray,
    dataset: GroupedBranchDataset,
    offsets: RecoveryConformalOffsets,
    search: RecoverySelectorSearchResult,
    selector_bundle: RecoverySelectorBundle,
    execution_lock_sha256: str,
) -> MatchedRandomPlaceboBundle:
    """Fit the placebo using predictions, decisions, and action geometry only."""
    selected, nominal_risk_lcb = _selector_decisions(
        member_risk=member_risk,
        dataset=dataset,
        offsets=offsets,
        search=search,
    )
    requested = np.asarray(dataset["candidate_requested"], dtype=np.float64)
    q_target = np.asarray(dataset["candidate_q_target"], dtype=np.float64)
    requested_rms = np.sqrt(np.mean(
        np.square(requested - requested[:, :1]), axis=2
    ))
    q_target_rms = np.sqrt(np.mean(
        np.square(q_target - q_target[:, :1]), axis=2
    ))
    # Placebo support uses only physical/current-state gates.  Risk, benefit,
    # epistemic uncertainty, and the Q_safe option ordering are deliberately
    # absent from this mask.
    support = (
        np.asarray(dataset["candidate_mask"], dtype=bool)
        & (requested_rms <= 0.50)
        & (q_target_rms <= 0.25)
    )
    support[:, 0] = True
    return fit_matched_random_placebo(
        nominal_risk_lcb=nominal_risk_lcb,
        qsafe_selected_index=selected,
        candidate_support_mask=support,
        candidate_duration_steps=np.asarray(
            dataset["candidate_behavior_steps"], dtype=np.int64
        ),
        first_action_distance=requested_rms,
        placebo_source_seed=np.asarray(dataset["source_seed"], dtype=np.uint64),
        group_fingerprint_sha256=np.asarray(dataset["state_hash"]).astype(str),
        placebo_draw_index=0,
        selector_config=search.selected_config,
        selector_bundle_sha256=selector_bundle.bundle_sha256,
        execution_lock=execution_lock_sha256,
    )


def _published_binding(
    *,
    path: Path,
    stage_b_root: Path,
    contract_sha256: str | None = None,
) -> dict[str, str]:
    result = {
        "relative_path": path.relative_to(stage_b_root).as_posix(),
        "file_sha256": _file_sha256(path, path.name),
    }
    if contract_sha256 is not None:
        result["contract_sha256"] = contract_sha256
    return result


def _failure_report(
    *,
    reason: str,
    frozen_identity: Mapping[str, Any],
    frozen_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    if reason not in ("no_feasible_selector", "placebo_balance_failed"):
        raise StageBFitError("unknown Stage-B development failure")
    return _self_hashed_report({
        "schema_version": DEVELOPMENT_FAILURE_SCHEMA_VERSION,
        "frozen_identity": frozen_identity,
        "status": "terminal_stage_b_development_failure",
        "reason": reason,
        "frozen_artifacts": frozen_artifacts,
        "model_test_outcomes_read": False,
        "model_test_consumed": False,
        "stage_B_pass": False,
        "stage_C_authorized": False,
        "objective1_pass": False,
        "phase2_authorized": False,
    })


def _require_outputs_absent(stage_b_root: Path) -> None:
    derived = [stage_b_root / name for name in _REPORT_FILENAMES.values()]
    derived.append(stage_b_root / "qsafe-artifact")
    derived.append(stage_b_root.parent / "state-dependent-recovery-stage-b-report.json")
    existing = [path for path in derived if os.path.lexists(path)]
    if existing:
        raise StageBFitError(
            f"Stage-B fitting is one-shot; output already exists: {existing[0]}")


def _rehash_inputs(inputs: StageBFrozenDevelopmentInputs) -> None:
    if require_clean_stage_b_generator() != inputs.generator_commit:
        raise StageBFitError("generator commit changed during Stage-B fitting")
    fixed = (
        (
            inputs.stage_b_root / "actor-bank-manifest.json",
            inputs.actor_bank_manifest_file_sha256,
            "actor-bank manifest",
        ),
        (
            inputs.stage_b_root / "stage-b-split-disjointness-report.json",
            inputs.split_disjointness_report_file_sha256,
            "split-disjointness report",
        ),
        (
            inputs.stage_b_root / "model-test-committed.json",
            inputs.model_test_commitment_file_sha256,
            "Model-Test commitment",
        ),
    )
    for path, expected, name in fixed:
        if _file_sha256(path, name) != expected:
            raise StageBFitError(f"{name} changed during fitting")
    for role in DEVELOPMENT_ROLES:
        item = inputs.roles[role]
        with stage_b_evidence_read_scope(
            scientific_role=role,
            evidence_kind="label",
            path=item.path,
        ):
            current = _file_sha256(item.path, f"{role} merged labels")
        if current != item.file_sha256:
            raise StageBFitError(f"{role} labels changed during fitting")


def run_stage_b_development_fit() -> StageBFitResult:
    """Execute the canonical one-shot development fitting workflow."""
    inputs = load_frozen_development_inputs()
    root = inputs.stage_b_root
    _require_outputs_absent(root)
    fit_input = inputs.roles["fit"]
    probability_input = inputs.roles["probability_calibration"]
    uncertainty_input = inputs.roles["uncertainty_calibration"]
    selector_input = inputs.roles["selector_calibration"]

    normalization = NormalizationStats.fit(fit_input.dataset)
    normalization_report = build_normalization_report(
        normalization=normalization,
        fit_input=fit_input,
        frozen_identity=inputs.frozen_identity,
    )
    normalization_path = root / _REPORT_FILENAMES["normalization"]
    normalization_file_sha256 = _atomic_no_clobber_json(
        normalization_path, normalization_report)

    train_view = TorchGroupedView(
        fit_input.dataset,
        normalization,
        action_view=RECOVERY_PROGRAM_VIEW,
        view_role="training",
    )
    probability_view = TorchGroupedView(
        probability_input.dataset,
        normalization,
        action_view=RECOVERY_PROGRAM_VIEW,
        view_role="calibration",
    )
    uncertainty_view = TorchGroupedView(
        uncertainty_input.dataset,
        normalization,
        action_view=RECOVERY_PROGRAM_VIEW,
        view_role="calibration",
    )
    selector_view = TorchGroupedView(
        selector_input.dataset,
        normalization,
        action_view=RECOVERY_PROGRAM_VIEW,
        view_role="calibration",
    )
    trained = train_qsafe_ensemble(
        train_view,
        RECOVERY_PROGRAM_V4_NETWORK_CONFIG,
        RECOVERY_PROGRAM_V4_TRAINING_CONFIG,
        RECOVERY_PROGRAM_V4_LOSS_CONFIG,
        probability_view,
    )
    probability_member = predict_recovery_member_risk(
        trained, probability_view, device="cpu", batch_size=256)
    probability_empirical = np.asarray(
        probability_input.dataset["fall"], dtype=np.float64
    ).mean(axis=2)
    probability_report = build_probability_report(
        trained=trained,
        member_risk=probability_member,
        empirical_risk=probability_empirical,
        probability_input=probability_input,
        normalization_report=normalization_report,
        normalization_report_file_sha256=normalization_file_sha256,
        frozen_identity=inputs.frozen_identity,
    )
    probability_path = root / _REPORT_FILENAMES["probability"]
    probability_file_sha256 = _atomic_no_clobber_json(
        probability_path, probability_report)

    conformal = fit_development_conformal(
        trained=trained,
        uncertainty_view=uncertainty_view,
        uncertainty_input=uncertainty_input,
        execution_lock_sha256=EXECUTION_PROTOCOL_CONTRACT_SHA256,
        production_contract=True,
    )
    uncertainty_report = build_uncertainty_report(
        calibration=conformal,
        uncertainty_input=uncertainty_input,
        probability_report=probability_report,
        probability_report_file_sha256=probability_file_sha256,
        frozen_identity=inputs.frozen_identity,
    )
    uncertainty_path = root / _REPORT_FILENAMES["uncertainty"]
    uncertainty_file_sha256 = _atomic_no_clobber_json(
        uncertainty_path, uncertainty_report)
    offsets = RecoveryConformalOffsets(
        nominal_lower=conformal.nominal_lower,
        risk_upper=conformal.risk_upper,
        benefit_lower=conformal.benefit_lower,
        calibration_report_sha256=uncertainty_report["report_sha256"],
    ).validated()

    selector_member, search = search_development_selector(
        trained=trained,
        selector_view=selector_view,
        selector_input=selector_input,
        offsets=offsets,
        execution_lock_sha256=EXECUTION_PROTOCOL_CONTRACT_SHA256,
        production_contract=True,
    )

    selector_report = build_selector_report(
        search=search,
        selector_input=selector_input,
        probability_report=probability_report,
        uncertainty_report=uncertainty_report,
        uncertainty_report_file_sha256=uncertainty_file_sha256,
        frozen_identity=inputs.frozen_identity,
    )
    selector_path = root / _REPORT_FILENAMES["selector"]
    selector_file_sha256 = _atomic_no_clobber_json(
        selector_path, selector_report)
    common_artifacts: dict[str, Any] = {
        "normalization_report": _published_binding(
            path=normalization_path,
            stage_b_root=root,
            contract_sha256=normalization_report["report_sha256"],
        ),
        "probability_calibration_report": _published_binding(
            path=probability_path,
            stage_b_root=root,
            contract_sha256=probability_report["report_sha256"],
        ),
        "uncertainty_calibration_report": _published_binding(
            path=uncertainty_path,
            stage_b_root=root,
            contract_sha256=uncertainty_report["report_sha256"],
        ),
        "selector_search_report": _published_binding(
            path=selector_path,
            stage_b_root=root,
            contract_sha256=selector_report["report_sha256"],
        ),
    }
    if search.selected_config is None:
        failure = _failure_report(
            reason="no_feasible_selector",
            frozen_identity=inputs.frozen_identity,
            frozen_artifacts=common_artifacts,
        )
        failure_path = root.parent / "state-dependent-recovery-stage-b-report.json"
        failure_file_sha256 = _atomic_no_clobber_json(failure_path, failure)
        return StageBFitResult(
            status="terminal_no_feasible_selector_before_model_test",
            selector_feasible=False,
            placebo_balanced=False,
            frozen_artifact_sha256={
                **{
                    name: value["file_sha256"]
                    for name, value in common_artifacts.items()
                },
                "stage_b_failure_report": failure_file_sha256,
            },
            failure_report=failure_path,
        )

    selector_bundle = RecoverySelectorBundle.create(
        offsets=offsets,
        selector_config=search.selected_config,
        probability_calibration_report_sha256=probability_report[
            "report_sha256"
        ],
        uncertainty_calibration_report_sha256=uncertainty_report[
            "report_sha256"
        ],
        selector_search_report_sha256=selector_report["report_sha256"],
    )
    selector_bundle_path = root / _REPORT_FILENAMES["selector_bundle"]
    selector_bundle_file_sha256 = _atomic_no_clobber_json(
        selector_bundle_path, selector_bundle.to_dict())
    common_artifacts["recovery_selector_bundle"] = _published_binding(
        path=selector_bundle_path,
        stage_b_root=root,
        contract_sha256=selector_bundle.bundle_sha256,
    )

    placebo = fit_outcome_free_placebo(
        member_risk=selector_member,
        dataset=selector_input.dataset,
        offsets=offsets,
        search=search,
        selector_bundle=selector_bundle,
        execution_lock_sha256=EXECUTION_PROTOCOL_CONTRACT_SHA256,
    )
    placebo_path = root / _REPORT_FILENAMES["placebo_bundle"]
    placebo_file_sha256 = _atomic_no_clobber_json(
        placebo_path, placebo.to_dict())
    common_artifacts["matched_random_placebo_bundle"] = _published_binding(
        path=placebo_path,
        stage_b_root=root,
        contract_sha256=placebo.bundle_sha256,
    )
    if not placebo.fit_metrics.eligible:
        failure = _failure_report(
            reason="placebo_balance_failed",
            frozen_identity=inputs.frozen_identity,
            frozen_artifacts=common_artifacts,
        )
        failure_path = root.parent / "state-dependent-recovery-stage-b-report.json"
        failure_file_sha256 = _atomic_no_clobber_json(failure_path, failure)
        return StageBFitResult(
            status="terminal_placebo_balance_failure_before_model_test",
            selector_feasible=True,
            placebo_balanced=False,
            frozen_artifact_sha256={
                **{
                    name: value["file_sha256"]
                    for name, value in common_artifacts.items()
                },
                "stage_b_failure_report": failure_file_sha256,
            },
            failure_report=failure_path,
        )

    _rehash_inputs(inputs)
    artifact_path = root / "qsafe-artifact"
    artifact_provenance = {
        **dict(inputs.frozen_identity),
        "command_vx": 0.30,
        "action_view": RECOVERY_PROGRAM_VIEW,
        "recovery_program": train_view.recovery_program_binding,
        "recovery_program_feature_contract": (
            train_view.recovery_program_feature_manifest
        ),
        "recovery_selector_bundle": selector_bundle.to_dict(),
        "recovery_selector_bundle_sha256": selector_bundle.bundle_sha256,
        "matched_random_placebo_bundle_sha256": placebo.bundle_sha256,
        "frozen_development_artifacts": common_artifacts,
        "model_test_commitment_file_sha256": (
            inputs.model_test_commitment_file_sha256
        ),
        "model_test_outcomes_read": False,
        "model_test_consumed": False,
    }
    save_qsafe_artifact(
        artifact_path,
        trained,
        normalization,
        RECOVERY_PROGRAM_V4_NETWORK_CONFIG,
        RECOVERY_PROGRAM_V4_TRAINING_CONFIG,
        RECOVERY_PROGRAM_V4_LOSS_CONFIG,
        provenance=artifact_provenance,
        recovery_selector_bundle=selector_bundle,
        pre_publish_check=lambda: _rehash_inputs(inputs),
    )
    artifact_manifest_file_sha256 = _file_sha256(
        artifact_path / "manifest.json", "Q_safe manifest")
    artifact_manifest_canonical_sha256 = _artifact_manifest_canonical_sha256(
        artifact_path)
    hashes = {
        name: value["file_sha256"]
        for name, value in common_artifacts.items()
    }
    hashes.update({
        "recovery_selector_bundle": selector_bundle_file_sha256,
        "matched_random_placebo_bundle": placebo_file_sha256,
        "qsafe_artifact_manifest_file_sha256": (
            artifact_manifest_file_sha256
        ),
        "qsafe_artifact_manifest_canonical_sha256": (
            artifact_manifest_canonical_sha256
        ),
    })
    return StageBFitResult(
        status="development_artifacts_frozen_model_test_not_consumed",
        selector_feasible=True,
        placebo_balanced=True,
        frozen_artifact_sha256=hashes,
    )


__all__ = [
    "DEVELOPMENT_FAILURE_SCHEMA_VERSION",
    "DEVELOPMENT_ROLES",
    "NORMALIZATION_REPORT_SCHEMA_VERSION",
    "PROBABILITY_REPORT_SCHEMA_VERSION",
    "SELECTOR_REPORT_SCHEMA_VERSION",
    "StageBFitError",
    "StageBFitResult",
    "StageBFrozenDevelopmentInputs",
    "StageBRoleInput",
    "UNCERTAINTY_REPORT_SCHEMA_VERSION",
    "build_normalization_report",
    "build_probability_report",
    "build_selector_report",
    "build_uncertainty_report",
    "fit_development_calibration",
    "fit_development_conformal",
    "fit_outcome_free_placebo",
    "load_frozen_development_inputs",
    "run_stage_b_development_fit",
    "search_development_selector",
]
