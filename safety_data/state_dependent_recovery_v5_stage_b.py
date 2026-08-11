"""Result-blind execution contract for V5 Stage B.

This module deliberately contains no model-test outcome evaluator.  It binds
the immutable V5 protocol, the passing Stage-A authorization, the Stage-B
execution supplement, actor/source roster, and the ten physical RNG domains
before any Stage-B outcome may be produced.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from safety_data.closed_loop_recovery_collector import canonical_protocol_sha256
from safety_data.state_dependent_recovery_v5 import (
    PROTOCOL_CONTRACT_SHA256 as PARENT_PROTOCOL_CONTRACT_SHA256,
    PROTOCOL_FILE_SHA256 as PARENT_PROTOCOL_FILE_SHA256,
    PROTOCOL_PATH as PARENT_PROTOCOL_PATH,
    REPORT_SCHEMA_VERSION as STAGE_A_REPORT_SCHEMA_VERSION,
    StateDependentRecoveryV5Error,
    load_state_dependent_recovery_v5_protocol,
)


EXECUTION_PROTOCOL_NAME = (
    "objective1_state_dependent_recovery_qsafe_v5_stage_b_execution"
)
EXECUTION_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "qsafe_state_dependent_recovery_v5_stage_b_execution.yaml"
)
EXECUTION_PROTOCOL_FILE_SHA256 = (
    "bb96cad57f79e16025a4a80254aaebee2e66289414994fa39f11865aa5a96687"
)
EXECUTION_PROTOCOL_CONTRACT_SHA256 = (
    "17c1ae4f8109f2d786f958830e838c52da7a03e61561dfaa9f790d75b9f70086"
)
STAGE_A_REPORT_SHA256 = (
    "e7ea56546bf8006cfc4d8ade4f5b2c26dbfcbc132e0568e054a98c2be3174b2e"
)
STAGE_A_DISPOSITION_COMMIT = (
    "959605d621163f2122d6947bbb8fd657a51d5f7f"
)
RECOVERY_LIBRARY_FINGERPRINT_SHA256 = (
    "fcfb1fa541acf316f87dacf82b1fdeb9188d7a4b9df7f69544b567fb2c5d1045"
)

ROLE_ORDER = (
    "fit",
    "probability_calibration",
    "uncertainty_calibration",
    "selector_calibration",
    "model_test",
)
ROLE_ACTOR_SEEDS: Mapping[str, tuple[int, ...]] = {
    "fit": (43, 44, 45, 46),
    "probability_calibration": (47, 48),
    "uncertainty_calibration": (49, 50),
    "selector_calibration": (51, 52),
    "model_test": (53, 54, 55, 56),
}
ROLE_SOURCE_SEEDS: Mapping[str, tuple[int, ...]] = {
    "fit": (8501, 8502, 8503, 8504, 8511, 8512, 8513, 8514,
            8521, 8522, 8523, 8524),
    "probability_calibration": (8601, 8602, 8611, 8612, 8621, 8622),
    "uncertainty_calibration": (8631, 8632, 8641, 8642, 8651, 8652),
    "selector_calibration": (8661, 8662, 8671, 8672, 8681, 8682),
    "model_test": (8701, 8702, 8703, 8704, 8711, 8712, 8713, 8714,
                   8721, 8722, 8723, 8724),
}
CHECKPOINT_STEPS = (25_000, 50_000, 100_000)
GROUPS_PER_SOURCE: Mapping[str, int] = {
    "fit": 128,
    "probability_calibration": 64,
    "uncertainty_calibration": 64,
    "selector_calibration": 64,
    "model_test": 64,
}
LABEL_REPLICAS: Mapping[str, int] = {
    "fit": 32,
    "probability_calibration": 32,
    "uncertainty_calibration": 32,
    "selector_calibration": 32,
    "model_test": 64,
}
ADMISSION_REPLICAS = 32
CANDIDATES = 9
HORIZON_POLICY_STEPS = 96
SPLIT_COLLISION_DIMENSIONS = (
    "policy_training_seed",
    "actor_checkpoint_sha256",
    "actor_state_dict_sha256",
    "policy_fingerprint_sha256",
    "checkpoint_fingerprint_sha256",
    "state_fingerprint_sha256",
    "trajectory_fingerprint_sha256",
    "crn_id",
    "rollout_seed",
    "perturbation_seed",
    "candidate_seed",
)
TRAJECTORY_FINGERPRINT_ARRAY = "trajectory_fingerprint_sha256"
TRAJECTORY_FINGERPRINT_CONTRACT = (
    "sha256_compound_post_settle_pre_source_trajectory_snapshot_v1"
)
SPLIT_IDENTITY_SOURCE_FIELDS: Mapping[str, str] = {
    "policy_training_seed": "policy_training_seed",
    "actor_checkpoint_sha256": (
        "actor_bank.identities[].actor_checkpoint_sha256"
    ),
    "actor_state_dict_sha256": (
        "actor_bank.identities[].actor_state_dict_sha256"
    ),
    "policy_fingerprint_sha256": (
        "actor_bank.identities[].policy_fingerprint_sha256"
    ),
    "checkpoint_fingerprint_sha256": (
        "actor_bank.identities[].checkpoint_fingerprint_sha256"
    ),
    "state_fingerprint_sha256": "state_hash",
    "trajectory_fingerprint_sha256": TRAJECTORY_FINGERPRINT_ARRAY,
    "crn_id": "crn_id",
    "rollout_seed": "rollout_seed",
    "perturbation_seed": "perturbation_seed",
    "candidate_seed": "candidate_seed",
}
_SPLIT_IDENTITY_ARRAY_NAMES = frozenset({
    "group_id",
    "source_seed",
    "policy_training_seed",
    "policy_source",
    "state_hash",
    TRAJECTORY_FINGERPRINT_ARRAY,
    "crn_id",
    "rollout_seed",
    "perturbation_seed",
    "candidate_seed",
})


@dataclass(frozen=True)
class StageBSplitIdentityView:
    """Outcome-inaccessible input accepted by the split-proof compiler."""

    manifest: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    content_sha256: str

    def __post_init__(self) -> None:
        if set(self.arrays) != _SPLIT_IDENTITY_ARRAY_NAMES:
            raise StageBExecutionError(
                "split identity view must expose exactly the frozen identity arrays"
            )
        if not isinstance(self.content_sha256, str) or _HEX64.fullmatch(
            self.content_sha256
        ) is None:
            raise StageBExecutionError(
                "split identity view content hash must be lowercase SHA-256"
            )
        object.__setattr__(self, "manifest", dict(self.manifest))
        object.__setattr__(self, "arrays", {
            str(name): np.asarray(value) for name, value in self.arrays.items()
        })

    @property
    def group_count(self) -> int:
        return int(np.asarray(self.arrays["group_id"]).shape[0])

    def validate(self, **_: Any) -> dict[str, Any]:
        """Compatibility surface for callers that only need the commitment."""
        return {"content_sha256": self.content_sha256, "groups": self.group_count}

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in _SPLIT_IDENTITY_ARRAY_NAMES:
            raise StageBExecutionError(
                f"split identity view forbids non-identity field {name!r}"
            )
        return self.arrays[name]


def make_split_identity_view(
    dataset: Any,
    *,
    content_sha256: str | None = None,
) -> StageBSplitIdentityView:
    """Project a deployable shard onto the outcome-inaccessible identity view.

    This helper intentionally copies only the frozen identity arrays.  It does
    not call ``GroupedBranchDataset.validate`` (which may summarize outcomes),
    and therefore is safe to use while the Model-Test producer capability is
    active.
    """
    arrays = getattr(dataset, "arrays", None)
    manifest = getattr(dataset, "manifest", None)
    if not isinstance(arrays, Mapping) or not isinstance(manifest, Mapping):
        raise StageBExecutionError("split identity source is not a dataset")
    missing = sorted(_SPLIT_IDENTITY_ARRAY_NAMES - set(arrays))
    if missing:
        raise StageBExecutionError(
            f"split identity source omits frozen arrays: {missing}")
    collection = manifest.get("collection_protocol")
    if not isinstance(collection, Mapping) or collection.get(
        "trajectory_fingerprint_array"
    ) != TRAJECTORY_FINGERPRINT_ARRAY or collection.get(
        "trajectory_fingerprint_contract"
    ) != TRAJECTORY_FINGERPRINT_CONTRACT:
        raise StageBExecutionError(
            "split identity source lacks the post-settle trajectory fingerprint contract")
    digest = content_sha256 or manifest.get("content_sha256")
    if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
        # The identity commitment is deliberately independent of outcome
        # columns; use a canonical digest of the projected arrays when a
        # caller is working with an in-memory shard before serialization.
        digest_payload = {
            "manifest": dict(manifest),
            "arrays": {
                name: np.asarray(arrays[name]).tolist()
                for name in sorted(_SPLIT_IDENTITY_ARRAY_NAMES)
            },
        }
        digest = canonical_sha256(digest_payload)
    return StageBSplitIdentityView(
        manifest=dict(manifest),
        arrays={name: np.asarray(arrays[name]) for name in _SPLIT_IDENTITY_ARRAY_NAMES},
        content_sha256=digest,
    )

_SEED_ALGORITHM = "high_bit_then_domain_low15_then_14_8_18_2_6_bitpack_v1"
_SEED_ROLE_TAGS: Mapping[str, int] = {
    "source_reset": 110,
    "source_impulse": 111,
    "source_action": 112,
    "admission": 120,
    "label": 130,
}
_ROLE_DOMAIN_LITERALS: Mapping[str, Mapping[str, bytes]] = {
    role: {
        partition: (
            f"qsafe_state_dependent_recovery_v4_stage_b_{role}_{partition}\0"
        ).encode("ascii")
        for partition in ("admission", "label")
    }
    for role in ROLE_ORDER
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class StageBExecutionError(StateDependentRecoveryV5Error):
    """The frozen Stage-B execution contract failed closed."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise StageBExecutionError(
                f"Stage-B execution YAML contains duplicate key {key!r}"
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def canonical_sha256(value: Any) -> str:
    """Return the stable canonical-JSON SHA-256 for JSON-compatible data."""
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StageBExecutionError(
            "value is not canonical JSON-compatible data"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StageBExecutionError(f"{name} must be a mapping")
    return value


def _require_equal(actual: object, expected: object, name: str) -> None:
    if actual != expected:
        raise StageBExecutionError(f"{name} has drifted")


def _decode_literal(value: object, name: str) -> bytes:
    if not isinstance(value, str) or not value.endswith("\\0"):
        raise StageBExecutionError(f"{name} must be ASCII ending in escaped NUL")
    try:
        result = value[:-2].encode("ascii") + b"\0"
    except UnicodeEncodeError as exc:
        raise StageBExecutionError(f"{name} must be ASCII") from exc
    return result


def load_stage_b_execution_protocol(
    path: str | os.PathLike[str] = EXECUTION_PROTOCOL_PATH,
    *,
    enforce_canonical_hash: bool = True,
) -> dict[str, Any]:
    """Load and fully bind the result-blind Stage-B execution supplement."""
    candidate = Path(path)
    try:
        raw = candidate.read_bytes()
        parsed = yaml.load(raw.decode("utf-8"), Loader=_UniqueKeySafeLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise StageBExecutionError(
            "could not load Stage-B execution protocol"
        ) from exc
    protocol = dict(_mapping(parsed, "Stage-B execution protocol"))
    raw_sha = hashlib.sha256(raw).hexdigest()
    contract_sha = canonical_protocol_sha256(protocol)
    if enforce_canonical_hash:
        _require_equal(
            raw_sha,
            EXECUTION_PROTOCOL_FILE_SHA256,
            "Stage-B execution protocol file SHA-256",
        )
        _require_equal(
            contract_sha,
            EXECUTION_PROTOCOL_CONTRACT_SHA256,
            "Stage-B execution protocol contract SHA-256",
        )
    _require_equal(
        protocol.get("protocol_name"),
        EXECUTION_PROTOCOL_NAME,
        "Stage-B execution protocol name",
    )
    _validate_execution_semantics(protocol)
    protocol["execution_protocol_file_sha256"] = raw_sha
    protocol["execution_protocol_contract_sha256"] = contract_sha
    return protocol


def _validate_execution_semantics(protocol: Mapping[str, Any]) -> None:
    authorization = _mapping(protocol.get("authorization"), "authorization")
    expected_authorization = {
        "parent_protocol_path": "config/qsafe_state_dependent_recovery_v5.yaml",
        "parent_protocol_file_sha256": PARENT_PROTOCOL_FILE_SHA256,
        "parent_protocol_contract_sha256": PARENT_PROTOCOL_CONTRACT_SHA256,
        "stage_a_report_relative_path": (
            "state-dependent-recovery-stage-a-report.json"
        ),
        "stage_a_report_sha256": STAGE_A_REPORT_SHA256,
        "stage_a_required_decision": "authorize_stage_B_only",
        "stage_a_required_pass": True,
        "stage_a_disposition_commit": STAGE_A_DISPOSITION_COMMIT,
        "recovery_library_fingerprint_sha256": (
            RECOVERY_LIBRARY_FINGERPRINT_SHA256
        ),
        "first_attempt_generator_rule": (
            "clean_HEAD_contains_execution_lock_implementation_and_tests"
        ),
        "generator_commit_capture": (
            "first_actor_bank_attempt_before_any_training_transition"
        ),
        "generator_commit_reuse": "exact_same_commit_all_stage_b_operations",
    }
    _require_equal(dict(authorization), expected_authorization, "authorization")

    actor_bank = _mapping(protocol.get("actor_bank"), "actor_bank")
    _require_equal(
        actor_bank.get("training_seeds_exact"),
        list(range(43, 57)),
        "actor_bank.training_seeds_exact",
    )
    _require_equal(
        actor_bank.get("checkpoint_steps_exact"),
        list(CHECKPOINT_STEPS),
        "actor_bank.checkpoint_steps_exact",
    )
    _require_equal(
        actor_bank.get("checkpoint_count_exact"), 42,
        "actor_bank.checkpoint_count_exact",
    )
    _require_equal(
        actor_bank.get("snapshot_timing"),
        "after_transition_and_scheduled_update_before_next_transition",
        "actor_bank.snapshot_timing",
    )
    _require_equal(
        actor_bank.get("retain_every_seed_and_checkpoint_without_return_or_fall_filter"),
        True,
        "actor_bank outcome-independent retention",
    )

    role_execution = _mapping(protocol.get("role_execution"), "role_execution")
    _require_equal(role_execution.get("role_order"), list(ROLE_ORDER), "role order")
    _require_equal(
        role_execution.get("label_replicas"),
        dict(LABEL_REPLICAS),
        "role label replicas",
    )
    _require_equal(
        role_execution.get("admission_replicas"),
        ADMISSION_REPLICAS,
        "role admission replicas",
    )
    _require_equal(
        role_execution.get("label_candidate_count"), CANDIDATES,
        "role candidate count",
    )
    _require_equal(
        role_execution.get("label_horizon_policy_steps"), HORIZON_POLICY_STEPS,
        "role horizon",
    )

    seeds = _mapping(protocol.get("seed_derivation"), "seed_derivation")
    _require_equal(seeds.get("algorithm"), _SEED_ALGORITHM, "seed algorithm")
    _require_equal(seeds.get("role_tags"), dict(_SEED_ROLE_TAGS), "seed role tags")
    domains = _mapping(seeds.get("domains"), "seed_derivation.domains")
    prefixes: list[int] = []
    for role in ROLE_ORDER:
        role_domains = _mapping(domains.get(role), f"domains.{role}")
        for partition in ("admission", "label"):
            entry = _mapping(
                role_domains.get(partition), f"domains.{role}.{partition}"
            )
            literal = _decode_literal(
                entry.get("literal_ascii_escaped"),
                f"domains.{role}.{partition}.literal_ascii_escaped",
            )
            _require_equal(
                literal,
                _ROLE_DOMAIN_LITERALS[role][partition],
                f"domains.{role}.{partition} literal",
            )
            prefix = int.from_bytes(
                hashlib.sha256(literal).digest()[:2], "little"
            ) & 0x7FFF
            _require_equal(
                entry.get("sha256_prefix_low15"),
                prefix,
                f"domains.{role}.{partition} prefix",
            )
            prefixes.append(prefix)
    if len(prefixes) != 10 or len(set(prefixes)) != 10:
        raise StageBExecutionError("the ten Stage-B RNG prefixes must be distinct")
    _require_equal(
        seeds.get("all_ten_domain_prefixes_pairwise_distinct"),
        True,
        "ten-domain disjointness declaration",
    )

    split = _mapping(
        protocol.get("split_and_normalization"),
        "split_and_normalization",
    )
    expected_split_proof = {
        "pairwise_role_pairs_exact": 10,
        "collision_dimensions": list(SPLIT_COLLISION_DIMENSIONS),
        "zero_collisions_required": True,
        "proof_may_read_outcome_values": False,
        "proof_publication_path": (
            "stage-b/stage-b-split-disjointness-report.json"
        ),
        "proof_publication_timing": (
            "blind_producer_after_model_test_merge_before_model_test_role_report"
        ),
        "model_test_identity_commitment_source": (
            "in_memory_merged_identity_columns_and_staged_label_byte_sha256"
        ),
        "proof_binds_all_five_aggregate_label_file_and_content_sha256": True,
        "model_test_role_report_published_after_proof_revokes_producer": True,
        "commitment_compiler_reads_model_test_role_report_only": True,
    }
    for field, expected in expected_split_proof.items():
        _require_equal(
            split.get(field), expected, f"split_and_normalization.{field}"
        )

    firewall = _mapping(protocol.get("model_test_firewall"), "model_test_firewall")
    _require_equal(
        firewall.get("consumed_marker_publication"),
        "atomic_no_clobber_before_first_forbidden_operation",
        "model-test consumption ordering",
    )
    _require_equal(
        firewall.get("crash_after_consumed_marker"),
        "permanently_consumed_stage_b_failure_no_rerun",
        "model-test crash policy",
    )

    conformal = _mapping(protocol.get("conformal"), "conformal")
    _require_equal(
        conformal.get("nonnominal_candidate_indices"),
        list(range(1, 9)),
        "conformal candidate indices",
    )
    _require_equal(conformal.get("per_option_alpha"), 0.00625, "conformal alpha")
    _require_equal(conformal.get("offsets_remain_signed"), True, "signed offsets")

    model_test = _mapping(
        protocol.get("model_test_statistics"), "model_test_statistics"
    )
    _require_equal(model_test.get("replicates"), 50_000, "model-test replicates")
    _require_equal(model_test.get("seed"), 20_260_812, "model-test seed")
    _require_equal(
        model_test.get("pair_accuracy_lcb"),
        "percentile_0.025_two_sided_95_lower_endpoint",
        "pair-accuracy lower endpoint",
    )
    _require_equal(
        model_test.get("empirical_pair_ties"),
        "excluded",
        "empirical pair-tie handling",
    )
    _require_equal(
        model_test.get("predicted_pair_ties"),
        "half_credit",
        "predicted pair-tie handling",
    )
    _require_equal(model_test.get("pass_authorizes"), "stage_C_only", "Stage-B authorization")
    _require_equal(model_test.get("objective1_pass_after_stage_b"), False, "Objective-1 guard")
    _require_equal(model_test.get("phase2_authorized_after_stage_b"), False, "Phase-2 guard")


def stage_b_artifact_root(parent_protocol: Mapping[str, Any]) -> Path:
    collection = _mapping(parent_protocol.get("collection"), "parent collection")
    root = Path(str(collection.get("artifact_root")))
    if not root.is_absolute():
        root = _REPOSITORY_ROOT / root
    return Path(os.path.abspath(os.fspath(root))) / "stage-b"


def validate_stage_a_authorization(
    execution_protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the canonical Stage-A report before any Stage-B mutation."""
    execution = dict(
        execution_protocol
        if execution_protocol is not None
        else load_stage_b_execution_protocol()
    )
    parent = load_state_dependent_recovery_v5_protocol()
    root = stage_b_artifact_root(parent).parent
    relative = str(execution["authorization"]["stage_a_report_relative_path"])
    report_path = root / relative
    if file_sha256(report_path) != STAGE_A_REPORT_SHA256:
        raise StageBExecutionError("canonical Stage-A report SHA-256 mismatch")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StageBExecutionError("canonical Stage-A report is unreadable") from exc
    checked = _mapping(report, "canonical Stage-A report")
    required = {
        "schema_version": STAGE_A_REPORT_SCHEMA_VERSION,
        "protocol_contract_sha256": PARENT_PROTOCOL_CONTRACT_SHA256,
        "protocol_file_sha256": PARENT_PROTOCOL_FILE_SHA256,
        "decision": "authorize_stage_B_only",
        "stage_B_authorized": True,
        "model_training_authorized": True,
        "model_training_triggered": False,
        "paired_closed_loop_authorized": False,
        "online_training_authorized": False,
        "objective1_pass": False,
        "phase2_authorized": False,
    }
    for name, expected in required.items():
        _require_equal(checked.get(name), expected, f"Stage-A report {name}")
    primary = _mapping(checked.get("stage_A_primary"), "Stage-A primary")
    _require_equal(primary.get("pass"), True, "Stage-A primary pass")
    return dict(checked)


def require_clean_stage_b_generator() -> str:
    """Return clean HEAD after proving the Stage-A disposition is its ancestor."""
    try:
        head = subprocess.run(
            ["git", "-C", str(_REPOSITORY_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(_REPOSITORY_ROOT), "status", "--porcelain=v1", "-z"],
            check=True,
            capture_output=True,
        ).stdout
        ancestry = subprocess.run(
            [
                "git", "-C", str(_REPOSITORY_ROOT), "merge-base", "--is-ancestor",
                STAGE_A_DISPOSITION_COMMIT, head,
            ],
            capture_output=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise StageBExecutionError("could not establish clean Stage-B HEAD") from exc
    if _HEX64.fullmatch(head) is None and re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise StageBExecutionError("Stage-B generator commit is malformed")
    if status or ancestry.returncode != 0:
        raise StageBExecutionError(
            "Stage-B operations require a clean descendant of the Stage-A disposition"
        )
    return head


@dataclass(frozen=True)
class StageBSourceAssignment:
    role: str
    actor_training_seed: int
    checkpoint_step: int
    source_seed: int
    groups: int
    admission_replicas: int
    label_replicas: int


def source_assignments() -> tuple[StageBSourceAssignment, ...]:
    """Return the immutable 42-source roster in role/age/actor order."""
    result: list[StageBSourceAssignment] = []
    for role in ROLE_ORDER:
        actors = ROLE_ACTOR_SEEDS[role]
        sources = ROLE_SOURCE_SEEDS[role]
        width = len(actors)
        if len(sources) != width * len(CHECKPOINT_STEPS):
            raise AssertionError("invalid frozen Stage-B source roster")
        for age_index, checkpoint_step in enumerate(CHECKPOINT_STEPS):
            block = sources[age_index * width:(age_index + 1) * width]
            for actor_seed, source_seed in zip(actors, block, strict=True):
                result.append(StageBSourceAssignment(
                    role=role,
                    actor_training_seed=actor_seed,
                    checkpoint_step=checkpoint_step,
                    source_seed=source_seed,
                    groups=GROUPS_PER_SOURCE[role],
                    admission_replicas=ADMISSION_REPLICAS,
                    label_replicas=LABEL_REPLICAS[role],
                ))
    if len(result) != 42 or len({row.source_seed for row in result}) != 42:
        raise AssertionError("Stage-B source roster must contain 42 unique sources")
    return tuple(result)


def assignment_for(role: str, source_seed: int) -> StageBSourceAssignment:
    matches = [
        row for row in source_assignments()
        if row.role == role and row.source_seed == source_seed
    ]
    if len(matches) != 1:
        raise StageBExecutionError("role/source_seed is not in the frozen roster")
    return matches[0]


def stage_b_seed(
    *,
    role: str,
    partition: str,
    source_seed: int,
    stream_role: str,
    identity: int,
    namespace: int,
    index: int = 0,
) -> int:
    """Derive one injective Stage-B seed from the frozen ten-domain mapping."""
    if role not in ROLE_ORDER or partition not in ("admission", "label"):
        raise StageBExecutionError("unknown Stage-B role or partition")
    if stream_role not in _SEED_ROLE_TAGS:
        raise StageBExecutionError("unknown Stage-B RNG stream role")
    if partition == "admission" and stream_role == "label":
        raise StageBExecutionError("label stream cannot use an admission domain")
    if partition == "label" and stream_role != "label":
        raise StageBExecutionError("label domains are reserved for label branches")
    fields = (
        (14, source_seed, "source_seed"),
        (8, _SEED_ROLE_TAGS[stream_role], "role_tag"),
        (18, identity, "identity"),
        (2, namespace, "namespace"),
        (6, index, "index"),
    )
    checked: list[tuple[int, int, str]] = []
    for width, value, name in fields:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ) or int(value) < 0 or int(value) >= 1 << width:
            raise StageBExecutionError(
                f"Stage-B seed {name} exceeds its {width}-bit contract"
            )
        checked.append((width, int(value), name))
    domain = _ROLE_DOMAIN_LITERALS[role][partition]
    packed = int.from_bytes(hashlib.sha256(domain).digest()[:2], "little") & 0x7FFF
    for width, value, _ in checked:
        packed = (packed << width) | value
    if packed >= 1 << 63:
        raise AssertionError("Stage-B seed payload exceeded 63 bits")
    return (1 << 63) | packed


def branch_randomness(
    *,
    role: str,
    partition: str,
    source_seed: int,
    proposal_index: int,
    replicas: int,
) -> dict[str, np.ndarray | np.uint64]:
    """Construct the complete physical admission or label seed bundle."""
    if isinstance(replicas, bool) or not isinstance(replicas, int) or not (
        0 < replicas <= 64
    ):
        raise StageBExecutionError("replicas must be an integer in [1,64]")
    stream_role = "admission" if partition == "admission" else "label"
    arrays: dict[str, np.ndarray | np.uint64] = {}
    for name, namespace in (("crn_id", 0), ("rollout_seed", 1),
                            ("perturbation_seed", 2)):
        arrays[name] = np.asarray([
            stage_b_seed(
                role=role,
                partition=partition,
                source_seed=source_seed,
                stream_role=stream_role,
                identity=proposal_index,
                namespace=namespace,
                index=replica,
            )
            for replica in range(replicas)
        ], dtype=np.uint64)
    arrays["candidate_seed"] = np.uint64(stage_b_seed(
        role=role,
        partition=partition,
        source_seed=source_seed,
        stream_role=stream_role,
        identity=proposal_index,
        namespace=3,
        index=0,
    ))
    return arrays


def validate_role_seed_disjointness() -> dict[str, Any]:
    """Prove domain-prefix and representative packed-stream disjointness."""
    prefixes: dict[str, int] = {}
    representative: dict[str, list[int]] = {}
    for role in ROLE_ORDER:
        assignment = next(row for row in source_assignments() if row.role == role)
        for partition in ("admission", "label"):
            key = f"{role}:{partition}"
            domain = _ROLE_DOMAIN_LITERALS[role][partition]
            prefixes[key] = (
                int.from_bytes(hashlib.sha256(domain).digest()[:2], "little")
                & 0x7FFF
            )
            bundle = branch_randomness(
                role=role,
                partition=partition,
                source_seed=assignment.source_seed,
                proposal_index=0,
                replicas=LABEL_REPLICAS[role] if partition == "label" else 32,
            )
            representative[key] = sorted(set(np.concatenate([
                np.asarray(bundle["crn_id"], dtype=np.uint64),
                np.asarray(bundle["rollout_seed"], dtype=np.uint64),
                np.asarray(bundle["perturbation_seed"], dtype=np.uint64),
                np.asarray([bundle["candidate_seed"]], dtype=np.uint64),
            ]).astype(object).tolist()))
    collisions: dict[str, list[int]] = {}
    keys = list(representative)
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1:]:
            overlap = sorted(set(representative[left]) & set(representative[right]))
            if overlap:
                collisions[f"{left}|{right}"] = overlap
    report = {
        "schema_version": (
            "qsafe.state_dependent_recovery_v5.stage_b_rng_disjointness.v1"
        ),
        "algorithm": _SEED_ALGORITHM,
        "domain_prefix_low15": prefixes,
        "domain_prefixes_pairwise_distinct": len(set(prefixes.values())) == 10,
        "representative_stream_collisions": collisions,
        "pass": len(set(prefixes.values())) == 10 and not collisions,
    }
    return report | {"report_sha256": canonical_sha256(report)}


def execution_identity(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact immutable identity copied into every Stage-B artifact."""
    return {
        "protocol_name": EXECUTION_PROTOCOL_NAME,
        "parent_protocol_file_sha256": PARENT_PROTOCOL_FILE_SHA256,
        "parent_protocol_contract_sha256": PARENT_PROTOCOL_CONTRACT_SHA256,
        "execution_protocol_file_sha256": protocol[
            "execution_protocol_file_sha256"
        ],
        "execution_protocol_contract_sha256": protocol[
            "execution_protocol_contract_sha256"
        ],
        "stage_a_report_sha256": STAGE_A_REPORT_SHA256,
        "stage_a_disposition_commit": STAGE_A_DISPOSITION_COMMIT,
        "recovery_library_fingerprint_sha256": (
            RECOVERY_LIBRARY_FINGERPRINT_SHA256
        ),
    }


def _actor_identities_by_role(
    actor_bank_manifest: Mapping[str, Any],
) -> dict[str, list[Mapping[str, Any]]]:
    identities = actor_bank_manifest.get("identities")
    if not isinstance(identities, list) or len(identities) != 42:
        raise StageBExecutionError("actor bank must contain exactly 42 identities")
    result: dict[str, list[Mapping[str, Any]]] = {role: [] for role in ROLE_ORDER}
    required = {
        "role", "actor_training_seed", "checkpoint_step",
        "actor_checkpoint_sha256", "actor_state_dict_sha256",
        "policy_fingerprint_sha256", "checkpoint_fingerprint_sha256",
    }
    for index, identity in enumerate(identities):
        item = _mapping(identity, f"actor identity {index}")
        if not required.issubset(item):
            raise StageBExecutionError("actor identity omits a collision field")
        role = item.get("role")
        if role not in result:
            raise StageBExecutionError("actor identity has an unknown role")
        if item.get("actor_training_seed") not in ROLE_ACTOR_SEEDS[str(role)] or (
            item.get("checkpoint_step") not in CHECKPOINT_STEPS
        ):
            raise StageBExecutionError("actor identity differs from frozen roster")
        for name in (
            "actor_checkpoint_sha256", "actor_state_dict_sha256",
            "policy_fingerprint_sha256", "checkpoint_fingerprint_sha256",
        ):
            if not isinstance(item.get(name), str) or _HEX64.fullmatch(
                str(item[name])
            ) is None:
                raise StageBExecutionError(f"actor identity {name} is malformed")
        result[str(role)].append(item)
    for role, items in result.items():
        expected = len(ROLE_ACTOR_SEEDS[role]) * len(CHECKPOINT_STEPS)
        observed = {
            (int(item["actor_training_seed"]), int(item["checkpoint_step"]))
            for item in items
        }
        if len(items) != expected or observed != {
            (seed, step)
            for seed in ROLE_ACTOR_SEEDS[role]
            for step in CHECKPOINT_STEPS
        }:
            raise StageBExecutionError(f"actor identities for {role} are incomplete")
    return result


def compile_split_disjointness(
    *,
    role_datasets: Mapping[str, StageBSplitIdentityView],
    actor_bank_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile the outcome-blind all-ten-role-pairs identity proof.

    Only identity arrays and actor-bank metadata are touched.  The caller is
    responsible for using the dedicated producer capability if ``model_test``
    has not yet been committed.
    """
    if tuple(role_datasets) != ROLE_ORDER:
        raise StageBExecutionError("split proof requires the five roles in order")
    actor_by_role = _actor_identities_by_role(actor_bank_manifest)
    values: dict[str, dict[str, set[Any]]] = {}
    commitments: dict[str, Any] = {}
    array_names = {
        dimension: source
        for dimension, source in SPLIT_IDENTITY_SOURCE_FIELDS.items()
        if not source.startswith("actor_bank.")
    }
    actor_names = (
        "actor_checkpoint_sha256", "actor_state_dict_sha256",
        "policy_fingerprint_sha256", "checkpoint_fingerprint_sha256",
    )
    for role in ROLE_ORDER:
        dataset = role_datasets[role]
        if not isinstance(dataset, StageBSplitIdentityView):
            dataset = make_split_identity_view(dataset)
        if dataset.group_count != sum(
            row.groups for row in source_assignments() if row.role == role
        ):
            raise StageBExecutionError(f"{role} group count has drifted")
        collection = _mapping(
            dataset.manifest.get("collection_protocol"),
            f"{role} collection protocol",
        )
        _require_equal(collection.get("role"), role, f"{role} manifest role")
        _require_equal(
            collection.get("trajectory_fingerprint_array"),
            TRAJECTORY_FINGERPRINT_ARRAY,
            f"{role} trajectory fingerprint array",
        )
        _require_equal(
            collection.get("trajectory_fingerprint_contract"),
            TRAJECTORY_FINGERPRINT_CONTRACT,
            f"{role} trajectory fingerprint contract",
        )
        observed_sources = set(map(int, np.asarray(dataset["source_seed"])))
        _require_equal(
            observed_sources,
            set(ROLE_SOURCE_SEEDS[role]),
            f"{role} source seeds",
        )
        observed_actors = set(
            map(int, np.asarray(dataset["policy_training_seed"]))
        )
        _require_equal(
            observed_actors,
            set(ROLE_ACTOR_SEEDS[role]),
            f"{role} actor seeds",
        )
        for source_seed, actor_seed in zip(
            np.asarray(dataset["source_seed"], dtype=np.int64),
            np.asarray(dataset["policy_training_seed"], dtype=np.int64),
            strict=True,
        ):
            expected = assignment_for(role, int(source_seed))
            if int(actor_seed) != expected.actor_training_seed:
                raise StageBExecutionError(
                    f"{role} actor/source assignment differs from protocol"
                )
        expected_fingerprints = {
            str(item["policy_fingerprint_sha256"])
            for item in actor_by_role[role]
        }
        observed_fingerprints = set(
            np.asarray(dataset["policy_source"]).astype(str).tolist()
        )
        if observed_fingerprints != expected_fingerprints:
            raise StageBExecutionError(
                f"{role} policy fingerprints differ from actor bank"
            )
        trajectory_fingerprints = np.asarray(
            dataset[TRAJECTORY_FINGERPRINT_ARRAY]
        )
        if trajectory_fingerprints.shape != (dataset.group_count,) or (
            trajectory_fingerprints.dtype.kind not in "US"
        ):
            raise StageBExecutionError(
                f"{role} trajectory fingerprints must be text [G]")
        trajectory_text = trajectory_fingerprints.astype(str)
        if any(_HEX64.fullmatch(value) is None for value in trajectory_text) or (
            len(np.unique(trajectory_text)) != dataset.group_count
        ):
            raise StageBExecutionError(
                f"{role} trajectory fingerprints are malformed or reused")
        role_values: dict[str, set[Any]] = {}
        for dimension, array_name in array_names.items():
            array = np.asarray(dataset[array_name])
            flattened = array.reshape(-1)
            role_values[dimension] = set(
                flattened.astype(str).tolist()
                if flattened.dtype.kind in "US" else flattened.tolist()
            )
        for dimension in actor_names:
            role_values[dimension] = {
                str(item[dimension]) for item in actor_by_role[role]
            }
        values[role] = role_values
        identity_payload = {
            dimension: sorted(map(str, role_values[dimension]))
            for dimension in SPLIT_COLLISION_DIMENSIONS
        }
        commitments[role] = {
            "groups": dataset.group_count,
            "source_seeds": sorted(observed_sources),
            "actor_training_seeds": sorted(observed_actors),
            "identity_commitment_sha256": canonical_sha256(identity_payload),
            "outcome_columns_read": False,
        }

    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(ROLE_ORDER):
        for right in ROLE_ORDER[left_index + 1:]:
            collisions = {
                dimension: len(values[left][dimension] & values[right][dimension])
                for dimension in SPLIT_COLLISION_DIMENSIONS
            }
            pair_pass = all(count == 0 for count in collisions.values())
            pairs.append({
                "left": left,
                "right": right,
                "collision_counts": collisions,
                "pass": pair_pass,
            })
            if not pair_pass:
                raise StageBExecutionError(
                    f"Stage-B split identities overlap for {left}/{right}"
                )
    if len(pairs) != 10:
        raise AssertionError("five roles must produce ten unordered pairs")
    report = {
        "schema_version": (
            "qsafe.state_dependent_recovery_v5.stage_b_split_disjointness.v2"
        ),
        "dimensions": list(SPLIT_COLLISION_DIMENSIONS),
        "identity_array_fields": dict(SPLIT_IDENTITY_SOURCE_FIELDS),
        "roles": commitments,
        "pairs_checked": 10,
        "pairs": pairs,
        "outcome_columns_read": False,
        "pass": True,
    }
    return report | {"report_sha256": canonical_sha256(report)}


def compile_partition_rng_disjointness(
    *,
    role_admissions: Mapping[str, Any],
    role_labels: Mapping[str, StageBSplitIdentityView],
) -> dict[str, Any]:
    """Prove all ten role×partition RNG namespaces are disjoint.

    Admission ledgers are accepted only through their four persisted seed
    arrays; label inputs are the already projected identity views.  No fall,
    first-failure, or other outcome field is accessed.
    """
    if tuple(role_admissions) != ROLE_ORDER or tuple(role_labels) != ROLE_ORDER:
        raise StageBExecutionError("partition proof requires the five roles in order")
    namespace_names = {
        "admission": (
            "admission_crn_id", "admission_rollout_seed",
            "admission_perturbation_seed", "admission_candidate_seed",
        ),
        "label": ("crn_id", "rollout_seed", "perturbation_seed", "candidate_seed"),
    }
    domains: dict[str, dict[str, set[int]]] = {}
    for role in ROLE_ORDER:
        admission = role_admissions[role]
        arrays = getattr(admission, "arrays", None)
        if not isinstance(arrays, Mapping):
            raise StageBExecutionError(f"{role} admission is not a ledger")
        for partition, names in namespace_names.items():
            source = arrays if partition == "admission" else role_labels[role].arrays
            values: dict[str, set[int]] = {}
            for name in names:
                if name not in source:
                    raise StageBExecutionError(
                        f"{role}/{partition} omits RNG identity {name}")
                vector = np.asarray(source[name], dtype=np.uint64).reshape(-1)
                if vector.size == 0 or len(np.unique(vector)) != vector.size:
                    raise StageBExecutionError(
                        f"{role}/{partition}/{name} is empty or reused")
                values[name] = set(map(int, vector.tolist()))
            union = set().union(*values.values())
            if sum(map(len, values.values())) != len(union):
                raise StageBExecutionError(
                    f"{role}/{partition} RNG namespaces overlap")
            domains[f"{role}/{partition}"] = values
    pairs: list[dict[str, Any]] = []
    domain_names = list(domains)
    for index, left in enumerate(domain_names):
        left_union = set().union(*domains[left].values())
        for right in domain_names[index + 1:]:
            right_union = set().union(*domains[right].values())
            collisions = len(left_union & right_union)
            record = {"left": left, "right": right,
                      "collision_count": collisions, "pass": collisions == 0}
            pairs.append(record)
            if collisions:
                raise StageBExecutionError(
                    f"partition RNG identities overlap for {left}/{right}")
    report = {
        "schema_version": (
            "qsafe.state_dependent_recovery_v5.stage_b_partition_rng_disjointness.v1"
        ),
        "domains": domain_names,
        "namespaces": {name: list(values) for name, values in namespace_names.items()},
        "pairs_checked": len(pairs),
        "pairs": pairs,
        "outcome_columns_read": False,
        "pass": True,
    }
    return report | {"report_sha256": canonical_sha256(report)}


def preflight_stage_b_execution() -> dict[str, Any]:
    """Perform the outcome-free authorization/RNG/roster preflight."""
    protocol = load_stage_b_execution_protocol()
    validate_stage_a_authorization(protocol)
    rng = validate_role_seed_disjointness()
    if not rng["pass"]:
        raise StageBExecutionError("Stage-B RNG disjointness preflight failed")
    assignments = source_assignments()
    totals = {
        role: sum(row.groups for row in assignments if row.role == role)
        for role in ROLE_ORDER
    }
    expected_totals = {
        "fit": 1536,
        "probability_calibration": 384,
        "uncertainty_calibration": 384,
        "selector_calibration": 384,
        "model_test": 768,
    }
    if totals != expected_totals:
        raise StageBExecutionError("Stage-B group totals have drifted")
    result = {
        "schema_version": (
            "qsafe.state_dependent_recovery_v5.stage_b_preflight.v1"
        ),
        "execution_identity": execution_identity(protocol),
        "source_assignments": len(assignments),
        "actor_training_seeds": list(range(43, 57)),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "group_totals": totals,
        "candidate_label_rollouts": sum(
            row.groups * CANDIDATES * row.label_replicas for row in assignments
        ),
        "rng_disjointness_report_sha256": rng["report_sha256"],
        "model_test_opened": False,
        "pass": True,
    }
    return result | {"preflight_sha256": canonical_sha256(result)}


__all__ = [
    "ADMISSION_REPLICAS",
    "CANDIDATES",
    "CHECKPOINT_STEPS",
    "EXECUTION_PROTOCOL_CONTRACT_SHA256",
    "EXECUTION_PROTOCOL_FILE_SHA256",
    "EXECUTION_PROTOCOL_NAME",
    "EXECUTION_PROTOCOL_PATH",
    "GROUPS_PER_SOURCE",
    "HORIZON_POLICY_STEPS",
    "LABEL_REPLICAS",
    "RECOVERY_LIBRARY_FINGERPRINT_SHA256",
    "ROLE_ACTOR_SEEDS",
    "ROLE_ORDER",
    "ROLE_SOURCE_SEEDS",
    "SPLIT_COLLISION_DIMENSIONS",
    "SPLIT_IDENTITY_SOURCE_FIELDS",
    "STAGE_A_DISPOSITION_COMMIT",
    "STAGE_A_REPORT_SHA256",
    "StageBSplitIdentityView",
    "make_split_identity_view",
    "StageBExecutionError",
    "StageBSourceAssignment",
    "TRAJECTORY_FINGERPRINT_ARRAY",
    "TRAJECTORY_FINGERPRINT_CONTRACT",
    "assignment_for",
    "branch_randomness",
    "canonical_sha256",
    "compile_split_disjointness",
    "compile_partition_rng_disjointness",
    "execution_identity",
    "file_sha256",
    "load_stage_b_execution_protocol",
    "preflight_stage_b_execution",
    "require_clean_stage_b_generator",
    "source_assignments",
    "stage_b_artifact_root",
    "stage_b_seed",
    "validate_role_seed_disjointness",
    "validate_stage_a_authorization",
]
