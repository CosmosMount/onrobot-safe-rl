from __future__ import annotations

import copy
import inspect

import pytest

import scripts.evaluate_state_dependent_recovery_v5_stage_b as evaluator


def _workflow_shaped_split_report() -> tuple[dict[str, object], dict[str, str]]:
    identity = {
        "generator": "d" * 40,
        "actor_file": "a" * 64,
        "actor_contract": "b" * 64,
        "model_test_label": "c" * 64,
    }
    aggregates: list[dict[str, object]] = []
    admission_aggregates: list[dict[str, object]] = []
    proof_roles: dict[str, dict[str, object]] = {}
    for index, role in enumerate(evaluator.ROLE_ORDER):
        replicas = evaluator.LABEL_REPLICAS[role]
        groups = (
            len(evaluator.ROLE_SOURCE_SEEDS[role])
            * evaluator.GROUPS_PER_SOURCE[role]
        )
        aggregates.append({
            "role": role,
            "path": (
                f"stage-b/{role.replace('_', '-')}/"
                f"labels-r{replicas}-deployable.npz"
            ),
            "file_sha256": (
                identity["model_test_label"]
                if role == "model_test"
                else f"{index + 1:064x}"
            ),
            "content_sha256": f"{index + 11:064x}",
            "groups": groups,
            "role_report_file_sha256": (
                None if role == "model_test" else f"{index + 21:064x}"
            ),
        })
        admission_aggregates.append({
            "role": role,
            "admission_path": (
                f"stage-b/{role.replace('_', '-')}/admission-r32.npz"),
            "admission_file_sha256": f"{index + 41:064x}",
            "admission_content_sha256": f"{index + 51:064x}",
            "admission_proposals": 32,
        })
        proof_roles[role] = {
            "groups": groups,
            "source_seeds": sorted(evaluator.ROLE_SOURCE_SEEDS[role]),
            "actor_training_seeds": sorted(evaluator.ROLE_ACTOR_SEEDS[role]),
            "identity_commitment_sha256": f"{index + 31:064x}",
            "outcome_columns_read": False,
        }
    pairs = [
        {
            "left": left,
            "right": right,
            "collision_counts": {
                dimension: 0
                for dimension in evaluator.SPLIT_COLLISION_DIMENSIONS
            },
            "pass": True,
        }
        for left_index, left in enumerate(evaluator.ROLE_ORDER)
        for right in evaluator.ROLE_ORDER[left_index + 1:]
    ]
    proof: dict[str, object] = {
        "schema_version": (
            "qsafe.state_dependent_recovery_v5."
            "stage_b_split_disjointness.v2"
        ),
        "dimensions": list(evaluator.SPLIT_COLLISION_DIMENSIONS),
        "identity_array_fields": dict(evaluator.SPLIT_IDENTITY_SOURCE_FIELDS),
        "roles": proof_roles,
        "pairs_checked": 10,
        "pairs": pairs,
        "outcome_columns_read": False,
        "pass": True,
    }
    proof["report_sha256"] = evaluator._canonical_object_sha256(proof)
    domains = [
        f"{role}/{partition}"
        for role in evaluator.ROLE_ORDER
        for partition in ("admission", "label")
    ]
    partition_pairs = [
        {"left": left, "right": right, "collision_count": 0, "pass": True}
        for index, left in enumerate(domains)
        for right in domains[index + 1:]
    ]
    partition = {
        "schema_version": (
            "qsafe.state_dependent_recovery_v5."
            "stage_b_partition_rng_disjointness.v1"),
        "domains": domains,
        "namespaces": {
            "admission": [
                "admission_crn_id", "admission_rollout_seed",
                "admission_perturbation_seed", "admission_candidate_seed",
            ],
            "label": ["crn_id", "rollout_seed", "perturbation_seed", "candidate_seed"],
        },
        "pairs_checked": 45,
        "pairs": partition_pairs,
        "outcome_columns_read": False,
        "pass": True,
    }
    partition["report_sha256"] = evaluator._canonical_object_sha256(partition)
    report: dict[str, object] = {
        "schema_version": (
            "qsafe.state_dependent_recovery_v5."
            "stage_b_split_disjointness_bound.v3"
        ),
        "parent_protocol_name": evaluator.STAGE_B_PROTOCOL_NAME,
        "parent_protocol_contract_sha256": (
            evaluator.PARENT_PROTOCOL_CONTRACT_SHA256
        ),
        "parent_protocol_file_sha256": evaluator.PARENT_PROTOCOL_FILE_SHA256,
        "execution_protocol_name": evaluator.STAGE_B_EXECUTION_PROTOCOL_NAME,
        "execution_protocol_contract_sha256": (
            evaluator.EXECUTION_PROTOCOL_CONTRACT_SHA256
        ),
        "execution_protocol_file_sha256": (
            evaluator.EXECUTION_PROTOCOL_FILE_SHA256
        ),
        "stage_a_report_sha256": evaluator.STAGE_A_REPORT_SHA256,
        "stage_a_disposition_commit": evaluator.STAGE_A_DISPOSITION_COMMIT,
        "generator_commit": identity["generator"],
        "actor_bank_manifest_file_sha256": identity["actor_file"],
        "actor_bank_contract_sha256": identity["actor_contract"],
        "role_order": list(evaluator.ROLE_ORDER),
        "role_aggregate_labels": aggregates,
        "role_aggregate_admissions": admission_aggregates,
        "identity_proof": proof,
        "partition_rng_proof": partition,
        "model_test_source": (
            "in_memory_merged_dataset_and_staged_label_bytes_before_role_report"
        ),
        "identity_proof_outcome_columns_read": False,
        "blind_mechanical_merge_outcome_statistics_computed": False,
        "pass": True,
    }
    report["report_sha256"] = evaluator._canonical_object_sha256(report)
    return report, identity


def _validate_split(
    report: dict[str, object], identity: dict[str, str]
) -> str:
    return evaluator._validate_split_disjointness_report(
        report,
        generator_commit=identity["generator"],
        actor_bank_manifest_file_sha256=identity["actor_file"],
        actor_bank_contract_sha256=identity["actor_contract"],
        committed_model_test_label_sha256=identity["model_test_label"],
    )


def _rehash_report(report: dict[str, object]) -> None:
    report.pop("report_sha256", None)
    report["report_sha256"] = evaluator._canonical_object_sha256(report)


def test_workflow_shaped_bound_split_report_is_accepted() -> None:
    report, identity = _workflow_shaped_split_report()
    assert _validate_split(report, identity) == report["report_sha256"]


def test_split_report_rejects_old_top_level_schema_and_wrong_model_test_hash() -> None:
    report, identity = _workflow_shaped_split_report()
    old_schema = copy.deepcopy(report)
    old_schema["schema_version"] = (
        "qsafe.state_dependent_recovery_v5.stage_b_split_disjointness_bound.v1"
    )
    _rehash_report(old_schema)
    with pytest.raises(evaluator.StageBModelTestError, match="schema version"):
        _validate_split(old_schema, identity)

    wrong_label = copy.deepcopy(report)
    records = wrong_label["role_aggregate_labels"]
    assert isinstance(records, list)
    assert isinstance(records[-1], dict)
    records[-1]["file_sha256"] = "f" * 64
    _rehash_report(wrong_label)
    with pytest.raises(evaluator.StageBModelTestError, match="Model-Test"):
        _validate_split(wrong_label, identity)


def test_split_proof_rejects_trajectory_id_alias_or_fingerprint_omission() -> None:
    report, identity = _workflow_shaped_split_report()
    aliased = copy.deepcopy(report)
    proof = aliased["identity_proof"]
    assert isinstance(proof, dict)
    fields = proof["identity_array_fields"]
    assert isinstance(fields, dict)
    fields["trajectory_fingerprint_sha256"] = "trajectory_id"
    _rehash_report(proof)
    _rehash_report(aliased)
    with pytest.raises(evaluator.StageBModelTestError, match="proof gate"):
        _validate_split(aliased, identity)

    omitted = copy.deepcopy(report)
    proof = omitted["identity_proof"]
    assert isinstance(proof, dict)
    dimensions = proof["dimensions"]
    fields = proof["identity_array_fields"]
    assert isinstance(dimensions, list)
    assert isinstance(fields, dict)
    dimensions.remove("trajectory_fingerprint_sha256")
    fields.pop("trajectory_fingerprint_sha256")
    _rehash_report(proof)
    _rehash_report(omitted)
    with pytest.raises(evaluator.StageBModelTestError, match="proof gate"):
        _validate_split(omitted, identity)


def test_production_cli_and_api_expose_no_statistical_override() -> None:
    parsed = evaluator._parser().parse_args([
        "--model-test-commitment-sha256",
        "a" * 64,
    ])
    assert vars(parsed) == {"model_test_commitment_sha256": "a" * 64}
    with pytest.raises(SystemExit):
        evaluator._parser().parse_args([
            "--model-test-commitment-sha256",
            "a" * 64,
            "--bootstrap-replicates",
            "10",
        ])
    assert tuple(inspect.signature(
        evaluator.evaluate_canonical_stage_b_model_test
    ).parameters) == ("expected_commitment_sha256",)


def test_generator_commit_alignment_is_exact() -> None:
    commit = "a" * 40
    evaluator._require_generator_commit_alignment(
        evaluator_commit=commit,
        actor_bank_generator_commit=commit,
        commitment_generator_commit=commit,
    )
    with pytest.raises(evaluator.StageBModelTestError, match="commits differ"):
        evaluator._require_generator_commit_alignment(
            evaluator_commit=commit,
            actor_bank_generator_commit="b" * 40,
            commitment_generator_commit=commit,
        )
