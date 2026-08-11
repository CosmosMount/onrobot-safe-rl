from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rl.qsafe.data import NormalizationStats
from rl.qsafe.recovery_calibration import (
    RecoverySelectorGridRow,
    RecoverySelectorSearchResult,
    recovery_selector_grid,
)
from rl.qsafe.recovery_program import RECOVERY_PROGRAM_BEHAVIOR_STEPS
from rl.qsafe.recovery_selector import (
    RecoveryConformalOffsets,
    RecoverySelectorBundle,
)
from safety_data.schema import GroupedBranchDataset
import safety_data.state_dependent_recovery_v5_stage_b_fit as stage_b_fit
from scripts.fit_state_dependent_recovery_v5_stage_b import build_parser


def _role_input(tmp_path: Path) -> stage_b_fit.StageBRoleInput:
    dataset = GroupedBranchDataset(
        manifest={"split": "stage-b-fit"},
        arrays={"group_id": np.asarray(["g-0", "g-1"])},
    )
    return stage_b_fit.StageBRoleInput(
        role="fit",
        path=tmp_path / "stage-b" / "fit" / "labels-r32-deployable.npz",
        dataset=dataset,
        file_sha256="a" * 64,
        content_sha256="b" * 64,
        role_report_path=tmp_path / "stage-b" / "fit" / "report.json",
        role_report_file_sha256="c" * 64,
        completion_marker_path=(
            tmp_path / "stage-b" / "fit" / "completed.json"),
        completion_marker_file_sha256="d" * 64,
    )


def test_normalization_report_is_fit_only_self_hashed_and_binds_commitment(
    tmp_path: Path,
):
    fit_input = _role_input(tmp_path)
    normalization = NormalizationStats(
        observation_mean=np.arange(46, dtype=np.float32),
        observation_std=np.ones(46, dtype=np.float32),
        fit_content_sha256=fit_input.content_sha256,
        fit_split="stage-b-fit",
    )
    report = stage_b_fit.build_normalization_report(
        normalization=normalization,
        fit_input=fit_input,
        frozen_identity={
            "generator_commit": "1" * 40,
            "model_test_commitment_file_sha256": "2" * 64,
            "model_test_outcomes_read": False,
        },
    )
    assert report["schema_version"] == (
        stage_b_fit.NORMALIZATION_REPORT_SCHEMA_VERSION)
    assert report["source_role"] == "fit"
    assert report["source_array_sha256"] == "a" * 64
    assert report["privileged_features_absent"] is True
    assert report["contract"]["accumulator_dtype"] == "float64"
    basis = dict(report)
    observed = basis.pop("report_sha256")
    assert observed == stage_b_fit.canonical_sha256(basis)
    assert report["frozen_identity"][
        "model_test_commitment_file_sha256"] == "2" * 64


def test_canonical_report_writer_is_no_clobber(tmp_path: Path):
    output = tmp_path / "report.json"
    value = {"z": 2, "a": [1, True]}
    digest = stage_b_fit._atomic_no_clobber_json(output, value)
    expected = b'{"a":[1,true],"z":2}\n'
    assert output.read_bytes() == expected
    assert digest == hashlib.sha256(expected).hexdigest()
    with pytest.raises(stage_b_fit.StageBFitError, match="overwrite|alias"):
        stage_b_fit._atomic_no_clobber_json(output, value)


def test_production_cli_has_no_paths_or_statistical_overrides():
    parser = build_parser()
    assert parser.parse_args([]).__dict__ == {}
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert option_strings == {"-h", "--help"}
    source = inspect.getsource(stage_b_fit.run_stage_b_development_fit)
    assert "model_test" not in source.casefold() or (
        "model_test_commitment" in source.casefold())


def test_production_selector_bootstrap_rejects_test_override_before_prediction():
    with pytest.raises(stage_b_fit.StageBFitError, match="immutable"):
        stage_b_fit.fit_development_calibration(
            trained=None,
            uncertainty_view=None,
            selector_view=None,
            uncertainty_input=None,
            selector_input=None,
            execution_lock_sha256="a" * 64,
            bootstrap_replicates=7,
            production_contract=True,
        )


class _OutcomeGuardDataset:
    def __init__(self, groups: int = 40) -> None:
        self.group_count = groups
        requested = np.zeros((groups, 9, 12), dtype=np.float32)
        for candidate in range(1, 9):
            requested[:, candidate] = candidate * 0.01
        self._arrays = {
            "candidate_requested": requested,
            "candidate_executed": requested.copy(),
            "candidate_q_target": requested.copy(),
            "candidate_mask": np.ones((groups, 9), dtype=np.bool_),
            "candidate_behavior_steps": np.broadcast_to(
                np.asarray(RECOVERY_PROGRAM_BEHAVIOR_STEPS, dtype=np.int64),
                (groups, 9),
            ).copy(),
            "source_seed": np.arange(9000, 9000 + groups, dtype=np.uint64),
            "state_hash": np.asarray([
                hashlib.sha256(f"state-{index}".encode()).hexdigest()
                for index in range(groups)
            ]),
        }

    def __getitem__(self, name: str) -> np.ndarray:
        if name == "fall":
            raise AssertionError("outcome-free placebo attempted to read fall")
        return self._arrays[name]


def _search_with_first_config() -> RecoverySelectorSearchResult:
    rows = tuple(
        RecoverySelectorGridRow(
            grid_index=index,
            config=config,
            absolute_fall_reduction=(0.04 if index == 0 else 0.0),
            simultaneous_lcb=(0.01 if index == 0 else -0.01),
            intervention_rate=(0.2 if index == 0 else 0.0),
            feasible=index == 0,
        )
        for index, config in enumerate(recovery_selector_grid())
    )
    return RecoverySelectorSearchResult(
        rows=rows,
        selected_grid_index=0,
        common_critical_value=0.03,
        bootstrap_replicates=5,
        bootstrap_seed=7,
        bootstrap_inner_unit="trajectory",
        bootstrap_middle_unit="retain_every_registered_source_stratum",
        execution_lock_sha256="e" * 64,
    )


def test_placebo_adapter_never_reads_outcome_columns():
    dataset = _OutcomeGuardDataset()
    member_risk = np.full((dataset.group_count, 5, 9), 0.1, dtype=np.float64)
    member_risk[:, :, 0] = 0.8
    offsets = RecoveryConformalOffsets(
        nominal_lower=0.0,
        risk_upper=np.zeros(9),
        benefit_lower=np.zeros(9),
        calibration_report_sha256="b" * 64,
    ).validated()
    search = _search_with_first_config()
    selector_bundle = RecoverySelectorBundle.create(
        offsets=offsets,
        selector_config=search.selected_config,
        probability_calibration_report_sha256="a" * 64,
        uncertainty_calibration_report_sha256="b" * 64,
        selector_search_report_sha256="c" * 64,
    )
    placebo = stage_b_fit.fit_outcome_free_placebo(
        member_risk=member_risk,
        dataset=dataset,
        offsets=offsets,
        search=search,
        selector_bundle=selector_bundle,
        execution_lock_sha256="e" * 64,
    )
    serialized = placebo.to_dict()
    assert serialized["outcome_based_reweighting"] == "forbidden"
    assert serialized["reads_qsafe_option_ranking"] is False


def test_artifact_expected_hash_is_canonical_manifest_not_file_bytes(
    tmp_path: Path,
):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    manifest = {"z": 2, "a": {"x": 1}}
    raw = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (artifact / "manifest.json").write_text(raw, encoding="utf-8")
    expected = hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert stage_b_fit._artifact_manifest_canonical_sha256(artifact) == expected
    assert expected != hashlib.sha256(raw.encode("utf-8")).hexdigest()


def test_split_report_binds_committed_model_test_hash_without_opening_it(
    tmp_path: Path,
):
    root = tmp_path / "stage-b"
    root.mkdir()
    generator = "1" * 40
    actor_file = "2" * 64
    actor_contract = "3" * 64
    development = {}
    aggregates = []
    proof_roles = {}
    for role in stage_b_fit.ROLE_ORDER:
        replicas = stage_b_fit.LABEL_REPLICAS[role]
        relative = (
            f"stage-b/{role.replace('_', '-')}/"
            f"labels-r{replicas}-deployable.npz"
        )
        file_hash = hashlib.sha256(f"file:{role}".encode()).hexdigest()
        content_hash = hashlib.sha256(f"content:{role}".encode()).hexdigest()
        report_hash = hashlib.sha256(f"report:{role}".encode()).hexdigest()
        groups = (
            len(stage_b_fit.ROLE_SOURCE_SEEDS[role])
            * stage_b_fit.GROUPS_PER_SOURCE[role]
        )
        aggregates.append({
            "role": role,
            "path": relative,
            "file_sha256": file_hash,
            "content_sha256": content_hash,
            "groups": groups,
            "role_report_file_sha256": (
                None if role == "model_test" else report_hash),
        })
        if role != "model_test":
            development[role] = SimpleNamespace(
                path=tmp_path / relative,
                file_sha256=file_hash,
                content_sha256=content_hash,
                role_report_file_sha256=report_hash,
            )
        proof_roles[role] = {
            "groups": groups,
            "source_seeds": sorted(stage_b_fit.ROLE_SOURCE_SEEDS[role]),
            "actor_training_seeds": sorted(
                stage_b_fit.ROLE_ACTOR_SEEDS[role]),
            "identity_commitment_sha256": hashlib.sha256(
                f"identity:{role}".encode()).hexdigest(),
            "outcome_columns_read": False,
        }
    pairs = []
    for index, left in enumerate(stage_b_fit.ROLE_ORDER):
        for right in stage_b_fit.ROLE_ORDER[index + 1:]:
            pairs.append({
                "left": left,
                "right": right,
                "collision_counts": {
                    name: 0 for name in stage_b_fit.SPLIT_COLLISION_DIMENSIONS
                },
                "pass": True,
            })
    proof_basis = {
        "schema_version": (
            "qsafe.state_dependent_recovery_v5."
            "stage_b_split_disjointness.v1"),
        "dimensions": list(stage_b_fit.SPLIT_COLLISION_DIMENSIONS),
        "roles": proof_roles,
        "pairs_checked": 10,
        "pairs": pairs,
        "outcome_columns_read": False,
        "pass": True,
    }
    proof = dict(proof_basis)
    proof["report_sha256"] = stage_b_fit.canonical_sha256(proof_basis)
    frozen = {
        "parent_protocol_name": stage_b_fit.STAGE_B_PROTOCOL_NAME,
        "parent_protocol_contract_sha256": (
            stage_b_fit.PARENT_PROTOCOL_CONTRACT_SHA256),
        "parent_protocol_file_sha256": stage_b_fit.PARENT_PROTOCOL_FILE_SHA256,
        "execution_protocol_name": stage_b_fit.STAGE_B_EXECUTION_PROTOCOL_NAME,
        "execution_protocol_contract_sha256": (
            stage_b_fit.EXECUTION_PROTOCOL_CONTRACT_SHA256),
        "execution_protocol_file_sha256": (
            stage_b_fit.EXECUTION_PROTOCOL_FILE_SHA256),
        "stage_a_report_sha256": stage_b_fit.STAGE_A_REPORT_SHA256,
        "stage_a_disposition_commit": stage_b_fit.STAGE_A_DISPOSITION_COMMIT,
        "generator_commit": generator,
    }
    report_basis = {
        "schema_version": (
            "qsafe.state_dependent_recovery_v5."
            "stage_b_split_disjointness_bound.v1"),
        **frozen,
        "actor_bank_manifest_file_sha256": actor_file,
        "actor_bank_contract_sha256": actor_contract,
        "role_order": list(stage_b_fit.ROLE_ORDER),
        "role_aggregate_labels": aggregates,
        "identity_proof": proof,
        "model_test_source": (
            "in_memory_merged_dataset_and_staged_label_bytes_before_role_report"),
        "outcome_columns_read": False,
        "pass": True,
    }
    report = dict(report_basis)
    report["report_sha256"] = stage_b_fit.canonical_sha256(report_basis)
    report_path = root / "stage-b-split-disjointness-report.json"
    stage_b_fit._atomic_no_clobber_json(report_path, report)
    model_test_label = root / "model-test" / "labels-r64-deployable.npz"
    assert not model_test_label.exists()
    commitment = {
        "evidence_artifacts": [{
            "kind": "label",
            "path": "stage-b/model-test/labels-r64-deployable.npz",
            "sha256": aggregates[-1]["file_sha256"],
        }],
    }
    loaded, _ = stage_b_fit._load_split_report(
        report_path,
        generator_commit=generator,
        actor_bank_manifest={"actor_bank_contract_sha256": actor_contract},
        actor_bank_manifest_file_sha256=actor_file,
        model_test_commitment=commitment,
        development_roles=development,
        stage_b_root=root,
    )
    assert loaded["pass"] is True
    assert not model_test_label.exists()

    bad_commitment = {
        "evidence_artifacts": [{
            "kind": "label",
            "path": "stage-b/model-test/labels-r64-deployable.npz",
            "sha256": "f" * 64,
        }],
    }
    with pytest.raises(stage_b_fit.StageBFitError, match="differs"):
        stage_b_fit._load_split_report(
            report_path,
            generator_commit=generator,
            actor_bank_manifest={
                "actor_bank_contract_sha256": actor_contract},
            actor_bank_manifest_file_sha256=actor_file,
            model_test_commitment=bad_commitment,
            development_roles=development,
            stage_b_root=root,
        )
