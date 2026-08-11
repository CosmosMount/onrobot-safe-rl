from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import train.state_dependent_recovery_v5_stage_b_actor_bank as actor_bank


def _stage_b_contract() -> dict[str, object]:
    return {
        "stage_B": {
            "actor_training_seeds": {
                role: list(seeds)
                for role, seeds in actor_bank.ROLE_SEEDS.items()
            },
            "actor_source_assignment": {
                "actor_checkpoint_steps": list(
                    actor_bank.EXACT_CHECKPOINT_STEPS),
            },
        },
    }


def _stage_b_execution_contract() -> dict[str, object]:
    return {
        "actor_bank": {
            "training_seeds_exact": list(range(43, 57)),
            "checkpoint_steps_exact": [25_000, 50_000, 100_000],
            "checkpoint_count_exact": 42,
            "snapshot_kind": "policy_only",
            "snapshot_timing": (
                "after_transition_and_scheduled_update_before_next_transition"),
            "nearby_or_episode_boundary_checkpoint_substitution": "forbidden",
            "retain_every_seed_and_checkpoint_without_return_or_fall_filter": True,
            "checkpoint_path_template": (
                "stage-b/actor-bank/seed-{training_seed}/"
                "step-{checkpoint_step}/agent"),
            "attempt_marker_path": (
                "stage-b/actor-bank-attempt-started.json"),
        },
    }


def _write_inputs(root: Path) -> dict[str, Path]:
    paths = {
        "supplement": root / "stage-b-execution.yaml",
        "protocol": root / "v5.yaml",
        "stage_a_report": root / "stage-a-report.json",
        "training_config": root / "sac.yaml",
    }
    paths["supplement"].write_text(
        yaml.safe_dump(_stage_b_execution_contract(), sort_keys=False),
        encoding="utf-8",
    )
    paths["protocol"].write_text(
        yaml.safe_dump(_stage_b_contract(), sort_keys=False),
        encoding="utf-8",
    )
    paths["stage_a_report"].write_text(
        json.dumps({"stage_A_pass": True}) + "\n", encoding="utf-8")
    paths["training_config"].write_text(
        "move_speed: 0.30\ntrain:\n  agent: droq\n",
        encoding="utf-8",
    )
    return paths


def _prepare(root: Path) -> tuple[dict[str, Path], Path, Path, list[Path]]:
    paths = _write_inputs(root)
    actor_root = root / "stage-b" / "actor-bank"
    contracts_root = root / "contracts"
    contracts = actor_bank.prepare_actor_run_contracts(
        supplement_path=paths["supplement"],
        protocol_path=paths["protocol"],
        stage_a_report_path=paths["stage_a_report"],
        training_config_path=paths["training_config"],
        actor_root=actor_root,
        contracts_root=contracts_root,
        generator_commit="a" * 40,
        require_clean_git=False,
        enforce_canonical_bindings=False,
    )
    return paths, actor_root, contracts_root, contracts


class _FakeDroQ:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.step = 0

    def export_inference_snapshot(self, *, snapshot_version: int):
        return {
            "agent_type": "droq",
            "snapshot_version": snapshot_version,
            "actor_steps": self.step - 999,
            "critic_steps": self.step - 999,
            "temperature_steps": self.step - 999,
            "auxiliary_steps": 0,
            "actor_state_dict": {
                "weight": torch.tensor(
                    [float(self.seed), float(self.step)], dtype=torch.float32),
            },
        }

    def get_update_counters(self):
        return {
            "policy_steps": self.step - 999,
            "actor_steps": self.step - 999,
            "critic_steps": self.step - 999,
            "target_steps": self.step - 999,
            "temperature_steps": self.step - 999,
            "auxiliary_steps": 0,
        }


def _export_seed(contract_path: Path, seed: int, run_dir: Path) -> None:
    cfg = SimpleNamespace(
        seed=seed,
        save_dir=str(run_dir),
        max_steps=100_000,
        resume_checkpoint=False,
    )
    exporter = actor_bank.ExactPolicyCheckpointExporter(
        contract_path,
        cfg=cfg,
        verify_live_bindings=False,
        require_clean_git=False,
    )
    agent = _FakeDroQ(seed)
    for step in actor_bank.EXACT_CHECKPOINT_STEPS:
        agent.step = step
        assert exporter.maybe_export(agent, step)
    exporter.require_complete()


def _fake_policy_inspector(
    checkpoint: Path,
    config: Path,
    observation_dim: int,
    action_dim: int,
    training_step: int,
):
    del config, observation_dim, action_dim
    actor_path = checkpoint / "actor.pt"
    payload = torch.load(actor_path, map_location="cpu", weights_only=True)
    state_sha = actor_bank._state_dict_sha256(payload["network_state_dict"])
    path_digest = hashlib.sha256(
        f"{checkpoint}:{training_step}".encode("utf-8")).hexdigest()
    return {
        "actor_sha256": actor_bank._sha256_file(actor_path, "test actor"),
        "actor_state_dict_sha256": state_sha,
        "policy_fingerprint_sha256": hashlib.sha256(
            f"policy:{path_digest}".encode("ascii")).hexdigest(),
        "checkpoint_fingerprint_sha256": path_digest,
    }


def test_frozen_roster_requires_all_five_roles_and_exact_steps():
    roster = actor_bank.validate_actor_roster(
        actor_bank.ROLE_SEEDS, actor_bank.EXACT_CHECKPOINT_STEPS)
    assert tuple(seed for seeds in roster.values() for seed in seeds) == tuple(
        range(43, 57))

    missing = dict(actor_bank.ROLE_SEEDS)
    missing["fit"] = (43, 44, 45)
    with pytest.raises(actor_bank.StageBActorBankError, match="must equal"):
        actor_bank.validate_actor_roster(
            missing, actor_bank.EXACT_CHECKPOINT_STEPS)
    with pytest.raises(actor_bank.StageBActorBankError, match="25000"):
        actor_bank.validate_actor_roster(
            actor_bank.ROLE_SEEDS, (25_001, 50_000, 100_000))


def test_actor_bank_attempt_is_first_and_no_clobber(tmp_path: Path):
    paths, actor_root, contracts_root, _ = _prepare(tmp_path)
    marker = tmp_path / "stage-b" / "actor-bank-attempt-started.json"
    value = json.loads(marker.read_text(encoding="utf-8"))
    assert value["created_before_first_training_transition"] is True
    assert value["expected_actor_identity_count"] == 42
    with pytest.raises(actor_bank.StageBActorBankError, match="clobber"):
        actor_bank.prepare_actor_run_contracts(
            supplement_path=paths["supplement"],
            protocol_path=paths["protocol"],
            stage_a_report_path=paths["stage_a_report"],
            training_config_path=paths["training_config"],
            actor_root=actor_root,
            contracts_root=contracts_root,
            generator_commit="a" * 40,
            require_clean_git=False,
            enforce_canonical_bindings=False,
        )


def test_exact_export_is_policy_only_no_clobber_and_after_update_semantic(
    tmp_path: Path,
):
    _, actor_root, _, contracts = _prepare(tmp_path)
    run_dir = actor_root / "seed-43"
    cfg = SimpleNamespace(
        seed=43,
        save_dir=str(run_dir),
        max_steps=100_000,
        resume_checkpoint=False,
    )
    exporter = actor_bank.ExactPolicyCheckpointExporter(
        contracts[0],
        cfg=cfg,
        verify_live_bindings=False,
        require_clean_git=False,
    )
    agent = _FakeDroQ(43)
    agent.step = 24_999
    assert not exporter.maybe_export(agent, 24_999)
    agent.step = 25_000
    assert exporter.maybe_export(agent, 25_000)

    checkpoint = run_dir / "step-25000"
    assert sorted(path.name for path in (checkpoint / "agent").iterdir()) == [
        "actor.pt"]
    payload = torch.load(
        checkpoint / "agent" / "actor.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert set(payload) == {
        "network_state_dict", "optimizer_state_dict",
        "scheduler_state_dict", "update_step",
    }
    assert payload["optimizer_state_dict"] is None
    assert payload["scheduler_state_dict"] is None
    manifest = json.loads(
        (checkpoint / "snapshot-manifest.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint_semantics"] == (
        "after_transition_and_scheduled_update_before_next_transition")
    assert manifest["update_counters_after_scheduled_update"][
        "critic_steps"] == 24_001
    with pytest.raises(actor_bank.StageBActorBankError, match="duplicate"):
        exporter.maybe_export(agent, 25_000)
    with pytest.raises(actor_bank.StageBActorBankError, match="already contains"):
        actor_bank.ExactPolicyCheckpointExporter(
            contracts[0],
            cfg=cfg,
            verify_live_bindings=False,
            require_clean_git=False,
        )


def test_exporter_rejects_skipped_exact_checkpoint(tmp_path: Path):
    _, actor_root, _, contracts = _prepare(tmp_path)
    cfg = SimpleNamespace(
        seed=43,
        save_dir=str(actor_root / "seed-43"),
        max_steps=100_000,
        resume_checkpoint=False,
    )
    exporter = actor_bank.ExactPolicyCheckpointExporter(
        contracts[0], cfg=cfg,
        verify_live_bindings=False, require_clean_git=False)
    with pytest.raises(actor_bank.StageBActorBankError, match="missed exact"):
        exporter.maybe_export(_FakeDroQ(43), 25_001)


def test_exact_export_fsyncs_every_directory_boundary_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _, actor_root, _, contracts = _prepare(tmp_path)
    run_dir = actor_root / "seed-43"
    observed: list[Path] = []
    monkeypatch.setattr(
        actor_bank,
        "_fsync_directory",
        lambda path: observed.append(Path(path)),
    )
    exporter = actor_bank.ExactPolicyCheckpointExporter(
        contracts[0],
        cfg=SimpleNamespace(
            seed=43,
            save_dir=str(run_dir),
            max_steps=100_000,
            resume_checkpoint=False,
        ),
        verify_live_bindings=False,
        require_clean_git=False,
    )
    agent = _FakeDroQ(43)
    agent.step = 25_000
    assert exporter.maybe_export(agent, 25_000)
    step_dir = run_dir / "step-25000"
    assert observed == [
        actor_root,
        run_dir,
        run_dir,
        step_dir,
        step_dir / "agent",
        step_dir,
        run_dir,
    ]


def test_exact_export_directory_fsync_failure_leaves_durable_fail_closed_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _, actor_root, _, contracts = _prepare(tmp_path)
    run_dir = actor_root / "seed-43"
    agent_dir = run_dir / "step-25000" / "agent"
    original = actor_bank._fsync_directory

    def fail_after_actor_write(path: Path) -> None:
        original(path)
        if Path(path) == agent_dir:
            raise OSError("injected directory fsync failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(actor_bank, "_fsync_directory", fail_after_actor_write)
        exporter = actor_bank.ExactPolicyCheckpointExporter(
            contracts[0],
            cfg=SimpleNamespace(
                seed=43,
                save_dir=str(run_dir),
                max_steps=100_000,
                resume_checkpoint=False,
            ),
            verify_live_bindings=False,
            require_clean_git=False,
        )
        agent = _FakeDroQ(43)
        agent.step = 25_000
        with pytest.raises(OSError, match="injected"):
            exporter.maybe_export(agent, 25_000)
    assert (agent_dir / "actor.pt").is_file()
    assert not (run_dir / "step-25000" / "snapshot-manifest.json").exists()
    with pytest.raises(actor_bank.StageBActorBankError, match="already contains"):
        actor_bank.ExactPolicyCheckpointExporter(
            contracts[0],
            cfg=SimpleNamespace(
                seed=43,
                save_dir=str(run_dir),
                max_steps=100_000,
                resume_checkpoint=False,
            ),
            verify_live_bindings=False,
            require_clean_git=False,
        )


def test_compiler_accepts_only_complete_42_identity_bank(tmp_path: Path):
    paths, actor_root, contracts_root, contracts = _prepare(tmp_path)
    for seed, contract_path in zip(
            actor_bank.ALL_ACTOR_SEEDS, contracts, strict=True):
        _export_seed(contract_path, seed, actor_root / f"seed-{seed}")
    manifest, file_sha = actor_bank.compile_actor_bank_manifest(
        supplement_path=paths["supplement"],
        protocol_path=paths["protocol"],
        stage_a_report_path=paths["stage_a_report"],
        training_config_path=paths["training_config"],
        actor_root=actor_root,
        contracts_root=contracts_root,
        output_path=tmp_path / "compiled-bank.json",
        observation_dim=2,
        action_dim=1,
        policy_inspector=_fake_policy_inspector,
        verify_live_bindings=False,
        require_clean_git=False,
        enforce_canonical_bindings=False,
    )
    assert manifest["identity_count"] == 42
    assert manifest["return_or_fall_filtering"] == "forbidden"
    assert manifest["checkpoint_steps"] == [25_000, 50_000, 100_000]
    assert len(file_sha) == 64
    assert (tmp_path / "stage-b" / "actor-bank-attempt-started.json").is_file()
    loaded = actor_bank.load_actor_bank_manifest(
        tmp_path / "compiled-bank.json",
        expected_bindings={
            "manifest_file_sha256": file_sha,
            "generator_commit": "a" * 40,
        },
        enforce_canonical_path=False,
    )
    identity = actor_bank.actor_identity_for(
        loaded, role="model_test", actor_seed=56, checkpoint_step=100_000)
    assert identity["checkpoint_path"].endswith(
        "stage-b/actor-bank/seed-56/step-100000/agent")
    duplicated = json.loads(json.dumps(loaded))
    duplicated["identities"].append(dict(identity))
    with pytest.raises(actor_bank.StageBActorBankError, match="duplicated"):
        actor_bank.actor_identity_for(
            duplicated, role="model_test", actor_seed=56,
            checkpoint_step=100_000)

    actor_path = Path(identity["checkpoint_path"]) / "actor.pt"
    actor_path.write_bytes(actor_path.read_bytes() + b"tampered")
    with pytest.raises(actor_bank.StageBActorBankError, match="bytes changed"):
        actor_bank.load_actor_bank_manifest(
            tmp_path / "compiled-bank.json",
            enforce_canonical_path=False,
        )


def test_compiler_rejects_nearby_checkpoint_without_substitution(
    tmp_path: Path,
):
    paths, actor_root, contracts_root, _ = _prepare(tmp_path)
    seed_root = actor_root / "seed-43"
    seed_root.mkdir(parents=True)
    (seed_root / "step-24999").mkdir()
    with pytest.raises(actor_bank.StageBActorBankError, match="nearby"):
        actor_bank.compile_actor_bank_manifest(
            supplement_path=paths["supplement"],
            protocol_path=paths["protocol"],
            stage_a_report_path=paths["stage_a_report"],
            training_config_path=paths["training_config"],
            actor_root=actor_root,
            contracts_root=contracts_root,
            output_path=tmp_path / "compiled-bank.json",
            observation_dim=2,
            action_dim=1,
            policy_inspector=_fake_policy_inspector,
            verify_live_bindings=False,
            require_clean_git=False,
            enforce_canonical_bindings=False,
        )


def test_canonical_actor_bank_dimensions_have_no_override_surface(
    tmp_path: Path,
):
    placeholder = tmp_path / "placeholder"
    with pytest.raises(
        actor_bank.StageBActorBankError,
        match="observation_dim=46 and action_dim=12",
    ):
        actor_bank.compile_actor_bank_manifest(
            supplement_path=placeholder,
            protocol_path=placeholder,
            stage_a_report_path=placeholder,
            training_config_path=placeholder,
            actor_root=placeholder,
            contracts_root=placeholder,
            output_path=placeholder,
            observation_dim=45,
            action_dim=12,
            enforce_canonical_bindings=True,
        )


def test_run_contract_hash_tampering_fails_closed(tmp_path: Path):
    _, _, _, contracts = _prepare(tmp_path)
    contract = json.loads(contracts[0].read_text(encoding="utf-8"))
    contract["actor_training_seed"] = 44
    contracts[0].write_text(json.dumps(contract) + "\n", encoding="utf-8")
    with pytest.raises(actor_bank.StageBActorBankError, match="invalid|hash"):
        actor_bank.load_actor_run_contract(
            contracts[0], verify_live_bindings=False,
            require_clean_git=False)
