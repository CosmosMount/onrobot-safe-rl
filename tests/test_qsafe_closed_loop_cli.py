from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from safety_data.recovery_behaviors import RecoveryBehaviorConfig
import scripts.collect_closed_loop_recovery_triage as collect_cli
from scripts.collect_closed_loop_recovery_triage import (
    _cohort_lock,
    _load_protocol,
    _outputs,
    _policy_for_seed,
    _policy_set_manifest,
    _start_attempt,
    _verify_runtime_contract,
)
from scripts.merge_closed_loop_recovery_admission import (
    _admission_output_paths,
)
from scripts.merge_grouped_qsafe_shards import (
    _require_v3_exact_discovery_gate,
    _v3_exact_discovery_gate,
    _v3_preopen_discovery_paths,
    main as merge_main,
)
import scripts.audit_closed_loop_recovery_triage as audit_cli


class _ExactGateDataset:
    def __init__(self, protocol):
        data = protocol["triage_gates"]["data"]
        candidate_protocol = copy.deepcopy(
            protocol["collection"]["candidates"])
        groups = int(data["independent_groups_exact"])
        candidates = int(data["candidates_per_group_exact"])
        discovery_replicas = int(data["discovery_replicas_exact"])
        audit_replicas = int(data["audit_replicas_exact"])
        names = np.asarray(candidate_protocol["ordered_names"], dtype=str)
        behavior_steps = np.asarray(
            candidate_protocol["behavior_override_steps"], dtype=np.int16)

        def matrix(start, replicas):
            return np.arange(
                start, start + groups * replicas, dtype=np.uint64,
            ).reshape(groups, replicas)

        self.manifest = {
            "collection_protocol": {"role": "discovery"},
            "candidate_protocol": candidate_protocol,
        }
        self.arrays = {
            "source_seed": np.repeat(
                np.asarray(data["required_source_seeds"], dtype=np.int64),
                int(data["groups_per_required_source_seed_exact"]),
            ),
            "trajectory_id": np.asarray([
                f"trajectory-{index}" for index in range(groups)]),
            "candidate_mask": np.ones((groups, candidates), dtype=bool),
            "candidate_kind": np.broadcast_to(
                names, (groups, candidates)).copy(),
            "candidate_behavior_steps": np.broadcast_to(
                behavior_steps, (groups, candidates)).copy(),
            "crn_id": matrix(0, discovery_replicas),
            "rollout_seed": matrix(100, discovery_replicas),
            "perturbation_seed": matrix(200, discovery_replicas),
            "candidate_seed": np.arange(
                300, 300 + groups, dtype=np.uint64),
            "preassigned_audit_crn_id": matrix(400, audit_replicas),
            "preassigned_audit_rollout_seed": matrix(
                500, audit_replicas),
            "preassigned_audit_perturbation_seed": matrix(
                600, audit_replicas),
            "preassigned_audit_candidate_seed": np.arange(
                700, 700 + groups, dtype=np.uint64),
        }
        self.group_count = groups
        self.candidate_count = candidates
        self.replica_count = discovery_replicas
        self.horizon_steps = int(data["horizon_policy_steps_exact"])

    def __getitem__(self, name):
        return self.arrays[name]


class ClosedLoopRecoveryCliContractTest(unittest.TestCase):
    def _exact_gate_fixture(self):
        protocol = copy.deepcopy(_load_protocol())
        data = protocol["triage_gates"]["data"]
        data.update({
            "independent_groups_exact": 4,
            "unique_source_trajectories_exact": 4,
            "required_source_seeds": [7801, 7802],
            "groups_per_required_source_seed_exact": 2,
            "discovery_replicas_exact": 2,
            "audit_replicas_exact": 3,
        })
        return protocol, _ExactGateDataset(protocol)

    def test_canonical_protocol_matches_runtime_library_and_seed_map(self):
        protocol = _load_protocol()
        self.assertEqual(
            protocol["collection"]["candidates"],
            RecoveryBehaviorConfig().manifest_protocol(),
        )
        expected = {
            7801: 25438,
            7802: 25438,
            7811: 50030,
            7812: 50030,
            7821: 100359,
            7822: 100359,
        }
        self.assertEqual({
            seed: int(_policy_for_seed(protocol, seed)["training_step"])
            for seed in expected
        }, expected)
        manifest = _policy_set_manifest(protocol)
        self.assertEqual(manifest["policy_training_seed"], 42)
        self.assertEqual(len(manifest["policies"]), 3)

    def test_unregistered_source_seed_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            _policy_for_seed(_load_protocol(), 7999)

    def test_attempt_marker_is_atomic_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source-7801.attempt-started.json"
            marker = _start_attempt(
                path,
                source_seed=7801,
                policy_training_step=25438,
                generator_commit="a" * 40,
                protocol_sha256="b" * 64,
                protocol_contract_sha256="d" * 64,
                cohort_lock_sha256="c" * 64,
            )
            self.assertTrue(path.is_file())
            self.assertEqual(
                marker["state"], "started_outcome_may_have_been_generated")
            with self.assertRaises(FileExistsError):
                _start_attempt(
                    path,
                    source_seed=7801,
                    policy_training_step=25438,
                    generator_commit="a" * 40,
                    protocol_sha256="b" * 64,
                    protocol_contract_sha256="d" * 64,
                    cohort_lock_sha256="c" * 64,
                )

    def test_collection_preflight_treats_dangling_symlink_as_consumed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dangling = root / "source-7801.discovery.npz"
            dangling.symlink_to(root / "missing-target.npz")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                _outputs(root, 7801)

    def test_collection_preflight_never_probes_audit_destinations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observed = []

            def lexists(path):
                observed.append(Path(path).name)
                return Path(path).name.endswith("attempt-started.json")

            with mock.patch(
                    "scripts.collect_closed_loop_recovery_triage.os.path.lexists",
                    side_effect=lexists):
                with self.assertRaisesRegex(
                        FileExistsError, "refusing to overwrite"):
                    _outputs(root, 7801)
            self.assertEqual(
                observed, ["source-7801.attempt-started.json"])
            self.assertFalse(any(".audit" in name for name in observed))

            observed.clear()
            with mock.patch(
                    "scripts.collect_closed_loop_recovery_triage.os.path.lexists",
                    side_effect=lambda path: observed.append(
                        Path(path).name) or False):
                outputs = _outputs(root, 7801)
            self.assertIn("audit", outputs)
            self.assertIn("audit_privileged", outputs)
            self.assertFalse(any(".audit" in name for name in observed))

    def test_preflight_only_returns_before_every_consuming_boundary(self):
        protocol = copy.deepcopy(_load_protocol())
        with tempfile.TemporaryDirectory() as directory:
            artifact_root = Path(directory) / "preflight-must-not-create"
            protocol["collection"]["artifact_root"] = str(artifact_root)
            policy_entry = _policy_for_seed(protocol, 7801)
            mature_entry = protocol["mature_recovery_policy"]

            def policy(manifest):
                value = mock.Mock()
                value.manifest.return_value = copy.deepcopy(manifest)
                return value

            early_policy = policy(policy_entry)
            mature_policy = policy(mature_entry)
            robot_cfg = SimpleNamespace(
                move_speed=protocol["target"]["command_speed_mps"],
                success_orientation_rad=protocol["target"]["failure"][
                    "max_abs_roll_pitch_rad"],
                obs_dim=46,
                num_joints=12,
            )
            train_cfg = SimpleNamespace(
                control_frequency=50.0,
                max_joint_delta=None,
                use_action_filter=False,
            )
            env = mock.Mock(name="preflight_environment")
            recovery_program = mock.Mock(name="preflight_recovery_program")
            recovery_program.manifest_protocol.return_value = copy.deepcopy(
                protocol["collection"]["candidates"])
            recovery_program.fingerprint.return_value = "f" * 64
            prepared = SimpleNamespace(recovery_program_binding={
                "fingerprint_sha256": "f" * 64,
            })

            def replace_config(value, **changes):
                return SimpleNamespace(**(vars(value) | changes))

            argv = [
                "collect_closed_loop_recovery_triage.py",
                "--source-seed", "7801",
                "--preflight-only",
            ]
            unreachable = AssertionError(
                "preflight-only crossed a consuming boundary")
            with mock.patch("sys.argv", argv), mock.patch.object(
                    collect_cli, "_load_protocol", return_value=protocol), \
                    mock.patch.object(
                        collect_cli, "_git_commit", return_value="a" * 40), \
                    mock.patch.object(
                        collect_cli, "load_app_config",
                        return_value=(robot_cfg, train_cfg, object())), \
                    mock.patch.object(
                        collect_cli, "replace", side_effect=replace_config), \
                    mock.patch.object(
                        collect_cli, "load_frozen_droq_policy",
                        side_effect=(early_policy, mature_policy)), \
                    mock.patch.object(
                        collect_cli, "MujocoSnapshotEnv", return_value=env), \
                    mock.patch.object(
                        collect_cli, "_verify_runtime_contract"), \
                    mock.patch.object(
                        collect_cli, "build_recovery_behavior_library",
                        return_value=recovery_program), \
                    mock.patch.object(
                        collect_cli,
                        "preflight_closed_loop_recovery_collection",
                        return_value=prepared) as preflight, \
                    mock.patch.object(
                        collect_cli, "_cohort_lock",
                        side_effect=unreachable) as cohort_lock, \
                    mock.patch.object(
                        collect_cli, "_start_attempt",
                        side_effect=unreachable) as start_attempt, \
                    mock.patch.object(
                        collect_cli,
                        "collect_preflighted_closed_loop_recovery_triage",
                        side_effect=unreachable) as collector, \
                    mock.patch.object(
                        collect_cli, "_prepare_staged_outputs",
                        side_effect=unreachable) as prepare_outputs, \
                    mock.patch.object(collect_cli.torch, "set_num_threads"), \
                    mock.patch("builtins.print"):
                self.assertEqual(collect_cli.main(), 0)

            preflight.assert_called_once()
            cohort_lock.assert_not_called()
            start_attempt.assert_not_called()
            collector.assert_not_called()
            prepare_outputs.assert_not_called()
            self.assertFalse(artifact_root.exists())

    def test_cohort_lock_accepts_identical_race_loser_but_rejects_symlink(self):
        protocol = _load_protocol()
        kwargs = {
            "generator_commit": "a" * 40,
            "protocol_sha256": "b" * 64,
            "protocol_contract_sha256": "c" * 64,
            "protocol": protocol,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            winner = root / "cohort-lock.json"
            first = _cohort_lock(winner, **kwargs)
            self.assertEqual(_cohort_lock(winner, **kwargs), first)

            symlink = root / "linked-cohort-lock.json"
            symlink.symlink_to(winner)
            with self.assertRaisesRegex(RuntimeError, "non-symlink"):
                _cohort_lock(symlink, **kwargs)

    def test_admission_outputs_reject_each_dangling_symlink_lexically(self):
        protocol = _load_protocol()
        keys = (
            "admission_deployable_filename",
            "admission_privileged_filename",
            "admission_merge_report_filename",
        )
        for key in keys:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / protocol["collection"][key]
                output.symlink_to(root / "missing-target")
                with self.assertRaisesRegex(FileExistsError, "overwrite"):
                    _admission_output_paths(protocol, root)
                self.assertTrue(output.is_symlink())

    def test_audit_cli_rejects_dangling_report_before_consumption(self):
        protocol = copy.deepcopy(_load_protocol())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol["collection"]["artifact_root"] = str(root)
            report = root / protocol["collection"]["triage_report_filename"]
            report.symlink_to(root / "missing-report-target.json")
            consumed = root / protocol["collection"]["audit_consumed_filename"]
            with mock.patch.object(
                    audit_cli, "_load_protocol", return_value=protocol), \
                    mock.patch("sys.argv", [
                        "audit_closed_loop_recovery_triage.py",
                        "--selection-lock-sha256", "a" * 64,
                    ]), mock.patch.object(
                        audit_cli, "consume_and_evaluate_audit",
                        side_effect=AssertionError("audit must not run")):
                with self.assertRaisesRegex(FileExistsError, "overwrite"):
                    audit_cli.main()
            self.assertFalse(consumed.exists())

    def test_runtime_contract_locks_timing_gains_and_projection(self):
        protocol = _load_protocol()
        robot = SimpleNamespace(
            kp=np.full(12, 60.0),
            kd=np.full(12, 5.0),
            num_joints=12,
            obs_dim=46,
        )
        train = SimpleNamespace(
            control_frequency=50.0,
            max_joint_delta=None,
            use_action_filter=False,
        )
        env = SimpleNamespace(
            policy_frequency=50.0,
            model=SimpleNamespace(opt=SimpleNamespace(timestep=0.002)),
            substeps=10,
            kp=np.full(12, 60.0),
            kd=np.full(12, 5.0),
            action_applier=SimpleNamespace(**{
                name: np.asarray(value, dtype=np.float32)
                for name, value in protocol["target"][
                    "action_application_contract"].items()
                if name in ("init_qpos", "action_offset", "joint_min", "joint_max")
            }),
            simulator_fingerprint=lambda: {
                "mjcf_xml_sha256": protocol["target"][
                    "model_mjcf_dependency_sha256"],
                "failure_measurement": {
                    "height_reference": "base_link_body_origin_world_z",
                    "cadence": "post_policy_step_after_all_low_level_substeps",
                    "low_level_substeps_per_policy_step": 10,
                },
            },
        )
        _verify_runtime_contract(env, robot, train, protocol)
        env.model.opt.timestep = 0.0025
        with self.assertRaisesRegex(ValueError, "timing"):
            _verify_runtime_contract(env, robot, train, protocol)

    def test_v3_merge_rejects_audit_paths_before_open(self):
        protocol = _load_protocol()
        collection = protocol["collection"]
        repository = Path(__file__).resolve().parents[1]
        root = repository / collection["artifact_root"]
        seeds = protocol["triage_gates"]["data"]["required_source_seeds"]

        def paths(template_key):
            return [str(root / str(collection[template_key]).format(
                source_seed=seed)) for seed in seeds]

        discovery = paths("discovery_shard_filename_template")
        privileged = paths("discovery_privileged_shard_filename_template")
        reports = paths("collection_report_shard_filename_template")
        kwargs = {
            "protocol": protocol,
            "shards": discovery,
            "privileged_shards": privileged,
            "collection_reports": reports,
            "output": root / collection["discovery_filename"],
            "privileged_output": root / collection[
                "discovery_privileged_filename"],
            "report_output": root / collection[
                "discovery_merge_report_filename"],
        }
        _v3_preopen_discovery_paths(**kwargs)
        with self.assertRaisesRegex(ValueError, "audit paths are forbidden"):
            _v3_preopen_discovery_paths(
                **(kwargs | {"shards": paths(
                    "audit_shard_filename_template")}))
        with self.assertRaisesRegex(ValueError, "source-seed order"):
            _v3_preopen_discovery_paths(
                **(kwargs | {"shards": list(reversed(discovery))}))

    def test_v3_merge_rejects_every_audit_path_role_without_a_probe(self):
        protocol = _load_protocol()
        collection = protocol["collection"]
        repository = Path(__file__).resolve().parents[1]
        root = repository / collection["artifact_root"]
        seeds = protocol["triage_gates"]["data"]["required_source_seeds"]

        def paths(template_key):
            return [str(root / str(collection[template_key]).format(
                source_seed=seed)) for seed in seeds]

        discovery = paths("discovery_shard_filename_template")
        privileged = paths("discovery_privileged_shard_filename_template")
        reports = paths("collection_report_shard_filename_template")
        audit = paths("audit_shard_filename_template")
        audit_privileged = paths("audit_privileged_shard_filename_template")
        kwargs = {
            "protocol": protocol,
            "shards": discovery,
            "privileged_shards": privileged,
            "collection_reports": reports,
            "output": root / collection["discovery_filename"],
            "privileged_output": root / collection[
                "discovery_privileged_filename"],
            "report_output": root / collection[
                "discovery_merge_report_filename"],
        }
        mutations = (
            {"shards": [audit[0], *discovery[1:]]},
            {"privileged_shards": [
                audit_privileged[0], *privileged[1:]]},
            {"collection_reports": [audit[0], *reports[1:]]},
            {"output": root / collection["audit_filename"]},
            {"privileged_output": root / collection[
                "audit_privileged_filename"]},
            {"report_output": Path(audit[0])},
        )
        with mock.patch.object(
                Path, "is_symlink",
                side_effect=AssertionError("unexpected filesystem probe")), \
                mock.patch.object(
                    Path, "resolve",
                    side_effect=AssertionError("unexpected filesystem probe")):
            for mutation in mutations:
                with self.subTest(mutation=next(iter(mutation))):
                    with self.assertRaisesRegex(
                            ValueError, "audit paths are forbidden"):
                        _v3_preopen_discovery_paths(**(kwargs | mutation))

    def test_v3_main_rejects_audit_output_before_no_clobber_probe(self):
        protocol = _load_protocol()
        collection = protocol["collection"]
        repository = Path(__file__).resolve().parents[1]
        root = repository / collection["artifact_root"]
        seeds = protocol["triage_gates"]["data"]["required_source_seeds"]

        def paths(template_key):
            return [str(root / str(collection[template_key]).format(
                source_seed=seed)) for seed in seeds]

        argv = [
            "merge_grouped_qsafe_shards.py",
            *paths("discovery_shard_filename_template"),
            "--privileged-shards",
            *paths("discovery_privileged_shard_filename_template"),
            "--collection-reports",
            *paths("collection_report_shard_filename_template"),
            "--output", str(root / collection["audit_filename"]),
            "--privileged-output", str(
                root / collection["discovery_privileged_filename"]),
            "--report", str(
                root / collection["discovery_merge_report_filename"]),
            "--protocol", str(repository / "config" /
                              "qsafe_closed_loop_recovery_triage_v3.yaml"),
        ]
        with mock.patch("sys.argv", argv), mock.patch(
                "scripts.merge_grouped_qsafe_shards.os.path.lexists",
                side_effect=AssertionError("unexpected output probe")), \
                mock.patch(
                    "scripts.merge_grouped_qsafe_shards._clean_git_commit",
                    side_effect=AssertionError("unexpected merge execution")):
            with self.assertRaisesRegex(ValueError, "audit paths are forbidden"):
                merge_main()

    def test_generic_merger_rejects_parent_alias_and_protocol_audit_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            audit = alias / "source-7801.audit.npz"
            normal = root / "normal.npz"
            base = [
                "merge_grouped_qsafe_shards.py", str(audit), str(normal),
                "--output", str(root / "merged.npz"),
            ]
            cases = (
                [*base, "--protocol", str(root / "protocol.yaml")],
                [
                    "merge_grouped_qsafe_shards.py", str(normal),
                    str(root / "other.npz"), "--output",
                    str(root / "merged.npz"), "--protocol", str(audit),
                ],
            )
            for argv in cases:
                with self.subTest(protocol_is_audit=argv[-1] == str(audit)), \
                        mock.patch("sys.argv", argv), mock.patch(
                            "scripts.merge_grouped_qsafe_shards."
                            "assert_development_path",
                            side_effect=AssertionError(
                                "audit rejection must precede path probing")):
                    with self.assertRaisesRegex(
                            ValueError, "audit paths are forbidden"):
                        merge_main()

    def test_generic_merger_rejects_protocol_symlink_alias_before_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "source-7801.audit.npz"
            audit.write_bytes(b"must-not-open")
            alias = root / "protocol.yaml"
            alias.symlink_to(audit)
            argv = [
                "merge_grouped_qsafe_shards.py",
                str(root / "first.npz"), str(root / "second.npz"),
                "--output", str(root / "merged.npz"),
                "--protocol", str(alias),
            ]
            with mock.patch("sys.argv", argv), mock.patch.object(
                    Path, "read_text",
                    side_effect=AssertionError("audit alias must not be read")):
                with self.assertRaisesRegex(
                        PermissionError, "refuse symlink inputs"):
                    merge_main()

    def test_v3_exact_gate_locks_k9_arrays_and_manifest_protocol(self):
        protocol, dataset = self._exact_gate_fixture()
        raw_gate = _v3_exact_discovery_gate(dataset, protocol)
        required_gate = _require_v3_exact_discovery_gate(dataset, protocol)
        self.assertTrue(raw_gate["pass"])
        self.assertTrue(required_gate["pass"])
        self.assertIsInstance(raw_gate["checks"]["candidates_exact"], bool)
        self.assertIsInstance(
            required_gate["checks"]["candidates_exact"], bool)
        json.dumps(required_gate)

        cases = (
            ("candidate_kind_exact", lambda value: value.arrays[
                "candidate_kind"].__setitem__((0, 1), "drift")),
            ("candidate_behavior_steps_exact", lambda value: value.arrays[
                "candidate_behavior_steps"].__setitem__((0, 1), 11)),
            ("candidate_protocol_exact", lambda value: value.manifest[
                "candidate_protocol"].update({"unregistered": True})),
        )
        for failed_check, mutate in cases:
            _, changed = self._exact_gate_fixture()
            mutate(changed)
            result = _v3_exact_discovery_gate(changed, protocol)
            with self.subTest(failed_check=failed_check):
                self.assertFalse(result["pass"])
                self.assertFalse(result["checks"][failed_check])
                with self.assertRaisesRegex(
                        ValueError, "before publication"):
                    _require_v3_exact_discovery_gate(changed, protocol)

    def test_v3_exact_gate_locks_audit_seed_shape_uniqueness_and_domain(self):
        protocol, _ = self._exact_gate_fixture()

        _, wrong_shape = self._exact_gate_fixture()
        wrong_shape.arrays["preassigned_audit_crn_id"] = wrong_shape.arrays[
            "preassigned_audit_crn_id"][:, :-1]
        shape_result = _v3_exact_discovery_gate(wrong_shape, protocol)
        self.assertFalse(shape_result["checks"][
            "audit_seed_preassignment_shape_exact"])

        _, duplicate = self._exact_gate_fixture()
        duplicate.arrays["preassigned_audit_rollout_seed"][0, 1] = (
            duplicate.arrays["preassigned_audit_rollout_seed"][0, 0])
        duplicate_result = _v3_exact_discovery_gate(duplicate, protocol)
        self.assertFalse(duplicate_result["checks"][
            "audit_seed_preassignment_unique"])

        _, collision = self._exact_gate_fixture()
        collision.arrays["preassigned_audit_candidate_seed"][0] = (
            collision.arrays["candidate_seed"][0])
        collision_result = _v3_exact_discovery_gate(collision, protocol)
        self.assertTrue(collision_result["checks"][
            "audit_seed_preassignment_unique"])
        self.assertFalse(collision_result["checks"][
            "discovery_audit_seed_domains_disjoint"])


if __name__ == "__main__":
    unittest.main()
