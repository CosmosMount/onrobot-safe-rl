from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from safety_data.candidates import (
    CANDIDATE_KINDS,
    CandidateProtocolError,
    EvidenceCandidateConfig,
    InsufficientCandidateSupportError,
)
from safety_data.collector import (
    CollectedGroup,
    GaussianImpulseSchedule,
    GroupIdentity,
    GroupRandomness,
    GroupedBranchAssembler,
    NativeCollectionConfig,
    collect_native_groups,
    group_randomness,
)
from safety_data.native import NativeGroupEvaluation
from safety_data.schema import GroupedBranchDataset, PrivilegedBranchView
from scripts.collect_native_grouped_qsafe import (
    _prepare_staged_outputs,
    _publish_staged_outputs,
    main as collect_main,
)
from tests.test_safety_data import synthetic_dataset


class _FakePolicy:
    def manifest(self):
        return {"policy_fingerprint_sha256": "fake-policy"}

    def fingerprint(self):
        return "fake-policy"

    def sample_action(self, observation, rng):
        del observation, rng
        return np.zeros(12, dtype=np.float32)

    def deterministic_action(self, observation):
        del observation
        return np.zeros(12, dtype=np.float32)


class _FakeNativeEnv:
    def __init__(self, events):
        self.events = events
        self.step_calls = 0
        self.capture_calls = 0
        self.action_applier = SimpleNamespace(
            init_qpos=np.zeros(12, dtype=np.float32),
            action_offset=np.ones(12, dtype=np.float32),
            joint_min=-np.ones(12, dtype=np.float32),
            joint_max=np.ones(12, dtype=np.float32),
            max_joint_delta=None,
            action_filter=None,
        )
        self.cfg = SimpleNamespace(
            fallen_orientation_rad=1.0,
            move_speed=0.3,
        )
        self.data = SimpleNamespace(qpos=np.zeros(12, dtype=np.float32))
        self.qpos_addresses = np.arange(12)
        self.previous_action_requested = np.zeros(12, dtype=np.float32)

    def simulator_fingerprint(self):
        return {"model": "fake"}

    def reset_standing(self, *, settle_seconds, rng):
        del settle_seconds, rng

    def record_observation(self):
        return np.zeros((5, 46), dtype=np.float32)

    def measurement(self):
        return SimpleNamespace(near_failure=False)

    def capture(self):
        self.capture_calls += 1
        digest = f"{self.capture_calls:064x}"
        return SimpleNamespace(compound_sha256=lambda: digest)

    def step(self, action):
        del action
        self.events.append("source_step")
        self.step_calls += 1
        return SimpleNamespace(failure=False)


class _RecordingAssembler:
    def __init__(self, events):
        self.events = events
        self.group_count = 0
        self.finalize_calls = 0

    def add(self, group):
        del group
        self.events.append("add")
        self.group_count += 1

    def finalize(self):
        self.finalize_calls += 1
        return "dataset", "privileged"


def _fake_candidates():
    requested = np.zeros((16, 12), dtype=np.float32)
    return SimpleNamespace(
        requested=requested,
        executed=requested.copy(),
        q_target=requested.copy(),
        kind=np.asarray(CANDIDATE_KINDS),
        mask=np.ones(16, dtype=bool),
        valid_count=16,
    )


def _fake_evaluation(candidates):
    return SimpleNamespace(
        candidate_requested=candidates.requested.copy(),
        candidate_executed=candidates.executed.copy(),
        candidate_q_target=candidates.q_target.copy(),
        fall=np.zeros((16, 2), dtype=bool),
    )


def _native_config():
    return NativeCollectionConfig(
        split="development_candidate_support_unit",
        target_groups=1,
        source_seed=91,
        policy_training_seed=42,
        horizon_steps=2,
        replicas=2,
        natural_acceptance_probability=1.0,
        max_episode_steps=3,
        max_groups_per_trajectory=3,
        max_source_steps=3,
        settle_seconds=0.0,
    )


class NativeCollectionCandidateSupportTest(unittest.TestCase):
    def test_insufficient_support_skips_before_outcome_and_advances_source(self):
        events = []
        env = _FakeNativeEnv(events)
        assembler = _RecordingAssembler(events)
        candidates = _fake_candidates()
        candidate_seeds = []

        def build(**kwargs):
            candidate_seeds.append(kwargs["candidate_seed"])
            if len(candidate_seeds) == 1:
                events.append("candidate_support_skip")
                raise InsufficientCandidateSupportError(7, 8)
            events.append("candidate_ready")
            return candidates

        def evaluate(*args, **kwargs):
            del args, kwargs
            events.append("branch_outcome")
            return _fake_evaluation(candidates)

        progress = []
        with patch(
            "safety_data.collector.GroupedBranchAssembler",
            return_value=assembler,
        ) as assembler_mock, patch(
            "safety_data.collector.privileged_features",
            return_value=np.zeros(38, dtype=np.float32),
        ), patch(
            "safety_data.candidates.build_evidence_candidates",
            side_effect=build,
        ) as build_mock, patch(
            "safety_data.native.evaluate_same_state_group",
            side_effect=evaluate,
        ) as evaluate_mock:
            result = collect_native_groups(
                env=env,
                source_policy=_FakePolicy(),
                continuation_policy=_FakePolicy(),
                candidate_config=EvidenceCandidateConfig(),
                branch_disturbance=GaussianImpulseSchedule(
                    policy_steps=(1,), linear_std_mps=0.1,
                    angular_std_radps=0.2),
                config=_native_config(),
                generator_commit="unit-test",
                progress=progress.append,
            )

        self.assertEqual(build_mock.call_count, 2)
        self.assertEqual(evaluate_mock.call_count, 1)
        self.assertEqual(candidate_seeds[0], candidate_seeds[1])
        self.assertEqual(result.source_steps, 2)
        self.assertEqual(result.randomly_accepted_groups, 1)
        self.assertEqual(result.skipped_candidate_support_groups, 1)
        self.assertEqual(assembler.group_count, 1)
        self.assertEqual(assembler.finalize_calls, 1)
        self.assertEqual(
            events,
            [
                "candidate_support_skip",
                "source_step",
                "candidate_ready",
                "branch_outcome",
                "add",
                "source_step",
            ],
        )
        self.assertEqual(
            progress[0]["skipped_candidate_support_groups"], 1)
        self.assertNotIn(
            "skipped_candidate_support_groups",
            assembler_mock.call_args.kwargs["collection_protocol"],
        )
        action_contract = assembler_mock.call_args.kwargs[
            "action_application_contract"]
        self.assertEqual(
            set(action_contract),
            {
                "q_target_semantic",
                "init_qpos",
                "action_offset",
                "joint_min",
                "joint_max",
                "projection",
            },
        )

    def test_unrelated_candidate_protocol_error_still_fails_closed(self):
        events = []
        env = _FakeNativeEnv(events)
        assembler = _RecordingAssembler(events)
        evaluate_mock = Mock()
        with patch(
            "safety_data.collector.GroupedBranchAssembler",
            return_value=assembler,
        ), patch(
            "safety_data.collector.privileged_features",
            return_value=np.zeros(38, dtype=np.float32),
        ), patch(
            "safety_data.candidates.build_evidence_candidates",
            side_effect=CandidateProtocolError("preview contract invalid"),
        ), patch(
            "safety_data.native.evaluate_same_state_group",
            evaluate_mock,
        ), self.assertRaisesRegex(
            CandidateProtocolError, "preview contract invalid"
        ):
            collect_native_groups(
                env=env,
                source_policy=_FakePolicy(),
                continuation_policy=_FakePolicy(),
                candidate_config=EvidenceCandidateConfig(),
                branch_disturbance=GaussianImpulseSchedule(
                    policy_steps=(1,), linear_std_mps=0.1,
                    angular_std_radps=0.2),
                config=_native_config(),
                generator_commit="unit-test",
            )

        evaluate_mock.assert_not_called()
        self.assertEqual(env.step_calls, 0)
        self.assertEqual(assembler.group_count, 0)
        self.assertEqual(assembler.finalize_calls, 0)


class GroupedBranchAssemblerTest(unittest.TestCase):
    def setUp(self):
        self.source, _ = synthetic_dataset()
        manifest = self.source.manifest
        self.assembler = GroupedBranchAssembler(
            split="development_native",
            horizon_steps=self.source.horizon_steps,
            generator_commit="test-commit",
            simulator_fingerprint=manifest["simulator_fingerprint"],
            source_policy=manifest["source_policy"],
            continuation_policy=manifest["continuation_policy"],
            candidate_protocol=manifest["candidate_protocol"],
            fall_definition=manifest["fall_definition"],
            action_application_contract=manifest["action_application_contract"],
            collection_protocol={"selection": "pre-outcome-random"},
            privileged_feature_names=("base_height", "tilt"),
        )

    def group(self, index: int) -> CollectedGroup:
        source = self.source
        crn_id = source["crn_id"][index]
        rollout_seed = source["rollout_seed"][index]
        perturbation_seed = source["perturbation_seed"][index]
        evaluation = NativeGroupEvaluation(
            candidate_requested=source["candidate_requested"][index],
            candidate_executed=source["candidate_executed"][index],
            candidate_q_target=source["candidate_q_target"][index],
            fall=source["fall"][index],
            first_failure_step=source["first_failure_step"][index],
            max_tilt_rad=source["max_tilt_rad"][index],
            min_height_m=source["min_height_m"][index],
            crn_id=crn_id,
            rollout_seed=rollout_seed,
            perturbation_seed=perturbation_seed,
            seed_contract="explicit_three_stream_v1",
        )
        return CollectedGroup(
            identity=GroupIdentity(
                group_id=str(source["group_id"][index]),
                state_hash=str(source["state_hash"][index]),
                trajectory_id=str(source["trajectory_id"][index]),
                episode_id=int(source["episode_id"][index]),
                episode_step=int(source["episode_step"][index]),
                policy_training_seed=int(source["policy_training_seed"][index]),
                source_seed=int(source["source_seed"][index]),
                policy_source=str(source["policy_source"][index]),
                command_vx=float(source["command_vx"][index]),
                acceptance_probability=float(
                    source["acceptance_probability"][index]),
            ),
            observation_history=source["obs_history"][index],
            candidate_kind=source["candidate_kind"][index],
            candidate_mask=source["candidate_mask"][index],
            evaluation=evaluation,
            randomness=GroupRandomness(
                crn_id=crn_id,
                rollout_seed=rollout_seed,
                perturbation_seed=perturbation_seed,
                candidate_seed=9000 + index,
            ),
            privileged_features=np.asarray([0.31, 0.12], np.float32),
        )

    def test_finalize_round_trips_deployable_and_privileged_views(self):
        for index in range(self.source.group_count):
            self.assembler.add(self.group(index))
        dataset, privileged = self.assembler.finalize()
        self.assertEqual(dataset.validate()["groups"], self.source.group_count)
        self.assertIsNotNone(privileged)
        assert privileged is not None
        privileged.validate(dataset)
        np.testing.assert_array_equal(
            dataset["nominal_action_requested"],
            dataset["candidate_requested"][:, 0],
        )
        self.assertEqual(dataset["candidate_seed"].dtype.kind, "u")
        self.assertEqual(dataset["episode_id"].dtype, np.dtype(np.int64))
        self.assertTrue(np.all(dataset["sampling_stratum"] == "unspecified"))
        with tempfile.TemporaryDirectory() as directory:
            deployable_path = dataset.save(Path(directory) / "native.npz")
            privileged_path = privileged.save(
                Path(directory) / "native-privileged.npz")
            restored = GroupedBranchDataset.load(deployable_path)
            restored_privileged = PrivilegedBranchView.load(
                privileged_path, deployable=restored)
        self.assertEqual(restored.group_count, self.source.group_count)
        self.assertEqual(restored_privileged.features.shape, (4, 2))

    def test_finalize_preserves_v4_high_bit_episode_identifier(self):
        group = self.group(0)
        episode_id = (1 << 63) | 123
        self.assembler.add(replace(
            group,
            identity=replace(group.identity, episode_id=episode_id),
        ))
        dataset, _ = self.assembler.finalize()
        self.assertEqual(dataset["episode_id"].dtype, np.dtype(np.uint64))
        self.assertEqual(int(dataset["episode_id"][0]), episode_id)
        self.assertEqual(dataset.validate()["groups"], 1)

    def test_add_copies_mutable_evaluation_arrays(self):
        group = self.group(0)
        expected_fall = group.evaluation.fall.copy()
        self.assembler.add(group)
        group.evaluation.fall[:] = ~group.evaluation.fall
        dataset, _ = self.assembler.finalize()
        np.testing.assert_array_equal(dataset["fall"][0], expected_fall)

    def test_sampling_stratum_controls_recorded_ipw_probability(self):
        self.assembler.manifest["collection_protocol"]["sampling_strata"] = {
            "physical_near_failure": {"acceptance_probability": 1.0},
            "random_accept": {"acceptance_probability": 0.5},
        }
        base = self.group(0)
        invalid = replace(
            base,
            identity=replace(
                base.identity,
                sampling_stratum="random_accept",
                acceptance_probability=1.0,
            ),
        )
        with self.assertRaisesRegex(ValueError, "disagrees with sampling stratum"):
            self.assembler.add(invalid)
        valid = replace(
            base,
            identity=replace(
                base.identity,
                sampling_stratum="random_accept",
                acceptance_probability=0.5,
            ),
        )
        self.assembler.add(valid)
        dataset, _ = self.assembler.finalize()
        self.assertEqual(dataset["sampling_stratum"].tolist(), ["random_accept"])
        self.assertEqual(dataset["acceptance_probability"].tolist(), [0.5])

    def test_duplicate_identity_and_nonrectangular_group_fail_early(self):
        first = self.group(0)
        self.assembler.add(first)
        with self.assertRaisesRegex(ValueError, "duplicate group_id"):
            self.assembler.add(first)
        second = self.group(1)
        shortened = replace(
            second,
            candidate_kind=second.candidate_kind[:2],
            candidate_mask=second.candidate_mask[:2],
            evaluation=NativeGroupEvaluation(
                candidate_requested=second.evaluation.candidate_requested[:2],
                candidate_executed=second.evaluation.candidate_executed[:2],
                candidate_q_target=second.evaluation.candidate_q_target[:2],
                fall=second.evaluation.fall[:2],
                first_failure_step=second.evaluation.first_failure_step[:2],
                max_tilt_rad=second.evaluation.max_tilt_rad[:2],
                min_height_m=second.evaluation.min_height_m[:2],
                crn_id=second.evaluation.crn_id,
                rollout_seed=second.evaluation.rollout_seed,
                perturbation_seed=second.evaluation.perturbation_seed,
                seed_contract=second.evaluation.seed_contract,
            ),
        )
        with self.assertRaisesRegex(ValueError, "all groups must share"):
            self.assembler.add(shortened)

    def test_explicit_seed_namespaces_are_deterministic_and_disjoint(self):
        first_bundle, first_record = group_randomness(
            source_seed=71, group_index=9, replicas=8)
        second_bundle, second_record = group_randomness(
            source_seed=71, group_index=9, replicas=8)
        np.testing.assert_array_equal(first_bundle.crn_id, second_bundle.crn_id)
        np.testing.assert_array_equal(
            first_bundle.rollout_seed, second_bundle.rollout_seed)
        np.testing.assert_array_equal(
            first_bundle.perturbation_seed, second_bundle.perturbation_seed)
        self.assertEqual(first_record.candidate_seed, second_record.candidate_seed)
        self.assertEqual(first_bundle.seed_contract, "explicit_three_stream_v1")
        self.assertTrue(set(map(int, first_bundle.crn_id)).isdisjoint(
            set(map(int, first_bundle.rollout_seed))))
        self.assertTrue(set(map(int, first_bundle.crn_id)).isdisjoint(
            set(map(int, first_bundle.perturbation_seed))))

    def test_impulse_schedule_consumes_only_scheduled_rng_draws(self):
        class RecordingEnv:
            def __init__(self):
                self.calls = []

            def apply_base_velocity_impulse(self, **value):
                self.calls.append(value)

        schedule = GaussianImpulseSchedule(
            policy_steps=(2,), linear_std_mps=0.1,
            angular_std_radps=0.2)
        env = RecordingEnv()
        rng = np.random.default_rng(4)
        before = repr(rng.bit_generator.state)
        schedule(env, 1, rng)
        self.assertEqual(repr(rng.bit_generator.state), before)
        schedule(env, 2, rng)
        self.assertEqual(len(env.calls), 1)
        self.assertEqual(schedule.manifest()["policy_steps"], [2])

    def test_staged_bundle_publication_rolls_back_on_destination_race(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destinations = tuple(root / name for name in (
                "dataset.npz", "privileged.npz", "report.json"))
            staged = _prepare_staged_outputs(destinations)
            for index, (staging, _) in enumerate(staged):
                staging.write_text(f"payload-{index}", encoding="utf-8")
            destinations[1].write_text("raced", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _publish_staged_outputs(staged)
            self.assertFalse(destinations[0].exists())
            self.assertEqual(
                destinations[1].read_text(encoding="utf-8"), "raced")
            self.assertFalse(destinations[2].exists())
            self.assertTrue(all(not staging.exists() for staging, _ in staged))


class CollectionCommandNoClobberTest(unittest.TestCase):
    def test_any_existing_bundle_target_refuses_before_collection(self):
        for occupied in ("dataset", "privileged", "report"):
            with self.subTest(occupied=occupied):
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / "development_no_clobber.npz"
                    targets = {
                        "dataset": output,
                        "privileged": output.with_name(
                            f"{output.stem}.privileged.npz"),
                        "report": output.with_name(
                            f"{output.stem}.report.json"),
                    }
                    sentinel = f"existing-{occupied}".encode()
                    targets[occupied].write_bytes(sentinel)
                    argv = [
                        "collect_native_grouped_qsafe.py",
                        "--checkpoint", "development_checkpoint",
                        "--split", "development_no_clobber",
                        "--groups", "1",
                        "--source-seed", "1",
                        "--output", str(output),
                    ]
                    with patch.object(sys, "argv", argv), patch(
                        "scripts.collect_native_grouped_qsafe.collect_native_groups"
                    ) as collect_mock, self.assertRaisesRegex(
                        FileExistsError, "refusing to overwrite outputs"
                    ):
                        collect_main()

                    collect_mock.assert_not_called()
                    self.assertEqual(targets[occupied].read_bytes(), sentinel)
                    for label, path in targets.items():
                        if label != occupied:
                            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
