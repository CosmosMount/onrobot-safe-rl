from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from safety_data.collector import (
    CollectedGroup,
    GaussianImpulseSchedule,
    GroupIdentity,
    GroupRandomness,
    GroupedBranchAssembler,
    group_randomness,
)
from safety_data.native import NativeGroupEvaluation
from safety_data.schema import GroupedBranchDataset, PrivilegedBranchView
from scripts.collect_native_grouped_qsafe import (
    _prepare_staged_outputs,
    _publish_staged_outputs,
)
from tests.test_safety_data import synthetic_dataset


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


if __name__ == "__main__":
    unittest.main()
