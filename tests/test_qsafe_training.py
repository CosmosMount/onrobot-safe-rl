from __future__ import annotations

import copy
from dataclasses import replace
import unittest
from unittest import mock

import numpy as np
import torch
from torch.nn import functional as F

from runtime.inference.actions import action_to_qpos, qpos_to_action
from rl.qsafe.data import (
    NormalizationStats,
    TorchGroupedView,
    trajectory_bootstrap_indices,
)
from rl.qsafe.loss import QSafeLossConfig, qsafe_group_loss
from rl.qsafe.network import QSafeNetworkConfig, SelectiveAdvantageQSafe
from rl.qsafe.recovery_program import (
    RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM,
    bind_recovery_program_manifest,
    make_recovery_program_feature_manifest,
)
from rl.qsafe.training import (
    QSafeTrainingConfig,
    RECOVERY_PROGRAM_V4_LOSS_CONFIG,
    RECOVERY_PROGRAM_V4_MEMBER_SEED_STRIDE,
    RECOVERY_PROGRAM_V4_NETWORK_CONFIG,
    RECOVERY_PROGRAM_V4_TRAINING_CONFIG,
    TrainedQSafeMember,
    _temperature_loss,
    fit_temperature,
    predict_qsafe_ensemble,
    train_qsafe_ensemble,
    train_qsafe_member,
)
from safety_data.splits import nested_trajectory_subsets
from safety_data.recovery_options import (
    RECOVERY_OPTION_KINDS,
    RECOVERY_OPTION_STEPS,
    RecoveryOptionCandidateConfig,
)
from safety_data.recovery_behaviors import RecoveryBehaviorLibrary
from tests.test_safety_data import synthetic_dataset
from tests.test_qsafe_closed_loop_schema import closed_loop_dataset
from tests.test_qsafe_recovery_behaviors import _MaturePolicy, _applier
from tests.test_selective_qsafe import inputs, outcome_tensors


def tiny_network_config(*, action_dim: int = 36) -> QSafeNetworkConfig:
    return QSafeNetworkConfig(
        action_dim=action_dim,
        frame_hidden_dim=8,
        state_hidden_dim=8,
        action_hidden_dim=8,
    )


def tiny_training_config(**changes) -> QSafeTrainingConfig:
    values = {
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "ensemble_members": 2,
        "calibration_steps": 2,
        "seed": 31,
        "device": "cpu",
    }
    values.update(changes)
    return QSafeTrainingConfig(**values)


def bind_recovery_dataset(dataset):
    library = RecoveryBehaviorLibrary(_MaturePolicy(), _applier())
    recovery_binding = bind_recovery_program_manifest(library.manifest())
    feature_manifest = make_recovery_program_feature_manifest(
        recovery_binding["fingerprint_sha256"])
    action_projection = recovery_binding["manifest"]["action_projection"]
    dataset.manifest["candidate_protocol"] = library.manifest_protocol()
    dataset.manifest["action_application_contract"] = {
        "q_target_semantic": "absolute_joint_position_sent",
        **{
            field: copy.deepcopy(action_projection[field])
            for field in (
                "init_qpos", "action_offset", "joint_min", "joint_max")
        },
        "projection": (
            "clip_normalized_then_joint_bounds_then_slew_then_filter"),
        "max_joint_delta": action_projection["max_joint_delta"],
        "use_action_filter": action_projection["use_action_filter"],
    }
    dataset.manifest["recovery_program"] = copy.deepcopy(recovery_binding)
    dataset.manifest["recovery_program_feature_contract"] = copy.deepcopy(
        feature_manifest)
    reproject_recovery_dataset(dataset)
    return recovery_binding, feature_manifest


def reproject_recovery_dataset(dataset):
    """Make a synthetic fixture reproduce the exact runtime application path."""
    contract = dataset.manifest["action_application_contract"]
    projection = {
        field: np.asarray(contract[field], dtype=np.float32)
        for field in ("init_qpos", "action_offset", "joint_min", "joint_max")
    }
    requested = np.asarray(dataset["candidate_requested"])
    q_target = np.empty_like(requested)
    executed = np.empty_like(requested)
    for group_index, candidate_index in np.ndindex(requested.shape[:2]):
        q_target[group_index, candidate_index] = action_to_qpos(
            requested[group_index, candidate_index], **projection)
        executed[group_index, candidate_index] = qpos_to_action(
            q_target[group_index, candidate_index],
            init_qpos=projection["init_qpos"],
            action_offset=projection["action_offset"],
        )
    dataset.arrays["candidate_q_target"] = q_target
    dataset.arrays["candidate_executed"] = executed


def v4_recovery_view():
    dataset = closed_loop_dataset()
    bind_recovery_dataset(dataset)
    normalization = NormalizationStats.fit(dataset)
    return TorchGroupedView(
        dataset, normalization, action_view="recovery_program_v1")


def synthetic_trained_member(
    view,
    network_config,
    training_config,
    loss_config,
    *,
    seed,
    bootstrap,
):
    del view, loss_config, bootstrap
    return TrainedQSafeMember(
        model=SelectiveAdvantageQSafe(network_config),
        seed=seed,
        bootstrap_trajectories=["synthetic-trajectory"],
        epoch_loss=[0.25] * training_config.epochs,
    )


class NormalizationAndViewTest(unittest.TestCase):
    def test_normalization_is_immutable_and_compares_privileged_stats(self):
        first = NormalizationStats(
            np.zeros(46), np.ones(46), np.zeros(3), np.ones(3))
        same = NormalizationStats(
            np.zeros(46), np.ones(46), np.zeros(3), np.ones(3))
        changed = NormalizationStats(
            np.zeros(46), np.ones(46), np.ones(3), np.ones(3))
        self.assertTrue(first.equivalent_to(same))
        self.assertFalse(first.equivalent_to(changed))
        with self.assertRaises(ValueError):
            first.observation_mean[0] = 3.0

    def test_fit_normalization_binds_dataset_content_and_split(self):
        dataset, _ = synthetic_dataset(split="development_train")
        report = dataset.validate()

        normalization = NormalizationStats.fit(dataset)

        self.assertEqual(
            normalization.fit_content_sha256, report["content_sha256"])
        self.assertEqual(normalization.fit_split, "development_train")
        self.assertTrue(normalization.equivalent_to(replace(normalization)))
        self.assertFalse(normalization.equivalent_to(replace(
            normalization, fit_content_sha256="0" * 64)))
        self.assertFalse(normalization.equivalent_to(replace(
            normalization, fit_split="development_calibration")))
        with self.assertRaisesRegex(ValueError, "both be present or absent"):
            NormalizationStats(
                np.zeros(46),
                np.ones(46),
                fit_content_sha256="0" * 64,
            )

    def test_masked_sentinels_are_sanitized_before_dense_model_forward(self):
        dataset, _ = synthetic_dataset()
        dataset.arrays["candidate_mask"][:, 2] = False
        dataset.arrays["candidate_requested"][:, 2] = np.nan
        dataset.arrays["candidate_executed"][:, 2] = np.nan
        dataset.arrays["candidate_q_target"][:, 2] = np.nan
        dataset.arrays["fall"] = dataset["fall"].astype(np.int8)
        dataset.arrays["fall"][:, 2] = 7
        dataset.arrays["first_failure_step"][:, 2] = 0
        dataset.arrays["max_tilt_rad"][:, 2] = np.nan
        dataset.arrays["min_height_m"][:, 2] = np.nan
        normalization = NormalizationStats.fit(dataset)
        view = TorchGroupedView(dataset, normalization)
        self.assertEqual(view.action_view, "application_concat")
        self.assertEqual(view.action_dim, 36)
        np.testing.assert_array_equal(view.candidate[:, 0], view.nominal)
        batch = view.batch(view.all_indices(), "cpu")
        for value in (
            batch.candidate_action,
            batch.fall,
            batch.max_tilt_rad,
            batch.min_height_m,
        ):
            self.assertTrue(bool(torch.all(torch.isfinite(value))))
        self.assertTrue(bool(torch.all(batch.fall[:, 2] == 0.0)))
        self.assertTrue(bool(torch.all(
            batch.first_failure_step[:, 2] == dataset.horizon_steps + 1)))

        trained = train_qsafe_ensemble(
            view,
            tiny_network_config(),
            tiny_training_config(ensemble_members=1, calibration_steps=0),
            QSafeLossConfig(),
        )
        self.assertTrue(np.isfinite(trained.members[0].epoch_loss).all())
        for parameter in trained.members[0].model.parameters():
            self.assertTrue(bool(torch.all(torch.isfinite(parameter))))

    def test_requested_only_action_view_is_explicit_12d_ablation(self):
        dataset, _ = synthetic_dataset()
        normalization = NormalizationStats.fit(dataset)
        view = TorchGroupedView(
            dataset, normalization, action_view="requested")
        self.assertEqual(view.action_view, "requested")
        self.assertEqual(view.action_dim, 12)
        np.testing.assert_array_equal(
            view.nominal, dataset["nominal_action_requested"])
        np.testing.assert_array_equal(view.candidate[:, 0], view.nominal)

    def test_unrepresentable_ipw_dynamic_range_fails_closed(self):
        dataset, _ = synthetic_dataset()
        dataset.arrays["acceptance_probability"] = np.asarray(
            [1e-300, 1.0, 1.0, 1.0], dtype=np.float64)
        normalization = NormalizationStats.fit(dataset)
        with self.assertRaisesRegex(ValueError, "cannot be represented"):
            TorchGroupedView(dataset, normalization)

    def test_v1_view_refuses_to_collapse_recovery_option_durations(self):
        dataset, _ = synthetic_dataset()
        for name in (
                "candidate_requested", "candidate_executed",
                "candidate_q_target", "candidate_kind", "candidate_mask",
                "fall", "first_failure_step", "max_tilt_rad",
                "min_height_m"):
            dataset.arrays[name] = np.repeat(
                dataset.arrays[name][:, :1], 29, axis=1)
        dataset.arrays["candidate_kind"] = np.repeat(
            np.asarray(RECOVERY_OPTION_KINDS)[None, :],
            dataset.group_count,
            axis=0,
        )
        dataset.arrays["candidate_mask"] = np.ones(
            (dataset.group_count, 29), dtype=bool)
        dataset.arrays["candidate_option_steps"] = np.repeat(
            np.asarray(RECOVERY_OPTION_STEPS, dtype=np.int8)[None, :],
            dataset.group_count,
            axis=0,
        )
        dataset.manifest["candidate_protocol"] = (
            RecoveryOptionCandidateConfig().manifest_protocol())
        normalization = NormalizationStats.fit(dataset)
        with self.assertRaisesRegex(ValueError, "duration-aware v2 model"):
            TorchGroupedView(dataset, normalization)

    def test_recovery_program_view_uses_shared_82d_contract(self):
        dataset = closed_loop_dataset()
        recovery_binding, feature_manifest = bind_recovery_dataset(dataset)
        normalization = NormalizationStats.fit(dataset)

        view = TorchGroupedView(
            dataset, normalization, action_view="recovery_program_v1")

        self.assertEqual(view.action_dim, RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM)
        self.assertEqual(view.action_view, "recovery_program_v1")
        self.assertEqual(view.view_role, "training")
        np.testing.assert_array_equal(
            view.mask, np.ones((dataset.group_count, 9), dtype=bool))
        np.testing.assert_array_equal(
            view.group_weight, np.ones(dataset.group_count, dtype=np.float32))
        np.testing.assert_array_equal(view.candidate[:, 0], view.nominal)
        self.assertEqual(
            view.recovery_program_feature_contract_sha256,
            feature_manifest["feature_contract_sha256"],
        )
        self.assertEqual(view.recovery_program_binding, recovery_binding)
        self.assertEqual(
            view.recovery_program_feature_manifest, feature_manifest)
        exposed = view.recovery_program_binding
        assert exposed is not None
        exposed["fingerprint_sha256"] = "0" * 64
        self.assertEqual(view.recovery_program_binding, recovery_binding)

    def test_recovery_program_view_rejects_non_float32_disk_actions(self):
        for field in (
                "candidate_requested", "candidate_executed",
                "candidate_q_target"):
            with self.subTest(field=field):
                dataset = closed_loop_dataset()
                bind_recovery_dataset(dataset)
                dataset.arrays[field] = dataset.arrays[field].astype(np.float64)
                normalization = NormalizationStats.fit(dataset)
                with self.assertRaisesRegex(ValueError, "dtype float32"):
                    TorchGroupedView(
                        dataset,
                        normalization,
                        action_view="recovery_program_v1",
                    )

    def test_recovery_program_view_fails_without_feature_binding(self):
        dataset = closed_loop_dataset()
        normalization = NormalizationStats.fit(dataset)
        with self.assertRaisesRegex(ValueError, "manifest bindings"):
            TorchGroupedView(
                dataset, normalization, action_view="recovery_program_v1")

    def test_recovery_program_view_requires_exact_action_projection_binding(self):
        for field, delta in (
            ("init_qpos", 0.01),
            ("action_offset", 0.01),
            ("joint_min", -0.01),
            ("joint_max", 0.01),
        ):
            with self.subTest(field=field):
                dataset = closed_loop_dataset()
                bind_recovery_dataset(dataset)
                dataset.manifest["action_application_contract"][field][0] += delta
                if field in ("init_qpos", "action_offset"):
                    reproject_recovery_dataset(dataset)
                normalization = NormalizationStats.fit(dataset)
                with self.assertRaisesRegex(
                        ValueError, "differs elementwise"):
                    TorchGroupedView(
                        dataset,
                        normalization,
                        action_view="recovery_program_v1",
                    )

        for field, value, error in (
            ("projection", "clip_only", "projection semantics"),
            ("max_joint_delta", 0.04, "max_joint_delta semantics"),
            ("use_action_filter", True, "filter semantics"),
        ):
            with self.subTest(field=field):
                dataset = closed_loop_dataset()
                bind_recovery_dataset(dataset)
                dataset.manifest["action_application_contract"][field] = value
                normalization = NormalizationStats.fit(dataset)
                with self.assertRaisesRegex(ValueError, error):
                    TorchGroupedView(
                        dataset,
                        normalization,
                        action_view="recovery_program_v1",
                    )

    def test_recovery_program_view_requires_exact_action_contract_keyset(self):
        for mutation in ("missing_max_joint_delta", "missing_filter", "extra"):
            with self.subTest(mutation=mutation):
                dataset = closed_loop_dataset()
                bind_recovery_dataset(dataset)
                contract = dataset.manifest["action_application_contract"]
                if mutation == "missing_max_joint_delta":
                    del contract["max_joint_delta"]
                elif mutation == "missing_filter":
                    del contract["use_action_filter"]
                else:
                    contract["unregistered_projection_field"] = "forbidden"
                normalization = NormalizationStats.fit(dataset)
                with self.assertRaisesRegex(ValueError, "exact locked keyset"):
                    TorchGroupedView(
                        dataset,
                        normalization,
                        action_view="recovery_program_v1",
                    )

    def test_recovery_program_view_replays_every_application_tuple_exactly(self):
        dataset = closed_loop_dataset()
        bind_recovery_dataset(dataset)
        dataset.arrays["candidate_requested"][0, 1, 0] = np.nextafter(
            dataset.arrays["candidate_requested"][0, 1, 0],
            np.float32(1.0),
        )
        normalization = NormalizationStats.fit(dataset)
        with self.assertRaisesRegex(ValueError, "exact runtime action_to_qpos"):
            TorchGroupedView(
                dataset,
                normalization,
                action_view="recovery_program_v1",
            )

        dataset = closed_loop_dataset()
        bind_recovery_dataset(dataset)
        dataset.arrays["candidate_executed"][0, 1, 0] = np.nextafter(
            dataset.arrays["candidate_executed"][0, 1, 0],
            np.float32(1.0),
        )
        normalization = NormalizationStats.fit(dataset)
        with self.assertRaisesRegex(ValueError, "exact runtime qpos_to_action"):
            TorchGroupedView(
                dataset,
                normalization,
                action_view="recovery_program_v1",
            )

    def test_recovery_program_view_requires_all_k9_candidates(self):
        dataset = closed_loop_dataset()
        bind_recovery_dataset(dataset)
        dataset.arrays["candidate_mask"][0, -1] = False
        normalization = NormalizationStats.fit(dataset)

        with self.assertRaisesRegex(ValueError, "every locked K9"):
            TorchGroupedView(
                dataset,
                normalization,
                action_view="recovery_program_v1",
            )

    def test_recovery_program_view_forbids_ipw(self):
        dataset = closed_loop_dataset()
        bind_recovery_dataset(dataset)
        dataset.arrays["acceptance_probability"][0] = 0.5
        normalization = NormalizationStats.fit(dataset)

        with self.assertRaisesRegex(ValueError, "unit acceptance probability"):
            TorchGroupedView(
                dataset,
                normalization,
                action_view="recovery_program_v1",
            )

    def test_recovery_normalization_provenance_is_role_aware(self):
        training_data = closed_loop_dataset()
        bind_recovery_dataset(training_data)
        normalization = NormalizationStats.fit(training_data)

        changed_content = closed_loop_dataset()
        bind_recovery_dataset(changed_content)
        changed_content.arrays["obs_history"][0, 0, 0] += 0.01
        with self.assertRaisesRegex(
                ValueError, "matching this dataset content and split"):
            TorchGroupedView(
                changed_content,
                normalization,
                action_view="recovery_program_v1",
            )

        wrong_split_provenance = replace(
            normalization, fit_split="development_calibration")
        with self.assertRaisesRegex(
                ValueError, "matching this dataset content and split"):
            TorchGroupedView(
                training_data,
                wrong_split_provenance,
                action_view="recovery_program_v1",
            )

        calibration_data = closed_loop_dataset()
        bind_recovery_dataset(calibration_data)
        calibration_data.manifest["split"] = "development_calibration"
        with self.assertRaisesRegex(
                ValueError, "matching this dataset content and split"):
            TorchGroupedView(
                calibration_data,
                normalization,
                action_view="recovery_program_v1",
            )

        calibration = TorchGroupedView(
            calibration_data,
            normalization,
            action_view="recovery_program_v1",
            view_role="calibration",
        )
        self.assertEqual(calibration.view_role, "calibration")

        test_data = closed_loop_dataset()
        bind_recovery_dataset(test_data)
        test_data.manifest["split"] = "development_test"
        test = TorchGroupedView(
            test_data,
            normalization,
            action_view="recovery_program_v1",
            view_role="test",
        )
        self.assertEqual(test.view_role, "test")

        without_provenance = NormalizationStats(
            normalization.observation_mean,
            normalization.observation_std,
        )
        with self.assertRaisesRegex(ValueError, "content/split provenance"):
            TorchGroupedView(
                training_data,
                without_provenance,
                action_view="recovery_program_v1",
            )

    def test_recovery_training_recomputes_exact_normalization_values(self):
        dataset = closed_loop_dataset()
        bind_recovery_dataset(dataset)
        fitted = NormalizationStats.fit(dataset)
        forged_mean = replace(
            fitted,
            observation_mean=fitted.observation_mean
            + np.full(46, np.float32(1e-3), dtype=np.float32),
        )
        with self.assertRaisesRegex(
                ValueError, "exactly equal NormalizationStats.fit"):
            TorchGroupedView(
                dataset,
                forged_mean,
                action_view="recovery_program_v1",
                view_role="training",
            )

        forged_std = replace(
            fitted,
            observation_std=fitted.observation_std
            + np.full(46, np.float32(1e-3), dtype=np.float32),
        )
        with self.assertRaisesRegex(
                ValueError, "exactly equal NormalizationStats.fit"):
            TorchGroupedView(
                dataset,
                forged_std,
                action_view="recovery_program_v1",
                view_role="training",
            )


class LossAndCalibrationTest(unittest.TestCase):
    def test_nonfinite_hyperparameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            QSafeLossConfig(ranking_weight=float("nan"))
        with self.assertRaisesRegex(ValueError, "optimizer"):
            tiny_training_config(learning_rate=float("nan"))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            QSafeNetworkConfig(action_dim=12.5)

    def test_nonfinite_valid_group_loss_is_not_silently_dropped(self):
        model = SelectiveAdvantageQSafe(tiny_network_config(action_dim=12))
        observation, nominal, candidate = inputs(batch=3)
        output = model(observation, nominal, candidate)
        bad_logits = output.risk_logits.clone()
        bad_logits[0, 1] = float("nan")
        output = replace(output, risk_logits=bad_logits)
        fall, failure_step, max_tilt, min_height = outcome_tensors(3, 3)
        with self.assertRaisesRegex(ValueError, "non-finite loss"):
            qsafe_group_loss(
                output,
                fall=fall,
                first_failure_step=failure_step,
                max_tilt_rad=max_tilt,
                min_height_m=min_height,
                candidate_mask=torch.ones(3, 3, dtype=torch.bool),
                horizon_steps=32,
            )

    def test_empirical_target_temperature_nll_equals_replica_nll(self):
        logits = torch.tensor([[0.3, -0.8], [1.1, 0.2]])
        fall = torch.tensor([
            [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            [[1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
        ])
        mask = torch.ones(2, 2, dtype=torch.bool)
        weight = torch.tensor([0.4, 1.0])
        temperature = torch.tensor(1.7)
        compact = _temperature_loss(
            logits, temperature, fall.mean(dim=2), mask, weight)
        replica = F.binary_cross_entropy_with_logits(
            (logits / temperature)[..., None].expand_as(fall),
            fall,
            reduction="none",
        ).mean(dim=2).mean(dim=1)
        expected = torch.sum(replica * weight) / weight.sum()
        torch.testing.assert_close(compact, expected)


class RecoveryProgramV4TrainingContractTest(unittest.TestCase):
    def test_preregistered_defaults_are_exact(self):
        self.assertEqual(
            RECOVERY_PROGRAM_V4_NETWORK_CONFIG,
            QSafeNetworkConfig(
                observation_dim=46,
                history_frames=5,
                action_dim=82,
                frame_hidden_dim=128,
                state_hidden_dim=128,
                action_hidden_dim=128,
                privileged_dim=0,
                action_mode="selective_advantage",
            ),
        )
        self.assertEqual(
            RECOVERY_PROGRAM_V4_TRAINING_CONFIG,
            QSafeTrainingConfig(
                epochs=100,
                batch_size=64,
                learning_rate=3e-4,
                weight_decay=1e-5,
                gradient_clip_norm=5.0,
                ensemble_members=5,
                seed=20260810,
                device="cpu",
                calibration_steps=100,
            ),
        )
        self.assertEqual(
            RECOVERY_PROGRAM_V4_LOSS_CONFIG, QSafeLossConfig())
        self.assertEqual(QSafeTrainingConfig().seed, 20260810)

    def test_v4_rejects_any_nonexact_config_family(self):
        view = v4_recovery_view()
        cases = (
            (
                "network",
                replace(
                    RECOVERY_PROGRAM_V4_NETWORK_CONFIG,
                    action_hidden_dim=127,
                ),
                RECOVERY_PROGRAM_V4_TRAINING_CONFIG,
                RECOVERY_PROGRAM_V4_LOSS_CONFIG,
            ),
            (
                "training",
                RECOVERY_PROGRAM_V4_NETWORK_CONFIG,
                replace(RECOVERY_PROGRAM_V4_TRAINING_CONFIG, epochs=99),
                RECOVERY_PROGRAM_V4_LOSS_CONFIG,
            ),
            (
                "loss",
                RECOVERY_PROGRAM_V4_NETWORK_CONFIG,
                RECOVERY_PROGRAM_V4_TRAINING_CONFIG,
                replace(
                    RECOVERY_PROGRAM_V4_LOSS_CONFIG,
                    ranking_weight=0.25,
                ),
            ),
        )
        for name, network, training, loss in cases:
            with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, f"exact V4 {name} configuration"):
                train_qsafe_ensemble(view, network, training, loss)

    def test_v4_uses_deterministic_cpu_and_exact_member_metadata(self):
        view = v4_recovery_view()
        with mock.patch(
                "rl.qsafe.training.torch.set_num_threads") as set_threads, \
                mock.patch(
                    "rl.qsafe.training.torch.use_deterministic_algorithms",
                ) as deterministic, mock.patch(
                    "rl.qsafe.training.train_qsafe_member",
                    side_effect=synthetic_trained_member,
                ) as train_member:
            trained = train_qsafe_ensemble(
                view,
                RECOVERY_PROGRAM_V4_NETWORK_CONFIG,
                RECOVERY_PROGRAM_V4_TRAINING_CONFIG,
            )

        set_threads.assert_called_once_with(1)
        deterministic.assert_called_once_with(True)
        expected_seeds = [
            20260810 + RECOVERY_PROGRAM_V4_MEMBER_SEED_STRIDE * index
            for index in range(5)
        ]
        self.assertEqual(
            [call.kwargs["seed"] for call in train_member.call_args_list],
            expected_seeds,
        )
        self.assertEqual([member.seed for member in trained.members], expected_seeds)
        self.assertTrue(all(
            len(member.epoch_loss) == 100 for member in trained.members))
        self.assertEqual(
            trained.network_config, RECOVERY_PROGRAM_V4_NETWORK_CONFIG)
        self.assertEqual(
            trained.training_config, RECOVERY_PROGRAM_V4_TRAINING_CONFIG)
        self.assertEqual(trained.loss_config, RECOVERY_PROGRAM_V4_LOSS_CONFIG)

    def test_member_epoch_metadata_must_match_config(self):
        view = v4_recovery_view()

        def incomplete_member(*args, **kwargs):
            member = synthetic_trained_member(*args, **kwargs)
            member.epoch_loss.pop()
            return member

        with mock.patch(
                "rl.qsafe.training.train_qsafe_member",
                side_effect=incomplete_member), self.assertRaisesRegex(
                    ValueError, "epoch metadata"):
            train_qsafe_ensemble(
                view,
                RECOVERY_PROGRAM_V4_NETWORK_CONFIG,
                RECOVERY_PROGRAM_V4_TRAINING_CONFIG,
            )

    def test_optimizers_lock_algorithm_parameters(self):
        dataset, _ = synthetic_dataset()
        normalization = NormalizationStats.fit(dataset)
        view = TorchGroupedView(dataset, normalization)
        network = tiny_network_config()
        training = tiny_training_config(
            ensemble_members=1, calibration_steps=0)
        with mock.patch(
                "rl.qsafe.training.torch.optim.AdamW",
                wraps=torch.optim.AdamW) as adamw:
            member = train_qsafe_member(
                view,
                network,
                training,
                QSafeLossConfig(),
                seed=training.seed,
                bootstrap=False,
            )
        for name, expected in (
            ("betas", (0.9, 0.999)),
            ("eps", 1e-8),
            ("amsgrad", False),
            ("foreach", False),
            ("fused", False),
        ):
            self.assertEqual(adamw.call_args.kwargs[name], expected)

        with mock.patch(
                "rl.qsafe.training.torch.optim.Adam",
                wraps=torch.optim.Adam) as adam:
            fit_temperature(
                member.model,
                view,
                device="cpu",
                steps=1,
                batch_size=2,
            )
        for name, expected in (
            ("betas", (0.9, 0.999)),
            ("eps", 1e-8),
            ("amsgrad", False),
            ("foreach", False),
            ("fused", False),
        ):
            self.assertEqual(adam.call_args.kwargs[name], expected)


class EnsembleTrainingTest(unittest.TestCase):
    def test_training_calibration_and_prediction_smoke(self):
        train_data, _ = synthetic_dataset(split="development_train", offset=0)
        calibration_data, _ = synthetic_dataset(
            split="development_calibration", offset=1)
        normalization = NormalizationStats.fit(train_data)
        train = TorchGroupedView(train_data, normalization)
        calibration = TorchGroupedView(calibration_data, normalization)
        trained = train_qsafe_ensemble(
            train,
            tiny_network_config(),
            tiny_training_config(),
            calibration=calibration,
        )
        self.assertEqual(len(trained.members), 2)
        self.assertTrue(trained.normalization.equivalent_to(normalization))
        for member in trained.members:
            self.assertTrue(np.isfinite(member.temperature))
            self.assertGreater(member.temperature, 0.0)
        prediction = predict_qsafe_ensemble(
            trained, calibration, device="cpu", batch_size=2)
        self.assertEqual(prediction.shape, (4, 3))
        self.assertTrue(np.isfinite(prediction).all())

    def test_cross_split_identity_leak_is_rejected_before_training(self):
        train_data, _ = synthetic_dataset(split="development_train")
        leaked_data, _ = synthetic_dataset(split="development_calibration")
        normalization = NormalizationStats.fit(train_data)
        train = TorchGroupedView(train_data, normalization)
        leaked = TorchGroupedView(leaked_data, normalization)
        with self.assertRaisesRegex(ValueError, "leaks across"):
            train_qsafe_ensemble(
                train,
                tiny_network_config(),
                tiny_training_config(ensemble_members=1),
                calibration=leaked,
            )

    def test_command_speed_mismatch_is_rejected(self):
        train_data, _ = synthetic_dataset(split="development_train", offset=0)
        calibration_data, _ = synthetic_dataset(
            split="development_calibration", offset=1)
        calibration_data.arrays["command_vx"][:] = 0.33
        normalization = NormalizationStats.fit(train_data)
        train = TorchGroupedView(train_data, normalization)
        calibration = TorchGroupedView(calibration_data, normalization)
        with self.assertRaisesRegex(ValueError, "command speeds differ"):
            train_qsafe_ensemble(
                train,
                tiny_network_config(),
                tiny_training_config(ensemble_members=1),
                calibration=calibration,
            )

    def test_prediction_rejects_nontraining_normalization(self):
        train_data, _ = synthetic_dataset(split="development_train", offset=0)
        normalization = NormalizationStats.fit(train_data)
        train = TorchGroupedView(train_data, normalization)
        trained = train_qsafe_ensemble(
            train,
            tiny_network_config(),
            tiny_training_config(ensemble_members=1, calibration_steps=0),
        )
        changed = NormalizationStats(
            normalization.observation_mean + 0.01,
            normalization.observation_std,
        )
        changed_view = TorchGroupedView(train_data, changed)
        with self.assertRaisesRegex(ValueError, "train-fitted"):
            predict_qsafe_ensemble(trained, changed_view, device="cpu")

    def test_recovery_contract_is_carried_and_checked_for_prediction(self):
        dataset = closed_loop_dataset()
        recovery_binding, feature_manifest = bind_recovery_dataset(dataset)
        normalization = NormalizationStats.fit(dataset)
        train = TorchGroupedView(
            dataset, normalization, action_view="recovery_program_v1")
        with mock.patch(
                "rl.qsafe.training.train_qsafe_member",
                side_effect=synthetic_trained_member):
            trained = train_qsafe_ensemble(
                train,
                RECOVERY_PROGRAM_V4_NETWORK_CONFIG,
                RECOVERY_PROGRAM_V4_TRAINING_CONFIG,
            )
        self.assertEqual(trained.recovery_program_binding, recovery_binding)
        self.assertEqual(
            trained.recovery_program_feature_manifest, feature_manifest)
        self.assertEqual(
            trained.recovery_program_feature_contract_sha256,
            feature_manifest["feature_contract_sha256"],
        )
        prediction = predict_qsafe_ensemble(
            trained, train, device="cpu", batch_size=2)
        self.assertEqual(prediction.shape, (dataset.group_count, 9))

        tampered = TorchGroupedView(
            dataset, normalization, action_view="recovery_program_v1")
        assert tampered._recovery_program_binding is not None
        tampered._recovery_program_binding["fingerprint_sha256"] = "0" * 64
        with self.assertRaisesRegex(
                ValueError, "prediction/training recovery-program contracts"):
            predict_qsafe_ensemble(trained, tampered, device="cpu")

    def test_calibration_recovery_contract_must_match_training(self):
        train_data = closed_loop_dataset()
        bind_recovery_dataset(train_data)
        calibration_data = closed_loop_dataset()
        bind_recovery_dataset(calibration_data)
        calibration_data.manifest["split"] = "development_calibration"
        normalization = NormalizationStats.fit(train_data)
        train = TorchGroupedView(
            train_data, normalization, action_view="recovery_program_v1")
        calibration = TorchGroupedView(
            calibration_data,
            normalization,
            action_view="recovery_program_v1",
            view_role="calibration",
        )
        assert calibration._recovery_program_binding is not None
        calibration._recovery_program_binding[
            "fingerprint_sha256"] = "0" * 64
        with mock.patch(
                "rl.qsafe.training.audit_split_disjointness",
                return_value={}), self.assertRaisesRegex(
                    ValueError,
                    "train/calibration recovery-program contracts differ",
                ):
            train_qsafe_ensemble(
                train,
                RECOVERY_PROGRAM_V4_NETWORK_CONFIG,
                RECOVERY_PROGRAM_V4_TRAINING_CONFIG,
                calibration=calibration,
            )


class ClusterAtomicSamplingTest(unittest.TestCase):
    def test_trajectory_bootstrap_draws_complete_clusters(self):
        trajectory = np.asarray(["a", "a", "b", "b", "b", "c"])
        indices, sampled = trajectory_bootstrap_indices(trajectory, seed=8)
        cursor = 0
        for name in sampled:
            expected = np.flatnonzero(trajectory == name)
            np.testing.assert_array_equal(
                indices[cursor:cursor + len(expected)], expected)
            cursor += len(expected)
        self.assertEqual(cursor, len(indices))

    def test_nested_subsets_are_monotone_and_never_cut_clusters(self):
        trajectory = np.asarray(["a"] * 3 + ["b"] * 3 + ["c"] * 4)
        subsets = nested_trajectory_subsets(
            trajectory, [2, 5, np.int64(10)], seed=4)
        previous: set[int] = set()
        for subset in subsets:
            selected = set(map(int, subset.indices))
            self.assertTrue(previous.issubset(selected))
            for name in np.unique(trajectory[list(selected)]):
                expected = set(map(int, np.flatnonzero(trajectory == name)))
                self.assertTrue(expected.issubset(selected))
            previous = selected
        self.assertEqual(subsets[-1].actual_groups, 10)

    def test_nested_subsets_reject_mislabeled_curve_points(self):
        trajectory = np.asarray(["a"] * 5 + ["b"] * 5)
        with self.assertRaisesRegex(ValueError, "must be integers"):
            nested_trajectory_subsets(trajectory, [2.5], seed=1)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            nested_trajectory_subsets(trajectory, [11], seed=1)
        with self.assertRaisesRegex(ValueError, "adds no trajectory"):
            nested_trajectory_subsets(trajectory, [2, 4], seed=1)


if __name__ == "__main__":
    unittest.main()
