from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from rl.qsafe.data import (
    NormalizationStats,
    TorchGroupedView,
    trajectory_bootstrap_indices,
)
from rl.qsafe.loss import QSafeLossConfig, qsafe_group_loss
from rl.qsafe.network import QSafeNetworkConfig, SelectiveAdvantageQSafe
from rl.qsafe.training import (
    QSafeTrainingConfig,
    _temperature_loss,
    predict_qsafe_ensemble,
    train_qsafe_ensemble,
)
from safety_data.splits import nested_trajectory_subsets
from safety_data.recovery_options import (
    RECOVERY_OPTION_KINDS,
    RECOVERY_OPTION_STEPS,
    RecoveryOptionCandidateConfig,
)
from tests.test_safety_data import synthetic_dataset
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
