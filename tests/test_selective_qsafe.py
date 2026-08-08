from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import torch

from rl.qsafe.loss import QSafeLossConfig, qsafe_group_loss
from rl.qsafe.network import (
    QSafeEnsemble,
    QSafeNetworkConfig,
    SelectiveAdvantageQSafe,
)


def inputs(batch: int = 4, candidates: int = 3):
    generator = torch.Generator().manual_seed(7)
    observation = torch.randn(batch, 5, 46, generator=generator)
    nominal = torch.rand(batch, 12, generator=generator) * 0.4 - 0.2
    candidate = torch.rand(
        batch, candidates, 12, generator=generator) * 0.8 - 0.4
    candidate[:, 0] = nominal
    return observation, nominal, candidate


def outcome_tensors(batch: int, candidates: int, replicas: int = 4):
    fall = torch.zeros(batch, candidates, replicas)
    fall[:, 0, :2] = 1.0
    fall[:, 1, :1] = 1.0
    fall[:, 2, :3] = 1.0
    failure_step = torch.where(
        fall.bool(), torch.full_like(fall, 2, dtype=torch.int64),
        torch.full_like(fall, 33, dtype=torch.int64))
    max_tilt = torch.where(fall.bool(), 1.2, 0.1)
    min_height = torch.where(fall.bool(), 0.1, 0.3)
    return fall, failure_step, max_tilt, min_height


class SelectiveAdvantageNetworkTest(unittest.TestCase):
    def config(self, **kwargs):
        return QSafeNetworkConfig(
            frame_hidden_dim=16,
            state_hidden_dim=16,
            action_hidden_dim=16,
            **kwargs,
        )

    def test_shapes_finite_and_nominal_anchor_is_exact(self):
        model = SelectiveAdvantageQSafe(self.config())
        observation, nominal, candidate = inputs()
        output = model(observation, nominal, candidate)
        self.assertEqual(output.risk.shape, (4, 3))
        self.assertEqual(output.ttf_fraction.shape, (4,))
        self.assertTrue(bool(torch.all(torch.isfinite(output.risk))))
        torch.testing.assert_close(
            output.advantage_logit[:, 0], torch.zeros(4), rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            output.relative_risk[:, 0], torch.zeros(4), rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            output.risk[:, 0], output.state_risk, rtol=0.0, atol=0.0)

    def test_candidate_permutation_is_equivariant(self):
        model = SelectiveAdvantageQSafe(self.config()).eval()
        observation, nominal, candidate = inputs()
        permutation = torch.tensor([0, 2, 1])
        with torch.no_grad():
            first = model(observation, nominal, candidate).risk
            second = model(
                observation, nominal, candidate[:, permutation]).risk
        torch.testing.assert_close(second, first[:, permutation])

    def test_privileged_input_is_explicit_and_deployable_rejects_it(self):
        observation, nominal, candidate = inputs()
        deployable = SelectiveAdvantageQSafe(self.config())
        with self.assertRaisesRegex(ValueError, "must not receive"):
            deployable(
                observation, nominal, candidate,
                privileged_state=torch.zeros(4, 6))
        privileged = SelectiveAdvantageQSafe(self.config(privileged_dim=6))
        output = privileged(
            observation, nominal, candidate,
            privileged_state=torch.zeros(4, 6))
        self.assertEqual(output.risk.shape, (4, 3))

    def test_ensemble_reports_member_mean_std_and_benefit(self):
        torch.manual_seed(1)
        members = [SelectiveAdvantageQSafe(self.config()) for _ in range(3)]
        ensemble = QSafeEnsemble(members, temperatures=[1.0, 1.2, 0.8])
        observation, nominal, candidate = inputs()
        result = ensemble.predict(observation, nominal, candidate)
        self.assertEqual(result.member_risk.shape, (3, 4, 3))
        torch.testing.assert_close(
            result.risk_mean, result.member_risk.mean(dim=0))
        torch.testing.assert_close(
            result.member_benefit[..., 0], torch.zeros((3, 4)),
            rtol=0.0, atol=0.0)


class QSafeLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model = SelectiveAdvantageQSafe(QSafeNetworkConfig(
            frame_hidden_dim=16,
            state_hidden_dim=16,
            action_hidden_dim=16,
        ))
        self.observation, self.nominal, self.candidate = inputs(batch=3)
        self.outcomes = outcome_tensors(3, 3)

    def loss(self, output, mask=None, config=None):
        fall, failure_step, max_tilt, min_height = self.outcomes
        return qsafe_group_loss(
            output,
            fall=fall,
            first_failure_step=failure_step,
            max_tilt_rad=max_tilt,
            min_height_m=min_height,
            candidate_mask=(
                torch.ones(3, 3, dtype=torch.bool) if mask is None else mask),
            horizon_steps=32,
            config=config,
        )

    def test_masked_candidate_does_not_change_loss(self):
        mask = torch.tensor([[1, 1, 0]] * 3, dtype=torch.bool)
        first = self.model(self.observation, self.nominal, self.candidate)
        changed = self.candidate.clone()
        changed[:, 2] = 1.0
        second = self.model(self.observation, self.nominal, changed)
        first_loss = self.loss(first, mask=mask).total
        second_loss = self.loss(second, mask=mask).total
        torch.testing.assert_close(first_loss, second_loss, rtol=0.0, atol=0.0)

    def test_correct_ranking_has_lower_loss_than_reversed_ranking(self):
        output = self.model(self.observation, self.nominal, self.candidate)
        correct_logits = torch.tensor([
            [2.0, -2.0, 0.0],
            [2.0, -2.0, 0.0],
            [2.0, -2.0, 0.0],
        ])
        reversed_logits = -correct_logits
        config = QSafeLossConfig(
            absolute_risk_weight=0.0,
            state_risk_weight=0.0,
            relative_risk_weight=0.0,
            ranking_weight=1.0,
            ttf_weight=0.0,
            max_tilt_weight=0.0,
            min_height_weight=0.0,
        )
        correct = replace(
            output,
            risk_logits=correct_logits,
            risk=torch.sigmoid(correct_logits),
        )
        reversed_output = replace(
            output,
            risk_logits=reversed_logits,
            risk=torch.sigmoid(reversed_logits),
        )
        self.assertLess(
            float(self.loss(correct, config=config).ranking),
            float(self.loss(reversed_output, config=config).ranking),
        )

    def test_all_model_heads_receive_gradient(self):
        output = self.model(self.observation, self.nominal, self.candidate)
        result = self.loss(output)
        result.total.backward()
        modules = (
            self.model.frame_encoder,
            self.model.temporal_encoder,
            self.model.state_risk_head,
            self.model.action_head,
            self.model.auxiliary_head,
        )
        for module in modules:
            gradient = sum(
                float(parameter.grad.abs().sum())
                for parameter in module.parameters()
                if parameter.grad is not None)
            self.assertGreater(gradient, 0.0)

    def test_tiny_grouped_problem_overfits_action_order(self):
        torch.manual_seed(11)
        model = SelectiveAdvantageQSafe(QSafeNetworkConfig(
            frame_hidden_dim=12,
            state_hidden_dim=12,
            action_hidden_dim=12,
        ))
        batch = 12
        observation = torch.randn(batch, 5, 46) * 0.1
        nominal = torch.zeros(batch, 12)
        candidate = nominal[:, None, :].repeat(1, 3, 1)
        candidate[:, 1, 0] = -0.8
        candidate[:, 2, 0] = 0.8
        fall = torch.zeros(batch, 3, 4)
        fall[:, 0, :2] = 1.0
        fall[:, 2, :] = 1.0
        failure_step = torch.where(
            fall.bool(), torch.full_like(fall, 2, dtype=torch.int64),
            torch.full_like(fall, 33, dtype=torch.int64))
        max_tilt = torch.where(fall.bool(), 1.2, 0.1)
        min_height = torch.where(fall.bool(), 0.1, 0.3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_config = QSafeLossConfig(
            ttf_weight=0.0, max_tilt_weight=0.0, min_height_weight=0.0)
        for _ in range(120):
            output = model(observation, nominal, candidate)
            loss = qsafe_group_loss(
                output,
                fall=fall,
                first_failure_step=failure_step,
                max_tilt_rad=max_tilt,
                min_height_m=min_height,
                candidate_mask=torch.ones(batch, 3, dtype=torch.bool),
                horizon_steps=32,
                config=loss_config,
            ).total
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            risk = model(observation, nominal, candidate).risk.mean(dim=0)
        self.assertLess(float(risk[1]), float(risk[0]))
        self.assertLess(float(risk[0]), float(risk[2]))


if __name__ == "__main__":
    unittest.main()
