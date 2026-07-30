from __future__ import annotations

import unittest
from collections import deque

import jax
import numpy as np

from jaxrl.agents.sac.droq.learner import DroQLearner
from jaxrl.agents.safety_critic import SafetyCritic
from jaxrl.agents.sqrl import (double_critic_replacement_decision,
                               select_sqrl_action)
from jaxrl.env.specs import BoxSpec


class SqrlSelectionTest(unittest.TestCase):

    def _agent_and_safety(self):
        obs_spec = BoxSpec((5,), np.float32)
        action_spec = BoxSpec(
            (2,), np.float32, np.full(2, -1), np.full(2, 1))
        agent = DroQLearner.create(
            0, obs_spec, action_spec, hidden_dims=(16, 16), num_qs=2,
            critic_dropout_rate=None)
        safety = SafetyCritic.create(1, 5, 2, hidden_dims=(16, 16))
        return agent, safety

    def test_selects_bounded_action_and_reports_metrics(self):
        agent, safety = self._agent_and_safety()
        action, info, rng = select_sqrl_action(
            agent, safety, np.zeros(5, dtype=np.float32),
            jax.random.PRNGKey(2), num_candidates=32, epsilon_safe=0.99)
        self.assertEqual(action.shape, (2,))
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))
        self.assertGreaterEqual(info['sqrl_rejected_fraction'], 0.0)
        self.assertLessEqual(info['sqrl_rejected_fraction'], 1.0)
        self.assertGreaterEqual(info['selected_Q_safe'], 0.0)
        self.assertLessEqual(info['selected_Q_safe'], 1.0)
        self.assertGreaterEqual(info['candidate_Q_safe_range'], 0.0)
        self.assertGreaterEqual(info['candidate_Q_safe_std'], 0.0)
        self.assertIn('policy_safe_coverage', info)
        self.assertIn('local_safe_coverage', info)
        self.assertIn('structured_safe_coverage', info)
        self.assertIn('sqrl_support_coverage', info)
        self.assertIn('selected_behavior_log_prob_per_dim', info)
        self.assertIn('selected_nominal_action_distance', info)
        self.assertEqual(np.asarray(rng).shape, (2,))

    def test_rejects_invalid_candidate_count(self):
        with self.assertRaises(ValueError):
            select_sqrl_action(
                None, None, np.zeros(1), jax.random.PRNGKey(0),
                num_candidates=0)

    def test_safe_policy_mean_is_kept_non_invasively(self):
        agent, safety = self._agent_and_safety()
        observation = np.zeros(5, dtype=np.float32)
        expected = agent.eval_actions(observation)
        action, info, _ = select_sqrl_action(
            agent, safety, observation, jax.random.PRNGKey(22),
            num_candidates=16, epsilon_safe=1.0,
            candidate_noise_std=0.0,
            previous_action=np.full(2, 0.8, dtype=np.float32))
        np.testing.assert_allclose(action, expected, atol=1e-6)
        self.assertEqual(info['sqrl_selected_group'], 2.0)

    def test_empty_safe_set_uses_structured_or_emergency_fallback(self):
        agent, safety = self._agent_and_safety()
        _, info, _ = select_sqrl_action(
            agent, safety, np.zeros(5, dtype=np.float32),
            jax.random.PRNGKey(3), num_candidates=16,
            epsilon_safe=-0.1)
        self.assertEqual(info['sqrl_no_safe_candidate'], 1.0)
        self.assertEqual(info['sqrl_fallback_min_risk'], 0.0)
        self.assertEqual(
            info['sqrl_fallback_structured']
            + info['sqrl_emergency_supervisor'], 1.0)
        self.assertEqual(info['sqrl_rejected_fraction'], 1.0)

    def test_support_gate_abstains_instead_of_selecting_ood_candidate(self):
        agent, safety = self._agent_and_safety()
        previous = np.asarray([0.8, -0.6], dtype=np.float32)
        action, info, _ = select_sqrl_action(
            agent, safety, np.zeros(5, dtype=np.float32),
            jax.random.PRNGKey(33), num_candidates=16,
            epsilon_safe=1.0, previous_action=previous,
            support_gate_enabled=True,
            # No finite-density action can pass this synthetic threshold.
            min_behavior_log_prob_per_dim=1.0e6,
            max_nominal_action_distance=10.0,
            fallback_contraction=0.9)
        np.testing.assert_allclose(action, 0.9 * previous, atol=1e-6)
        self.assertEqual(info['sqrl_support_coverage'], 0.0)
        self.assertEqual(info['sqrl_unsupported_candidate_rate'], 1.0)
        self.assertEqual(info['sqrl_support_abstention'], 1.0)

    def test_independent_validator_rejects_selector_underestimate(self):
        # A calls candidate 1 much safer than nominal 0; independent B says
        # the candidate is worse. B must reject and abstain to index 2.
        result = double_critic_replacement_decision(
            np.asarray([0.8, 0.1, 0.7], dtype=np.float32),
            np.asarray([0.8, 0.9, 0.7], dtype=np.float32),
            np.asarray([True, True, True]),
            selected_index=1, nominal_index=0, abstain_index=2,
            epsilon_safe=0.95, improvement_margin=0.05)
        final_index, nominal_safe, replacement, reject, _, _ = result
        self.assertEqual(int(final_index), 0)
        # Nominal is safe to both critics, so it is retained non-invasively.
        self.assertTrue(bool(nominal_safe))
        self.assertFalse(bool(replacement))
        self.assertFalse(bool(reject))

        result = double_critic_replacement_decision(
            np.asarray([0.8, 0.1, 0.7], dtype=np.float32),
            np.asarray([0.8, 0.9, 0.7], dtype=np.float32),
            np.asarray([True, True, True]),
            selected_index=1, nominal_index=0, abstain_index=2,
            epsilon_safe=0.5, improvement_margin=0.05)
        final_index, nominal_safe, replacement, reject, _, _ = result
        self.assertEqual(int(final_index), 2)
        self.assertFalse(bool(nominal_safe))
        self.assertFalse(bool(replacement))
        self.assertTrue(bool(reject))

    def test_double_critic_uses_independent_parameter_trees(self):
        agent, critic_a = self._agent_and_safety()
        critic_b = SafetyCritic.create(
            20_001, 5, 2, hidden_dims=(16, 16))
        self.assertIsNot(critic_a.critic.params, critic_b.critic.params)
        self.assertFalse(all(
            np.array_equal(a, b)
            for a, b in zip(
                jax.tree_util.tree_leaves(critic_a.critic.params),
                jax.tree_util.tree_leaves(critic_b.critic.params))))
        _, info, _ = select_sqrl_action(
            agent, critic_a, np.zeros(5, dtype=np.float32),
            jax.random.PRNGKey(44), validation_critic=critic_b,
            num_candidates=8, epsilon_safe=1.0)
        self.assertEqual(info['sqrl_double_critic_enabled'], 1.0)
        self.assertIn('selected_Q_safe_B', info)
        self.assertIn('sqrl_validation_reject', info)


class SqrlActorUpdateTest(unittest.TestCase):

    def test_finetune_update_keeps_nonnegative_nu(self):
        obs_spec = BoxSpec((4,), np.float32)
        action_spec = BoxSpec(
            (2,), np.float32, np.full(2, -1), np.full(2, 1))
        agent = DroQLearner.create(
            0, obs_spec, action_spec, hidden_dims=(16, 16), num_qs=2,
            critic_dropout_rate=None)
        safety = SafetyCritic.create(1, 4, 2, hidden_dims=(16, 16))
        batch = {
            'observations': np.zeros((8, 4), dtype=np.float32),
            'actions': np.zeros((8, 2), dtype=np.float32),
            'rewards': np.zeros((8,), dtype=np.float32),
            'masks': np.ones((8,), dtype=np.float32),
            'next_observations': np.zeros((8, 4), dtype=np.float32),
        }
        new_agent, info = agent.update(
            batch, utd_ratio=1, safety_critic=safety,
            epsilon_safe=0.1, sqrl_use_lagrange=True)
        self.assertIn('sqrl_nu', info)
        self.assertGreaterEqual(float(info['sqrl_nu']), 0.0)
        nu = float(new_agent.safety_lagrange.apply_fn(
            {'params': new_agent.safety_lagrange.params}))
        self.assertGreaterEqual(nu, 0.0)


class SqrlGateTest(unittest.TestCase):

    @staticmethod
    def _ready_metrics():
        return {
            'Q_safe_AUROC': 0.90,
            'Q_safe_label_pos': 0.80,
            'Q_safe_label_neg': 0.10,
            'Q_safe_calibration_ece': 0.05,
            'Q_safe_brier': 0.08,
            'Q_safe_num_samples': 256.0,
            'Q_safe_positive_rate': 0.25,
        }

    @staticmethod
    def _ready_control_metrics():
        return {
            'control_pairwise_risk_ranking_accuracy': 0.9,
            'control_selected_false_safe_rate': 0.02,
            'control_coverage': 0.8,
            'control_nominal_relative_failure_reduction': 0.2,
        }

    def test_gate_requires_calibration_and_candidate_coverage(self):
        from train.config import load_app_config
        from train.loop import _sqrl_gate_decision

        _, cfg, _ = load_app_config(path='config/go2.yaml')
        no_safe = deque(
            [0.0] * cfg.sqrl_gate_candidate_window,
            maxlen=cfg.sqrl_gate_candidate_window)
        ranges = deque(
            [0.2] * cfg.sqrl_gate_candidate_window,
            maxlen=cfg.sqrl_gate_candidate_window)
        ready, reason = _sqrl_gate_decision(
            self._ready_metrics(), no_safe, ranges, cfg,
            self._ready_control_metrics())
        self.assertTrue(ready)
        self.assertEqual(reason, 'ready')

        saturated = self._ready_metrics()
        saturated['Q_safe_positive_rate'] = 1.0
        ready, reason = _sqrl_gate_decision(
            saturated, no_safe, ranges, cfg)
        self.assertFalse(ready)
        self.assertEqual(reason, 'calibration-missing-class')

        ready, reason = _sqrl_gate_decision(
            self._ready_metrics(),
            deque([1.0] * cfg.sqrl_gate_candidate_window),
            ranges, cfg, self._ready_control_metrics())
        self.assertFalse(ready)
        self.assertEqual(reason, 'no-safe-rate')

    def test_gate_blocks_uncalibrated_warm_start(self):
        from train.config import load_app_config
        from train.loop import _sqrl_gate_decision

        _, cfg, _ = load_app_config(path='config/go2.yaml')
        ready, reason = _sqrl_gate_decision(
            None, deque(), deque(), cfg)
        self.assertFalse(ready)
        self.assertEqual(reason, 'no-natural-calibration')

    def test_p15_prevalidated_gate_can_activate_on_first_step(self):
        from train.config import TrainConfig
        from train.loop import _sqrl_gate_decision

        cfg = TrainConfig()
        cfg.sqrl_prevalidated_control_gate = True
        ready, reason = _sqrl_gate_decision(
            None, deque(), deque(), cfg, {
                'protocol': 'P15',
                'p15_gate_passed': True,
            })
        self.assertTrue(ready)
        self.assertEqual(reason, 'ready:prevalidated-P15')
        ready, reason = _sqrl_gate_decision(
            None, deque(), deque(), cfg, {
                'protocol': 'P15',
                'p15_gate_passed': False,
            })
        self.assertFalse(ready)
        self.assertEqual(reason, 'prevalidated-gate-failed')
        ready, reason = _sqrl_gate_decision(
            None, deque(), deque(), cfg, {
                'protocol': 'P16',
                'p16_gate_passed': True,
            })
        self.assertTrue(ready)
        self.assertEqual(reason, 'ready:prevalidated-P16')

    def test_gate_rejects_perfect_natural_metrics_with_reversed_control_ranking(self):
        from train.config import load_app_config
        from train.loop import _sqrl_gate_decision

        _, cfg, _ = load_app_config(path='config/go2.yaml')
        no_safe = deque(
            [0.0] * cfg.sqrl_gate_candidate_window,
            maxlen=cfg.sqrl_gate_candidate_window)
        ranges = deque(
            [0.2] * cfg.sqrl_gate_candidate_window,
            maxlen=cfg.sqrl_gate_candidate_window)
        control = self._ready_control_metrics()
        control['control_pairwise_risk_ranking_accuracy'] = 0.0
        ready, reason = _sqrl_gate_decision(
            self._ready_metrics(), no_safe, ranges, cfg, control)
        self.assertFalse(ready)
        self.assertEqual(reason, 'control-pairwise-ranking')

    def test_gate_hysteresis_ignores_short_calibration_noise(self):
        from train.loop import _sqrl_gate_with_hysteresis

        ready, reason, latched, streak = _sqrl_gate_with_hysteresis(
            True, 'ready', False, 0, 3)
        self.assertTrue(ready)
        self.assertTrue(latched)
        self.assertEqual(streak, 0)

        ready, reason, latched, streak = _sqrl_gate_with_hysteresis(
            False, 'calibration-missing-class', latched, streak, 3)
        self.assertTrue(ready)
        self.assertEqual(reason, 'latched:calibration-missing-class')
        self.assertEqual(streak, 1)

        ready, _, latched, streak = _sqrl_gate_with_hysteresis(
            False, 'auroc', latched, streak, 3)
        self.assertTrue(ready)
        self.assertEqual(streak, 2)

        ready, reason, latched, streak = _sqrl_gate_with_hysteresis(
            False, 'auroc', latched, streak, 3)
        self.assertFalse(ready)
        self.assertFalse(latched)
        self.assertEqual(reason, 'revoked:auroc')
        self.assertEqual(streak, 0)

    def test_gate_hysteresis_cannot_latch_a_failed_initial_gate(self):
        from train.loop import _sqrl_gate_with_hysteresis

        ready, reason, latched, streak = _sqrl_gate_with_hysteresis(
            False, 'no-control-evaluation', False, 0, 64)
        self.assertFalse(ready)
        self.assertFalse(latched)
        self.assertEqual(reason, 'no-control-evaluation')
        self.assertEqual(streak, 0)


class SqrlFromScratchConfigTest(unittest.TestCase):

    def test_from_scratch_clears_warm_start(self):
        import argparse
        from train.config import load_app_config
        from train.main import _configure_sqrl_mode, apply_move_speed

        robot_cfg, train_cfg, droq_cfg = load_app_config(path='config/go2.yaml')
        robot_cfg = apply_move_speed(robot_cfg, 0.30)
        self.assertAlmostEqual(robot_cfg.move_speed, 0.30)
        ns = argparse.Namespace(
            mode='sqrl_pretrain', checkpoint=None,
            save_dir='saved/checkpoints_sqrl_transfer_pre',
            from_scratch=True)
        cfg, _ = _configure_sqrl_mode(ns, train_cfg, dict(droq_cfg))
        self.assertIsNone(cfg.warm_start_checkpoint)
        self.assertFalse(cfg.resume_checkpoint)
        self.assertEqual(cfg.experiment_name, 'sqrl_pretrain')
        self.assertTrue(cfg.sqrl_enabled)
        self.assertEqual(cfg.sqrl_phase, 'pretrain')

    def test_default_pretrain_still_warm_starts_sac(self):
        import argparse
        from train.config import load_app_config
        from train.main import _DEFAULT_SQRL_WARM_START, _configure_sqrl_mode

        _, train_cfg, droq_cfg = load_app_config(path='config/go2.yaml')
        ns = argparse.Namespace(
            mode='sqrl_pretrain', checkpoint=None, save_dir=None,
            from_scratch=False)
        cfg, _ = _configure_sqrl_mode(ns, train_cfg, dict(droq_cfg))
        self.assertEqual(cfg.warm_start_checkpoint, _DEFAULT_SQRL_WARM_START)
        self.assertTrue(cfg.resume_checkpoint)

    def test_from_scratch_rejected_on_finetune(self):
        import argparse
        from train.config import load_app_config
        from train.main import _configure_sqrl_mode

        _, train_cfg, droq_cfg = load_app_config(path='config/go2.yaml')
        ns = argparse.Namespace(
            mode='sqrl_finetune',
            checkpoint='saved/checkpoints_sqrl/training_snapshot_000000016584.pkl',
            save_dir='saved/checkpoints_sqrl_transfer_ft',
            from_scratch=True)
        with self.assertRaises(SystemExit):
            _configure_sqrl_mode(ns, train_cfg, dict(droq_cfg))


if __name__ == '__main__':
    unittest.main()
