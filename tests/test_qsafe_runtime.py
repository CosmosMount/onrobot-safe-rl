from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from rl.qsafe.artifact import LoadedQSafeArtifact
from rl.qsafe.data import NormalizationStats
from rl.qsafe.network import QSafeNetworkConfig
from rl.qsafe.runtime import run_qsafe_step
from rl.qsafe.selector import SelectorConfig
from safety_data.candidates import (
    CANDIDATE_COUNT,
    CANDIDATE_KINDS,
    CandidateSet,
    EvidenceCandidateConfig,
)


class _RecordingEnsemble(torch.nn.Module):
    def __init__(self, member_risk: np.ndarray):
        super().__init__()
        self.register_buffer(
            "fixed_member_risk",
            torch.as_tensor(member_risk, dtype=torch.float32),
        )
        self.seen_history: torch.Tensor | None = None
        self.seen_nominal: torch.Tensor | None = None
        self.seen_candidates: torch.Tensor | None = None

    def predict(self, history, nominal, candidates):
        self.seen_history = history.detach().cpu().clone()
        self.seen_nominal = nominal.detach().cpu().clone()
        self.seen_candidates = candidates.detach().cpu().clone()
        risk = self.fixed_member_risk[:, None, :].to(history)
        return SimpleNamespace(
            member_risk=risk,
            member_state_risk=risk[:, :, 0],
        )


def _member_risk() -> np.ndarray:
    risk = np.vstack([
        np.full(CANDIDATE_COUNT, 0.90),
        np.full(CANDIDATE_COUNT, 0.91),
    ])
    risk[:, 0] = (0.80, 0.82)
    risk[:, 1] = (0.10, 0.12)
    return risk


def _candidates(*, mask_candidate_one: bool = False) -> CandidateSet:
    requested = np.zeros((CANDIDATE_COUNT, 12), dtype=np.float32)
    requested[:, 0] = np.arange(CANDIDATE_COUNT, dtype=np.float32) / 20.0
    executed = requested * 0.8
    q_target = requested * 0.5
    mask = np.ones(CANDIDATE_COUNT, dtype=bool)
    mask[1] = not mask_candidate_one
    return CandidateSet(
        requested=requested,
        executed=executed,
        q_target=q_target,
        kind=np.asarray(CANDIDATE_KINDS),
        mask=mask,
        candidate_seed=17,
        manifest_protocol=EvidenceCandidateConfig().manifest_protocol(),
    )


def _artifact(ensemble: torch.nn.Module) -> LoadedQSafeArtifact:
    contract = {
        "view": "application_concat",
        "components_in_order": ["requested", "executed", "q_target"],
        "joint_width_per_component": 12,
        "total_width": 36,
    }
    return LoadedQSafeArtifact(
        ensemble=ensemble,  # type: ignore[arg-type]
        normalization=NormalizationStats(
            np.arange(46, dtype=np.float32),
            np.full(46, 2.0, dtype=np.float32),
        ),
        network_config=QSafeNetworkConfig(
            action_dim=36,
            frame_hidden_dim=4,
            state_hidden_dim=4,
            action_hidden_dim=4,
        ),
        action_view="application_concat",
        action_components=("requested", "executed", "q_target"),
        manifest={
            "feature_view": "deployable",
            "action_feature_contract": contract,
            "provenance": {
                "command_vx": 0.30,
                "action_feature_contract": {
                    "view": "application_concat",
                    "components_in_order": [
                        "requested", "executed", "q_target"],
                    "total_width": 36,
                },
            },
        },
        path=Path("/tmp/development-qsafe-test"),
    )


def _selector() -> SelectorConfig:
    return SelectorConfig(
        nominal_risk_lcb_trigger=0.5,
        min_benefit_lcb=0.1,
        max_risk_ucb=0.5,
        max_epistemic_std=0.2,
        max_action_delta_rms=1.0,
        max_q_target_delta_rms=1.0,
        reward_q_margin=0.5,
    )


class QSafeRuntimeTest(unittest.TestCase):
    def test_training_features_selection_and_rng_are_exact(self):
        ensemble = _RecordingEnsemble(_member_risk())
        artifact = _artifact(ensemble)
        history = np.arange(46, dtype=np.float32)[None, :] + 6.0
        history = np.repeat(history, 5, axis=0)
        candidates = _candidates()

        np.random.seed(23)
        torch.manual_seed(29)
        numpy_before = np.random.get_state()
        torch_before = torch.random.get_rng_state().clone()
        result = run_qsafe_step(
            artifact,
            history,
            candidates,
            np.zeros(CANDIDATE_COUNT),
            _selector(),
            expected_command_speed_mps=0.30,
        )
        numpy_after = np.random.get_state()
        torch_after = torch.random.get_rng_state()

        self.assertEqual(result.selected_index, 1)
        self.assertTrue(result.intervened)
        np.testing.assert_array_equal(
            result.selected_requested_action, candidates.requested[1])
        np.testing.assert_allclose(ensemble.seen_history.numpy(), 3.0)
        expected_features = np.concatenate([
            candidates.requested,
            candidates.executed,
            candidates.q_target,
        ], axis=1)
        np.testing.assert_array_equal(
            ensemble.seen_nominal.numpy()[0], expected_features[0])
        np.testing.assert_array_equal(
            ensemble.seen_candidates.numpy()[0], expected_features)
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        torch.testing.assert_close(torch_before, torch_after)
        self.assertFalse(result.selected_requested_action.flags.writeable)

    def test_mask_reaches_selector_and_malformed_masked_reward_is_rejected(self):
        ensemble = _RecordingEnsemble(_member_risk())
        artifact = _artifact(ensemble)
        candidates = _candidates(mask_candidate_one=True)
        history = np.repeat(
            np.arange(46, dtype=np.float32)[None, :], 5, axis=0)
        result = run_qsafe_step(
            artifact,
            history,
            candidates,
            np.zeros(CANDIDATE_COUNT),
            _selector(),
            expected_command_speed_mps=0.30,
        )
        self.assertEqual(result.selected_index, 0)
        self.assertEqual(result.selection.reason, "no_eligible")
        self.assertFalse(result.selection.support_gate[1])
        np.testing.assert_array_equal(
            ensemble.seen_candidates.numpy()[0, 1],
            ensemble.seen_nominal.numpy()[0],
        )

        bad_reward = np.zeros(CANDIDATE_COUNT)
        bad_reward[1] = np.nan
        with self.assertRaisesRegex(ValueError, "reward_q.*finite"):
            run_qsafe_step(
                artifact,
                history,
                candidates,
                bad_reward,
                _selector(),
                expected_command_speed_mps=0.30,
            )

    def test_rejects_speed_privilege_contract_and_shape_mismatches(self):
        artifact = _artifact(_RecordingEnsemble(_member_risk()))
        candidates = _candidates()
        history = np.zeros((5, 46), dtype=np.float32)
        reward_q = np.zeros(CANDIDATE_COUNT)

        with self.assertRaisesRegex(ValueError, "command speed mismatch"):
            run_qsafe_step(
                artifact, history, candidates, reward_q, _selector(),
                expected_command_speed_mps=0.31)
        privileged_manifest = dict(artifact.manifest)
        privileged_manifest["feature_view"] = "privileged_diagnostic_only"
        with self.assertRaisesRegex(ValueError, "privileged"):
            run_qsafe_step(
                replace(artifact, manifest=privileged_manifest),
                history, candidates, reward_q, _selector(),
                expected_command_speed_mps=0.30)
        with self.assertRaisesRegex(ValueError, "action contract"):
            run_qsafe_step(
                replace(
                    artifact,
                    action_view="requested",
                    action_components=("requested",),
                ),
                history, candidates, reward_q, _selector(),
                expected_command_speed_mps=0.30)
        with self.assertRaisesRegex(ValueError, "shape"):
            run_qsafe_step(
                artifact, history[:4], candidates, reward_q, _selector(),
                expected_command_speed_mps=0.30)


if __name__ == "__main__":
    unittest.main()
