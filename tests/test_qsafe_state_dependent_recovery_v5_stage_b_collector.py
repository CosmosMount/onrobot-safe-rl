from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import SimpleNamespace
import unittest

import numpy as np

from runtime.inference.actions import ActionApplier
from safety_data.closed_loop_recovery_collector import (
    ClosedLoopRecoveryCollectionConfig,
)
from safety_data.schema import (
    CLOSED_LOOP_RECOVERY_BEHAVIOR_STEPS,
    CLOSED_LOOP_RECOVERY_CANDIDATE_KINDS,
    CLOSED_LOOP_RECOVERY_CANDIDATE_PROTOCOL_VERSION,
)
from safety_data.state_dependent_recovery_v5 import SEED_ROLE_TAGS
from safety_data.state_dependent_recovery_v5_stage_b import StageBExecutionError
from safety_data.state_dependent_recovery_v5_stage_b_collector import (
    COLLECTION_PROTOCOL_VERSION,
    collect_preflighted_stage_b_role,
    preflight_stage_b_role_collection,
    production_collection_config,
)


_INIT = np.asarray([0.05, 0.7, -1.4] * 4, dtype=np.float32)
_OFFSET = np.asarray([0.2, 0.4, 0.4] * 4, dtype=np.float32)
_LOWER = np.asarray([-1.05, -1.57, -2.72] * 4, dtype=np.float32)
_UPPER = np.asarray([1.05, 3.49, -0.84] * 4, dtype=np.float32)


@dataclass(frozen=True)
class _Snapshot:
    token: int

    def compound_sha256(self) -> str:
        return f"{self.token:064x}"


class _Policy:
    def __init__(self, events: list[str]):
        self.events = events

    def manifest(self) -> dict[str, object]:
        return {
            "training_step": 25_000,
            "policy_fingerprint_sha256": "stage-b-test-policy",
        }

    def fingerprint(self) -> str:
        return "stage-b-test-policy"

    def deterministic_action(self, observation: np.ndarray) -> np.ndarray:
        del observation
        self.events.append("deterministic")
        return np.zeros(12, dtype=np.float32)

    def sample_action(
        self, observation: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        del observation, rng
        self.events.append("sample")
        return np.zeros(12, dtype=np.float32)

    def __call__(
        self,
        history: np.ndarray,
        step: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del history, step, rng
        return np.zeros(12, dtype=np.float32)


class _Program:
    def __init__(self, applier: ActionApplier, events: list[str]):
        self.action_applier = applier
        self.events = events
        self.behavior_steps = np.asarray(
            CLOSED_LOOP_RECOVERY_BEHAVIOR_STEPS, dtype=np.int64
        )

    def manifest_protocol(self) -> dict[str, object]:
        return {
            "protocol_version": CLOSED_LOOP_RECOVERY_CANDIDATE_PROTOCOL_VERSION,
            "count": 9,
            "ordered_names": list(CLOSED_LOOP_RECOVERY_CANDIDATE_KINDS),
            "behavior_steps_array": "candidate_behavior_steps",
            "behavior_override_steps": list(CLOSED_LOOP_RECOVERY_BEHAVIOR_STEPS),
        }

    def manifest(self) -> dict[str, object]:
        return {
            "candidate_protocol": self.manifest_protocol(),
            "test_program": "stage-b-single-label",
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.manifest(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def __call__(
        self,
        candidate_index: int,
        history: np.ndarray,
        step: int,
        nominal_action: np.ndarray,
    ) -> np.ndarray:
        del candidate_index, history, step
        return np.asarray(nominal_action, dtype=np.float32).copy()

    def preview_projected(
        self, history: np.ndarray, nominal_action: np.ndarray
    ) -> SimpleNamespace:
        self.events.append("preview")
        requested = np.repeat(
            np.asarray(nominal_action, dtype=np.float32)[None, :], 9, axis=0
        )
        projected = self.action_applier.preview_many(
            requested, np.asarray(history[-1, :12], dtype=np.float32)
        )
        return SimpleNamespace(
            requested=np.stack([value.action_requested for value in projected]),
            executed=np.stack([value.action_executed for value in projected]),
            q_target=np.stack([value.action_q_target for value in projected]),
            kind=np.asarray(CLOSED_LOOP_RECOVERY_CANDIDATE_KINDS),
            mask=np.ones(9, dtype=bool),
            behavior_steps=self.behavior_steps,
        )


class _Env:
    def __init__(self, events: list[str]):
        self.events = events
        self.cfg = SimpleNamespace(
            num_joints=12,
            fallen_orientation_rad=0.523599,
            move_speed=0.30,
        )
        self.action_applier = ActionApplier(
            init_qpos=_INIT.copy(),
            action_offset=_OFFSET.copy(),
            joint_min=_LOWER.copy(),
            joint_max=_UPPER.copy(),
        )
        self.qpos_addresses = np.arange(7, 19)
        self.qvel_addresses = np.arange(6, 18)
        self.data = SimpleNamespace(
            qpos=np.concatenate([
                np.asarray([0.0, 0.0, 0.30, 1.0, 0.0, 0.0, 0.0]),
                _INIT.astype(np.float64),
            ]),
            qvel=np.zeros(18, dtype=np.float64),
            ncon=4,
        )
        self._history = self._make_history()

    @staticmethod
    def _make_history() -> np.ndarray:
        frame = np.zeros(46, dtype=np.float32)
        frame[:12] = _INIT
        frame[30] = 1.0
        frame[34:46] = _INIT
        return np.repeat(frame[None, :], 5, axis=0)

    def simulator_fingerprint(self) -> dict[str, object]:
        return {"backend": "fake-stage-b", "version": 1}

    def reset_standing(
        self, settle_seconds: float, rng: np.random.Generator
    ) -> None:
        del settle_seconds, rng
        self.events.append("reset")
        self._history = self._make_history()

    def apply_base_velocity_impulse(self, **kwargs: object) -> None:
        del kwargs

    def record_observation(self) -> np.ndarray:
        return self._history.copy()

    def observation_history(self) -> np.ndarray:
        return self._history.copy()

    def measurement(self) -> SimpleNamespace:
        return SimpleNamespace(
            failure=False, near_failure=True, tilt_rad=0.20, height_m=0.30
        )

    def capture(self) -> _Snapshot:
        return _Snapshot(1)

    def restore(self, snapshot: _Snapshot) -> None:
        del snapshot
        self._history = self._make_history()

    def step(self, action: np.ndarray) -> SimpleNamespace:
        self.events.append("step")
        projection = self.action_applier.project(action, _INIT)
        self._history[-1, 34:46] = projection.action_q_target
        return SimpleNamespace(
            application=SimpleNamespace(
                action_requested=projection.action_requested,
                action_executed=projection.action_executed,
                action_q_target=projection.action_q_target,
            ),
            failure=False,
            tilt_rad=0.20,
            height_m=0.30,
        )


def _tiny_config() -> ClosedLoopRecoveryCollectionConfig:
    return ClosedLoopRecoveryCollectionConfig(
        source_seed=8501,
        policy_training_step=25_000,
        policy_training_seed=43,
        target_groups=1,
        horizon_steps=50,
        admission_replicas=1,
        admission_min_falls=0,
        admission_max_falls=1,
        discovery_replicas=1,
        audit_replicas=1,
        max_episode_steps=10,
        max_proposals=1,
        max_trajectories=1,
        seed_domain=b"qsafe_state_dependent_recovery_v4_stage_b_fit_admission\0",
        seed_role_tags=SEED_ROLE_TAGS,
        seed_algorithm=(
            "high_bit_then_domain_low15_then_14_8_18_2_6_bitpack_v1"
        ),
        dataset_split_prefix="stage_b_test",
        collection_protocol_version=COLLECTION_PROTOCOL_VERSION,
        trajectory_id_prefix="stage-b-fit-test",
        explicit_filter_settings_in_action_contract=True,
    )


class StageBRoleCollectorTest(unittest.TestCase):
    def test_single_label_partition_after_admission(self) -> None:
        events: list[str] = []
        env = _Env(events)
        policy = _Policy(events)
        program = _Program(env.action_applier, events)
        prepared = preflight_stage_b_role_collection(
            role="fit",
            env=env,
            early_policy=policy,
            recovery_program=program,
            policy_set_manifest={"type": "stage-b-test-actor-bank"},
            config=_tiny_config(),
            generator_commit="test-clean-stage-b-commit",
            parent_protocol_sha256="a" * 64,
            parent_protocol_contract_sha256="b" * 64,
            production_contract=False,
        )
        self.assertEqual(events, [])
        result = collect_preflighted_stage_b_role(preflight=prepared)
        self.assertEqual(result.role, "fit")
        self.assertEqual(result.admission.validate()["accepted"], 1)
        self.assertEqual(result.labels.validate()["groups"], 1)
        self.assertEqual(result.labels.replica_count, 1)
        self.assertEqual(
            result.labels.manifest["collection_protocol"]["partition"],
            "label",
        )
        self.assertNotIn("preassigned_audit_crn_id", result.labels.arrays)
        self.assertGreater(events.index("preview"), events.index("deterministic"))
        self.assertEqual(events.count("step"), 500)
        self.assertEqual(result.trajectories, 1)

        admission_seeds = set(np.concatenate([
            np.asarray(result.admission[name]).reshape(-1)
            for name in (
                "admission_crn_id",
                "admission_rollout_seed",
                "admission_perturbation_seed",
            )
        ]).tolist())
        label_seeds = set(np.concatenate([
            np.asarray(result.labels[name]).reshape(-1)
            for name in (
                "crn_id", "rollout_seed", "perturbation_seed", "candidate_seed"
            )
        ]).tolist())
        self.assertFalse(admission_seeds & label_seeds)

    def test_production_builder_matches_frozen_source_assignment(self) -> None:
        config = production_collection_config(
            role="model_test",
            source_seed=8722,
            max_episode_steps=100,
            max_trajectories=2048,
            proposal_cooldown_steps=5,
            settle_seconds=0.04,
            source_impulse_interval_steps=10,
            source_linear_std_mps=1.0,
            source_angular_std_radps=4.0,
            proposal_min_tilt_rad=0.10,
            proposal_max_height_m=0.32,
        )
        self.assertEqual(config.policy_training_seed, 49)
        self.assertEqual(config.policy_training_step, 100_000)
        self.assertEqual(config.target_groups, 64)
        self.assertEqual(config.admission_replicas, 32)
        self.assertEqual(config.discovery_replicas, 64)
        self.assertEqual(config.max_proposals, 4096)

    def test_production_preflight_rejects_tiny_or_nearby_contract(self) -> None:
        events: list[str] = []
        env = _Env(events)
        with self.assertRaisesRegex(StageBExecutionError, "target_groups"):
            preflight_stage_b_role_collection(
                role="fit",
                env=env,
                early_policy=_Policy(events),
                recovery_program=_Program(env.action_applier, events),
                policy_set_manifest={"type": "stage-b-test-actor-bank"},
                config=_tiny_config(),
                generator_commit="test-clean-stage-b-commit",
                parent_protocol_sha256="a" * 64,
                parent_protocol_contract_sha256="b" * 64,
                production_contract=True,
            )
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
