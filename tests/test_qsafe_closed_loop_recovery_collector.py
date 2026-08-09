from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from runtime.inference.actions import ActionApplier
from safety_data.closed_loop_recovery_collector import (
    AdmissionLedger,
    AdmissionPrivilegedView,
    ClosedLoopRecoveryCollectionConfig,
    collect_closed_loop_recovery_triage,
    merge_admission_ledgers,
    merge_admission_privileged_views,
    role_randomness,
)
from safety_data.merge import merge_grouped_shards, merge_privileged_shards
from safety_data.schema import (
    CLOSED_LOOP_RECOVERY_BEHAVIOR_STEPS,
    CLOSED_LOOP_RECOVERY_CANDIDATE_KINDS,
    CLOSED_LOOP_RECOVERY_CANDIDATE_PROTOCOL_VERSION,
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
    def __init__(self, training_step: int, events: list[str]):
        self.training_step = training_step
        self.events = events

    def manifest(self):
        return {
            "training_step": self.training_step,
            "policy_fingerprint_sha256": f"policy-{self.training_step}",
        }

    def fingerprint(self):
        return f"policy-{self.training_step}"

    def deterministic_action(self, observation):
        self.events.append("deterministic")
        return np.zeros(12, dtype=np.float32)

    def sample_action(self, observation, rng):
        del observation, rng
        self.events.append("sample")
        return np.zeros(12, dtype=np.float32)

    def __call__(self, history, step, rng):
        del history, step, rng
        self.events.append("continuation")
        return np.zeros(12, dtype=np.float32)


class _Program:
    def __init__(self, applier: ActionApplier, events: list[str]):
        self.action_applier = applier
        self.events = events
        self.behavior_steps = np.asarray(
            CLOSED_LOOP_RECOVERY_BEHAVIOR_STEPS, dtype=np.int64)

    def manifest_protocol(self):
        return {
            "protocol_version": CLOSED_LOOP_RECOVERY_CANDIDATE_PROTOCOL_VERSION,
            "count": 9,
            "ordered_names": list(CLOSED_LOOP_RECOVERY_CANDIDATE_KINDS),
            "behavior_steps_array": "candidate_behavior_steps",
            "behavior_override_steps": list(
                CLOSED_LOOP_RECOVERY_BEHAVIOR_STEPS),
        }

    def manifest(self):
        return {
            "candidate_protocol": self.manifest_protocol(),
            "test_program": "closed-loop-recovery",
        }

    def fingerprint(self):
        payload = json.dumps(
            self.manifest(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def __call__(self, candidate_index, history, step, nominal_action):
        del candidate_index, history, step
        return np.asarray(nominal_action, dtype=np.float32).copy()

    def preview_projected(self, history, nominal_action):
        self.events.append("preview")
        requested = np.repeat(
            np.asarray(nominal_action, dtype=np.float32)[None, :], 9, axis=0)
        projections = self.action_applier.preview_many(
            requested, np.asarray(history[-1, :12], dtype=np.float32))
        return SimpleNamespace(
            requested=np.stack([value.action_requested for value in projections]),
            executed=np.stack([value.action_executed for value in projections]),
            q_target=np.stack([value.action_q_target for value in projections]),
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
        self._snapshot_token = 1

    def _make_history(self):
        frame = np.zeros(46, dtype=np.float32)
        frame[:12] = _INIT
        frame[30] = 1.0
        frame[34:46] = _INIT
        return np.repeat(frame[None, :], 5, axis=0)

    def simulator_fingerprint(self):
        return {"backend": "fake-v3", "version": 1}

    def reset_standing(self, settle_seconds, rng):
        del settle_seconds, rng
        self.events.append("reset")
        self._history = self._make_history()

    def apply_base_velocity_impulse(self, **kwargs):
        del kwargs
        self.events.append("impulse")

    def record_observation(self):
        return self._history.copy()

    def observation_history(self):
        return self._history.copy()

    def measurement(self):
        return SimpleNamespace(
            failure=False,
            near_failure=True,
            tilt_rad=0.20,
            height_m=0.30,
        )

    def capture(self):
        return _Snapshot(self._snapshot_token)

    def restore(self, snapshot):
        del snapshot
        self._history = self._make_history()

    def step(self, action):
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


class ClosedLoopRecoveryCollectorTest(unittest.TestCase):
    def _collect(self, source_seed: int, snapshot_token: int = 1):
        events: list[str] = []
        env = _Env(events)
        env._snapshot_token = snapshot_token
        policy = _Policy(training_step=1, events=events)
        program = _Program(env.action_applier, events)
        result = collect_closed_loop_recovery_triage(
            env=env,
            early_policy=policy,
            recovery_program=program,
            policy_set_manifest={"type": "locked-test-policy-set"},
            config=ClosedLoopRecoveryCollectionConfig(
                source_seed=source_seed,
                policy_training_step=1,
                target_groups=1,
                horizon_steps=50,
                admission_replicas=1,
                admission_min_falls=0,
                admission_max_falls=1,
                discovery_replicas=1,
                audit_replicas=1,
                max_episode_steps=50,
                max_proposals=1,
                max_trajectories=1,
            ),
            generator_commit="test-clean-commit",
            protocol_sha256="a" * 64,
            protocol_contract_sha256="b" * 64,
        )
        return events, result

    def test_config_rejects_impossible_group_cap_and_int16_horizon(self):
        with self.assertRaisesRegex(ValueError, "max_proposals"):
            ClosedLoopRecoveryCollectionConfig(
                source_seed=7801,
                policy_training_step=1,
                target_groups=2,
                max_proposals=1,
            )
        with self.assertRaisesRegex(ValueError, "int16.max - 1"):
            ClosedLoopRecoveryCollectionConfig(
                source_seed=7801,
                policy_training_step=1,
                horizon_steps=np.iinfo(np.int16).max,
            )

    def test_role_seed_namespaces_are_pairwise_disjoint(self):
        values = {}
        for role in ("admission", "discovery", "audit"):
            bundle, randomness = role_randomness(
                source_seed=7801,
                proposal_index=3,
                replicas=4,
                role=role,
            )
            values[role] = set(np.concatenate([
                bundle.crn_id, bundle.rollout_seed,
                bundle.perturbation_seed,
                np.asarray([randomness.candidate_seed], dtype=np.uint64),
            ]).tolist())
        self.assertFalse(values["admission"] & values["discovery"])
        self.assertFalse(values["admission"] & values["audit"])
        self.assertFalse(values["discovery"] & values["audit"])

    def test_admission_manifest_binds_height_frame_and_sampling_cadence(self):
        _, result = self._collect(7801)
        contract = result.admission.manifest["fall_definition"]
        self.assertEqual(
            contract["height_reference"], "base_link_body_origin_world_z")
        self.assertEqual(contract["height_comparator"], "strict_less_than")
        self.assertEqual(
            contract["sampling_cadence"],
            "first_failing_50Hz_post_action_boundary_after_10_low_level_substeps",
        )
        result.admission.manifest["fall_definition"]["height_reference"] = (
            "imu_site_world_z")
        with self.assertRaisesRegex(ValueError, "sampling/reference"):
            result.admission.validate(verify_hash=False)

    def test_admission_precedes_candidate_outcomes_and_files_align(self):
        events, result = self._collect(7801)
        self.assertEqual(result.admission.validate()["accepted"], 1)
        self.assertEqual(result.discovery.validate()["groups"], 1)
        self.assertEqual(result.audit.validate()["groups"], 1)
        self.assertEqual(events.count("step"), 50 + 9 * 50 * 2)
        self.assertGreater(events.index("preview"), 49)
        np.testing.assert_array_equal(
            result.discovery["group_id"], result.audit["group_id"])
        self.assertFalse(np.array_equal(
            result.discovery["rollout_seed"], result.audit["rollout_seed"]))
        np.testing.assert_array_equal(
            result.discovery["preassigned_audit_crn_id"],
            result.audit["crn_id"])
        np.testing.assert_array_equal(
            result.discovery["preassigned_audit_rollout_seed"],
            result.audit["rollout_seed"])
        np.testing.assert_array_equal(
            result.discovery["preassigned_audit_perturbation_seed"],
            result.audit["perturbation_seed"])
        np.testing.assert_array_equal(
            result.discovery["preassigned_audit_candidate_seed"],
            result.audit["candidate_seed"])
        self.assertEqual(
            int(result.discovery["candidate_behavior_steps"][0, 0]), 0)
        self.assertEqual(result.trajectories, 1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger_path = result.admission.save(root / "admission.npz")
            privileged_path = result.admission_privileged.save(
                root / "admission-privileged.npz", result.admission)
            restored = AdmissionLedger.load(ledger_path)
            restored_privileged = AdmissionPrivilegedView.load(
                privileged_path, ledger=restored)
        self.assertEqual(restored.validate()["accepted"], 1)
        self.assertEqual(restored_privileged.validate(restored)["proposals"], 1)

    def test_admission_merge_reindexes_and_preserves_privileged_link(self):
        _, first = self._collect(7801, snapshot_token=1)
        _, second = self._collect(7802, snapshot_token=2)
        merged = merge_admission_ledgers([
            first.admission, second.admission])
        merged_privileged = merge_admission_privileged_views(
            [first.admission_privileged, second.admission_privileged],
            [first.admission, second.admission],
            merged,
        )
        report = merged.validate()
        self.assertEqual(report["proposals"], 2)
        self.assertEqual(report["accepted"], 2)
        np.testing.assert_array_equal(
            merged["proposal_index"], np.asarray([0, 1]))
        np.testing.assert_array_equal(
            merged["accepted_group_index"], np.asarray([0, 1]))
        self.assertEqual(
            merged_privileged.validate(merged)["proposals"], 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            leaf_datasets = []
            leaf_views = []
            for index, result in enumerate((first, second)):
                dataset_path = result.discovery.save(
                    root / f"discovery-{index}.npz")
                view_path = result.discovery_privileged.save(
                    root / f"discovery-{index}.privileged.npz")
                dataset = type(result.discovery).load(dataset_path)
                view = type(result.discovery_privileged).load(
                    view_path, deployable=dataset)
                leaf_datasets.append(dataset)
                leaf_views.append(view)
            merged_discovery = merge_grouped_shards(leaf_datasets)
            merged_discovery_privileged = merge_privileged_shards(
                leaf_views, leaf_datasets, merged_discovery)
        self.assertEqual(merged_discovery.validate()["groups"], 2)
        self.assertEqual(
            merged_discovery_privileged.validate(
                merged_discovery)["groups"], 2)


if __name__ == "__main__":
    unittest.main()
