from __future__ import annotations

from pathlib import Path

import yaml


def test_natural_ppo_state_risk_protocol_has_disjoint_sac_roles():
    path = Path("config/qsafe_natural_ppo_state_risk_v1.yaml")
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    roles = protocol["sac_natural_state_roles"]
    names = (
        "probability_calibration", "uncertainty_calibration",
        "selector_and_recovery_calibration", "protected_model_test",
    )
    seeds = [set(roles[name]["actor_seeds"]) for name in names]
    assert all(seeds[i].isdisjoint(seeds[j])
               for i in range(len(seeds)) for j in range(i + 1, len(seeds)))
    assert roles["checkpoint_steps"] == [25000, 50000, 100000]
    protected = roles["protected_model_test"]
    assert protected["exposure_policy_steps_per_source"] == 15000
    assert protected["minimum_snapshot_spacing_policy_steps"] > protocol[
        "target"]["risk_horizon_policy_steps"]
    assert protocol["target"]["fall_predicate"] == {
        "minimum_base_link_origin_height_m": 0.18,
        "maximum_abs_roll_or_pitch_rad": 1.047198,
    }
    recovery = protocol["fixed_nonpolicy_recovery_candidates"]
    assert recovery["allowed_k9_indices"] == [4, 5, 6, 7, 8]
    assert recovery["mature_actor_options_allowed"] is False
    assert recovery["original_getup_motion_allowed_as_prevention"] is False
