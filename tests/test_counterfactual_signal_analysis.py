from __future__ import annotations

import numpy as np

from safety_data.counterfactual_signal_analysis import (
    horizon_analysis, replica_scaling_analysis,
)


def test_replica_scaling_detects_stable_r16_not_noisy_r4() -> None:
    first = np.full((40, 16, 16), 97, np.int16)
    # R4 contains weak, contradictory half-sample orderings. Candidate 15 is
    # consistently risky in later replicas, so R16 becomes strong and stable.
    first[:, 15, 0] = 20
    first[:, 14, 2] = 20
    first[:, 15, 4:] = 20
    report = replica_scaling_analysis(first)
    assert report["R4"]["strong_pair_state_coverage"] == 0
    assert report["R16"]["strong_pair_state_coverage"] == 1
    assert report["r4_label_noise_likely"] is True


def test_horizon_labels_are_derived_from_first_fall_step() -> None:
    first = np.full((60, 16, 16), 97, np.int16)
    first[:, 8:, :] = 20
    strata = np.asarray(["boundary", "medium", "normal"] * 20)
    seeds = np.asarray([137, 138] * 30)
    report = horizon_analysis(first, strata, seeds)
    assert report["H16"]["overall"]["candidate_fall_risk_mean"] == 0
    assert report["H32"]["overall"]["candidate_fall_risk_mean"] == 0.5
    assert report["H96"]["overall"]["candidate_fall_risk_mean"] == 0.5
    assert report["H32"]["boundary"]["strong_pair_state_coverage"] == 1
