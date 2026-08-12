import numpy as np
import hashlib

from safety_data.short_option_analysis import (
    Bootstrap, analyze_short_option_oracle, validate_short_option_dataset,
)


def _inputs():
    shape = (600, 16, 8)
    fall = np.ones(shape, bool)
    # L1 offers no rescue. Direction zero at L4 and L8 is consistently safe.
    fall[:, 6, :] = False
    fall[:, 11, :] = False
    duration = np.broadcast_to(
        np.asarray([0] + [1] * 5 + [4] * 5 + [8] * 5), (600, 16))
    zeros = np.zeros(shape, np.float32)
    ones = np.ones(shape, np.float32)
    return dict(
        h96_fall=fall, candidate_duration=duration,
        collector_seed=np.repeat([137, 138], 300),
        replacement_sum=ones, replacement_max=ones,
        projection_saturation_count=zeros,
        joint_limit_saturation_count=zeros,
        active_steps=ones,
        max_abs_roll=zeros, max_abs_pitch=zeros,
        max_angular_velocity=zeros, min_base_height=ones,
    )


def test_long_options_pass_independent_oracle_and_timescale_gate() -> None:
    report = analyze_short_option_oracle(
        **_inputs(), bootstrap=Bootstrap(replicates=100, seed=4))
    assert report["duration_families"]["L4"][
        "independent_oracle_reduction"]["mean"] == 1.0
    assert report["duration_families"]["L4"]["rescue_states"] == 600
    assert report["short_option_candidate_space_supported"] is True
    assert report["one_step_action_timescale_insufficient"] is True


def test_analysis_rejects_wrong_replica_shape() -> None:
    values = _inputs()
    values["h96_fall"] = values["h96_fall"][:, :, :7]
    try:
        analyze_short_option_oracle(
            **values, bootstrap=Bootstrap(replicates=10, seed=4))
    except ValueError as error:
        assert "[600,16,8]" in str(error)
    else:
        raise AssertionError("wrong replica count was accepted")


def test_dataset_validation_enforces_paired_crn_and_exact_residuals() -> None:
    shape = (600, 16, 8)
    state_id = np.asarray([f"{value:064x}" for value in range(600)], "S64")
    episode_id = np.asarray([f"{value + 1000:064x}" for value in range(600)], "S64")
    direction = np.broadcast_to(np.asarray([-1] + list(range(5)) * 3), (600, 16))
    duration = np.broadcast_to(np.asarray([0] + [1] * 5 + [4] * 5 + [8] * 5),
                               (600, 16))
    action = np.zeros((600, 16, 12), np.float32)
    for candidate in range(1, 16):
        action[:, candidate, (candidate - 1) % 5] = (candidate - 1) % 5 + 1
    residual = action - action[:, :1]
    residual[:, 6:11] = residual[:, 1:6]
    residual[:, 11:16] = residual[:, 1:6]
    action[:, 6:11] = action[:, :1] + residual[:, 6:11]
    action[:, 11:16] = action[:, :1] + residual[:, 11:16]
    crn = np.asarray([[
        hashlib.sha256(
            b"qsafe.short-option.crn.id.v1\0" + bytes(state_id[state])
            + replica.to_bytes(2, "little")).hexdigest()
        for replica in range(8)] for state in range(600)], "S64")
    crn = np.broadcast_to(crn[:, None, :], shape)
    zeros = np.zeros(shape, np.float32)
    arrays = {
        "state_id": state_id, "episode_id": episode_id,
        "collector_seed": np.repeat([137, 138], 300),
        "candidate_index": np.broadcast_to(np.arange(16), (600, 16)),
        "candidate_duration": duration, "candidate_direction": direction,
        "critic_action": action, "residual": residual,
        "replica_id": np.broadcast_to(np.arange(1, 9), shape), "crn_id": crn,
        "h96_fall": np.zeros(shape, bool),
        "first_fall_step": np.full(shape, 97, np.int16),
        "replacement_magnitude_sum": zeros,
        "replacement_magnitude_max": zeros,
        "projection_saturation_count": zeros,
        "joint_limit_saturation_count": zeros,
        "option_active_steps_executed": np.ones(shape, np.int8),
        "option_max_abs_roll": zeros, "option_max_abs_pitch": zeros,
        "option_max_angular_velocity": zeros,
        "option_min_base_height": np.ones(shape, np.float32),
    }
    report = validate_short_option_dataset(arrays)
    assert report["branches"] == 76_800
    broken = dict(arrays)
    broken["crn_id"] = crn.copy()
    broken["crn_id"][0, 2, 0] = b"bad"
    try:
        validate_short_option_dataset(broken)
    except ValueError as error:
        assert "paired CRN" in str(error)
    else:
        raise AssertionError("candidate-specific CRN was accepted")
