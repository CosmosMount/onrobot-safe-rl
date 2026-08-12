import numpy as np

from safety_data.short_option_analysis import Bootstrap, analyze_short_option_oracle


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
