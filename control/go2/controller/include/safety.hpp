#pragma once

#include <cmath>

#include <unitree/idl/go2/LowState_.hpp>

#include "motions.hpp"

namespace control::safety
{
    struct orientation_sample
    {
        float roll{0.f};
        float pitch{0.f};
        float up_cos{1.f};
        float acc_z{9.8f};
    };

    inline orientation_sample measure(
        const unitree_go::msg::dds_::LowState_& state)
    {
        orientation_sample out;
        const auto& q = state.imu_state().quaternion();
        float w = q[0];
        float x = q[1];
        float y = q[2];
        float z = q[3];
        const float norm_sq = w * w + x * x + y * y + z * z;
        if (norm_sq >= 1e-6f)
        {
            const float inv_norm = 1.0f / std::sqrt(norm_sq);
            w *= inv_norm;
            x *= inv_norm;
            y *= inv_norm;
            z *= inv_norm;
            out.roll = std::atan2(
                2.0f * (w * x + y * z),
                1.0f - 2.0f * (x * x + y * y));
            const float sinp = 2.0f * (w * y - z * x);
            constexpr float half_pi = 1.57079632679f;
            out.pitch = std::abs(sinp) >= 1.0f
                ? std::copysign(half_pi, sinp)
                : std::asin(sinp);
            out.up_cos = 1.0f - 2.0f * (x * x + y * y);
        }
        out.acc_z = state.imu_state().accelerometer()[2];
        return out;
    }

    inline bool is_fallen(
        const orientation_sample& sample,
        const motions::imu_thresholds& thresholds)
    {
        return std::abs(sample.roll) >
                   thresholds.fallen_roll_pitch_limit_rad ||
               std::abs(sample.pitch) >
                   thresholds.fallen_roll_pitch_limit_rad;
    }

    inline bool is_upside_down(
        const orientation_sample& sample,
        const motions::imu_thresholds& thresholds)
    {
        const bool pose_inverted =
            sample.up_cos < thresholds.upside_down_up_cos_on;
        const bool accel_inverted =
            sample.acc_z < thresholds.upside_down_acc_z_on &&
            sample.up_cos < 0.0f;
        return pose_inverted || accel_inverted;
    }
}
