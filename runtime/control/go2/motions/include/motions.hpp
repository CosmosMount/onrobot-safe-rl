#pragma once

#include <algorithm>
#include <cmath>

namespace motions
{
    inline int seconds_to_ticks(float seconds, float control_hz)
    {
        return std::max(1, static_cast<int>(std::lround(seconds * control_hz)));
    }

    inline float clamp01(float v)
    {
        return std::max(0.f, std::min(1.f, v));
    }

    struct imu_thresholds
    {
        float upside_down_acc_z_on = -1.f;   // m/s², enter belly-up
        float upside_down_acc_z_off = 3.f;  // m/s², exit belly-up (Schmitt)
        float fallen_acc_z_off = 7.f;        // m/s², upright when acc_z above
        float upside_down_up_cos_on = -0.7f; // body-up dot world-up, enter belly-up
        float fallen_roll_pitch_limit_rad = 0.523599f; // upstream Run terminates at 30 deg
    };
}
