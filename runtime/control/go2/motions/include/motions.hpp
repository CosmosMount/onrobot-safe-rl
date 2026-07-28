#pragma once

namespace motions
{
    int seconds_to_ticks(float seconds, float control_hz)
    {
        return std::max(1, static_cast<int>(std::lround(seconds * control_hz)));
    }

    float clamp01(float v)
    {
        return std::max(0.f, std::min(1.f, v));
    }
}