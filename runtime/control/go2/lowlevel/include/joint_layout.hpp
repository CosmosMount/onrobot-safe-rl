#pragma once

#include <array>

namespace go2_layout
{
    // Policy and Unitree LowCmd/LowState order are currently identical.
    // Keep the direction in the name so a future hardware permutation cannot
    // be introduced as an unnamed index in a control loop.
    constexpr std::array<int, 12> kPolicyToMotorIndex = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    constexpr std::array<int, 12> kMotorToPolicyIndex = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
}
