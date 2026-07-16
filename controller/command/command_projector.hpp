#pragma once

#include <array>
#include <algorithm>
#include <cstddef>

class command_projector
{
public:
    command_projector(std::array<float, 12> joint_min,
                      std::array<float, 12> joint_max)
        : joint_min_(joint_min), joint_max_(joint_max)
    {}

    std::array<float, 12> project(const std::array<float, 12>& q) const
    {
        std::array<float, 12> out{};
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] = std::max(joint_min_[i], std::min(joint_max_[i], q[i]));
        }
        return out;
    }

private:
    std::array<float, 12> joint_min_{};
    std::array<float, 12> joint_max_{};
};
