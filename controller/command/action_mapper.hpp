#pragma once

#include <array>
#include <algorithm>
#include <cstddef>

class action_mapper
{
public:
    action_mapper(std::array<float, 12> init_qpos,
                  std::array<float, 12> action_offset,
                  std::array<float, 12> joint_min,
                  std::array<float, 12> joint_max)
        : init_qpos_(init_qpos),
          action_offset_(action_offset),
          joint_min_(joint_min),
          joint_max_(joint_max)
    {}

    std::array<float, 12> to_q_target(
        const std::array<float, 12>& action) const
    {
        std::array<float, 12> out{};
        for (size_t i = 0; i < out.size(); ++i) {
            const float clipped_action = std::max(-1.0f, std::min(1.0f, action[i]));
            const float q = init_qpos_[i] + clipped_action * action_offset_[i];
            out[i] = std::max(joint_min_[i], std::min(joint_max_[i], q));
        }
        return out;
    }

private:
    std::array<float, 12> init_qpos_{};
    std::array<float, 12> action_offset_{};
    std::array<float, 12> joint_min_{};
    std::array<float, 12> joint_max_{};
};
