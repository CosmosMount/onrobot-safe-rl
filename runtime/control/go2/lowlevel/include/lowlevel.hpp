#pragma once

#include <array>
#include <string>

#include <unitree/idl/go2/LowCmd_.hpp>
#include "crc32.hpp"

namespace lowlevel
{
    struct config 
    {
        float kp = 60.f;
        float kd = 10.f;
        std::array<float, 12> init_qpos{};
        std::array<float, 12> joint_min{};
        std::array<float, 12> joint_max{};
        int policy_timeout_ms = 200;
        int policy_delay_ms = 0;
    };

    class cmd 
    {
    public:
        explicit cmd(const config& _config);

        void init(unitree_go::msg::dds_::LowCmd_& _cmd) const;
        void fill(unitree_go::msg::dds_::LowCmd_& _cmd,
                    const std::array<float, 12>& q_target) const;

        const std::array<float, 12>& init_qpos() const { return config_.init_qpos; }

    private:
        static constexpr double stopcmd_q = 2.146E+9;
        static constexpr double stopcmd_dq = 16000.0;
        float clip_joints(int i, float q) const;
        config config_;
    };

}