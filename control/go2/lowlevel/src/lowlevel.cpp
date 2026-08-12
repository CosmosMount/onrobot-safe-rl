#include "lowlevel.hpp"

namespace lowlevel
{
    cmd::cmd(const config& _config) : config_(_config) {}

    void cmd::init(unitree_go::msg::dds_::LowCmd_& _cmd) const 
    {
        _cmd.head()[0] = 0xFE;
        _cmd.head()[1] = 0xEF;
        _cmd.level_flag() = 0xFF;
        _cmd.gpio() = 0;

        for (int i = 0; i < 20; i++) 
        {
            _cmd.motor_cmd()[i].mode() = 0x01;
            _cmd.motor_cmd()[i].q() = stopcmd_q;
            _cmd.motor_cmd()[i].kp() = 0;
            _cmd.motor_cmd()[i].dq() = stopcmd_dq;
            _cmd.motor_cmd()[i].kd() = 0;
            _cmd.motor_cmd()[i].tau() = 0;
        }
    }

    float cmd::clip_joints(int i, float q) const 
    {
        if (q < config_.joint_min[i]) {
            return config_.joint_min[i];
        }
        if (q > config_.joint_max[i]) {
            return config_.joint_max[i];
        }
        return q;
    }

    void cmd::fill(unitree_go::msg::dds_::LowCmd_& _cmd,
                                    const std::array<float, 12>& q_target) const
    {
        for (int policy_index = 0; policy_index < 12; policy_index++)
        {
            const int motor_index = go2_layout::kPolicyToMotorIndex[policy_index];
            _cmd.motor_cmd()[motor_index].mode() = 0x01;
            _cmd.motor_cmd()[motor_index].q() = clip_joints(motor_index, q_target[policy_index]);
            _cmd.motor_cmd()[motor_index].dq() = 0.f;
            _cmd.motor_cmd()[motor_index].kp() = config_.kp;
            _cmd.motor_cmd()[motor_index].kd() = config_.kd;
            _cmd.motor_cmd()[motor_index].tau() = 0.f;
        }
        _cmd.crc() = crc32_core(reinterpret_cast<uint32_t*>(&_cmd),
                            (sizeof(_cmd) >> 2) - 1);
    }

}
