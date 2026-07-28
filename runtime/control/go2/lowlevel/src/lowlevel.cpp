#include "lowlevel.hpp"

namespace lowlevel
{
    cmd::cmd(const config& _config) : config_(config) {}

    void cmd::init(unitree_go::msg::dds_::LowCmd_& _cmd) const 
    {
        _cmd.head()[0] = 0xFE;
        _cmd.head()[1] = 0xEF;
        _cmd.level_flag() = 0xFF;
        _cmd.gpio() = 0;

        for (int i = 0; i < 20; i++) 
        {
            _cmd._motor_cmd()[i].mode() = 0x01;
            _cmd._motor_cmd()[i].q() = _stopcmd_q;
            _cmd._motor_cmd()[i].kp() = 0;
            _cmd._motor_cmd()[i].dq() = _stopcmd_dq;
            _cmd._motor_cmd()[i].kd() = 0;
            _cmd._motor_cmd()[i].tau() = 0;
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
        for (int i = 0; i < 12; i++)
        {
            _cmd._motor_cmd()[i].mode() = 0x01;
            _cmd._motor_cmd()[i].q() = clip_joints(i, q_target[i]);
            _cmd._motor_cmd()[i].dq() = 0.f;
            _cmd._motor_cmd()[i].kp() = kp;
            _cmd._motor_cmd()[i].kd() = kd;
            _cmd._motor_cmd()[i].tau() = 0.f;
        }
        _cmd.crc() = crc32_core(reinterpret_cast<uint32_t*>(&_cmd),
                            (sizeof(_cmd) >> 2) - 1);
    }

}