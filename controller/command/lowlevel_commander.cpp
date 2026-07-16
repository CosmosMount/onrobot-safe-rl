#include "lowlevel_commander.hpp"

lowlevel_commander::lowlevel_commander(const control_config& config) : config_(config) {}

void lowlevel_commander::init(unitree_go::msg::dds_::LowCmd_& cmd) const 
{
    cmd.head()[0] = 0xFE;
    cmd.head()[1] = 0xEF;
    cmd.level_flag() = 0xFF;
    cmd.gpio() = 0;

    for (int i = 0; i < 20; i++) 
    {
        cmd.motor_cmd()[i].mode() = 0x01;
        cmd.motor_cmd()[i].q() = stopcmd_q;
        cmd.motor_cmd()[i].kp() = 0;
        cmd.motor_cmd()[i].dq() = stopcmd_dq;
        cmd.motor_cmd()[i].kd() = 0;
        cmd.motor_cmd()[i].tau() = 0;
    }
}

float lowlevel_commander::clip_joints(int i, float q) const 
{
    if (q < config_.joint_min[i]) {
        return config_.joint_min[i];
    }
    if (q > config_.joint_max[i]) {
        return config_.joint_max[i];
    }
    return q;
}

void lowlevel_commander::fill_cmd(unitree_go::msg::dds_::LowCmd_& cmd,
                                  const std::array<float, 12>& q_target) const
{
    fill_cmd(cmd, q_target, config_.kp, config_.kd);
}

void lowlevel_commander::fill_cmd(unitree_go::msg::dds_::LowCmd_& cmd,
                                  const std::array<float, 12>& q_target,
                                  float kp,
                                  float kd) const
{
    for (int i = 0; i < 12; i++)
    {
        cmd.motor_cmd()[i].mode() = 0x01;
        cmd.motor_cmd()[i].q() = clip_joints(i, q_target[i]);
        cmd.motor_cmd()[i].dq() = 0.f;
        cmd.motor_cmd()[i].kp() = kp;
        cmd.motor_cmd()[i].kd() = kd;
        cmd.motor_cmd()[i].tau() = 0.f;
    }
    cmd.crc() = crc32_core(reinterpret_cast<uint32_t*>(&cmd),
                           (sizeof(cmd) >> 2) - 1);
}
