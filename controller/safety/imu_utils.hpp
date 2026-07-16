#pragma once

#include <cmath>

#include <unitree/idl/go2/LowState_.hpp>

struct imu_orientation_config
{
    float upside_down_acc_z_on = -3.f;   // m/s², enter belly-up
    float upside_down_acc_z_off = -1.f;  // m/s², exit belly-up (Schmitt)
    float fallen_acc_z_off = 7.f;        // m/s², upright when acc_z above
};

class belly_up_detector
{
public:
    explicit belly_up_detector(const imu_orientation_config& cfg) : cfg_(cfg) {}

    bool update(float acc_z)
    {
        if (!belly_up_ && acc_z < cfg_.upside_down_acc_z_on) {
            belly_up_ = true;
        } else if (belly_up_ && acc_z > cfg_.upside_down_acc_z_off) {
            belly_up_ = false;
        }
        return belly_up_;
    }

    bool belly_up() const { return belly_up_; }

private:
    imu_orientation_config cfg_;
    bool belly_up_{false};
};

namespace imu_utils
{

inline float gravity_acc_z(const unitree_go::msg::dds_::IMUState_& imu)
{
    return imu.accelerometer()[2];
}

inline bool is_upside_down(const unitree_go::msg::dds_::LowState_& state,
                           belly_up_detector& detector)
{
    return detector.update(gravity_acc_z(state.imu_state()));
}

inline bool is_upright(float acc_z, float fallen_acc_z_off)
{
    return acc_z > fallen_acc_z_off;
}

inline bool is_clearly_belly_up(float acc_z, float upside_down_acc_z_on)
{
    return acc_z < upside_down_acc_z_on;
}

}  // namespace imu_utils
