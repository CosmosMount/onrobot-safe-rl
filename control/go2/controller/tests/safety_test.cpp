#include <cassert>

#include "safety.hpp"

int main()
{
    motions::imu_thresholds thresholds;
    thresholds.upside_down_acc_z_on = -3.0f;

    control::safety::orientation_sample upright{};
    assert(!control::safety::is_fallen(upright, thresholds));
    assert(!control::safety::is_upside_down(upright, thresholds));

    control::safety::orientation_sample side_fall{};
    side_fall.roll = 0.7f;
    side_fall.up_cos = 0.75f;
    side_fall.acc_z = 7.0f;
    assert(control::safety::is_fallen(side_fall, thresholds));
    assert(!control::safety::is_upside_down(side_fall, thresholds));

    control::safety::orientation_sample belly_up{};
    belly_up.roll = 3.14159f;
    belly_up.up_cos = -1.0f;
    belly_up.acc_z = -9.8f;
    assert(control::safety::is_fallen(belly_up, thresholds));
    assert(control::safety::is_upside_down(belly_up, thresholds));
}
