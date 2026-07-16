#pragma once

#include <cstdint>

class policy_scheduler
{
public:
    policy_scheduler(float control_hz, float policy_hz)
        : ticks_per_policy_step_(
              static_cast<uint32_t>(control_hz / policy_hz + 0.5f))
    {
        if (ticks_per_policy_step_ == 0) {
            ticks_per_policy_step_ = 1;
        }
    }

    bool tick()
    {
        ++control_tick_;
        if (ticks_since_policy_ + 1 >= ticks_per_policy_step_) {
            ticks_since_policy_ = 0;
            ++policy_sequence_;
            return true;
        }
        ++ticks_since_policy_;
        return false;
    }

    uint64_t policy_sequence() const { return policy_sequence_; }
    uint32_t ticks_per_policy_step() const { return ticks_per_policy_step_; }

private:
    uint64_t control_tick_{0};
    uint64_t policy_sequence_{0};
    uint32_t ticks_per_policy_step_{25};
    uint32_t ticks_since_policy_{0};
};
