#pragma once

#include <cstdint>

#include "dds_bridge.hpp"
#include "state_snapshot.hpp"

struct robot_state_snapshot
{
    low_state_snapshot low{};
    state_freshness freshness{};
};

class state_synchronizer
{
public:
    void write_low_state(const unitree_go::msg::dds_::LowState_& state,
                         uint64_t receive_time_ns)
    {
        low_buffer_.write(state, receive_time_ns);
    }

    robot_state_snapshot read(uint64_t now_ns) const
    {
        robot_state_snapshot snapshot{};
        snapshot.low = low_buffer_.read();
        if (snapshot.low.receive_time_ns > 0 && now_ns >= snapshot.low.receive_time_ns) {
            snapshot.freshness.lowstate_age_us =
                (now_ns - snapshot.low.receive_time_ns) / 1000ULL;
        } else {
            snapshot.freshness.stale_mask |= 0x1;
        }
        return snapshot;
    }

private:
    low_state_double_buffer low_buffer_{};
};
