#pragma once

#include <array>
#include <atomic>
#include <cstdint>

#include <unitree/idl/go2/LowState_.hpp>

struct low_state_snapshot
{
    unitree_go::msg::dds_::LowState_ low_state{};
    uint64_t receive_time_ns{0};
    uint64_t sequence{0};
};

class low_state_double_buffer
{
public:
    void write(const unitree_go::msg::dds_::LowState_& state,
               uint64_t receive_time_ns)
    {
        const int next = 1 - active_index_.load(std::memory_order_relaxed);
        buffers_[next].low_state = state;
        buffers_[next].receive_time_ns = receive_time_ns;
        buffers_[next].sequence = sequence_.fetch_add(1) + 1;
        active_index_.store(next, std::memory_order_release);
    }

    low_state_snapshot read() const
    {
        const int index = active_index_.load(std::memory_order_acquire);
        return buffers_[index];
    }

private:
    std::array<low_state_snapshot, 2> buffers_{};
    std::atomic<int> active_index_{0};
    std::atomic<uint64_t> sequence_{0};
};
