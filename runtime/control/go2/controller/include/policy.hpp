#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

namespace control
{
    #pragma pack(push, 1)
    struct policy_packet_t
    {
        static constexpr uint8_t magicSOF = 0xA5;
        static constexpr uint8_t FLAG_STAND_UP = 0x01;
        static constexpr uint8_t FLAG_RECOVERY = 0x02;

        uint8_t SOF;
        uint8_t flags;
        double timestamp;
        float q_target[12];
    };
    #pragma pack(pop)

    class policy_receiver
    {
    public:
        explicit policy_receiver(std::string socket_path);
        ~policy_receiver();

        void start();
        void stop();

        bool get_latest_target(std::array<float, 12>& out, double& timestamp,
                            uint8_t& flags) const;
        bool has_fresh_target(int timeout_ms) const;
        uint8_t consume_pending_motion_flags();
        void clear_pending_motion_flags();

    private:
        void loop();

        std::string socket_path_;
        std::thread server_thread_;
        std::atomic<bool> running_{false};

        mutable std::mutex mutex_;
        std::array<float, 12> latest_target_{};
        double latest_timestamp_{0.0};
        uint8_t latest_flags_{0};
        uint8_t pending_motion_flags_{0};
        std::atomic<int64_t> last_update_ns_{0};

        int64_t get_now_ns() const 
        {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                .count();
        }
    };

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
}
