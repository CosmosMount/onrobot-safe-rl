#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

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
